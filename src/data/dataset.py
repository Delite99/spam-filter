"""PyTorch Dataset, Tokenizer, and DataModule for SpamGuard."""
import json
import collections
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from ..config import DEFAULT_CONFIG, PROCESSED_DATA_DIR

class TokenizerVocab:
    """Vocabulary and tokenizer for from-scratch PyTorch models."""
    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    CLS_TOKEN = "<cls>"
    SEP_TOKEN = "<sep>"

    def __init__(self, max_vocab_size: int = 25000, min_freq: int = 2):
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self.word2id: Dict[str, int] = {}
        self.id2word: Dict[int, str] = {}
        self._init_special_tokens()

    def _init_special_tokens(self):
        for idx, token in enumerate([self.PAD_TOKEN, self.UNK_TOKEN, self.CLS_TOKEN, self.SEP_TOKEN]):
            self.word2id[token] = idx
            self.id2word[idx] = token

    @property
    def pad_id(self) -> int:
        return self.word2id[self.PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.word2id[self.UNK_TOKEN]

    def __len__(self) -> int:
        return len(self.word2id)

    def build_vocab(self, texts: List[str]) -> "TokenizerVocab":
        """Build vocabulary from a collection of raw/cleaned texts."""
        counter: collections.Counter = collections.Counter()
        for text in texts:
            if not isinstance(text, str):
                text = "" if text is None or (isinstance(text, float) and np.isnan(text)) else str(text)
            tokens = text.lower().split()
            counter.update(tokens)

        # Sort by frequency
        most_common = counter.most_common()
        for word, count in most_common:
            if count < self.min_freq:
                break
            if len(self.word2id) >= self.max_vocab_size:
                break
            if word not in self.word2id:
                new_id = len(self.word2id)
                self.word2id[word] = new_id
                self.id2word[new_id] = word
        return self

    def encode(self, text: str, max_length: int = 256) -> Tuple[List[int], int]:
        """Convert text string into token id sequence with padding/truncation."""
        if not isinstance(text, str):
            text = "" if text is None or (isinstance(text, float) and np.isnan(text)) else str(text)
        tokens = text.lower().split()
        seq = [self.word2id.get(w, self.unk_id) for w in tokens]
        original_len = min(len(seq), max_length)
        if len(seq) < max_length:
            seq = seq + [self.pad_id] * (max_length - len(seq))
        else:
            seq = seq[:max_length]
        return seq, max(1, original_len)

    def save(self, file_path: Union[str, Path]):
        """Serialize vocabulary to JSON."""
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump({"word2id": self.word2id, "max_vocab_size": self.max_vocab_size}, f, indent=2)

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> "TokenizerVocab":
        """Load vocabulary from JSON."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab = cls(max_vocab_size=data.get("max_vocab_size", 25000))
        vocab.word2id = data["word2id"]
        vocab.id2word = {int(v): k for k, v in vocab.word2id.items()}
        return vocab


class SpamDataModule:
    """Manages train, validation, and test datasets and splits."""

    def __init__(
        self,
        data_path: Optional[Path] = None,
        test_size: float = 0.15,
        val_size: float = 0.15,
        random_state: int = 42,
    ):
        self.data_path = data_path or (PROCESSED_DATA_DIR / "spam_corpus.csv")
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.df: Optional[pd.DataFrame] = None
        self.train_df: Optional[pd.DataFrame] = None
        self.val_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None
        self.vocab: Optional[TokenizerVocab] = None

    def prepare_data(self) -> "SpamDataModule":
        """Load data and create stratified train / val / test splits."""
        if not self.data_path.exists():
            from .download import ingest_and_prepare_dataset
            self.df = ingest_and_prepare_dataset()
        else:
            self.df = pd.read_csv(self.data_path)

        # Sanitize texts and drop empty / NaN rows
        self.df["text"] = self.df["text"].fillna("").astype(str)
        if "cleaned_text" not in self.df.columns:
            from .preprocessor import TextPreprocessor
            prep = TextPreprocessor()
            self.df["cleaned_text"] = prep.clean_series(self.df["text"].tolist())
        else:
            self.df["cleaned_text"] = self.df["cleaned_text"].fillna("").astype(str)

        # Ensure non-empty cleaned texts
        self.df = self.df[self.df["cleaned_text"].str.strip().str.len() > 1].reset_index(drop=True)

        # Stratified train / test split
        train_val_df, test_df = train_test_split(
            self.df,
            test_size=self.test_size,
            stratify=self.df["label"],
            random_state=self.random_state,
        )


        # Stratified train / validation split
        relative_val_size = self.val_size / (1.0 - self.test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=relative_val_size,
            stratify=train_val_df["label"],
            random_state=self.random_state,
        )

        self.train_df = train_df.reset_index(drop=True)
        self.val_df = val_df.reset_index(drop=True)
        self.test_df = test_df.reset_index(drop=True)

        return self

    def build_vocab(self, max_vocab_size: int = 25000) -> TokenizerVocab:
        """Build tokenizer vocabulary on the training split only."""
        if self.train_df is None:
            self.prepare_data()
        self.vocab = TokenizerVocab(max_vocab_size=max_vocab_size)
        texts = self.train_df["cleaned_text"].dropna().tolist()
        self.vocab.build_vocab(texts)
        return self.vocab

    def get_class_weights(self) -> float:
        """Compute positive class weight (neg_count / pos_count) for BCEWithLogitsLoss."""
        if self.train_df is None:
            self.prepare_data()
        labels = self.train_df["label"].values
        pos_count = (labels == 1).sum()
        neg_count = (labels == 0).sum()
        return float(neg_count / max(1, pos_count))
