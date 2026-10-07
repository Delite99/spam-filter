"""Hugging Face Transformer (DistilBERT/MiniLM) classifier for SpamGuard."""
import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List, Union
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import numpy as np

from ..config import TransformerConfig, DEFAULT_CONFIG
from ..data.preprocessor import TextPreprocessor

logger = logging.getLogger(__name__)

class TransformerSpamClassifier:
    """Wrapper around Hugging Face Transformer for email spam classification."""

    def __init__(
        self,
        model_name_or_path: str = "distilbert-base-uncased",
        num_labels: int = 2,
        device: Optional[torch.device] = None,
    ):
        self.model_name_or_path = model_name_or_path
        self.num_labels = num_labels
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None

    def initialize_new_model(self):
        """Initialize a new transformer model and tokenizer from Hugging Face hub."""
        logger.info(f"Loading pretrained tokenizer & model: {self.model_name_or_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name_or_path,
            num_labels=self.num_labels,
            id2label={0: "ham", 1: "spam"},
            label2id={"ham": 0, "spam": 1},
        )
        self.model.to(self.device)
        return self

    def predict_proba(self, texts: List[str], max_length: int = 256, batch_size: int = 32) -> np.ndarray:
        """Compute predicted spam probability for a list of texts."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model or tokenizer is not loaded.")
        self.model.eval()

        all_probs = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                all_probs.extend(probs)

        return np.array(all_probs)

    def predict(self, texts: List[str], threshold: float = 0.5, max_length: int = 256) -> np.ndarray:
        """Predict binary labels (1 = spam, 0 = ham)."""
        probs = self.predict_proba(texts, max_length=max_length)
        return (probs >= threshold).astype(int)

    def save_pretrained(self, save_directory: Union[str, Path], metadata: Optional[Dict[str, Any]] = None):
        """Save model weights, tokenizer, and metadata to directory."""
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        if self.model is not None:
            self.model.save_pretrained(save_path)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(save_path)
        if metadata:
            with open(save_path / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
        logger.info(f"Saved transformer model to {save_path}")

    @classmethod
    def from_pretrained(cls, save_directory: Union[str, Path], device: Optional[torch.device] = None) -> "TransformerSpamClassifier":
        """Load fine-tuned transformer and tokenizer from disk."""
        path = Path(save_directory)
        target_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        instance = cls(model_name_or_path=str(path), device=target_device)
        instance.tokenizer = AutoTokenizer.from_pretrained(path)
        instance.model = AutoModelForSequenceClassification.from_pretrained(path)
        instance.model.to(target_device)
        instance.model.eval()
        return instance
