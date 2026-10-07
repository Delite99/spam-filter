"""PyTorch Bidirectional LSTM with Self-Attention for SpamGuard."""
import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ..config import LSTMConfig, DEFAULT_CONFIG
from ..data.dataset import TokenizerVocab
from ..data.preprocessor import TextPreprocessor

logger = logging.getLogger(__name__)

class AdditiveAttention(nn.Module):
    """Additive Self-Attention mechanism (Bahdanau-style) for sequence pooling."""

    def __init__(self, hidden_dim: int, attention_dim: int = 64):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1, bias=False)
        )

    def forward(self, rnn_outputs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            rnn_outputs: Tensor of shape [batch_size, seq_len, hidden_dim]
            mask: Optional boolean or float mask of shape [batch_size, seq_len], where 1 is valid and 0 is padding.
        Returns:
            context_vector: Pooled representation [batch_size, hidden_dim]
            weights: Attention weights [batch_size, seq_len]
        """
        # [batch_size, seq_len, 1]
        scores = self.projection(rnn_outputs)
        scores = scores.squeeze(-1) # [batch_size, seq_len]

        if mask is not None:
            # Mask out padding tokens with large negative number
            scores = scores.masked_fill(~mask, -1e9)

        weights = F.softmax(scores, dim=-1) # [batch_size, seq_len]
        weights_expanded = weights.unsqueeze(-1) # [batch_size, seq_len, 1]
        context = torch.sum(rnn_outputs * weights_expanded, dim=1) # [batch_size, hidden_dim]
        return context, weights


class BiLSTMSpamClassifier(nn.Module):
    """Bi-directional LSTM with Self-Attention classifier."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.3,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.pad_idx = pad_idx

        # Embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_idx
        )
        self.embed_dropout = nn.Dropout(dropout)

        # BiLSTM layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        direction_factor = 2 if bidirectional else 1
        lstm_out_dim = hidden_dim * direction_factor

        # Attention layer
        self.attention = AdditiveAttention(hidden_dim=lstm_out_dim, attention_dim=64)

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, input_ids: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids: [batch_size, seq_len]
            lengths: [batch_size] length of each sequence before padding
        Returns:
            logits: [batch_size, 1]
            attention_weights: [batch_size, seq_len]
        """
        # Create padding mask (True for real tokens, False for pad_idx)
        mask = (input_ids != self.pad_idx)

        # Embedding lookup
        embeds = self.embed_dropout(self.embedding(input_ids)) # [batch, seq_len, embed_dim]

        # LSTM forward pass
        lstm_out, _ = self.lstm(embeds) # [batch, seq_len, lstm_out_dim]

        # Self-attention pooling
        context, attn_weights = self.attention(lstm_out, mask=mask)

        # Classification logit
        logits = self.classifier(context) # [batch, 1]
        return logits, attn_weights

    def predict_proba_single(
        self,
        text: str,
        vocab: TokenizerVocab,
        preprocessor: TextPreprocessor,
        device: torch.device,
        max_length: int = 256,
    ) -> Tuple[float, List[Tuple[str, float]]]:
        """Inference on a single text string; returns probability and attention weights per word."""
        self.eval()
        cleaned = preprocessor.clean_text(text)
        words = cleaned.split()[:max_length]
        token_ids, valid_len = vocab.encode(cleaned, max_length=max_length)
        
        inp = torch.tensor([token_ids], dtype=torch.long, device=device)
        lengths = torch.tensor([valid_len], dtype=torch.long, device=device)

        with torch.no_grad():
            logit, attn_weights = self.forward(inp, lengths)
            prob = float(torch.sigmoid(logit).item())
            weights = attn_weights[0][:len(words)].cpu().numpy().tolist()

        # Normalize attention over existing words for display
        sum_w = sum(weights) or 1.0
        normalized_w = [round(w / sum_w, 4) for w in weights]
        word_attentions = list(zip(words, normalized_w))

        return prob, word_attentions

    def save(self, filepath: Union[str, Path], metadata: Optional[Dict[str, Any]] = None):
        """Save PyTorch weights and model configuration."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": {
                "vocab_size": self.vocab_size,
                "embedding_dim": self.embedding_dim,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "bidirectional": self.bidirectional,
                "pad_idx": self.pad_idx,
            },
            "metadata": metadata or {}
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved BiLSTM checkpoint to {path}")

    @classmethod
    def load(cls, filepath: Union[str, Path], device: Optional[torch.device] = None) -> Tuple["BiLSTMSpamClassifier", Dict[str, Any]]:
        """Load trained BiLSTM model from disk."""
        target_device = device or torch.device("cpu")
        checkpoint = torch.load(filepath, map_location=target_device)
        cfg = checkpoint["config"]
        model = cls(
            vocab_size=cfg["vocab_size"],
            embedding_dim=cfg["embedding_dim"],
            hidden_dim=cfg["hidden_dim"],
            num_layers=cfg["num_layers"],
            bidirectional=cfg["bidirectional"],
            pad_idx=cfg["pad_idx"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(target_device)
        model.eval()
        return model, checkpoint.get("metadata", {})
