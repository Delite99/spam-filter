"""Configuration module for SpamGuard."""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Tuple

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
STATIC_DIR = BASE_DIR / "src" / "api" / "static"

for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

@dataclass
class PreprocessingConfig:
    lowercase: bool = True
    normalize_urls: bool = True
    normalize_emails: bool = True
    normalize_numbers: bool = True
    normalize_currency: bool = True
    strip_html: bool = True
    remove_special_chars: bool = True
    min_length: int = 3
    max_length: int = 512

@dataclass
class BaselineConfig:
    max_features: int = 10000
    ngram_range: Tuple[int, int] = (1, 2)
    sublinear_tf: bool = True
    random_state: int = 42

@dataclass
class LSTMConfig:
    vocab_size: int = 25000
    embedding_dim: int = 128
    hidden_dim: int = 128
    num_layers: int = 2
    bidirectional: bool = True
    dropout: float = 0.3
    max_seq_length: int = 256
    batch_size: int = 64
    learning_rate: float = 0.001
    epochs: int = 10
    early_stopping_patience: int = 3

@dataclass
class TransformerConfig:
    pretrained_name: str = "distilbert-base-uncased"
    max_seq_length: int = 256
    batch_size: int = 16
    learning_rate: float = 2e-5
    epochs: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    early_stopping_patience: int = 2

@dataclass
class Config:
    seed: int = 42
    test_size: float = 0.15
    val_size: float = 0.15
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    lstm: LSTMConfig = field(default_factory=LSTMConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    default_model_name: str = "distilbert"

DEFAULT_CONFIG = Config()
