"""Model architectures and baseline classifiers for SpamGuard."""
from .baseline import BaselineSpamClassifier

__all__ = ["BaselineSpamClassifier"]

def get_bilstm_classifier():
    from .lstm_model import BiLSTMSpamClassifier
    return BiLSTMSpamClassifier

def get_transformer_classifier():
    from .transformer_model import TransformerSpamClassifier
    return TransformerSpamClassifier
