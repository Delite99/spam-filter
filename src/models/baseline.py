"""Classical ML baselines for SpamGuard."""
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from ..config import BaselineConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)

class BaselineSpamClassifier:
    """TF-IDF vectorizer paired with MultinomialNB or LogisticRegression."""

    def __init__(self, model_type: str = "logistic_regression", config: Optional[BaselineConfig] = None):
        self.model_type = model_type.lower()
        self.config = config or DEFAULT_CONFIG.baseline
        
        self.vectorizer = TfidfVectorizer(
            max_features=self.config.max_features,
            ngram_range=self.config.ngram_range,
            sublinear_tf=self.config.sublinear_tf,
        )

        if self.model_type in ["nb", "naive_bayes", "multinomial_nb"]:
            self.model = MultinomialNB(alpha=0.1)
        else:
            self.model = LogisticRegression(
                C=2.0,
                max_iter=1000,
                random_state=self.config.random_state,
                class_weight="balanced"
            )
        self.is_fitted = False

    def fit(self, texts: list, labels: list) -> "BaselineSpamClassifier":
        """Train the TF-IDF vectorizer and classifier."""
        clean_texts = ["" if t is None or (isinstance(t, float) and np.isnan(t)) else str(t) for t in texts]
        logger.info(f"Fitting TF-IDF vectorizer on {len(clean_texts)} samples...")
        X = self.vectorizer.fit_transform(clean_texts)
        y = np.array(labels)
        logger.info(f"Fitting {self.model_type} classifier...")
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict_proba(self, texts: list) -> np.ndarray:
        """Predict spam probabilities."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        clean_texts = ["" if t is None or (isinstance(t, float) and np.isnan(t)) else str(t) for t in texts]
        X = self.vectorizer.transform(clean_texts)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)[:, 1]
        elif hasattr(self.model, "decision_function"):
            decision = self.model.decision_function(X)
            return 1.0 / (1.0 + np.exp(-decision))
        return self.model.predict(X).astype(float)

    def predict(self, texts: list, threshold: float = 0.5) -> np.ndarray:
        """Predict binary labels (1 = spam, 0 = ham)."""
        probs = self.predict_proba(texts)
        return (probs >= threshold).astype(int)

    def evaluate(self, texts: list, labels: list, threshold: float = 0.5) -> Dict[str, Any]:
        """Compute comprehensive evaluation metrics."""
        y_true = np.array(labels)
        y_prob = self.predict_proba(texts)
        y_pred = (y_prob >= threshold).astype(int)

        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        try:
            roc_auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            roc_auc = 0.0

        cm = confusion_matrix(y_true, y_pred).tolist()

        return {
            "model_type": f"baseline_{self.model_type}",
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
            "confusion_matrix": cm,
            "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
        }

    def save(self, filepath: Union[str, Path]):
        """Save model and vectorizer bundle."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model_type": self.model_type,
            "config": self.config,
            "vectorizer": self.vectorizer,
            "model": self.model,
            "is_fitted": self.is_fitted
        }, path)
        logger.info(f"Saved baseline model to {path}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "BaselineSpamClassifier":
        """Load trained baseline model from disk."""
        data = joblib.load(filepath)
        instance = cls(model_type=data["model_type"], config=data["config"])
        instance.vectorizer = data["vectorizer"]
        instance.model = data["model"]
        instance.is_fitted = data["is_fitted"]
        return instance
