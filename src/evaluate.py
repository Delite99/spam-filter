"""Evaluation, visualization, and metric reporting for SpamGuard."""
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
)
from .config import MODELS_DIR, REPORTS_DIR, PROCESSED_DATA_DIR
from .data.dataset import SpamDataModule, TokenizerVocab
from .data.preprocessor import TextPreprocessor
from .models.baseline import BaselineSpamClassifier


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def plot_confusion_matrices(cm_dict: Dict[str, np.ndarray], output_file: Path):
    """Plot side-by-side confusion matrices for evaluated models."""
    n_models = len(cm_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5), squeeze=False)
    
    classes = ["Ham (Legit)", "Spam"]
    for idx, (name, cm) in enumerate(cm_dict.items()):
        ax = axes[0][idx]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues" if idx == 0 else "Purples",
            cbar=False,
            xticklabels=classes,
            yticklabels=classes,
            ax=ax
        )
        ax.set_title(f"{name}\nConfusion Matrix", fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("True Label", fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved confusion matrix plot to {output_file}")

def plot_roc_curves(eval_data: Dict[str, Dict[str, Any]], output_file: Path):
    """Plot ROC curves comparison for multiple models."""
    plt.figure(figsize=(7, 6))
    
    for name, data in eval_data.items():
        y_true = data["y_true"]
        y_prob = data["y_prob"]
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})", linewidth=2)

    plt.plot([0, 1], [0, 1], "k--", label="Random Chance (AUC = 0.50)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (FPR)", fontsize=11)
    plt.ylabel("True Positive Rate (TPR / Recall)", fontsize=11)
    plt.title("ROC Curves Comparison - SpamGuard", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower right", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved ROC curves plot to {output_file}")

def plot_pr_curves(eval_data: Dict[str, Dict[str, Any]], output_file: Path):
    """Plot Precision-Recall curves comparison."""
    plt.figure(figsize=(7, 6))

    for name, data in eval_data.items():
        y_true = data["y_true"]
        y_prob = data["y_prob"]
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        f1 = data["f1"]
        plt.plot(recall, precision, label=f"{name} (F1 = {f1:.4f})", linewidth=2)

    plt.xlabel("Recall", fontsize=11)
    plt.ylabel("Precision", fontsize=11)
    plt.title("Precision-Recall Curves - SpamGuard", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower left", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved PR curves plot to {output_file}")

def plot_metrics_bar_chart(metrics_df: pd.DataFrame, output_file: Path):
    """Plot comparison bar chart of Accuracy, Precision, Recall, and F1."""
    plt.figure(figsize=(9, 5))
    metrics_to_plot = ["accuracy", "precision", "recall", "f1"]
    melted = metrics_df.melt(id_vars=["Model"], value_vars=metrics_to_plot, var_name="Metric", value_name="Score")
    
    palette = ["#3B82F6", "#10B981", "#F59E0B", "#8B5CF6"]
    ax = sns.barplot(data=melted, x="Metric", y="Score", hue="Model", palette=palette)
    plt.title("Model Performance Benchmark Comparison", fontsize=13, fontweight="bold", pad=12)
    plt.ylim([0.80, 1.02])
    plt.ylabel("Score", fontsize=11)
    plt.xlabel("Evaluation Metric", fontsize=11)
    plt.legend(title="Model", frameon=True)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    
    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(f"{height:.3f}", (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved metrics bar chart to {output_file}")

def evaluate_all_models():
    """Run full evaluation suite on all trained models on held-out test split."""
    logger.info("Starting SpamGuard comprehensive evaluation...")
    dm = SpamDataModule().prepare_data()
    test_df = dm.test_df
    texts = test_df["cleaned_text"].tolist()
    y_true = test_df["label"].values

    eval_data = {}
    metrics_rows = []
    cms = {}

    # 1. Baseline: Logistic Regression
    lr_path = MODELS_DIR / "baseline_logistic_regression.joblib"
    if lr_path.exists():
        lr_clf = BaselineSpamClassifier.load(lr_path)
        y_prob = lr_clf.predict_proba(texts)
        y_pred = (y_prob >= 0.5).astype(int)
        
        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        auc = float(roc_auc_score(y_true, y_prob))
        cm = confusion_matrix(y_true, y_pred)

        eval_data["TF-IDF + LogisticRegression"] = {"y_true": y_true, "y_prob": y_prob, "y_pred": y_pred, "f1": f1}
        cms["TF-IDF + LogisticRegression"] = cm
        metrics_rows.append({"Model": "Logistic Regression", "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": auc})

    # 2. Baseline: Naive Bayes
    nb_path = MODELS_DIR / "baseline_naive_bayes.joblib"
    if nb_path.exists():
        nb_clf = BaselineSpamClassifier.load(nb_path)
        y_prob = nb_clf.predict_proba(texts)
        y_pred = (y_prob >= 0.5).astype(int)

        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        auc = float(roc_auc_score(y_true, y_prob))
        cm = confusion_matrix(y_true, y_pred)

        eval_data["TF-IDF + NaiveBayes"] = {"y_true": y_true, "y_prob": y_prob, "y_pred": y_pred, "f1": f1}
        cms["TF-IDF + NaiveBayes"] = cm
        metrics_rows.append({"Model": "Naive Bayes", "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": auc})

    # 3. BiLSTM with Attention
    lstm_path = MODELS_DIR / "bilstm_best.pt"
    vocab_path = MODELS_DIR / "bilstm_vocab.json"
    if lstm_path.exists() and vocab_path.exists():
        try:
            import torch
            from .models.lstm_model import BiLSTMSpamClassifier
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            vocab = TokenizerVocab.load(vocab_path)
            model, _ = BiLSTMSpamClassifier.load(lstm_path, device=device)
            model.eval()

            encoded_tuples = [vocab.encode(t, max_length=256) for t in texts]
            input_ids = torch.tensor([t[0] for t in encoded_tuples], dtype=torch.long, device=device)
            lengths = torch.tensor([t[1] for t in encoded_tuples], dtype=torch.long, device=device)

            with torch.no_grad():
                # Process in batches of 128
                y_probs = []
                for b in range(0, len(texts), 128):
                    b_ids = input_ids[b : b + 128]
                    b_lens = lengths[b : b + 128]
                    logits, _ = model(b_ids, b_lens)
                    probs = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
                    y_probs.extend(probs)
                y_prob = np.array(y_probs)

            y_pred = (y_prob >= 0.5).astype(int)
            acc = float(accuracy_score(y_true, y_pred))
            prec = float(precision_score(y_true, y_pred, zero_division=0))
            rec = float(recall_score(y_true, y_pred, zero_division=0))
            f1 = float(f1_score(y_true, y_pred, zero_division=0))
            auc = float(roc_auc_score(y_true, y_prob))
            cm = confusion_matrix(y_true, y_pred)

            eval_data["BiLSTM + Self-Attention"] = {"y_true": y_true, "y_prob": y_prob, "y_pred": y_pred, "f1": f1}
            cms["BiLSTM + Self-Attention"] = cm
            metrics_rows.append({"Model": "BiLSTM + Attention", "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": auc})
        except Exception as e:
            logger.warning(f"Could not evaluate BiLSTM: {e}")



    if not metrics_rows:
        logger.warning("No trained models found in models/ directory. Run src/train.py first!")
        return

    metrics_df = pd.DataFrame(metrics_rows)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save visual reports
    plot_confusion_matrices(cms, REPORTS_DIR / "confusion_matrix.png")
    plot_roc_curves(eval_data, REPORTS_DIR / "roc_curve.png")
    plot_pr_curves(eval_data, REPORTS_DIR / "precision_recall_curve.png")
    plot_metrics_bar_chart(metrics_df, REPORTS_DIR / "metrics_comparison.png")

    report_json = {
        "benchmark_summary": metrics_df.to_dict(orient="records"),
        "test_dataset_size": len(y_true),
        "test_spam_ratio": float(np.mean(y_true)),
    }
    with open(REPORTS_DIR / "evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(report_json, f, indent=2)

    logger.info("Evaluation complete! Benchmark Summary:\n" + metrics_df.to_string(index=False))

if __name__ == "__main__":
    evaluate_all_models()
