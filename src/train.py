"""Training pipeline for SpamGuard."""
import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score

from .config import DEFAULT_CONFIG, MODELS_DIR, REPORTS_DIR, PROCESSED_DATA_DIR
from .data.dataset import SpamDataModule, TokenizerVocab
from .data.preprocessor import TextPreprocessor
from .models.baseline import BaselineSpamClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def get_torch_device():
    try:
        import torch
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except Exception:
        return "cpu"


def train_baselines(dm: SpamDataModule) -> Dict[str, Any]:
    """Train and evaluate baseline models (Logistic Regression & Naive Bayes)."""
    logger.info("--- Starting Baseline Models Training ---")
    train_texts = dm.train_df["cleaned_text"].tolist()
    train_labels = dm.train_df["label"].tolist()
    val_texts = dm.val_df["cleaned_text"].tolist()
    val_labels = dm.val_df["label"].tolist()
    test_texts = dm.test_df["cleaned_text"].tolist()
    test_labels = dm.test_df["label"].tolist()

    results = {}
    for m_type in ["logistic_regression", "naive_bayes"]:
        logger.info(f"Training Baseline: {m_type}")
        clf = BaselineSpamClassifier(model_type=m_type)
        clf.fit(train_texts, train_labels)

        val_metrics = clf.evaluate(val_texts, val_labels)
        test_metrics = clf.evaluate(test_texts, test_labels)
        logger.info(f"[{m_type}] Val F1: {val_metrics['f1']:.4f} | Test F1: {test_metrics['f1']:.4f}")

        save_path = MODELS_DIR / f"baseline_{m_type}.joblib"
        clf.save(save_path)
        results[m_type] = {
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "saved_model": str(save_path)
        }
    return results


def train_lstm(
    dm: SpamDataModule,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 0.001,
    max_vocab_size: int = 25000,
    max_seq_length: int = 256
) -> Dict[str, Any]:
    """Train BiLSTM with Self-Attention model."""
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from .models.lstm_model import BiLSTMSpamClassifier

    device = get_torch_device()
    logger.info(f"--- Starting BiLSTM with Self-Attention Training on {device} ---")
    vocab = dm.build_vocab(max_vocab_size=max_vocab_size)
    vocab_path = MODELS_DIR / "bilstm_vocab.json"
    vocab.save(vocab_path)
    logger.info(f"Vocabulary size: {len(vocab)} words. Saved to {vocab_path}")

    class PyTorchTextDataset(Dataset):
        def __init__(self, texts: List[str], labels: List[int], vocab: TokenizerVocab, max_length: int = 256):
            self.labels = labels
            self.encoded = [vocab.encode(t, max_length=max_length) for t in texts]

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            token_ids, length = self.encoded[idx]
            return (
                torch.tensor(token_ids, dtype=torch.long),
                torch.tensor(length, dtype=torch.long),
                torch.tensor(self.labels[idx], dtype=torch.float32)
            )

    # Datasets & Loaders
    train_ds = PyTorchTextDataset(dm.train_df["cleaned_text"].tolist(), dm.train_df["label"].tolist(), vocab, max_seq_length)
    val_ds = PyTorchTextDataset(dm.val_df["cleaned_text"].tolist(), dm.val_df["label"].tolist(), vocab, max_seq_length)
    test_ds = PyTorchTextDataset(dm.test_df["cleaned_text"].tolist(), dm.test_df["label"].tolist(), vocab, max_seq_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Class balance weighting
    pos_weight = torch.tensor([dm.get_class_weights()], dtype=torch.float32, device=device)
    logger.info(f"Computed positive class weight: {pos_weight.item():.3f}")

    model = BiLSTMSpamClassifier(
        vocab_size=len(vocab),
        embedding_dim=DEFAULT_CONFIG.lstm.embedding_dim,
        hidden_dim=DEFAULT_CONFIG.lstm.hidden_dim,
        num_layers=DEFAULT_CONFIG.lstm.num_layers,
        bidirectional=DEFAULT_CONFIG.lstm.bidirectional,
        dropout=DEFAULT_CONFIG.lstm.dropout,
        pad_idx=vocab.pad_id,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)

    best_val_f1 = 0.0
    best_checkpoint_path = MODELS_DIR / "bilstm_best.pt"
    patience = DEFAULT_CONFIG.lstm.early_stopping_patience
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        start_time = time.time()

        for input_ids, lengths, labels in train_loader:
            input_ids = input_ids.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits, _ = model(input_ids, lengths)
            loss = criterion(logits.squeeze(-1), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(labels)

        train_loss /= len(train_ds)

        # Validation phase
        model.eval()
        val_preds, val_probs, val_trues = [], [], []
        with torch.no_grad():
            for input_ids, lengths, labels in val_loader:
                input_ids = input_ids.to(device)
                lengths = lengths.to(device)
                logits, _ = model(input_ids, lengths)
                probs = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
                preds = (probs >= 0.5).astype(int)

                val_probs.extend(probs)
                val_preds.extend(preds)
                val_trues.extend(labels.numpy())

        val_f1 = float(f1_score(val_trues, val_preds, zero_division=0))
        val_acc = float(accuracy_score(val_trues, val_preds))
        scheduler.step(val_f1)
        epoch_sec = time.time() - start_time

        logger.info(f"Epoch {epoch:02d}/{epochs:02d} [{epoch_sec:.1f}s] - Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            model.save(
                best_checkpoint_path,
                metadata={
                    "model_type": "BiLSTM_Attention",
                    "best_val_f1": best_val_f1,
                    "epoch": epoch,
                    "max_seq_length": max_seq_length,
                }
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch} (no improvement for {patience} epochs).")
                break

    # Evaluate on held-out test set using best checkpoint
    logger.info(f"Evaluating best checkpoint on held-out test set: {best_checkpoint_path}")
    best_model, meta = BiLSTMSpamClassifier.load(best_checkpoint_path, device=device)
    best_model.eval()

    test_preds, test_probs, test_trues = [], [], []
    inference_times = []

    with torch.no_grad():
        for input_ids, lengths, labels in test_loader:
            input_ids = input_ids.to(device)
            lengths = lengths.to(device)
            
            t0 = time.perf_counter()
            logits, _ = best_model(input_ids, lengths)
            latency = (time.perf_counter() - t0) * 1000.0 / len(labels)
            inference_times.append(latency)

            probs = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
            test_probs.extend(probs)
            test_preds.extend((probs >= 0.5).astype(int))
            test_trues.extend(labels.numpy())

    test_metrics = {
        "model_type": "BiLSTM_Attention",
        "accuracy": round(float(accuracy_score(test_trues, test_preds)), 4),
        "precision": round(float(precision_score(test_trues, test_preds, zero_division=0)), 4),
        "recall": round(float(recall_score(test_trues, test_preds, zero_division=0)), 4),
        "f1": round(float(f1_score(test_trues, test_preds, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(test_trues, test_probs)), 4),
        "avg_latency_ms": round(float(np.mean(inference_times)), 2),
        "checkpoint_path": str(best_checkpoint_path)
    }
    logger.info(f"BiLSTM Test Results: Acc={test_metrics['accuracy']} | Prec={test_metrics['precision']} | Rec={test_metrics['recall']} | F1={test_metrics['f1']} | Latency={test_metrics['avg_latency_ms']}ms")
    return test_metrics

def train_transformer(
    dm: SpamDataModule,
    model_name: str = "distilbert-base-uncased",
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 2e-5,
    max_length: int = 256
) -> Dict[str, Any]:
    """Fine-tune Hugging Face DistilBERT/MiniLM for spam classification."""
    import torch
    from torch.utils.data import Dataset, DataLoader
    from .models.transformer_model import TransformerSpamClassifier

    device = get_torch_device()
    logger.info(f"--- Starting Transformer Fine-Tuning ({model_name}) on {device} ---")
    save_dir = MODELS_DIR / "distilbert_best"
    
    classifier = TransformerSpamClassifier(model_name_or_path=model_name, device=device)
    classifier.initialize_new_model()

    class HFTransformerDataset(Dataset):
        def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 256):
            self.labels = labels
            self.encodings = tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt"
            )

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
            item = {key: val[idx] for key, val in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item

    train_ds = HFTransformerDataset(dm.train_df["cleaned_text"].tolist(), dm.train_df["label"].tolist(), classifier.tokenizer, max_length)
    val_ds = HFTransformerDataset(dm.val_df["cleaned_text"].tolist(), dm.val_df["label"].tolist(), classifier.tokenizer, max_length)
    test_ds = HFTransformerDataset(dm.test_df["cleaned_text"].tolist(), dm.test_df["label"].tolist(), classifier.tokenizer, max_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)


    optimizer = torch.optim.AdamW(classifier.model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * 0.1)

    best_val_f1 = 0.0

    for epoch in range(1, epochs + 1):
        classifier.model.train()
        total_loss = 0.0
        start_time = time.time()

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = classifier.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        classifier.model.eval()
        val_preds, val_trues = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                outputs = classifier.model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
                val_preds.extend(preds)
                val_trues.extend(batch["labels"].numpy())

        val_f1 = float(f1_score(val_trues, val_preds, zero_division=0))
        val_acc = float(accuracy_score(val_trues, val_preds))
        epoch_sec = time.time() - start_time
        logger.info(f"Epoch {epoch}/{epochs} [{epoch_sec:.1f}s] - Train Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        if val_f1 >= best_val_f1:
            best_val_f1 = val_f1
            classifier.save_pretrained(
                save_dir,
                metadata={
                    "model_type": "DistilBERT",
                    "best_val_f1": best_val_f1,
                    "epoch": epoch,
                    "max_length": max_length,
                }
            )

    # Evaluate on test set
    best_classifier = TransformerSpamClassifier.from_pretrained(save_dir, device=device)
    test_probs = best_classifier.predict_proba(dm.test_df["cleaned_text"].tolist(), max_length=max_length)
    test_preds = (test_probs >= 0.5).astype(int)
    test_trues = dm.test_df["label"].values

    test_metrics = {
        "model_type": "DistilBERT",
        "accuracy": round(float(accuracy_score(test_trues, test_preds)), 4),
        "precision": round(float(precision_score(test_trues, test_preds, zero_division=0)), 4),
        "recall": round(float(recall_score(test_trues, test_preds, zero_division=0)), 4),
        "f1": round(float(f1_score(test_trues, test_preds, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(test_trues, test_probs)), 4),
        "checkpoint_path": str(save_dir)
    }
    logger.info(f"DistilBERT Test Results: Acc={test_metrics['accuracy']} | Prec={test_metrics['precision']} | Rec={test_metrics['recall']} | F1={test_metrics['f1']}")
    return test_metrics

def main():
    parser = argparse.ArgumentParser(description="SpamGuard Model Training Pipeline")
    parser.add_argument("--model", type=str, default="all", choices=["all", "baseline", "lstm", "transformer"], help="Model to train")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    args = parser.parse_args()

    dm = SpamDataModule().prepare_data()
    summary = {}

    if args.model in ["all", "baseline"]:
        baseline_results = train_baselines(dm)
        summary["baselines"] = baseline_results

    if args.model in ["all", "lstm"]:
        lstm_results = train_lstm(dm, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
        summary["lstm"] = lstm_results

    if args.model in ["all", "transformer"]:
        try:
            transformer_results = train_transformer(dm, epochs=min(3, args.epochs), batch_size=16)
            summary["transformer"] = transformer_results
        except Exception as e:
            logger.warning(f"Could not fine-tune transformer (possibly offline or memory constraint): {e}")

    summary_file = REPORTS_DIR / "training_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Training pipeline complete! Summary written to {summary_file}")

if __name__ == "__main__":
    main()
