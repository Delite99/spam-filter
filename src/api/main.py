"""FastAPI REST service and web server for SpamGuard."""
import io
import os
import time
import json
import logging
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form, Query
from fastapi.responses import HTMLResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from ..config import MODELS_DIR, REPORTS_DIR, BASE_DIR, PROCESSED_DATA_DIR, PreprocessingConfig
from ..data.preprocessor import TextPreprocessor
from ..data.dataset import TokenizerVocab
from ..models.baseline import BaselineSpamClassifier
from .schemas import (
    PredictRequest,
    PredictResponse,
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    FeatureSignals,
    WordAttention,
    CleanOptions,
    DataUploadResponse,
    CleanDataRequest,
    CleanSampleDiff,
    CleanDataResponse,
    TrainJobRequest,
    TrainStatusResponse,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Safe PyTorch import
TORCH_AVAILABLE = False
DEVICE = "cpu"
try:
    import torch
    TORCH_AVAILABLE = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from ..models.lstm_model import BiLSTMSpamClassifier
    from ..models.transformer_model import TransformerSpamClassifier
except Exception as e:
    logger.warning(f"PyTorch could not be initialized ({e}). PyTorch models will be disabled; Scikit-Learn baselines and cleaning remain active.")

# Global registry of loaded models and preprocessors
MODELS: Dict[str, Any] = {}
PREPROCESSOR = TextPreprocessor()

# In-memory storage for uploaded / cleaned data
DATA_STORE: Dict[str, Any] = {
    "raw_df": None,
    "clean_df": None,
    "columns": [],
    "cleaned_path": PROCESSED_DATA_DIR / "custom_cleaned_dataset.csv",
}

# Background training state
TRAINING_STATE: Dict[str, Any] = {
    "status": "idle",  # "idle" | "running" | "completed" | "error"
    "progress": 0,
    "current_step": "Ready to train",
    "logs": [],
    "model_type": None,
    "metrics": None,
    "error_message": None,
    "saved_model_path": None,
}
TRAINING_LOCK = threading.Lock()


def load_available_models():
    """Discover and load saved model checkpoints from models/."""
    # 1. Baseline models (TF-IDF + LR, TF-IDF + NB)
    for m_name, fname in [("baseline_lr", "baseline_logistic_regression.joblib"), ("baseline_nb", "baseline_naive_bayes.joblib")]:
        p = MODELS_DIR / fname
        if p.exists():
            try:
                clf = BaselineSpamClassifier.load(p)
                MODELS[m_name] = {"model": clf}
                logger.info(f"Loaded {m_name} model successfully.")
            except Exception as e:
                logger.error(f"Failed to load {m_name}: {e}")

    # 2. PyTorch BiLSTM model
    if TORCH_AVAILABLE:
        bilstm_path = MODELS_DIR / "bilstm_best.pt"
        vocab_path = MODELS_DIR / "bilstm_vocab.json"
        if bilstm_path.exists() and vocab_path.exists():
            try:
                vocab = TokenizerVocab.load(vocab_path)
                lstm_model, meta = BiLSTMSpamClassifier.load(bilstm_path, device=DEVICE)
                MODELS["bilstm"] = {"model": lstm_model, "vocab": vocab, "metadata": meta}
                logger.info("Loaded BiLSTM model successfully.")
            except Exception as e:
                logger.error(f"Failed to load BiLSTM model: {e}")

        # 3. Transformer model (DistilBERT)
        distilbert_path = MODELS_DIR / "distilbert_best"
        if distilbert_path.exists() and ((distilbert_path / "model.safetensors").exists() or (distilbert_path / "pytorch_model.bin").exists() or (distilbert_path / "config.json").exists()):
            try:
                trans_model = TransformerSpamClassifier.from_pretrained(distilbert_path, device=DEVICE)
                MODELS["distilbert"] = {"model": trans_model}
                logger.info("Loaded DistilBERT transformer model successfully.")
            except Exception as e:
                logger.error(f"Failed to load DistilBERT model: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event to load models on startup."""
    logger.info("Starting SpamGuard service...")
    load_available_models()
    yield
    logger.info("Shutting down SpamGuard service...")


app = FastAPI(
    title="SpamGuard AI Suite",
    description="Production Email Spam Classification, Automated Cleaning, and Training Studio API",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static and reports directories
STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

if REPORTS_DIR.exists():
    app.mount("/reports", StaticFiles(directory=str(REPORTS_DIR)), name="reports")


def get_active_model(requested_name: Optional[str] = "default") -> Tuple[str, Any]:
    """Resolve which model to use for inference."""
    priority = ["distilbert", "bilstm", "baseline_lr", "baseline_nb"]
    
    if requested_name and requested_name != "default":
        alias = requested_name.lower().replace("-", "_")
        if alias in MODELS:
            return alias, MODELS[alias]
        if alias == "lstm" and "bilstm" in MODELS:
            return "bilstm", MODELS["bilstm"]
        if alias in ["lr", "logistic_regression"] and "baseline_lr" in MODELS:
            return "baseline_lr", MODELS["baseline_lr"]
        if alias in ["nb", "naive_bayes"] and "baseline_nb" in MODELS:
            return "baseline_nb", MODELS["baseline_nb"]

    for p in priority:
        if p in MODELS:
            return p, MODELS[p]

    return "heuristic_fallback", None


def run_single_inference(
    text: str,
    model_name: Optional[str] = "default",
    explain: bool = False
) -> PredictResponse:
    """Execute classification pipeline and calculate latency & signals."""
    t0 = time.perf_counter()
    active_name, model_obj = get_active_model(model_name)
    cleaned = PREPROCESSOR.clean_text(text)
    signals_dict = PREPROCESSOR.extract_features(text)
    signals = FeatureSignals(**signals_dict) if explain else None

    word_attns: Optional[List[WordAttention]] = None
    prob: float = 0.5

    if model_obj is not None:
        if active_name == "bilstm" and TORCH_AVAILABLE:
            model = model_obj["model"]
            vocab = model_obj["vocab"]
            prob, attn_pairs = model.predict_proba_single(
                text=text,
                vocab=vocab,
                preprocessor=PREPROCESSOR,
                device=DEVICE
            )
            if explain:
                word_attns = [WordAttention(word=w, weight=wt) for w, wt in attn_pairs]

        elif active_name in ["baseline_lr", "baseline_nb"]:
            clf = model_obj["model"]
            probs = clf.predict_proba([cleaned])
            prob = float(probs[0])
            if explain:
                tokens = set(cleaned.split())
                word_attns = [WordAttention(word=t, weight=1.0) for t in list(tokens)[:15]]

        elif active_name == "distilbert" and TORCH_AVAILABLE:
            trans_model = model_obj["model"]
            probs = trans_model.predict_proba([cleaned])
            prob = float(probs[0])
    else:
        # Heuristic fallback if models have not yet been trained
        spam_score = (
            signals_dict["uppercase_ratio"] * 2.0
            + min(1.0, signals_dict["exclamation_count"] * 0.2)
            + min(1.0, signals_dict["currency_count"] * 0.3)
            + min(1.0, signals_dict["url_count"] * 0.25)
        )
        prob = min(0.99, max(0.01, 1.0 / (1.0 + 2.718 ** (-2.0 * (spam_score - 0.5)))))

    # Integrate Phishing Threat Detection: if credential harvesting or urgent closure is detected, elevate probability
    phish_risk = signals_dict.get("phishing_risk_score", 0.0)
    if phish_risk >= 0.70:
        prob = max(prob, phish_risk)

    prob = round(prob, 4)
    label = "spam" if prob >= 0.5 else "ham"
    confidence = round(prob if prob >= 0.5 else (1.0 - prob), 4)
    latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    return PredictResponse(
        label=label,
        probability=prob,
        confidence=confidence,
        model_version=f"{active_name}:v1.0",
        latency_ms=latency_ms,
        signals=signals,
        word_attentions=word_attns,
    )


# -------------------------------------------------------------
# Base Endpoints
# -------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serve the SpamGuard interactive web demo dashboard."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        response = FileResponse(str(index_path))
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response
    return HTMLResponse("<h1>SpamGuard API is live</h1><p>Visit <a href='/docs'>/docs</a> for Swagger UI.</p>")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """System and model health check endpoint."""
    active_name, _ = get_active_model()
    return HealthResponse(
        status="healthy",
        version="1.1.0",
        active_model=active_name,
        available_models=list(MODELS.keys()),
        device=str(DEVICE),
        torch_available=TORCH_AVAILABLE,
    )


@app.post("/predict", response_model=PredictResponse)
async def predict_single(req: PredictRequest):
    """Predict whether a given email is spam or ham."""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    return run_single_inference(
        text=req.text,
        model_name=req.model_name,
        explain=bool(req.explain)
    )


@app.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(req: BatchPredictRequest):
    """Classify a batch of emails efficiently."""
    if not req.texts:
        raise HTTPException(status_code=400, detail="Texts list cannot be empty.")
    
    t0 = time.perf_counter()
    predictions = [
        run_single_inference(text=t, model_name=req.model_name, explain=False)
        for t in req.texts
    ]
    total_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    spam_count = sum(1 for p in predictions if p.label == "spam")
    ham_count = len(predictions) - spam_count

    return BatchPredictResponse(
        predictions=predictions,
        total_processed=len(predictions),
        total_latency_ms=total_ms,
        spam_count=spam_count,
        ham_count=ham_count,
    )


@app.get("/metrics")
async def get_model_metrics():
    """Retrieve evaluation benchmark report."""
    report_file = REPORTS_DIR / "evaluation_report.json"
    if report_file.exists():
        with open(report_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"status": "pending", "message": "Evaluation report will be available once models are trained."}


# -------------------------------------------------------------
# Data Ingestion & Automated Cleaning Pipeline Endpoints
# -------------------------------------------------------------

def infer_columns(columns: List[str]) -> Tuple[str, str]:
    """Auto-detect text column and label column names."""
    cols_lower = [c.lower() for c in columns]
    text_cands = ["text", "cleaned_text", "message", "email", "body", "sms", "content", "sentence", "v2"]
    label_cands = ["label", "target", "class", "category", "type", "tag", "is_spam", "v1"]

    suggested_text = columns[0]
    for cand in text_cands:
        for idx, col in enumerate(cols_lower):
            if cand in col:
                suggested_text = columns[idx]
                break
        if suggested_text != columns[0]:
            break

    suggested_label = columns[-1]
    for cand in label_cands:
        for idx, col in enumerate(cols_lower):
            if cand in col and columns[idx] != suggested_text:
                suggested_label = columns[idx]
                break
        if suggested_label != columns[-1]:
            break

    return suggested_text, suggested_label


@app.post("/api/data/upload", response_model=DataUploadResponse)
async def upload_dataset(
    file: Optional[UploadFile] = File(None),
    raw_text: Optional[str] = Form(None)
):
    """Upload training data from a file (CSV/TSV/JSON) or raw text input."""
    df = None
    try:
        if file is not None:
            contents = await file.read()
            text_data = contents.decode("utf-8", errors="replace")
            # Try delimiter detection
            if file.filename.endswith(".tsv") or "\t" in text_data.splitlines()[0]:
                df = pd.read_csv(io.StringIO(text_data), sep="\t")
            elif file.filename.endswith(".json"):
                df = pd.read_json(io.StringIO(text_data))
            else:
                df = pd.read_csv(io.StringIO(text_data))
        elif raw_text and raw_text.strip():
            # Parse raw text input
            lines = raw_text.strip().splitlines()
            first_line = lines[0]
            sep = "\t" if "\t" in first_line else (";" if ";" in first_line else ",")
            try:
                df = pd.read_csv(io.StringIO(raw_text), sep=sep)
            except Exception:
                # Fallback to single text column if no tabular structure
                df = pd.DataFrame({"text": lines, "label": ["ham"] * len(lines)})
        else:
            raise HTTPException(status_code=400, detail="No file or text payload provided.")

        if df is None or len(df) == 0:
            raise HTTPException(status_code=400, detail="Uploaded dataset is empty.")

        DATA_STORE["raw_df"] = df
        DATA_STORE["columns"] = list(df.columns)
        
        suggested_text, suggested_label = infer_columns(list(df.columns))

        sample_rows = df.head(5).to_dict(orient="records")

        return DataUploadResponse(
            status="success",
            row_count=len(df),
            columns=list(df.columns),
            suggested_text_column=suggested_text,
            suggested_label_column=suggested_label,
            sample_rows=sample_rows,
            message=f"Successfully loaded {len(df):,} rows with {len(df.columns)} columns."
        )
    except Exception as e:
        logger.error(f"Error reading dataset: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to parse dataset: {str(e)}")


@app.post("/api/data/load-sample", response_model=DataUploadResponse)
async def load_sample_dataset(type: str = Query("corpus", description="'corpus' (5,163 records) or 'demo' (50 records)")):
    """Load the bundled verified dataset directly into the training studio."""
    corpus_file = PROCESSED_DATA_DIR / "spam_corpus.csv"
    if not corpus_file.exists():
        # Fallback creation of a rich dataset if file doesn't exist
        from ..data.download import ingest_and_prepare_dataset
        df = ingest_and_prepare_dataset()
    else:
        df = pd.read_csv(corpus_file)

    if type == "demo":
        df = df.sample(min(50, len(df)), random_state=42).reset_index(drop=True)

    DATA_STORE["raw_df"] = df
    DATA_STORE["columns"] = list(df.columns)

    suggested_text, suggested_label = infer_columns(list(df.columns))
    sample_rows = df.head(5).to_dict(orient="records")

    return DataUploadResponse(
        status="success",
        row_count=len(df),
        columns=list(df.columns),
        suggested_text_column=suggested_text,
        suggested_label_column=suggested_label,
        sample_rows=sample_rows,
        message=f"Loaded bundled dataset with {len(df):,} records ready for cleaning and training."
    )


@app.post("/api/data/clean", response_model=CleanDataResponse)
async def clean_dataset(request: CleanDataRequest):
    """Execute end-to-end cleaning, regex normalization, deduplication, and feature extraction."""
    df = DATA_STORE.get("raw_df")
    
    # If raw data string was sent directly in request
    if request.raw_data and request.raw_data.strip():
        lines = request.raw_data.strip().splitlines()
        sep = "\t" if "\t" in lines[0] else ","
        try:
            df = pd.read_csv(io.StringIO(request.raw_data), sep=sep)
        except Exception:
            df = pd.DataFrame({"text": lines, "label": ["ham"] * len(lines)})
        DATA_STORE["raw_df"] = df
        DATA_STORE["columns"] = list(df.columns)

    if df is None:
        # Auto-load the bundled corpus if user clicked clean without prior upload
        corpus_file = PROCESSED_DATA_DIR / "spam_corpus.csv"
        if corpus_file.exists():
            df = pd.read_csv(corpus_file)
            DATA_STORE["raw_df"] = df
            DATA_STORE["columns"] = list(df.columns)
        else:
            raise HTTPException(status_code=400, detail="No dataset loaded. Please upload or load a sample dataset first.")

    opts = request.options or CleanOptions()
    text_col = opts.text_column or infer_columns(list(df.columns))[0]
    label_col = opts.label_column or infer_columns(list(df.columns))[1]

    if text_col not in df.columns:
        text_col = df.columns[0]
    if label_col not in df.columns:
        label_col = df.columns[min(1, len(df.columns) - 1)]

    total_raw_rows = len(df)
    work_df = df.copy()

    # Normalize labels into binary 0 (ham) / 1 (spam)
    spam_tokens = [s.strip().lower() for s in (opts.spam_identifier or "spam,1,phishing,threat,yes,true").split(",") if s.strip()]

    def parse_label(val: Any) -> Tuple[int, str]:
        s = str(val).strip().lower()
        if s in spam_tokens or s == "1" or s == "1.0":
            return 1, "spam"
        return 0, "ham"

    parsed_labels = [parse_label(x) for x in work_df[label_col]]
    work_df["binary_label"] = [p[0] for p in parsed_labels]
    work_df["label_str"] = [p[1] for p in parsed_labels]
    work_df["raw_text"] = work_df[text_col].fillna("").astype(str)

    # Initialize preprocessor with selected options
    custom_prep = TextPreprocessor(
        config=PreprocessingConfig(
            lowercase=opts.lowercase,
            normalize_urls=opts.normalize_urls,
            normalize_emails=opts.normalize_emails,
            normalize_numbers=opts.normalize_numbers,
            normalize_currency=opts.normalize_currency,
            strip_html=opts.strip_html,
            remove_special_chars=True,
            min_length=opts.min_length,
        )
    )

    # Clean text strings
    cleaned_texts = []
    dropped_empty = 0
    replacement_counts = {"urls": 0, "emails": 0, "phones": 0, "currency": 0, "numbers": 0}

    for raw_t in work_df["raw_text"]:
        c_text = custom_prep.clean_text(raw_t)
        if len(c_text.strip()) < opts.min_length:
            dropped_empty += 1
        cleaned_texts.append(c_text)

        # Count tokens
        replacement_counts["urls"] += c_text.count("[URL]")
        replacement_counts["emails"] += c_text.count("[EMAIL]")
        replacement_counts["phones"] += c_text.count("[PHONE]")
        replacement_counts["currency"] += c_text.count("[CURRENCY]")
        replacement_counts["numbers"] += c_text.count("[NUMBER]")

    work_df["cleaned_text"] = cleaned_texts

    # Filter out empty or too short entries
    clean_filtered = work_df[work_df["cleaned_text"].str.strip().str.len() >= opts.min_length].copy()

    # Deduplication
    duplicates_removed = 0
    if opts.remove_duplicates:
        before_dedup = len(clean_filtered)
        clean_filtered = clean_filtered.drop_duplicates(subset=["cleaned_text"]).reset_index(drop=True)
        duplicates_removed = before_dedup - len(clean_filtered)

    clean_filtered = clean_filtered.reset_index(drop=True)
    DATA_STORE["clean_df"] = clean_filtered

    # Persist cleaned dataset
    clean_filtered[["raw_text", "cleaned_text", "binary_label", "label_str"]].rename(
        columns={"binary_label": "label", "raw_text": "text"}
    ).to_csv(DATA_STORE["cleaned_path"], index=False)

    # Calculate statistics
    cleaned_rows = len(clean_filtered)
    ham_count = int((clean_filtered["binary_label"] == 0).sum())
    spam_count = int((clean_filtered["binary_label"] == 1).sum())
    ham_pct = round((ham_count / max(1, cleaned_rows)) * 100, 1)
    spam_pct = round((spam_count / max(1, cleaned_rows)) * 100, 1)

    avg_chars_before = round(float(work_df["raw_text"].str.len().mean()), 1) if total_raw_rows > 0 else 0
    avg_chars_after = round(float(clean_filtered["cleaned_text"].str.len().mean()), 1) if cleaned_rows > 0 else 0

    # Build before & after diff previews (up to 30 sample rows for UI table)
    sample_diffs: List[CleanSampleDiff] = []
    preview_df = clean_filtered.head(30)
    for idx, row in preview_df.iterrows():
        raw_str = str(row["raw_text"])
        signals_dict = custom_prep.extract_features(raw_str)
        diff_item = CleanSampleDiff(
            id=int(idx + 1),
            raw_text=raw_str,
            cleaned_text=str(row["cleaned_text"]),
            raw_label=str(row.get(label_col, row["label_str"])),
            label=int(row["binary_label"]),
            label_str=str(row["label_str"]),
            signals=FeatureSignals(**signals_dict)
        )
        sample_diffs.append(diff_item)

    return CleanDataResponse(
        status="success",
        total_raw_rows=total_raw_rows,
        cleaned_rows=cleaned_rows,
        duplicates_removed=duplicates_removed,
        dropped_empty_rows=dropped_empty,
        ham_count=ham_count,
        spam_count=spam_count,
        ham_pct=ham_pct,
        spam_pct=spam_pct,
        avg_chars_before=avg_chars_before,
        avg_chars_after=avg_chars_after,
        replacements=replacement_counts,
        samples=sample_diffs,
        message=f"Dataset cleaned successfully! Retained {cleaned_rows:,} high-quality samples ({duplicates_removed} duplicates removed)."
    )


@app.get("/api/data/download-cleaned")
async def download_cleaned_dataset():
    """Download the processed and normalized dataset as a CSV."""
    p = DATA_STORE.get("cleaned_path")
    if p and Path(p).exists():
        return FileResponse(
            str(p),
            media_type="text/csv",
            filename="cleaned_spam_dataset.csv"
        )
    # Fallback to standard corpus
    fallback = PROCESSED_DATA_DIR / "spam_corpus.csv"
    if fallback.exists():
        return FileResponse(str(fallback), media_type="text/csv", filename="cleaned_spam_dataset.csv")
    raise HTTPException(status_code=404, detail="No cleaned dataset found to download.")


# -------------------------------------------------------------
# Background Model Training Studio Endpoints
# -------------------------------------------------------------

def run_training_worker(
    model_type: str,
    test_size: float,
    val_size: float,
    epochs: int,
    batch_size: int,
    learning_rate: float
):
    """Background worker executing model training and updating progress."""
    global TRAINING_STATE

    def log(msg: str):
        timestamp = time.strftime("%H:%M:%S")
        entry = f"[{timestamp}] {msg}"
        TRAINING_STATE["logs"].append(entry)
        logger.info(entry)

    try:
        with TRAINING_LOCK:
            TRAINING_STATE["status"] = "running"
            TRAINING_STATE["progress"] = 5
            TRAINING_STATE["current_step"] = "Ingesting and splitting dataset"
            TRAINING_STATE["model_type"] = model_type
            TRAINING_STATE["metrics"] = None
            TRAINING_STATE["error_message"] = None

        log(f"--- Starting Training Job: {model_type.upper()} ---")

        # ── Select data source ────────────────────────────────────────────────
        MIN_TOTAL   = 20   # absolute floor: at least 20 samples
        MIN_PER_CLS = 5    # need at least 5 of each class for valid splits

        def _load_corpus_fallback(reason: str) -> pd.DataFrame:
            log(f"⚠️  {reason} — falling back to the bundled SpamGuard corpus.")
            corpus_p = PROCESSED_DATA_DIR / "spam_corpus.csv"
            if not corpus_p.exists():
                raise FileNotFoundError("Bundled corpus not found. Upload a larger dataset first.")
            df = pd.read_csv(corpus_p)
            if "cleaned_text" in df.columns:
                df["cleaned_text"] = df["cleaned_text"].fillna("").astype(str)
            else:
                df["cleaned_text"] = PREPROCESSOR.clean_series(
                    df.get("text", df.iloc[:, 0]).fillna("").astype(str).tolist()
                )
            df = df[df["cleaned_text"].str.strip().str.len() > 0].reset_index(drop=True)
            return df

        clean_df = DATA_STORE.get("clean_df")

        # Fall back to saved CSV if in-memory store is empty
        if clean_df is None or len(clean_df) == 0:
            cleaned_p = DATA_STORE.get("cleaned_path")
            if cleaned_p and Path(cleaned_p).exists():
                clean_df = pd.read_csv(cleaned_p)
            else:
                clean_df = _load_corpus_fallback("No dataset in memory or on disk")

        # Standardize column names
        if "cleaned_text" not in clean_df.columns:
            text_src = clean_df["text"].fillna("").astype(str).tolist()
            clean_df["cleaned_text"] = PREPROCESSOR.clean_series(text_src)
        else:
            clean_df["cleaned_text"] = clean_df["cleaned_text"].fillna("").astype(str)

        if "label" not in clean_df.columns and "binary_label" in clean_df.columns:
            clean_df["label"] = clean_df["binary_label"]

        # Ensure non-empty texts
        clean_df = clean_df[clean_df["cleaned_text"].str.strip().str.len() > 0].reset_index(drop=True)

        # ── Robust label encoding ─────────────────────────────────────────────
        # Handles: "spam"/"ham", "yes"/"no", "1"/"0", 1/0, 1.0/0.0, etc.
        SPAM_TOKENS = {"spam", "1", "1.0", "yes", "true", "phishing", "threat", "junk", "unsolicited"}

        def encode_label(val) -> int:
            s = str(val).strip().lower()
            try:
                f = float(s)
                return 1 if f >= 0.5 else 0
            except (ValueError, TypeError):
                return 1 if s in SPAM_TOKENS else 0

        if "binary_label" in clean_df.columns:
            labels_raw = clean_df["binary_label"].apply(encode_label).tolist()
        elif "label" in clean_df.columns:
            labels_raw = clean_df["label"].apply(encode_label).tolist()
        else:
            raise ValueError("Dataset has no 'label' or 'binary_label' column. Please clean the dataset first.")

        texts_raw = [str(t) for t in clean_df["cleaned_text"].tolist()]
        spam_n = sum(labels_raw)
        ham_n  = len(labels_raw) - spam_n

        log(f"Total available samples: {len(texts_raw):,} ({spam_n:,} spam, {ham_n:,} ham)")

        # ── Auto-fallback if dataset is too small ─────────────────────────────
        if len(texts_raw) < MIN_TOTAL or spam_n < MIN_PER_CLS or ham_n < MIN_PER_CLS:
            reason = (
                f"Dataset too small to train ({len(texts_raw)} samples, "
                f"{spam_n} spam, {ham_n} ham — need ≥{MIN_TOTAL} total and ≥{MIN_PER_CLS} per class)"
            )
            fb_df = _load_corpus_fallback(reason)
            lbl_col = "binary_label" if "binary_label" in fb_df.columns else "label"
            texts_raw  = [str(t) for t in fb_df["cleaned_text"].tolist()]
            labels_raw = fb_df[lbl_col].apply(encode_label).tolist()
            spam_n = sum(labels_raw)
            ham_n  = len(labels_raw) - spam_n
            log(f"Corpus loaded: {len(texts_raw):,} samples ({spam_n:,} spam, {ham_n:,} ham)")

        # Final sanitization guarantee: ensure no float, nan, or empty text
        sanitized = [(t, l) for t, l in zip(texts_raw, labels_raw) if len(t.strip()) > 0]
        texts  = [s[0] for s in sanitized]
        labels = [s[1] for s in sanitized]

        TRAINING_STATE["progress"] = 15
        TRAINING_STATE["current_step"] = "Creating stratified Train / Val / Test splits"

        # ── Stratified splits (gracefully degrade if classes are too small) ───
        def safe_split(X, y, test_size, stratify_y, seed=42):
            """Try stratified split; fall back to non-stratified if classes are too small."""
            min_cls = min(y.count(0), y.count(1)) if isinstance(y, list) else 0
            use_strat = stratify_y if min_cls >= 2 else None
            if use_strat is None and stratify_y is not None:
                log("⚠️  Falling back to non-stratified split (minority class has < 2 samples).")
            return train_test_split(X, y, test_size=test_size, stratify=use_strat, random_state=seed)

        train_val_texts, test_texts, train_val_labels, test_labels = safe_split(
            texts, labels, test_size=test_size, stratify_y=labels
        )
        relative_val = val_size / max(1e-6, 1.0 - test_size)
        train_texts, val_texts, train_labels, val_labels = safe_split(
            train_val_texts, train_val_labels, test_size=relative_val, stratify_y=train_val_labels
        )

        log(f"Stratified partition → Train: {len(train_texts):,} | Val: {len(val_texts):,} | Test: {len(test_texts):,}")
        TRAINING_STATE["progress"] = 25

        models_trained = {}

        # -------------------------------------------------------------
        # 1. Baseline: Logistic Regression
        # -------------------------------------------------------------
        if model_type in ["baseline_lr", "all"]:
            TRAINING_STATE["current_step"] = "Training Baseline Logistic Regression"
            log("Fitting TF-IDF Vectorizer and Logistic Regression...")
            clf_lr = BaselineSpamClassifier(model_type="logistic_regression")
            clf_lr.fit(train_texts, train_labels)

            val_metrics = clf_lr.evaluate(val_texts, val_labels)
            test_metrics = clf_lr.evaluate(test_texts, test_labels)
            save_path = MODELS_DIR / "baseline_logistic_regression.joblib"
            clf_lr.save(save_path)
            MODELS["baseline_lr"] = {"model": clf_lr}
            models_trained["baseline_lr"] = test_metrics

            log(f"[Logistic Regression] Test Accuracy: {test_metrics['accuracy']:.4f} | F1: {test_metrics['f1']:.4f} | ROC-AUC: {test_metrics['roc_auc']:.4f}")
            TRAINING_STATE["progress"] = 50 if model_type == "baseline_lr" else 40

        # -------------------------------------------------------------
        # 2. Baseline: Naive Bayes
        # -------------------------------------------------------------
        if model_type in ["baseline_nb", "all"]:
            TRAINING_STATE["current_step"] = "Training Baseline Multinomial Naive Bayes"
            log("Fitting TF-IDF Vectorizer and Naive Bayes...")
            clf_nb = BaselineSpamClassifier(model_type="naive_bayes")
            clf_nb.fit(train_texts, train_labels)

            val_metrics = clf_nb.evaluate(val_texts, val_labels)
            test_metrics = clf_nb.evaluate(test_texts, test_labels)
            save_path = MODELS_DIR / "baseline_naive_bayes.joblib"
            clf_nb.save(save_path)
            MODELS["baseline_nb"] = {"model": clf_nb}
            models_trained["baseline_nb"] = test_metrics

            log(f"[Naive Bayes] Test Accuracy: {test_metrics['accuracy']:.4f} | F1: {test_metrics['f1']:.4f} | ROC-AUC: {test_metrics['roc_auc']:.4f}")
            TRAINING_STATE["progress"] = 80 if model_type == "baseline_nb" else 60

        # -------------------------------------------------------------
        # 3. BiLSTM with Self-Attention (PyTorch)
        # -------------------------------------------------------------
        if model_type in ["bilstm", "all"]:
            if not TORCH_AVAILABLE:
                log("⚠️ PyTorch is not available on this environment (missing C++ runtime). Skipping BiLSTM training.")
            else:
                TRAINING_STATE["current_step"] = "Training PyTorch BiLSTM with Attention"
                log(f"Compiling PyTorch BiLSTM on {DEVICE} (Epochs: {epochs}, Batch size: {batch_size})...")
                
                # Build vocabulary
                vocab = TokenizerVocab(max_vocab_size=25000)
                vocab.build_vocab(train_texts)
                vocab_path = MODELS_DIR / "bilstm_vocab.json"
                vocab.save(vocab_path)
                log(f"Vocabulary constructed: {len(vocab):,} words. Saved to {vocab_path}")

                from torch.utils.data import Dataset, DataLoader

                class SimpleTextDataset(Dataset):
                    def __init__(self, t_list, l_list, v, max_len=256):
                        self.labels = l_list
                        self.encoded = [v.encode(t, max_length=max_len) for t in t_list]
                    def __len__(self):
                        return len(self.labels)
                    def __getitem__(self, idx):
                        toks, length = self.encoded[idx]
                        return (
                            torch.tensor(toks, dtype=torch.long),
                            torch.tensor(length, dtype=torch.long),
                            torch.tensor(self.labels[idx], dtype=torch.float32)
                        )

                train_ds = SimpleTextDataset(train_texts, train_labels, vocab)
                val_ds = SimpleTextDataset(val_texts, val_labels, vocab)
                test_ds = SimpleTextDataset(test_texts, test_labels, vocab)

                train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
                test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

                pos_count = sum(train_labels)
                neg_count = len(train_labels) - pos_count
                pos_weight = torch.tensor([neg_count / max(1, pos_count)], dtype=torch.float32, device=DEVICE)

                lstm_model = BiLSTMSpamClassifier(
                    vocab_size=len(vocab),
                    embedding_dim=128,
                    hidden_dim=128,
                    num_layers=2,
                    bidirectional=True,
                    dropout=0.3,
                    pad_idx=vocab.pad_id
                ).to(DEVICE)

                criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                optimizer = torch.optim.AdamW(lstm_model.parameters(), lr=learning_rate, weight_decay=1e-4)

                best_f1 = 0.0
                save_checkpoint = MODELS_DIR / "bilstm_best.pt"

                for ep in range(1, epochs + 1):
                    lstm_model.train()
                    t_loss = 0.0
                    for ids, lengths, lbls in train_loader:
                        ids, lengths, lbls = ids.to(DEVICE), lengths.to(DEVICE), lbls.to(DEVICE)
                        optimizer.zero_grad()
                        logits, _ = lstm_model(ids, lengths)
                        loss = criterion(logits.squeeze(-1), lbls)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(lstm_model.parameters(), 1.0)
                        optimizer.step()
                        t_loss += loss.item() * len(lbls)
                    t_loss /= len(train_ds)

                    # Validation
                    lstm_model.eval()
                    v_preds, v_trues = [], []
                    with torch.no_grad():
                        for ids, lengths, lbls in val_loader:
                            ids, lengths = ids.to(DEVICE), lengths.to(DEVICE)
                            logits, _ = lstm_model(ids, lengths)
                            probs = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
                            v_preds.extend((probs >= 0.5).astype(int))
                            v_trues.extend(lbls.numpy())
                    v_f1 = float(f1_score(v_trues, v_preds, zero_division=0))
                    v_acc = float(accuracy_score(v_trues, v_preds))

                    step_pct = 60 + int((ep / epochs) * 25)
                    TRAINING_STATE["progress"] = min(85, step_pct)
                    log(f"BiLSTM Epoch {ep}/{epochs} - Train Loss: {t_loss:.4f} | Val Acc: {v_acc:.4f} | Val F1: {v_f1:.4f}")

                    if v_f1 >= best_f1:
                        best_f1 = v_f1
                        lstm_model.save(save_checkpoint, metadata={"epoch": ep, "val_f1": v_f1})

                # Test evaluation
                best_lstm, meta = BiLSTMSpamClassifier.load(save_checkpoint, device=DEVICE)
                best_lstm.eval()
                test_preds, test_probs, test_trues = [], [], []
                with torch.no_grad():
                    for ids, lengths, lbls in test_loader:
                        ids, lengths = ids.to(DEVICE), lengths.to(DEVICE)
                        logits, _ = best_lstm(ids, lengths)
                        probs = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
                        test_probs.extend(probs)
                        test_preds.extend((probs >= 0.5).astype(int))
                        test_trues.extend(lbls.numpy())

                lstm_metrics = {
                    "accuracy": round(float(accuracy_score(test_trues, test_preds)), 4),
                    "precision": round(float(precision_score(test_trues, test_preds, zero_division=0)), 4),
                    "recall": round(float(recall_score(test_trues, test_preds, zero_division=0)), 4),
                    "f1": round(float(f1_score(test_trues, test_preds, zero_division=0)), 4),
                    "roc_auc": round(float(roc_auc_score(test_trues, test_probs)), 4),
                }
                models_trained["bilstm"] = lstm_metrics
                MODELS["bilstm"] = {"model": best_lstm, "vocab": vocab, "metadata": meta}
                log(f"[BiLSTM Test Evaluation] Acc: {lstm_metrics['accuracy']} | F1: {lstm_metrics['f1']} | ROC-AUC: {lstm_metrics['roc_auc']}")

        # -------------------------------------------------------------
        # 4. DistilBERT Transformer Fine-Tuning
        # -------------------------------------------------------------
        if model_type in ["distilbert", "all"]:
            if not TORCH_AVAILABLE:
                log("⚠️ PyTorch is not available. Skipping DistilBERT transformer training.")
            else:
                try:
                    from ..train import train_transformer
                    from ..data.dataset import SpamDataModule
                    dm = SpamDataModule().prepare_data()
                    t_metrics = train_transformer(dm, epochs=min(3, epochs), batch_size=16)
                    models_trained["distilbert"] = t_metrics
                    load_available_models()
                    log(f"[DistilBERT Evaluation] Acc: {t_metrics['accuracy']} | F1: {t_metrics['f1']}")
                except Exception as ex:
                    log(f"⚠️ Transformer fine-tuning skipped or failed: {ex}")

        # Choose primary model for return scorecard
        primary_key = "baseline_lr" if "baseline_lr" in models_trained else (list(models_trained.keys())[0] if models_trained else "none")
        primary_metrics = models_trained.get(primary_key, {})

        # Hot reload all models in memory
        load_available_models()

        TRAINING_STATE["progress"] = 100
        TRAINING_STATE["status"] = "completed"
        TRAINING_STATE["current_step"] = "Training Complete! Model deployed and active."
        TRAINING_STATE["metrics"] = primary_metrics
        TRAINING_STATE["saved_model_path"] = str(MODELS_DIR / f"{primary_key}.joblib")

        log(f"🎉 Pipeline finished! Primary Model: {primary_key} (F1: {primary_metrics.get('f1', 0):.4f}). Active in live scoring.")

    except Exception as e:
        logger.error(f"Training worker failed: {e}", exc_info=True)
        TRAINING_STATE["status"] = "error"
        TRAINING_STATE["error_message"] = str(e)
        TRAINING_STATE["current_step"] = f"Training failed: {str(e)}"
        log(f"❌ Error during training: {str(e)}")


@app.post("/api/train", response_model=TrainStatusResponse)
async def start_training(req: TrainJobRequest):
    """Trigger background model training on the cleaned dataset."""
    if TRAINING_STATE["status"] == "running":
        return TrainStatusResponse(
            status="running",
            progress=TRAINING_STATE["progress"],
            current_step=TRAINING_STATE["current_step"],
            logs=TRAINING_STATE["logs"][-20:],
            model_type=TRAINING_STATE["model_type"],
            metrics=TRAINING_STATE["metrics"],
            error_message=None,
            saved_model_path=TRAINING_STATE["saved_model_path"]
        )

    # Reset state
    TRAINING_STATE["status"] = "running"
    TRAINING_STATE["progress"] = 0
    TRAINING_STATE["current_step"] = "Initializing training pipeline..."
    TRAINING_STATE["logs"] = []
    TRAINING_STATE["metrics"] = None
    TRAINING_STATE["error_message"] = None
    TRAINING_STATE["model_type"] = req.model_type

    thread = threading.Thread(
        target=run_training_worker,
        kwargs={
            "model_type": req.model_type,
            "test_size": req.test_size,
            "val_size": req.val_size,
            "epochs": req.epochs,
            "batch_size": req.batch_size,
            "learning_rate": req.learning_rate
        },
        daemon=True
    )
    thread.start()

    return TrainStatusResponse(
        status="running",
        progress=5,
        current_step="Training worker spawned in background",
        logs=["Training job launched."],
        model_type=req.model_type,
        metrics=None,
        error_message=None,
        saved_model_path=None
    )


@app.get("/api/train/status", response_model=TrainStatusResponse)
async def get_training_status():
    """Poll the real-time status and logs of the training pipeline."""
    return TrainStatusResponse(
        status=TRAINING_STATE["status"],
        progress=TRAINING_STATE["progress"],
        current_step=TRAINING_STATE["current_step"],
        logs=TRAINING_STATE["logs"][-35:],
        model_type=TRAINING_STATE["model_type"],
        metrics=TRAINING_STATE["metrics"],
        error_message=TRAINING_STATE["error_message"],
        saved_model_path=TRAINING_STATE["saved_model_path"],
    )


@app.post("/api/models/reload")
async def reload_models():
    """Reload all available model checkpoints into the active server registry."""
    load_available_models()
    return {
        "status": "success",
        "loaded_models": list(MODELS.keys()),
        "active_model": get_active_model()[0],
        "device": str(DEVICE),
    }
