"""Pydantic schemas for the SpamGuard REST API."""
from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Raw email text, subject, or message body to classify")
    model_name: Optional[str] = Field("default", description="Model architecture: 'bilstm', 'distilbert', 'baseline_lr', 'baseline_nb', or 'default'")
    explain: Optional[bool] = Field(False, description="Whether to include attention weights and extracted heuristic signals")

class FeatureSignals(BaseModel):
    char_length: int
    word_count: int
    uppercase_ratio: float
    exclamation_count: int
    question_count: int
    currency_count: int
    url_count: int
    phishing_risk_score: Optional[float] = 0.0
    phishing_indicators: Optional[List[str]] = []

class WordAttention(BaseModel):
    word: str
    weight: float

class PredictResponse(BaseModel):
    label: str = Field(..., description="'spam' or 'ham'")
    probability: float = Field(..., description="Estimated probability of spam (0.0 to 1.0)")
    confidence: float = Field(..., description="Classification confidence (0.5 to 1.0)")
    model_version: str = Field(..., description="Active model identifier and version")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")
    signals: Optional[FeatureSignals] = Field(None, description="Extracted heuristic spam signals")
    word_attentions: Optional[List[WordAttention]] = Field(None, description="Token-level attention weights for interpretability")

class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, description="List of emails or messages to score")
    model_name: Optional[str] = Field("default", description="Model architecture to use")

class BatchPredictResponse(BaseModel):
    predictions: List[PredictResponse]
    total_processed: int
    total_latency_ms: float
    spam_count: int
    ham_count: int

class HealthResponse(BaseModel):
    status: str
    version: str
    active_model: str
    available_models: List[str]
    device: str
    torch_available: bool = True

class CleanOptions(BaseModel):
    strip_html: bool = True
    normalize_urls: bool = True
    normalize_emails: bool = True
    normalize_phone: bool = True
    normalize_currency: bool = True
    normalize_numbers: bool = True
    lowercase: bool = True
    remove_duplicates: bool = True
    min_length: int = 2
    text_column: Optional[str] = None
    label_column: Optional[str] = None
    spam_identifier: Optional[str] = "spam,1,phishing,threat,yes,true"

class DataUploadResponse(BaseModel):
    status: str
    row_count: int
    columns: List[str]
    suggested_text_column: str
    suggested_label_column: str
    sample_rows: List[Dict[str, Any]]
    message: str

class CleanDataRequest(BaseModel):
    raw_data: Optional[str] = None
    options: Optional[CleanOptions] = None

class CleanSampleDiff(BaseModel):
    id: int
    raw_text: str
    cleaned_text: str
    raw_label: str
    label: int
    label_str: str
    signals: FeatureSignals

class CleanDataResponse(BaseModel):
    status: str
    total_raw_rows: int
    cleaned_rows: int
    duplicates_removed: int
    dropped_empty_rows: int
    ham_count: int
    spam_count: int
    ham_pct: float
    spam_pct: float
    avg_chars_before: float
    avg_chars_after: float
    replacements: Dict[str, int]
    samples: List[CleanSampleDiff]
    message: str

class TrainJobRequest(BaseModel):
    model_type: str = "baseline_lr"  # baseline_lr, baseline_nb, bilstm, distilbert, all
    test_size: float = 0.15
    val_size: float = 0.15
    epochs: int = 5
    batch_size: int = 64
    learning_rate: float = 0.001
    use_cleaned_dataset: bool = True

class TrainStatusResponse(BaseModel):
    status: str  # idle, running, completed, error
    progress: int
    current_step: str
    logs: List[str]
    model_type: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    saved_model_path: Optional[str] = None
