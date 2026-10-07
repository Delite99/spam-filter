"""API schemas and web service for SpamGuard."""
from .schemas import PredictRequest, PredictResponse, BatchPredictRequest, BatchPredictResponse, HealthResponse

__all__ = [
    "PredictRequest",
    "PredictResponse",
    "BatchPredictRequest",
    "BatchPredictResponse",
    "HealthResponse",
]
