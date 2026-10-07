"""Unit and integration tests for FastAPI service."""
import pytest
from fastapi.testclient import TestClient
from src.api.main import app

@pytest.fixture
def client():
    return TestClient(app)

def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "active_model" in data

def test_predict_endpoint_single(client):
    payload = {
        "text": "URGENT: Claim your $1,000,000 lottery cash prize at http://lottery-win.com",
        "explain": True
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["label"] in ["spam", "ham"]
    assert 0.0 <= data["probability"] <= 1.0
    assert 0.5 <= data["confidence"] <= 1.0
    assert data["latency_ms"] >= 0
    assert data["signals"] is not None
    assert data["signals"]["currency_count"] >= 1

def test_predict_batch(client):
    payload = {
        "texts": [
            "Win $500 cash now!",
            "Can we review the quarterly budget spreadsheet tomorrow?",
            "Your PayPal account was locked. Click here to verify."
        ]
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_processed"] == 3
    assert len(data["predictions"]) == 3
    assert data["spam_count"] + data["ham_count"] == 3
    assert data["total_latency_ms"] > 0

def test_predict_empty_text_returns_400_or_422(client):
    response = client.post("/predict", json={"text": ""})
    assert response.status_code in [400, 422]
