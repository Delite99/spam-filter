# SpamGuard: Email Spam Filter (PyTorch-based)
**Product Requirements Document (PRD)**

- **Version**: 1.0
- **Date**: October 1, 2026
- **Status**: Final

---

## 1. Overview
- **Product Name**: SpamGuard
- **One-liner**: A production-ready email spam classification system powered by PyTorch that accurately distinguishes spam from legitimate (ham) emails and exposes a simple API for real-time inference.
- **Goal**: Build an end-to-end data science product demonstrating modern NLP techniques (classical baselines + deep learning) while delivering a usable, deployable spam filter.
- **Success Criteria**:
  - Achieves $\ge$ 97% F1-score on a held-out test set from public spam corpora.
  - Provides a FastAPI endpoint returning spam/ham prediction + confidence score in < 100 ms (CPU) for short emails.
  - Fully reproducible via Docker + requirements.
  - Clean, documented codebase suitable for portfolio and technical interviews.

---

## 2. Problem Statement
Email spam remains a high-volume nuisance and security risk. Traditional rule-based filters are brittle; modern ML (especially transformer models) significantly improves detection of sophisticated phishing and marketing spam while keeping false positives low.
SpamGuard addresses this with an end-to-end PyTorch-based text classifier integrated into email pipelines, personal tools, and demos.

---

## 3. Scope & Requirements
- **FR-01**: Ingestion & preprocessing of public datasets (SMS Spam Collection, SpamAssassin, Enron subsets).
- **FR-02**: Text cleaning (lowercasing, URL/email/phone/number normalization, HTML tag stripping, tokenization).
- **FR-03**: Classical ML baselines (MultinomialNB, LogisticRegression with TF-IDF).
- **FR-04**: PyTorch deep learning models (train/val/test splits 70/15/15 stratified).
- **FR-05**: Support both from-scratch PyTorch model (BiLSTM + Self-Attention) and Hugging Face fine-tuning (DistilBERT / MiniLM).
- **FR-06**: Model checkpointing & metadata serialization.
- **FR-07**: FastAPI REST service (`GET /health`, `POST /predict`, `POST /predict/batch`, interactive docs & Web UI).
- **FR-08**: Confidence score & inference latency reporting.
- **FR-09**: Docker containerization (`docker build`, `docker run`).
- **FR-10**: Evaluation reports, metrics table, and visual artifacts (Confusion Matrix, ROC Curve, PR Curve).
- **FR-11**: Batch prediction endpoint.

---

## 4. Architecture
- **Data Layer**: Cleaners, normalizers, tokenizers, PyTorch `Dataset` & `DataLoader`.
- **Modeling Layer**:
  - Baselines: TF-IDF + Naive Bayes / Logistic Regression.
  - PyTorch: Bi-directional LSTM with additive self-attention pooling.
  - Transformer: Fine-tuned `distilbert-base-uncased` with sequence classification head.
- **Serving Layer**: FastAPI with async inference, pydantic request validation, latency metrics, and interactive dashboard UI.
- **Packaging**: Docker multi-stage / slim python container with healthchecks.
