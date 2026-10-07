"""Unit tests for TextPreprocessor."""
import pytest
from src.data.preprocessor import TextPreprocessor
from src.config import PreprocessingConfig

def test_clean_html_tags():
    preprocessor = TextPreprocessor()
    raw = "<p>Click <b>here</b> to claim your <a href='http://spam.com'>prize</a></p>"
    cleaned = preprocessor.clean_text(raw)
    assert "<p>" not in cleaned
    assert "<b>" not in cleaned
    assert "href" not in cleaned
    assert "[url]" in cleaned
    assert "prize" in cleaned

def test_normalize_urls_and_emails():
    preprocessor = TextPreprocessor()
    raw = "Contact support@service.com or visit https://example.org/verify"
    cleaned = preprocessor.clean_text(raw)
    assert "[email]" in cleaned
    assert "[url]" in cleaned
    assert "support@service.com" not in cleaned

def test_normalize_currency_and_numbers():
    preprocessor = TextPreprocessor()
    raw = "You won $10,000 cash! Call 1-800-555-0199 now!"
    cleaned = preprocessor.clean_text(raw)
    assert "[currency]" in cleaned
    assert "[number]" in cleaned or "[phone]" in cleaned

def test_extract_features():
    preprocessor = TextPreprocessor()
    raw = "URGENT WINNER!!! Claim $500 now at https://prize.com"
    features = preprocessor.extract_features(raw)
    assert features["char_length"] > 0
    assert features["word_count"] > 0
    assert features["uppercase_ratio"] >= 0.20
    assert features["exclamation_count"] >= 3
    assert features["currency_count"] >= 1
    assert features["url_count"] >= 1


def test_empty_string_handling():
    preprocessor = TextPreprocessor()
    assert preprocessor.clean_text("") == ""
    assert preprocessor.clean_text("   ") == ""
    features = preprocessor.extract_features("")
    assert features["char_length"] == 0
    assert features["word_count"] == 0
