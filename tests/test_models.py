"""Unit tests for models, attention mechanisms, and vocabularies."""
import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile

from src.data.dataset import TokenizerVocab
from src.models.lstm_model import BiLSTMSpamClassifier, AdditiveAttention
from src.models.baseline import BaselineSpamClassifier

def test_tokenizer_vocab():
    texts = ["win cash prize now", "urgent account verify", "hi friend how are you"]
    vocab = TokenizerVocab(max_vocab_size=50, min_freq=1)
    vocab.build_vocab(texts)
    
    assert len(vocab) > 4
    assert vocab.pad_id == 0
    assert vocab.unk_id == 1

    seq, length = vocab.encode("win cash urgent", max_length=10)
    assert len(seq) == 10
    assert length == 3
    assert seq[0] != vocab.unk_id
    assert seq[3] == vocab.pad_id

    # Test serialization
    with tempfile.TemporaryDirectory() as tmpdir:
        save_file = Path(tmpdir) / "vocab.json"
        vocab.save(save_file)
        loaded = TokenizerVocab.load(save_file)
        assert len(loaded) == len(vocab)

def test_additive_attention():
    batch_size = 4
    seq_len = 10
    hidden_dim = 64
    attention = AdditiveAttention(hidden_dim=hidden_dim, attention_dim=32)

    rnn_out = torch.randn(batch_size, seq_len, hidden_dim)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    mask[:, 7:] = False # last 3 tokens are padding

    context, weights = attention(rnn_out, mask=mask)
    assert context.shape == (batch_size, hidden_dim)
    assert weights.shape == (batch_size, seq_len)
    
    # Check that weights sum to ~1.0
    weight_sums = torch.sum(weights, dim=-1)
    assert torch.allclose(weight_sums, torch.ones(batch_size), atol=1e-4)

def test_bilstm_forward_pass():
    vocab_size = 100
    model = BiLSTMSpamClassifier(
        vocab_size=vocab_size,
        embedding_dim=32,
        hidden_dim=32,
        num_layers=1,
        bidirectional=True
    )
    
    input_ids = torch.randint(0, vocab_size, (2, 16))
    lengths = torch.tensor([16, 12])
    logits, attn_weights = model(input_ids, lengths)

    assert logits.shape == (2, 1)
    assert attn_weights.shape == (2, 16)

def test_baseline_classifier():
    texts = [
        "claim your free cash prize lottery winner",
        "urgent account access suspended verify password",
        "meeting agenda for team standup tomorrow",
        "let's grab lunch and talk about project roadmap",
    ]
    labels = [1, 1, 0, 0]

    clf = BaselineSpamClassifier(model_type="logistic_regression")
    clf.fit(texts, labels)
    probs = clf.predict_proba(["win free cash", "project standup notes"])
    assert len(probs) == 2
    assert probs[0] > probs[1]
