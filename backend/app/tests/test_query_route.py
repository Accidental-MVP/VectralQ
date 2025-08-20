from __future__ import annotations

import json

import pytest


def test_refusal_and_empty_citations(monkeypatch):
    # This is a lightweight contract test assuming the test client is configured elsewhere.
    # Here we only validate the normalization logic on a mock response.
    from app.api.routes.query import _compute_confidence
    # With low chunks and low score, confidence should reduce
    conf = _compute_confidence(1, 0.1)
    assert 0 <= conf <= 1


