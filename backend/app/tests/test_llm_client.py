from __future__ import annotations

import json

import pytest

from app.services.llm_client import _validate_or_fallback


def test_llm_validate_accepts_valid_json():
    raw = json.dumps({"answer": "ok", "citations": [], "confidence": 0.9})
    out = _validate_or_fallback(raw, [])
    obj = json.loads(out)
    assert obj["answer"] == "ok"
    assert obj["citations"] == []
    assert 0 <= obj["confidence"] <= 1


def test_llm_validate_fallbacks_on_invalid():
    raw = "not json"
    out = _validate_or_fallback(raw, [{"doc_file_id": "d", "chunk_id": "c"}])
    obj = json.loads(out)
    # Fallback answer is deterministic; at minimum it returns valid JSON with citations
    assert "answer" in obj and "citations" in obj and "confidence" in obj
    assert isinstance(obj["citations"], list)


