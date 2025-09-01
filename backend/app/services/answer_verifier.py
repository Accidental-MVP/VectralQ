from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple


REFUSAL_OBJ = {"answer": "I don’t have enough information.", "citations": [], "confidence": 0.0}


def _split_sentences(text: str) -> List[str]:
    s = (text or "").strip()
    if not s:
        return []
    parts = re.split(r"(?<=[.!?])\s+", s)
    return [p.strip() for p in parts if p and p.strip()]


def _extract_numbers_and_dates(text: str) -> List[str]:
    tokens: List[str] = []
    # numbers with optional separators and decimals, currencies, percents
    for m in re.finditer(r"(?i)(\$|€|£|₹)?\s?\d{1,3}(?:,\d{3})*(?:\.\d+)?\s?(USD|CAD|INR|EUR|%)?", text or ""):
        val = (m.group(0) or "").strip()
        if val:
            tokens.append(val)
    # simple date patterns
    for m in re.finditer(r"(?i)\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|q[1-4])\b", text or ""):
        tokens.append((m.group(0) or "").strip())
    return [t for t in tokens if t]


def _contains_any(hay: str, needles: List[str]) -> bool:
    h = (hay or "").lower()
    for n in needles:
        if (n or "").lower() in h:
            return True
    return False


def verify_and_normalize_answer(
    raw_text: str,
    spans_texts: List[str],
    citations_hint: List[Dict[str, str]],
) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate streamed final JSON against evidence constraints.
    - Must be valid JSON with keys answer, citations, confidence
    - Every sentence in answer must include a tag like [D1:S3]
    - If answer contains numeric/date tokens, at least one must appear in any provided span text
    Returns (ok, normalized_obj_or_refusal)
    """
    spans_texts = spans_texts or []
    try:
        obj = json.loads(raw_text or "")
        if not isinstance(obj, dict):
            return False, REFUSAL_OBJ.copy()
    except Exception:
        return False, REFUSAL_OBJ.copy()

    answer = str(obj.get("answer") or "").strip()
    citations = obj.get("citations")
    confidence = obj.get("confidence")

    if not answer or not isinstance(citations, list) or confidence is None:
        return False, REFUSAL_OBJ.copy()

    # Verify tags per sentence
    sentences = _split_sentences(answer)
    if not sentences:
        return False, REFUSAL_OBJ.copy()
    tag_ok = all(re.search(r"\[D\d+:S\d+\]", s) for s in sentences)
    if not tag_ok:
        return False, REFUSAL_OBJ.copy()

    # Optional numeric/date check: if present in answer, ensure present in some span
    tokens = _extract_numbers_and_dates(answer)
    if tokens:
        found = any(_contains_any(span, tokens) for span in spans_texts)
        if not found:
            return False, REFUSAL_OBJ.copy()

    # Normalize citations to subset of hint when possible
    hint_set = {f"{c.get('doc_file_id')}#{c.get('chunk_id')}" for c in (citations_hint or [])}
    cleaned: List[Dict[str, str]] = []
    for c in citations:
        if isinstance(c, dict):
            key = f"{c.get('doc_file_id')}#{c.get('chunk_id')}"
            if not hint_set or key in hint_set:
                cleaned.append({"doc_file_id": c.get("doc_file_id"), "chunk_id": c.get("chunk_id")})
    if not cleaned and citations_hint:
        cleaned = list(citations_hint)

    out = {
        "answer": answer,
        "citations": cleaned,
        "confidence": float(confidence) if isinstance(confidence, (int, float)) else 0.5,
    }
    return True, out


