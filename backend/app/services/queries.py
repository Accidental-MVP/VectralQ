from __future__ import annotations

from html import escape
from typing import List, Tuple

import numpy as np
from sqlalchemy.sql import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.embeddings import encode_texts


def build_plainto_tsquery(query: str) -> Tuple[str, dict]:
    # Use plainto_tsquery with 'simple' config; no special escaping needed beyond trimming
    q = (query or "").strip()
    return "plainto_tsquery('simple', :q)", {"q": q}


def vector_literal_from_numpy(vec: np.ndarray) -> str:
    # Convert a 1-D numpy vector to pgvector textual literal "[v1, v2, ...]"
    vals = vec.astype(np.float32).tolist()
    return "[" + ",".join(f"{x:.6f}" for x in vals) + "]"


def embed_query(query: str) -> np.ndarray:
    # Reuse the sentence-transformers model to create a single embedding
    vecs = encode_texts([query], batch_size=1)
    return vecs[0]


def html_safe_snippet(text_val: str, max_chars: int = 300) -> str:
    s = escape(text_val or "")
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 1] + "\u2026"


