from __future__ import annotations

import threading
from functools import lru_cache
from typing import Iterable, List

import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import get_settings


_model_lock = threading.Lock()
_model: SentenceTransformer | None = None


def _load_model() -> SentenceTransformer:
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                settings = get_settings()
                model_name = settings.embedding_model_name or "sentence-transformers/all-MiniLM-L6-v2"
                device = settings.embedding_device or "cpu"
                _model = SentenceTransformer(model_name, device=device)
    return _model  # type: ignore[return-value]


def get_embedding_dimension() -> int:
    model = _load_model()
    return int(model.get_sentence_embedding_dimension())


def encode_texts(texts: List[str], batch_size: int | None = None) -> np.ndarray:
    model = _load_model()
    settings = get_settings()
    bs = batch_size or settings.embedding_batch_size or 64
    # Convert to list of strings and normalize whitespace
    clean_texts = [t.replace("\x00", "").strip() for t in texts]
    vectors = model.encode(clean_texts, batch_size=bs, convert_to_numpy=True, normalize_embeddings=True)
    # Ensure float32
    if vectors.dtype != np.float32:
        vectors = vectors.astype(np.float32)
    return vectors


