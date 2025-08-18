from __future__ import annotations

import asyncio
from typing import Iterable, List, Tuple

import numpy as np
from sentence_transformers import CrossEncoder

from app.core.config import get_settings


_cross_encoder: CrossEncoder | None = None


def _get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        model_name = get_settings().cross_encoder_model or "cross-encoder/ms-marco-MiniLM-L-6-v2"
        _cross_encoder = CrossEncoder(model_name)
    return _cross_encoder


async def rerank_with_cross_encoder(pairs: List[Tuple[str, str]]) -> List[float]:
    if not pairs:
        return []
    model = _get_cross_encoder()
    loop = asyncio.get_event_loop()
    # Run in threadpool to avoid blocking the event loop
    scores: np.ndarray = await loop.run_in_executor(None, lambda: model.predict(pairs))
    # Ensure Python floats
    return [float(x) for x in scores.tolist()]


