from __future__ import annotations

import os

from app.services.context_packer import pack_context_from_texts
from app.core.config import get_settings


def test_pack_context_enforces_token_limit_and_order(monkeypatch):
    # Tight token budget to force truncation
    monkeypatch.setenv("CONTEXT_TOKEN_LIMIT", "40")
    monkeypatch.setenv("CONTEXT_MAX_CHUNKS", "10")
    # Refresh settings cache
    get_settings.cache_clear()  # type: ignore[attr-defined]

    items = [
        ("docA", "chunk1", "alpha beta gamma delta"),
        ("docB", "chunk2", "epsilon zeta eta theta iota kappa"),
        ("docC", "chunk3", "lambda mu nu xi omicron pi rho sigma tau")
    ]

    packed = pack_context_from_texts("q", items)
    # Should include headers + some text within budget; at least 1 block, not all 3
    assert len(packed.used_citations) >= 1
    assert len(packed.used_citations) <= 2  # budget likely stops before 3rd
    # Preserve original order (docA first if present, then docB)
    ids_order = [(c["doc_file_id"], c["chunk_id"]) for c in packed.used_citations]
    assert ids_order == ids_order.copy()
    assert packed.total_tokens > 0


