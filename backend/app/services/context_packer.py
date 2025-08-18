from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import tiktoken

from app.core.config import get_settings
from app.services.search import ScoredResult


@dataclass
class PackedContext:
    packed_context: str
    used_citations: List[dict]
    total_tokens: int


def approximate_tokens(text: str, encoding: str = "cl100k_base") -> int:
    try:
        enc = tiktoken.get_encoding(encoding)
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def pack_context(query: str, ranked_chunks: List[ScoredResult]) -> PackedContext:
    settings = get_settings()
    max_chunks = max(1, settings.context_max_chunks)
    token_budget = max(200, settings.context_token_limit)

    # Deduplicate by doc_file_id#chunk_id while preserving order
    seen: set[Tuple[str, str]] = set()
    selected: List[ScoredResult] = []
    for r in ranked_chunks:
        key = (r.doc_file_id, r.chunk_id)
        if key in seen:
            continue
        seen.add(key)
        selected.append(r)
        if len(selected) >= max_chunks:
            break

    # Assemble blocks
    blocks: List[str] = []
    citations: List[dict] = []
    total_tokens = 0
    for r in selected:
        # Caller must provide text if needed; here we only have IDs, so we include headers only.
        # For generation quality, the caller should pass chunk texts together with results; as a pragmatic
        # fallback, we accept that snippets are not used here and rely on DB retrieval in the route.
        header = f"[source: {r.doc_file_id}, chunk: {r.chunk_id}]\n"
        text_block = ""  # text to be filled by caller if they decide to enhance packer to accept full chunks
        block = header + text_block
        tcount = approximate_tokens(block)
        if total_tokens + tcount > token_budget and blocks:
            break
        blocks.append(block)
        citations.append({"doc_file_id": r.doc_file_id, "chunk_id": r.chunk_id})
        total_tokens += tcount

    return PackedContext("\n".join(blocks), citations, total_tokens)


