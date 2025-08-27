from __future__ import annotations

import hashlib
from typing import Iterable, List, Tuple

import tiktoken


def tokenize(text: str, model_encoding: str = "cl100k_base") -> List[int]:
    enc = tiktoken.get_encoding(model_encoding)
    return enc.encode(text)


def det_chunk_id_sha1(tenant_id: str, doc_file_id: str, chunk_text: str) -> str:
    normalized = "\n".join(line.strip() for line in chunk_text.splitlines()).strip()
    h = hashlib.sha1()
    h.update(tenant_id.encode())
    h.update(doc_file_id.encode())
    h.update(normalized.encode())
    return h.hexdigest()


def chunk_text(text: str, tenant_id: str, doc_file_id: str, target_tokens: int = 300, overlap_ratio: float = 0.15) -> List[Tuple[str, int, str]]:
    tokens = tokenize(text)
    if not tokens:
        return []
    overlap = max(1, int(target_tokens * overlap_ratio))
    step = max(1, target_tokens - overlap)
    enc = tiktoken.get_encoding("cl100k_base")
    chunks: List[Tuple[str, int, str]] = []
    for start in range(0, len(tokens), step):
        window = tokens[start : start + target_tokens]
        if not window:
            break
        chunk_text_str = enc.decode(window)
        token_count = len(window)
        chunk_id = det_chunk_id_sha1(tenant_id, doc_file_id, chunk_text_str)
        chunks.append((chunk_text_str, token_count, chunk_id))
        if start + target_tokens >= len(tokens):
            break
    return chunks


def split_sentences(text: str) -> List[Tuple[int, int, str]]:
    # Very lightweight splitter: split on period/question/exclamation; keep indices
    s: List[Tuple[int, int, str]] = []
    start = 0
    for i, ch in enumerate(text):
        if ch in ".!?":
            seg = text[start : i + 1].strip()
            if seg:
                s.append((start, i + 1, seg))
            start = i + 1
    if start < len(text):
        tail = text[start:].strip()
        if tail:
            s.append((start, len(text), tail))
    return s


