from __future__ import annotations

import asyncio
import hashlib
import time
from typing import Any, List, Tuple

import numpy as np
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import request_id_var, tenant_id_var
from app.services.embeddings import encode_texts


def sha1_normalized(text_val: str) -> str:
    normalized = "\n".join(line.strip() for line in text_val.splitlines()).strip()
    return hashlib.sha1(normalized.encode()).hexdigest()


async def _fetch_chunks(session: AsyncSession, where_sql: str, params: dict[str, Any], limit: int) -> List[tuple]:
    q = text(
        f"""
        SELECT id, tenant_id, doc_file_id, chunk_id_sha1, text, embedding, text_checksum
        FROM app.doc_chunks
        WHERE {where_sql}
        ORDER BY created_at ASC
        LIMIT :lim
        """
    )
    res = await session.execute(q, {**params, "lim": limit})
    return list(res.fetchall())


async def _update_embeddings(session: AsyncSession, rows: List[Tuple[str, str, str, str, str]], vectors: np.ndarray) -> None:
    # Executemany safe update per row. Convert vectors to pgvector textual literal: "[v1, v2, ...]"
    updates: List[dict] = []
    for idx, row in enumerate(rows):
        chunk_id = row[0]
        text_val = row[4]
        checksum = sha1_normalized(text_val)
        vec = vectors[idx].tolist()
        embedding_str = "[" + ",".join(f"{x:.6f}" for x in vec) + "]"
        updates.append({"id": chunk_id, "embedding": embedding_str, "checksum": checksum})

    # Run updates per-row to avoid asyncpg executemany named bind issues
    stmt = text(
        """
        UPDATE app.doc_chunks
        SET embedding = CAST(:embedding AS vector), text_checksum = :checksum
        WHERE id = :id
        """
    )
    for upd in updates:
        await session.execute(stmt, upd)


async def embed_missing_for_tenant(session: AsyncSession, tenant_id: str, batch_size: int = 64, page_size: int = 5000) -> dict[str, Any]:
    start = time.perf_counter()
    total_processed = 0
    total_skipped = 0
    req_id = request_id_var.get() or "-"

    while True:
        # For the 'missing' flow, only fetch rows with NULL embedding
        rows = await _fetch_chunks(session, "embedding IS NULL", {}, page_size)
        if not rows:
            break

        texts: List[str] = []
        to_embed: List[Tuple[str, str, str, str, str]] = []
        for row in rows:
            chunk_id, _t_id, _doc_file_id, _chunk_sha1, text_val, embedding_val, existing_checksum = row
            checksum = sha1_normalized(text_val)
            needs_embedding = embedding_val is None or existing_checksum != checksum
            if needs_embedding:
                to_embed.append((chunk_id, _t_id, _doc_file_id, _chunk_sha1, text_val))
                texts.append(text_val)
            else:
                total_skipped += 1

        if not to_embed:
            break

        for i in range(0, len(texts), batch_size):
            batch_rows = to_embed[i : i + batch_size]
            batch_texts = texts[i : i + batch_size]
            vectors = encode_texts(batch_texts, batch_size=batch_size)
            await _update_embeddings(session, batch_rows, vectors)
            total_processed += len(batch_rows)
            await session.commit()

    took_ms = int((time.perf_counter() - start) * 1000)
    return {
        "tenant_id": tenant_id,
        "processed": total_processed,
        "skipped": total_skipped,
        "batch_size": batch_size,
        "took_ms": took_ms,
    }


async def embed_for_doc_file(session: AsyncSession, doc_file_id: str, batch_size: int = 64, page_size: int = 5000) -> dict[str, Any]:
    start = time.perf_counter()
    total_processed = 0
    total_skipped = 0
    rows = await _fetch_chunks(session, "doc_file_id = :doc_file_id", {"doc_file_id": doc_file_id}, page_size)
    if not rows:
        return {"processed": 0, "skipped": 0, "took_ms": 0}

    texts: List[str] = []
    to_embed: List[Tuple[str, str, str, str, str]] = []
    for row in rows:
        chunk_id, _t_id, _doc_file_id, _chunk_sha1, text_val, embedding_val, existing_checksum = row
        checksum = sha1_normalized(text_val)
        if embedding_val is None or existing_checksum != checksum:
            to_embed.append((chunk_id, _t_id, _doc_file_id, _chunk_sha1, text_val))
            texts.append(text_val)
        else:
            total_skipped += 1

    for i in range(0, len(texts), batch_size):
        batch_rows = to_embed[i : i + batch_size]
        batch_texts = texts[i : i + batch_size]
        vectors = encode_texts(batch_texts, batch_size=batch_size)
        await _update_embeddings(session, batch_rows, vectors)
        total_processed += len(batch_rows)
        await session.commit()

    took_ms = int((time.perf_counter() - start) * 1000)
    return {
        "doc_file_id": doc_file_id,
        "processed": total_processed,
        "skipped": total_skipped,
        "batch_size": batch_size,
        "took_ms": took_ms,
    }


