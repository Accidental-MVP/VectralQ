from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logging import request_id_var, tenant_id_var
from app.services.queries import build_plainto_tsquery, embed_query, vector_literal_from_numpy, html_safe_snippet
from app.services.xrerank import rerank_with_cross_encoder


@dataclass
class Candidate:
    id: str
    tenant_id: str
    doc_file_id: str
    chunk_id_sha1: str
    text: str
    token_count: int | None
    source: str | None
    created_at: str
    bm25_score: float | None = None
    semantic_score: float | None = None
    headline: str | None = None


@dataclass
class ScoredResult:
    doc_file_id: str
    chunk_id: str
    score: float
    snippet: str
    source: str | None
    token_count: int | None


def _apply_filters_sql(filters: dict[str, Any] | None) -> Tuple[str, Dict[str, Any]]:
    if not filters:
        return "", {}
    clauses: List[str] = []
    params: Dict[str, Any] = {}
    # Filter by source via app.doc_files.source
    if (sources := filters.get("source")):
        clauses.append("f.source = ANY(:source)")
        params["source"] = list(sources)
    if (doc_ids := filters.get("doc_file_ids")):
        clauses.append("c.doc_file_id = ANY(:doc_ids)")
        params["doc_ids"] = list(doc_ids)
    if not clauses:
        return "", {}
    return " AND " + " AND ".join(clauses), params


async def run_bm25(session: AsyncSession, query: str, limit: int, filters: dict[str, Any] | None) -> List[Candidate]:
    ts_sql, ts_params = build_plainto_tsquery(query)
    filter_sql, filter_params = _apply_filters_sql(filters)
    q = text(
        f"""
        WITH q AS (
            SELECT {ts_sql} AS query
        )
        SELECT c.id, c.tenant_id, c.doc_file_id, c.chunk_id_sha1, c.text, c.token_count,
               f.source, c.created_at,
               ts_rank_cd(c.text_tsv, q.query) AS rank,
               ts_headline('simple', c.text, q.query, 'StartSel=<b>,StopSel=</b>,MaxFragments=2,ShortWord=2') AS headline
        FROM app.doc_chunks c
        JOIN app.doc_files f ON f.id = c.doc_file_id AND f.deleted_at IS NULL
        CROSS JOIN q
        WHERE c.text_tsv @@ q.query{filter_sql}
        ORDER BY rank DESC
        LIMIT :limit
        """
    )
    res = await session.execute(q, {**ts_params, **filter_params, "limit": int(limit)})
    rows = res.fetchall()
    out: List[Candidate] = []
    for r in rows:
        out.append(
            Candidate(
                id=str(r[0]),
                tenant_id=str(r[1]),
                doc_file_id=str(r[2]),
                chunk_id_sha1=str(r[3]),
                text=r[4] or "",
                token_count=int(r[5]) if r[5] is not None else None,
                source=str(r[6]) if r[6] is not None else None,
                created_at=str(r[7]),
                bm25_score=float(r[8]) if r[8] is not None else 0.0,
                headline=(r[9] or None),
            )
        )
    return out


async def run_vector(session: AsyncSession, query: str, limit: int, filters: dict[str, Any] | None) -> List[Candidate]:
    vec = embed_query(query)
    vec_lit = vector_literal_from_numpy(vec)
    filter_sql, filter_params = _apply_filters_sql(filters)
    # Convert cosine distance to similarity via 1 - dist, assuming embeddings are normalized
    q = text(
        f"""
        SELECT c.id, c.tenant_id, c.doc_file_id, c.chunk_id_sha1, c.text, c.token_count,
               f.source, c.created_at,
               (1.0 - (c.embedding <=> CAST(:vec AS vector))) AS sim
        FROM app.doc_chunks c
        JOIN app.doc_files f ON f.id = c.doc_file_id AND f.deleted_at IS NULL
        WHERE c.embedding IS NOT NULL{filter_sql}
        ORDER BY sim DESC
        LIMIT :limit
        """
    )
    res = await session.execute(q, {**filter_params, "limit": int(limit), "vec": vec_lit})
    rows = res.fetchall()
    out: List[Candidate] = []
    for r in rows:
        out.append(
            Candidate(
                id=str(r[0]),
                tenant_id=str(r[1]),
                doc_file_id=str(r[2]),
                chunk_id_sha1=str(r[3]),
                text=r[4] or "",
                token_count=int(r[5]) if r[5] is not None else None,
                source=str(r[6]) if r[6] is not None else None,
                created_at=str(r[7]),
                semantic_score=float(r[8]) if r[8] is not None else 0.0,
            )
        )
    return out


def _min_max_normalize(values: List[float]) -> List[float]:
    if not values:
        return []
    vmin = min(values)
    vmax = max(values)
    if math.isclose(vmax, vmin):
        return [0.0 for _ in values]
    scale = vmax - vmin
    return [(v - vmin) / scale for v in values]


def fuse_candidates(
    bm25_list: List[Candidate],
    vector_list: List[Candidate],
    mode: str,
    linear_lambda: float,
    rrf_k: int,
) -> List[ScoredResult]:
    # Index by chunk_id_sha1 to deduplicate
    by_chunk: Dict[str, Candidate] = {}

    # Collect scores and ranks
    bm25_ranks: Dict[str, int] = {}
    vector_ranks: Dict[str, int] = {}
    for idx, c in enumerate(bm25_list, start=1):
        bm25_ranks[c.chunk_id_sha1] = idx
        existing = by_chunk.get(c.chunk_id_sha1)
        if existing is None or (c.bm25_score or 0) > (existing.bm25_score or 0):
            by_chunk[c.chunk_id_sha1] = c
        else:
            # merge bm25 score onto existing
            if existing.bm25_score is None:
                existing.bm25_score = c.bm25_score
    for idx, c in enumerate(vector_list, start=1):
        vector_ranks[c.chunk_id_sha1] = idx
        existing = by_chunk.get(c.chunk_id_sha1)
        if existing is None or (c.semantic_score or 0) > (existing.semantic_score or 0):
            by_chunk[c.chunk_id_sha1] = c
        else:
            if existing.semantic_score is None:
                existing.semantic_score = c.semantic_score

    items = list(by_chunk.values())

    if mode == "rrf":
        fused_pairs: List[Tuple[float, Candidate]] = []
        for c in items:
            rrf = 0.0
            if c.chunk_id_sha1 in bm25_ranks:
                rrf += 1.0 / (rrf_k + bm25_ranks[c.chunk_id_sha1])
            if c.chunk_id_sha1 in vector_ranks:
                rrf += 1.0 / (rrf_k + vector_ranks[c.chunk_id_sha1])
            fused_pairs.append((rrf, c))
    else:
        # linear (fallback). Note: we do not zero-fill missing values into normalization baselines.
        present_bm = [c.bm25_score for c in items if c.bm25_score is not None]
        present_sem = [c.semantic_score for c in items if c.semantic_score is not None]
        bm_minmax = _min_max_normalize([float(x) for x in present_bm]) if present_bm else []
        sem_minmax = _min_max_normalize([float(x) for x in present_sem]) if present_sem else []
        # Build lookup for normalized scores only where present
        bm_lookup: Dict[str, float] = {}
        sem_lookup: Dict[str, float] = {}
        bi = 0
        for c in items:
            if c.bm25_score is not None and bm_minmax:
                bm_lookup[c.chunk_id_sha1] = bm_minmax[bi]
                bi += 1
        si = 0
        for c in items:
            if c.semantic_score is not None and sem_minmax:
                sem_lookup[c.chunk_id_sha1] = sem_minmax[si]
                si += 1
        fused_pairs = []
        for c in items:
            b = bm_lookup.get(c.chunk_id_sha1)
            s = sem_lookup.get(c.chunk_id_sha1)
            parts: List[float] = []
            if s is not None:
                parts.append(linear_lambda * s)
            if b is not None:
                parts.append((1.0 - linear_lambda) * b)
            score = sum(parts) if parts else 0.0
            fused_pairs.append((score, c))

    # Tie-breakers: higher semantic score, then BM25 rank, then created_at DESC
    def sort_key(item: Tuple[float, Candidate]) -> Tuple[float, float, int, str]:
        score, c = item
        sem = c.semantic_score or 0.0
        bm_rank = bm25_ranks.get(c.chunk_id_sha1, 10**9)
        return (score, sem, -bm_rank, c.created_at)

    fused_pairs.sort(key=sort_key, reverse=True)
    results: List[ScoredResult] = []
    for score, c in fused_pairs:
        results.append(
            ScoredResult(
                doc_file_id=c.doc_file_id,
                chunk_id=c.chunk_id_sha1,
                score=float(score),
                snippet="",  # filled by caller
                source=c.source,
                token_count=c.token_count,
            )
        )
    return results


async def maybe_rerank_with_cross_encoder(
    query: str, results: List[ScoredResult], candidate_map: Dict[str, Candidate]
) -> List[ScoredResult]:
    settings = get_settings()
    if not settings.search_enable_cross_encoder:
        return results
    top_n = max(1, min(settings.cross_encoder_top_n, len(results)))
    top = results[:top_n]
    pairs: List[Tuple[str, str]] = []
    for r in top:
        c = candidate_map.get(r.chunk_id)
        if c is None:
            pairs.append((query, ""))
        else:
            pairs.append((query, c.text))
    scores = await rerank_with_cross_encoder(pairs)
    # Replace score with cross-encoder score (or blend later if desired)
    for i, s in enumerate(scores):
        top[i].score = float(s)
    # Resort combined list by updated scores
    top_sorted = sorted(top, key=lambda r: r.score, reverse=True)
    return top_sorted + results[top_n:]


def build_snippet(result: Candidate, ts_terms: Optional[str]) -> str:
    # If BM25 present, we expect caller to supply tsquery string; server-side headline would be better
    # but we keep a lightweight HTML-safe snippet here for semantic-only fallback.
    if result.headline:
        return result.headline
    if ts_terms:
        # Simple highlight: bold occurrences case-insensitive; escape first
        base = html_safe_snippet(result.text, max_chars=300)
        try:
            # Very lightweight highlighting: replace exact occurrences of terms
            for term in filter(None, ts_terms.split()):
                base = base.replace(term, f"<b>{term}</b>")
                base = base.replace(term.capitalize(), f"<b>{term.capitalize()}</b>")
        except Exception:
            pass
        return base
    # Semantic-only
    return html_safe_snippet(result.text, max_chars=300)


