from __future__ import annotations

import math
import time
import asyncio
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
    title: str | None = None
    heading: str | None = None
    # Optional pre-fusion bonus (e.g., from phrase lane)
    phrase_bonus: float | None = None


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
    settings = get_settings()
    # Use English websearch parser and weighted tsvector when enabled
    if settings.bm25_weighted:
        ts_sql, ts_params = "websearch_to_tsquery('english', :q)", {"q": (query or "").strip()}
        tsv_col = "c.text_tsv_enw"
        headline_cfg = "english"
    else:
        ts_sql, ts_params = build_plainto_tsquery(query)
        tsv_col = "c.text_tsv"
        headline_cfg = "simple"
    filter_sql, filter_params = _apply_filters_sql(filters)
    q = text(
        f"""
        WITH q AS (
            SELECT {ts_sql} AS query
        )
        SELECT c.id, c.tenant_id, c.doc_file_id, c.chunk_id_sha1, c.text, c.token_count,
               f.source, c.created_at,
               ts_rank_cd({tsv_col}, q.query) AS rank,
               ts_headline('{headline_cfg}', c.text, q.query, 'StartSel=<b>,StopSel=</b>,MaxFragments=2,ShortWord=2') AS headline,
               c.title, c.heading
        FROM app.doc_chunks c
        JOIN app.doc_files f ON f.id = c.doc_file_id AND f.deleted_at IS NULL
        CROSS JOIN q
        WHERE {tsv_col} @@ q.query{filter_sql}
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
                title=(r[10] or None),
                heading=(r[11] or None),
            )
        )
    return out


def _extract_phrase(query: str) -> str | None:
    # Prefer quoted phrase if present
    q = (query or "").strip()
    if not q:
        return None
    import re
    m = re.search(r'"([^"]{2,100})"', q)
    if m:
        return m.group(1)
    # Fallback: longest 2-4 word span (very cheap heuristic)
    tokens = [t for t in re.split(r"\W+", q) if t]
    best = []
    for n in (4, 3, 2):
        for i in range(0, max(0, len(tokens) - n + 1)):
            span = tokens[i : i + n]
            if len(" ".join(span)) >= len(" ".join(best)):
                best = span
        if best:
            break
    return " ".join(best) if best else None


async def run_phrase_lane(
    session: AsyncSession,
    query: str,
    limit: int,
    filters: dict[str, Any] | None,
) -> Dict[str, float]:
    """Return chunk_id_sha1 -> phrase rank for exact/phrase matches."""
    settings = get_settings()
    if not settings.phrase_lane_enabled:
        return {}
    phrase = _extract_phrase(query)
    if not phrase:
        return {}
    filter_sql, filter_params = _apply_filters_sql(filters)
    tsv_col = "c.text_tsv_enw" if settings.bm25_weighted else "c.text_tsv"
    q = text(
        f"""
        WITH p AS (
            SELECT phraseto_tsquery('english', :phrase) AS tsq
        )
        SELECT c.chunk_id_sha1, ts_rank_cd({tsv_col}, p.tsq) AS phrase_rank
        FROM app.doc_chunks c
        JOIN app.doc_files f ON f.id = c.doc_file_id AND f.deleted_at IS NULL, p
        WHERE {tsv_col} @@ p.tsq{filter_sql}
        ORDER BY phrase_rank DESC
        LIMIT :limit
        """
    )
    res = await session.execute(q, {"phrase": phrase, **filter_params, "limit": int(limit)})
    rows = res.fetchall()
    out: Dict[str, float] = {}
    for r in rows:
        cid = str(r[0])
        rank = float(r[1]) if r[1] is not None else 0.0
        if rank > 0:
            out[cid] = min(1.0, rank)
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
        # carry phrase bonus if present
        if existing is not None and c.phrase_bonus:
            existing.phrase_bonus = max(existing.phrase_bonus or 0.0, c.phrase_bonus)
    for idx, c in enumerate(vector_list, start=1):
        vector_ranks[c.chunk_id_sha1] = idx
        existing = by_chunk.get(c.chunk_id_sha1)
        if existing is None or (c.semantic_score or 0) > (existing.semantic_score or 0):
            by_chunk[c.chunk_id_sha1] = c
        else:
            if existing.semantic_score is None:
                existing.semantic_score = c.semantic_score
        if existing is not None and c.phrase_bonus:
            existing.phrase_bonus = max(existing.phrase_bonus or 0.0, c.phrase_bonus)

    items = list(by_chunk.values())

    if mode == "rrf":
        fused_pairs: List[Tuple[float, Candidate]] = []
        for c in items:
            rrf = 0.0
            if c.chunk_id_sha1 in bm25_ranks:
                rrf += 1.0 / (rrf_k + bm25_ranks[c.chunk_id_sha1])
            if c.chunk_id_sha1 in vector_ranks:
                rrf += 1.0 / (rrf_k + vector_ranks[c.chunk_id_sha1])
            if c.phrase_bonus:
                rrf += float(c.phrase_bonus)
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
            if c.phrase_bonus:
                score += float(c.phrase_bonus)
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
    # Build compact CE input: [title > heading] + snippet (~400 chars)
    for r in top:
        c = candidate_map.get(r.chunk_id)
        if c is None:
            pairs.append((query, ""))
            continue
        snippet = build_snippet(c, query) if c is not None else ""
        prefix_title = (c.title or "").strip()
        prefix_heading = (c.heading or "").strip()
        prefix = f"[{prefix_title} > {prefix_heading}] ".strip()
        ce_text = (prefix + (snippet or c.text or "")).strip()
        if len(ce_text) > 300:
            ce_text = ce_text[:300]
        pairs.append((query, ce_text))
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


