from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_tenant_scoped_session, get_tenant_id
from app.core.config import get_settings
from app.core.logging import request_id_var
from app.services.search import run_bm25, run_vector, fuse_candidates, build_snippet, maybe_rerank_with_cross_encoder


router = APIRouter(prefix="/search", tags=["search"])


class SearchFilters(BaseModel):
    source: Optional[List[str]] = None
    doc_file_ids: Optional[List[str]] = Field(default=None, alias="doc_file_ids")


class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = None
    filters: Optional[SearchFilters] = None


@router.post("")
@router.post("/")
async def search(
    body: SearchRequest,
    session: AsyncSession = Depends(get_tenant_scoped_session),
    tenant_id: str = Depends(get_tenant_id),
) -> Any:
    settings = get_settings()
    q = (body.query or "").strip()
    if not q:
        raise HTTPException(status_code=422, detail="query must be non-empty")
    top_k = int(body.top_k or settings.search_top_k_default)
    top_k = max(1, min(top_k, 50))
    t0 = time.perf_counter()

    bm25_limit = settings.search_bm25_limit
    vec_limit = settings.search_vector_limit
    filters = body.filters.dict(by_alias=True, exclude_none=True) if body.filters else None

    bm25_results, vector_results = await _run_candidates(session, q, bm25_limit, vec_limit, filters)

    fused_results = fuse_candidates(
        bm25_results,
        vector_results,
        mode=settings.search_fusion_mode,
        linear_lambda=settings.search_linear_lambda,
        rrf_k=settings.search_rrf_k,
    )

    # Optional cross-encoder rerank
    cand_index = {c.chunk_id_sha1: c for c in (bm25_results + vector_results)}
    reranked_results = await maybe_rerank_with_cross_encoder(q, fused_results, cand_index)

    # Build response up to top_k
    out: List[Dict[str, Any]] = []
    ts_terms = q
    for r in reranked_results[:top_k]:
        cand = cand_index.get(r.chunk_id)
        snippet = build_snippet(cand, ts_terms) if cand is not None else ""
        out.append(
            {
                "doc_file_id": r.doc_file_id,
                "chunk_id": r.chunk_id,
                "score": float(r.score),
                "snippet": snippet,
                "source": (cand.source if cand else "") or "",
                "token_count": (cand.token_count if cand else 0) or 0,
            }
        )

    took_ms = int((time.perf_counter() - t0) * 1000)
    # Minimal structured log
    req_id = request_id_var.get() or "-"
    reranked = get_settings().search_enable_cross_encoder
    print(
        {
            "event": "search",
            "request_id": req_id,
            "tenant_id": tenant_id,
            "top_k": top_k,
            "bm25_n": len(bm25_results),
            "vector_n": len(vector_results),
            "fusion_mode": settings.search_fusion_mode,
            "took_ms": took_ms,
            "reranked": reranked,
        }
    )

    return {"results": out}


async def _run_candidates(
    session: AsyncSession, q: str, bm25_limit: int, vec_limit: int, filters: Optional[Dict[str, Any]]
):
    # Run in parallel if desired; for simplicity, sequential to avoid connection contention
    bm = await run_bm25(session, q, bm25_limit, filters)
    ve = await run_vector(session, q, vec_limit, filters)
    return bm, ve


