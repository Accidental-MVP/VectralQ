from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional
import os

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from app.api.deps import get_tenant_scoped_session, get_tenant_id
from app.core.config import get_settings
from app.core.logging import request_id_var
from app.services.search import run_bm25, run_vector, fuse_candidates, run_phrase_lane
from app.services.context_packer import pack_context_from_texts
from app.services.llm_client import SYSTEM_PROMPT, generate_answer


router = APIRouter(prefix="/query", tags=["query"])


class QueryOptions(BaseModel):
    top_k: Optional[int] = 6
    max_tokens: Optional[int] = 600
    sources: Optional[List[str]] = Field(default=None, description="Limit retrieval to sources, e.g., ['google_drive']")


class QueryRequest(BaseModel):
    question: str
    options: Optional[QueryOptions] = None


def _compute_confidence(num_chunks: int, top_score: float) -> float:
    conf = 0.6 if num_chunks >= 3 else 0.4
    if top_score >= 0.6:
        conf += 0.1
    if num_chunks <= 1:
        conf -= 0.2
    return max(0.0, min(1.0, conf))


def _truncate(s: str, max_len: int = 220) -> str:
    if len(s) <= max_len:
        return s
    return s[: max_len - 1].rstrip() + "…"


@router.post("")
@router.post("/")
async def query(
    body: QueryRequest,
    session: AsyncSession = Depends(get_tenant_scoped_session),
    tenant_id: str = Depends(get_tenant_id),
) -> Any:
    settings = get_settings()
    q = (body.question or "").strip()
    if not q:
        raise HTTPException(status_code=422, detail="question must be non-empty")

    opts = body.options or QueryOptions()
    top_k = int(opts.top_k or settings.context_max_chunks)
    top_k = max(1, min(top_k, 12))

    t0 = time.perf_counter()
    # Retrieval via existing services
    filters: Dict[str, Any] | None = None
    if opts.sources:
        filters = {"source": list(opts.sources)}
    # Optional phrase lane bonus
    phrase_map: Dict[str, float] = {}
    if settings.phrase_lane_enabled:
        phrase_map = await run_phrase_lane(session, q, settings.bm25_k or settings.search_bm25_limit, filters)

    bm25_results = await run_bm25(session, q, settings.bm25_k or settings.search_bm25_limit, filters)
    vector_results = await run_vector(session, q, settings.vec_k or settings.search_vector_limit, filters)
    # Apply phrase bonus pre-fusion
    if phrase_map:
        boost = get_settings().phrase_boost
        for c in bm25_results:
            if c.chunk_id_sha1 in phrase_map:
                c.phrase_bonus = boost
        for c in vector_results:
            if c.chunk_id_sha1 in phrase_map:
                c.phrase_bonus = boost
    fused_results = fuse_candidates(
        bm25_results,
        vector_results,
        mode=settings.search_fusion_mode,
        linear_lambda=settings.search_linear_lambda,
        rrf_k=settings.search_rrf_k,
    )
    fused_results = fused_results[:top_k]

    # Build BM25 rank map for gating (position-based)
    bm25_rank_map: Dict[str, int] = {}
    for idx, c in enumerate(bm25_results, start=1):
        bm25_rank_map[c.chunk_id_sha1] = idx
    # Vector similarity map for top fused lookups
    vec_sim_map: Dict[str, float] = {c.chunk_id_sha1: float(c.semantic_score or 0.0) for c in vector_results}
    # Consider BM25 hit within a relaxed window (default 30) across top fused window (8)
    bm25_window = int(os.getenv("BM25_RANK_WINDOW", "30"))
    top_fused_window = int(os.getenv("FUSED_TOP_WINDOW", "8"))
    bm25_rank_ok = any(bm25_rank_map.get(r.chunk_id, 10**9) <= bm25_window for r in fused_results[:top_fused_window])
    # Strong vector signal gate for top fused candidate
    vec_strong_min = float(os.getenv("VEC_STRONG_MIN", "0.78"))
    top_vec_sim = vec_sim_map.get(fused_results[0].chunk_id, 0.0) if fused_results else 0.0
    vec_strong = fused_results and (top_vec_sim >= vec_strong_min)

    # Refusal / hallucination guard (configurable)
    min_results = max(1, settings.refusal_min_results)
    min_top = max(0.0, settings.refusal_min_top_score)
    # Optional: RRF-based gating – minimal RRF if configured (kept but default 0.0)
    min_rrf = max(0.0, getattr(settings, "min_rrf", 0.0))
    early_refuse = False
    reason = ""
    if len(fused_results) < min_results:
        early_refuse = True
        reason = "min_results"
    elif fused_results and fused_results[0].score < min_top:
        early_refuse = True
        reason = "min_top"
    # Temporary evidence gate: allow if either BM25 is present (within window) OR vector similarity is strong
    elif not (bm25_rank_ok or vec_strong):
        early_refuse = True
        reason = "no_evidence"
    if early_refuse:
        top_score = float(fused_results[0].score) if fused_results else 0.0
        req_id = request_id_var.get() or "-"
        print({
            "event": "refusal",
            "request_id": req_id,
            "refusal_reason": reason,
            "len": len(fused_results),
            "top_score": top_score,
            "min_results": min_results,
            "min_top": min_top,
            "bm25_rank_ok": bm25_rank_ok,
            "vec_strong": bool(vec_strong),
            "top_vec_sim": float(top_vec_sim),
            "bm25_window": bm25_window,
            "top_fused_window": top_fused_window,
        })
        return {
            "answer": "I don’t have enough information.",
            "citations": [],
            "confidence": 0.0,
            "latency_ms": int((time.perf_counter() - t0) * 1000),
        }

    # Fetch chunk texts for selected citations (ensure RLS via session)
    id_pairs = [(r.doc_file_id, r.chunk_id) for r in fused_results]
    texts_map: Dict[str, str] = {}
    for doc_id, chunk_id in id_pairs:
        res = await session.execute(
            text(
                """
                SELECT text FROM app.doc_chunks
                WHERE doc_file_id = :doc AND chunk_id_sha1 = :chunk
                """
            ),
            {"doc": doc_id, "chunk": chunk_id},
        )
        row = res.first()
        if row:
            texts_map[f"{doc_id}#{chunk_id}"] = row[0]

    # Build packed context via packer with token budgeting
    triples = [(r.doc_file_id, r.chunk_id, texts_map.get(f"{r.doc_file_id}#{r.chunk_id}", "")) for r in fused_results]
    packed = pack_context_from_texts(q, triples)
    used_citations = packed.used_citations
    packed_context = packed.packed_context

    user_prompt = f"QUESTION:\n{q}\n\nCONTEXT:\n{packed_context}"

    # Generate
    raw_text = await generate_answer(SYSTEM_PROMPT, user_prompt, citations_hint=used_citations)

    # Validate/normalize output
    answer: str
    citations: List[dict]
    confidence = _compute_confidence(len(used_citations), fused_results[0].score if fused_results else 0.0)
    parsed_ok = True
    try:
        obj = json.loads(raw_text)
        answer = str(obj.get("answer") or "I don’t have enough information.")
        citations = obj.get("citations")
        if citations is None:
            citations = used_citations
        # Enforce citations subset of used_citations
        allowed = {f"{c['doc_file_id']}#{c['chunk_id']}" for c in used_citations}
        clean: List[dict] = []
        for c in citations:
            if not isinstance(c, dict):
                continue
            key = f"{c.get('doc_file_id')}#{c.get('chunk_id')}"
            if key in allowed:
                clean.append({"doc_file_id": c.get("doc_file_id"), "chunk_id": c.get("chunk_id")})
        citations = clean
    except Exception:
        parsed_ok = False
        answer = raw_text.strip()
        if not answer:
            answer = "I don’t have enough information."
        citations = used_citations

    # If model explicitly refused
    if parsed_ok and answer.strip() == "I don’t have enough information.":
        req_id = request_id_var.get() or "-"
        top_score = float(fused_results[0].score) if fused_results else 0.0
        print({
            "event": "refusal",
            "request_id": req_id,
            "refusal_reason": "llm_refusal",
            "len": len(citations or []),
            "top_score": top_score,
        })

    # Post-generation refusal: require citations subset non-empty unless explicit refusal
    if (answer.strip().lower() != "i don’t have enough information." and not citations):
        req_id = request_id_var.get() or "-"
        top_score = float(fused_results[0].score) if fused_results else 0.0
        print({
            "event": "refusal",
            "request_id": req_id,
            "refusal_reason": "no_citations",
            "len": 0,
            "top_score": top_score,
        })
        answer = "I don’t have enough information."
        confidence = 0.0

    latency_ms = int((time.perf_counter() - t0) * 1000)
    # Log
    req_id = request_id_var.get() or "-"
    try:
        await session.execute(
            text(
                """
                INSERT INTO app.query_logs (tenant_id, request_id, question, top_k, selected_chunks, model, generation_ms, confidence, refused, streaming)
                VALUES (:tenant_id, :request_id, :question, :top_k, :selected_chunks, :model, :generation_ms, :confidence, :refused, false)
                """
            ),
            {
                "tenant_id": tenant_id,
                "request_id": req_id,
                "question": q,
                "top_k": top_k,
                "selected_chunks": len(used_citations),
                "model": settings.llm_model or "fallback",
                "generation_ms": latency_ms,
                "confidence": confidence,
                "refused": answer == "I don’t have enough information.",
            },
        )
        await session.commit()
        print({"event": "telemetry_query_ok", "request_id": req_id})
    except Exception as exc:
        print({"event": "telemetry_query_err", "error": str(exc)})

    # Extractive fallback: only if model failed JSON and enabled in settings
    try_extractive = settings.refusal_enable_extractive
    if (not parsed_ok) and try_extractive:
        if fused_results:
            top = fused_results[0]
            txt = texts_map.get(f"{top.doc_file_id}#{top.chunk_id}") or ""
            if txt:
                answer = _truncate(txt, 220)
                citations = [
                    {"doc_file_id": top.doc_file_id, "chunk_id": top.chunk_id}
                ]
                confidence = 1.0

    # Stub coverage tokens metric
    coverage = 0.0
    if used_citations:
        cited = {f"{c['doc_file_id']}#{c['chunk_id']}" for c in citations}
        coverage = round(len(cited) / max(1, len(used_citations)), 3)

    return {
        "answer": answer,
        "citations": citations,
        "confidence": confidence,
        "latency_ms": latency_ms,
        "coverage": coverage,
    }


