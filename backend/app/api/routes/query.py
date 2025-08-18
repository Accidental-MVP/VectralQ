from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from app.api.deps import get_tenant_scoped_session, get_tenant_id
from app.core.config import get_settings
from app.core.logging import request_id_var
from app.services.search import run_bm25, run_vector, fuse_candidates
from app.services.context_packer import pack_context
from app.services.llm_client import SYSTEM_PROMPT, generate_answer


router = APIRouter(prefix="/query", tags=["query"])


class QueryOptions(BaseModel):
    top_k: Optional[int] = 6
    max_tokens: Optional[int] = 600


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
    bm25_results = await run_bm25(session, q, settings.search_bm25_limit, None)
    vector_results = await run_vector(session, q, settings.search_vector_limit, None)
    fused_results = fuse_candidates(
        bm25_results,
        vector_results,
        mode=settings.search_fusion_mode,
        linear_lambda=settings.search_linear_lambda,
        rrf_k=settings.search_rrf_k,
    )
    fused_results = fused_results[:top_k]

    # Hallucination guard
    if len(fused_results) < 2 or (fused_results and fused_results[0].score < 0.25):
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

    # Build packed context (include text blocks)
    # Re-construct ScoredResult-like objects with text for packing
    # Extend the packer to build blocks with provided texts
    blocks: List[str] = []
    used_citations: List[dict] = []
    total_tokens = 0
    for r in fused_results:
        key = f"{r.doc_file_id}#{r.chunk_id}"
        text_val = texts_map.get(key, "")
        header = f"[source: {r.doc_file_id}, chunk: {r.chunk_id}]\n"
        block = header + text_val
        blocks.append(block)
        used_citations.append({"doc_file_id": r.doc_file_id, "chunk_id": r.chunk_id})
    packed_context = "\n\n".join(blocks)

    user_prompt = f"QUESTION:\n{q}\n\nCONTEXT:\n{packed_context}"

    # Generate
    raw_text = await generate_answer(SYSTEM_PROMPT, user_prompt, citations_hint=used_citations)

    # Validate/normalize output
    answer: str
    citations: List[dict]
    confidence = _compute_confidence(len(used_citations), fused_results[0].score if fused_results else 0.0)
    try:
        obj = json.loads(raw_text)
        answer = str(obj.get("answer") or "I don’t have enough information.")
        citations = obj.get("citations") or used_citations
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
        answer = raw_text.strip()
        if not answer:
            answer = "I don’t have enough information."
        citations = used_citations

    latency_ms = int((time.perf_counter() - t0) * 1000)
    # Log
    req_id = request_id_var.get() or "-"
    print(
        {
            "event": "query",
            "request_id": req_id,
            "tenant_id": tenant_id,
            "top_k": top_k,
            "selected_chunks": len(used_citations),
            "model": settings.llm_model or "fallback",
            "generation_ms": latency_ms,
            "confidence": confidence,
            "refused": answer == "I don’t have enough information.",
        }
    )

    return {"answer": answer, "citations": citations, "confidence": confidence, "latency_ms": latency_ms}


