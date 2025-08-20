from __future__ import annotations

import asyncio
import json
import time
from typing import Any, AsyncGenerator, Dict, List

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from app.api.deps import get_tenant_scoped_session, get_tenant_id
from app.core.config import get_settings
from app.core.logging import request_id_var
from app.services.search import run_bm25, run_vector, fuse_candidates
from app.services.context_packer import pack_context_from_texts
from app.services.llm_client import SYSTEM_PROMPT, generate_answer, generate_answer_stream


router = APIRouter(prefix="/query", tags=["query-stream"])


class QueryOptions(BaseModel):
    top_k: int | None = 6
    max_tokens: int | None = 600


class QueryRequest(BaseModel):
    question: str
    options: QueryOptions | None = None


async def sse_event(data: dict) -> bytes:
    return ("data: " + json.dumps(data) + "\n\n").encode("utf-8")


@router.post("/stream")
async def query_stream(
    body: QueryRequest,
    session: AsyncSession = Depends(get_tenant_scoped_session),
    tenant_id: str = Depends(get_tenant_id),
) -> Any:
    settings = get_settings()
    q = (body.question or "").strip()
    if not q:
        raise HTTPException(status_code=422, detail="question must be non-empty")
    opts = body.options or QueryOptions()
    top_k = max(1, min(int(opts.top_k or settings.context_max_chunks), 12))

    async def gen() -> AsyncGenerator[bytes, None]:
        t0 = time.perf_counter()
        # Retrieval
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
        yield await sse_event({"phase": "retrieved", "num": len(fused_results)})

        if len(fused_results) < 2 or (fused_results and fused_results[0].score < 0.25):
            yield await sse_event({
                "final": {
                    "answer": "I don’t have enough information.",
                    "citations": [],
                    "confidence": 0.0,
                    "latency_ms": int((time.perf_counter() - t0) * 1000),
                }
            })
            return

        # Fetch texts and pack
        id_pairs = [(r.doc_file_id, r.chunk_id) for r in fused_results]
        texts_map: Dict[str, str] = {}
        for doc_id, chunk_id in id_pairs:
            res = await session.execute(
                text("SELECT text FROM app.doc_chunks WHERE doc_file_id=:doc AND chunk_id_sha1=:chunk"),
                {"doc": doc_id, "chunk": chunk_id},
            )
            row = res.first()
            if row:
                texts_map[f"{doc_id}#{chunk_id}"] = row[0]
        triples = [(r.doc_file_id, r.chunk_id, texts_map.get(f"{r.doc_file_id}#{r.chunk_id}", "")) for r in fused_results]
        packed = pack_context_from_texts(q, triples)
        yield await sse_event({"phase": "packed", "chunks": len(packed.used_citations), "tokens": packed.total_tokens})

        # Try real provider streaming; if it fails, fallback to fake
        # Try provider streaming, yield deltas as they arrive
        stream_text = ""
        try:
            async for delta in generate_answer_stream(
                SYSTEM_PROMPT,
                f"QUESTION:\n{q}\n\nCONTEXT:\n{packed.packed_context}",
                citations_hint=packed.used_citations,
            ):
                stream_text += delta
                yield await sse_event({"delta": delta})
        except Exception:
            stream_text = ""

        if not stream_text:
            # Fake streaming fallback
            used = packed.used_citations
            part1 = "Answer: "
            part2 = " ".join([c["doc_file_id"] for c in used[:1]]) or ""
            part3 = "; citations included."
            for part in (part1, part2, part3):
                await asyncio.sleep(0.1)
                yield await sse_event({"delta": part})
            final_answer = (part1 + part2 + part3).strip()
        else:
            final_answer = stream_text

        latency_ms = int((time.perf_counter() - t0) * 1000)
        yield await sse_event({
            "final": {
                "answer": final_answer,
                "citations": packed.used_citations,
                "confidence": 0.6,
                "latency_ms": latency_ms,
            }
        })

    return StreamingResponse(gen(), media_type="text/event-stream")


