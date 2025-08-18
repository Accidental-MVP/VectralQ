from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_tenant_scoped_session, get_tenant_id
from app.core.config import get_settings
from app.services.search import run_bm25, run_vector


router = APIRouter(prefix="/search", tags=["search-debug"])


class DebugRequest(BaseModel):
    query: str


@router.post("/debug")
async def search_debug(
    body: DebugRequest,
    session: AsyncSession = Depends(get_tenant_scoped_session),
    tenant_id: str = Depends(get_tenant_id),
) -> Any:
    settings = get_settings()
    if not settings.debug_search or settings.environment != "local":
        raise HTTPException(status_code=403, detail="Debug disabled")
    q = (body.query or "").strip()
    if not q:
        raise HTTPException(status_code=422, detail="query must be non-empty")
    bm = await run_bm25(session, q, settings.search_bm25_limit, None)
    ve = await run_vector(session, q, settings.search_vector_limit, None)
    return {
        "bm25": [
            {
                "id": c.id,
                "chunk_id": c.chunk_id_sha1,
                "doc_file_id": c.doc_file_id,
                "rank": c.bm25_score,
                "source": c.source,
            }
            for c in bm
        ],
        "vector": [
            {
                "id": c.id,
                "chunk_id": c.chunk_id_sha1,
                "doc_file_id": c.doc_file_id,
                "sim": c.semantic_score,
                "source": c.source,
            }
            for c in ve
        ],
    }


