from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_tenant_scoped_session, get_tenant_id
from app.services.embedder_service import embed_missing_for_tenant, embed_for_doc_file


router = APIRouter(prefix="/embeddings", tags=["embeddings"])


@router.post("/rebuild")
async def rebuild_embeddings(
    body: dict[str, Any],
    session: AsyncSession = Depends(get_tenant_scoped_session),
    tenant_id: str = Depends(get_tenant_id),
) -> Any:
    scope = body.get("scope")
    if scope not in ("tenant", "doc"):
        raise HTTPException(status_code=422, detail="scope must be 'tenant' or 'doc'")
    t0 = time.perf_counter()
    if scope == "tenant":
        stats = await embed_missing_for_tenant(session, tenant_id)
    else:
        doc_id = body.get("id")
        if not doc_id:
            raise HTTPException(status_code=422, detail="id is required for scope='doc'")
        # Validate doc belongs to tenant via RLS (no rows means 404)
        from sqlalchemy import text

        res = await session.execute(text("SELECT 1 FROM app.doc_files WHERE id=:id"), {"id": doc_id})
        if res.first() is None:
            raise HTTPException(status_code=404, detail="doc not found")
        stats = await embed_for_doc_file(session, doc_id)
    stats["took_ms_total"] = int((time.perf_counter() - t0) * 1000)
    return stats


@router.post("/missing")
async def embed_missing(
    session: AsyncSession = Depends(get_tenant_scoped_session),
    tenant_id: str = Depends(get_tenant_id),
) -> Any:
    stats = await embed_missing_for_tenant(session, tenant_id)
    return stats


