from __future__ import annotations

from typing import AsyncGenerator

from fastapi import Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.db.session import get_db_session
from app.db.tenancy import set_tenant_id, unset_tenant_id
from app.core.logging import tenant_id_var


async def get_tenant_id() -> str:
    # Prefer context set by middleware
    tenant_id = tenant_id_var.get()
    if tenant_id is None:
        raise HTTPException(status_code=400, detail="Missing X-Tenant-ID header")
    return tenant_id


async def get_tenant_scoped_session(
    session: AsyncSession = Depends(get_db_session), tenant_id: str | None = Depends(get_tenant_id)
) -> AsyncGenerator[AsyncSession, None]:
    await set_tenant_id(session, tenant_id)
    try:
        yield session
    finally:
        # Ensure we don't leak the tenant GUC on connection reuse
        await unset_tenant_id(session)


