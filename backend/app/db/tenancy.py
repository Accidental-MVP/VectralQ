from __future__ import annotations

from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text


async def set_tenant_id(session: AsyncSession, tenant_id: Optional[str]) -> None:
    # Sets a per-connection parameter used by Postgres RLS policies
    await session.execute(
        text("SELECT set_config('app.tenant_id', :tenant_id, true)"),
        {"tenant_id": tenant_id or ""},
    )


async def unset_tenant_id(session: AsyncSession) -> None:
    await session.execute(text("SELECT set_config('app.tenant_id', '', true)"))


