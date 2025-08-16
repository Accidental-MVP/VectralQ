from __future__ import annotations

import argparse
import asyncio
import os

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from app.db.session import AsyncSessionLocal
from app.db.tenancy import set_tenant_id, unset_tenant_id
from app.services.embedder_service import embed_missing_for_tenant, embed_for_doc_file


async def main() -> None:
    parser = argparse.ArgumentParser(description="Embed chunks")
    parser.add_argument("--tenant", dest="tenant", help="Tenant UUID")
    parser.add_argument("--doc", dest="doc", help="Doc file UUID")
    parser.add_argument("--missing", action="store_true", help="Only missing embeddings")
    args = parser.parse_args()

    if not args.tenant:
        raise SystemExit("--tenant is required")

    async with AsyncSessionLocal() as session:
        await set_tenant_id(session, args.tenant)
        try:
            if args.doc:
                # Validate doc file exists via RLS
                res = await session.execute(text("SELECT 1 FROM app.doc_files WHERE id=:id"), {"id": args.doc})
                if res.first() is None:
                    raise SystemExit("doc not found or not in tenant")
                stats = await embed_for_doc_file(session, args.doc)
            else:
                stats = await embed_missing_for_tenant(session, args.tenant)
            print(stats)
        finally:
            await unset_tenant_id(session)


if __name__ == "__main__":
    asyncio.run(main())


