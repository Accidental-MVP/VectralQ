from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import RedirectResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_tenant_scoped_session, get_tenant_id
from app.core.config import get_settings
from app.services.token_crypto import encrypt_json, decrypt_json
from app.services.extract import pdf_to_text, txt_to_text
from app.services.chunk import chunk_text
from app.db.session import get_db_session
from app.db.tenancy import set_tenant_id, unset_tenant_id


router = APIRouter(prefix="/integrations/google", tags=["integrations", "google"])


GOOGLE_AUTH = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN = "https://oauth2.googleapis.com/token"
DRIVE_BASE = "https://www.googleapis.com/drive/v3"


def _auth_url(state: str) -> str:
    s = get_settings()
    q = dict(
        client_id=s.google_client_id,
        response_type="code",
        redirect_uri=s.google_redirect_uri,
        scope=s.google_drive_scopes,
        access_type="offline",
        prompt="consent",
        include_granted_scopes="true",
        state=state,
    )
    return f"{GOOGLE_AUTH}?{urlencode(q)}"


async def _exchange_code(code: str) -> dict[str, Any]:
    s = get_settings()
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.post(
            GOOGLE_TOKEN,
            data=dict(
                code=code,
                client_id=s.google_client_id,
                client_secret=s.google_client_secret,
                redirect_uri=s.google_redirect_uri,
                grant_type="authorization_code",
            ),
        )
        r.raise_for_status()
        return r.json()


async def _refresh_token(refresh_token: str) -> dict[str, Any]:
    s = get_settings()
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.post(
            GOOGLE_TOKEN,
            data=dict(
                client_id=s.google_client_id,
                client_secret=s.google_client_secret,
                refresh_token=refresh_token,
                grant_type="refresh_token",
            ),
        )
        r.raise_for_status()
        return r.json()


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


async def _ensure_access_token(session: AsyncSession, tenant_id: str) -> str:
    res = await session.execute(
        text(
            """
            SELECT access_token, refresh_token, expiry, scope
            FROM app.oauth_credentials
            WHERE tenant_id = :t AND provider='google_drive'
            """
        ),
        {"t": tenant_id},
    )
    row = res.first()
    if not row:
        raise HTTPException(status_code=404, detail="No Google Drive credentials for tenant")
    enc_access, enc_refresh, expiry, _scope = row
    tokens = decrypt_json(enc_access)
    access_token = tokens.get("access_token")
    if expiry and expiry > _now_utc() + timedelta(seconds=60):
        return access_token
    if not enc_refresh:
        return access_token
    refresh_token = decrypt_json(enc_refresh).get("refresh_token")
    refreshed = await _refresh_token(refresh_token)
    # Save new access token and expiry
    new_expiry = _now_utc() + timedelta(seconds=int(refreshed.get("expires_in", 3600)))
    enc = encrypt_json({"access_token": refreshed.get("access_token")})
    await session.execute(
        text(
            """
            UPDATE app.oauth_credentials
            SET access_token=:a, expiry=:e, updated_at=NOW()
            WHERE tenant_id=:t AND provider='google_drive'
            """
        ),
        {"a": enc, "e": new_expiry, "t": tenant_id},
    )
    await session.commit()
    return refreshed.get("access_token")


@router.get("/oauth/begin")
async def oauth_begin(tenant_id: str = Depends(get_tenant_id)) -> Any:
    # Minimal state; in production sign HMAC with tenant_id
    state = tenant_id
    return {"redirect": _auth_url(state)}


@router.get("/oauth/callback")
async def oauth_callback(
    code: str, state: str, session: AsyncSession = Depends(get_db_session)
) -> Any:
    # The middleware allows this path without X-Tenant-ID; derive tenant from state and set GUC
    tenant_id = state  # TODO: verify signature in production
    await set_tenant_id(session, tenant_id)
    try:
        tokens = await _exchange_code(code)
        access_token = tokens.get("access_token")
        refresh_token = tokens.get("refresh_token")
        expires_in = int(tokens.get("expires_in", 3600))
        scope = tokens.get("scope")
        expiry_dt = _now_utc() + timedelta(seconds=expires_in)
        enc_access = encrypt_json({"access_token": access_token})
        enc_refresh = encrypt_json({"refresh_token": refresh_token}) if refresh_token else None

        # Upsert oauth_credentials
        await session.execute(
            text(
                """
                INSERT INTO app.oauth_credentials (tenant_id, provider, access_token, refresh_token, expiry, scope)
                VALUES (:t, 'google_drive', :a, :r, :e, :s)
                ON CONFLICT (tenant_id, provider)
                DO UPDATE SET access_token=EXCLUDED.access_token, refresh_token=EXCLUDED.refresh_token, expiry=EXCLUDED.expiry, scope=EXCLUDED.scope, updated_at=NOW()
                """
            ),
            {"t": tenant_id, "a": enc_access, "r": enc_refresh, "e": expiry_dt, "s": scope},
        )
        # Ensure connector row
        await session.execute(
            text(
                """
                INSERT INTO app.connectors (tenant_id, provider, state, active)
                VALUES (:t, 'google_drive', '{}'::jsonb, true)
                ON CONFLICT (tenant_id, provider)
                DO NOTHING
                """
            ),
            {"t": tenant_id},
        )
        await session.commit()
    finally:
        try:
            await unset_tenant_id(session)
        except Exception:
            pass
    return RedirectResponse(url="/dashboard/connectors?connected=google_drive")


async def _get_start_page_token(tok: str) -> str:
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(
            f"{DRIVE_BASE}/changes/startPageToken",
            headers={"Authorization": f"Bearer {tok}"},
        )
        r.raise_for_status()
        return r.json()["startPageToken"]


async def _list_changes(tok: str, page_token: str, page_size: int) -> dict[str, Any]:
    params = {
        "pageToken": page_token,
        "pageSize": page_size,
        "fields": "nextPageToken,newStartPageToken,changes(fileId,file(name,mimeType,modifiedTime,md5Checksum,trashed))",
        "includeRemoved": True,
        "restrictToMyDrive": True,
        "supportsAllDrives": False,
    }
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(
            f"{DRIVE_BASE}/changes",
            params=params,
            headers={"Authorization": f"Bearer {tok}"},
        )
        r.raise_for_status()
        return r.json()


async def _download_file(tok: str, file_id: str, mime: str) -> bytes:
    export_map = {
        "application/vnd.google-apps.document": "text/plain",
        "application/vnd.google-apps.presentation": "text/plain",
        "application/vnd.google-apps.spreadsheet": "text/plain",
    }
    if mime == "application/pdf" or mime.startswith("text/"):
        url = f"{DRIVE_BASE}/files/{file_id}?alt=media"
    else:
        export = export_map.get(mime)
        if not export:
            raise HTTPException(status_code=415, detail="Unsupported MIME for export")
        url = f"{DRIVE_BASE}/files/{file_id}/export?mimeType={export}"
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.get(url, headers={"Authorization": f"Bearer {tok}"})
        r.raise_for_status()
        return r.content


def _is_supported_mime(mime: str) -> bool:
    if mime == "application/pdf":
        return True
    if mime.startswith("text/"):
        return True
    if mime.startswith("application/vnd.google-apps."):
        return True
    return False


@router.post("/sync")
async def sync_now(
    session: AsyncSession = Depends(get_tenant_scoped_session),
    tenant_id: str = Depends(get_tenant_id),
) -> Any:
    settings = get_settings()
    tok = await _ensure_access_token(session, tenant_id)

    # Load connector state
    res = await session.execute(
        text("SELECT state FROM app.connectors WHERE tenant_id=:t AND provider='google_drive'"),
        {"t": tenant_id},
    )
    row = res.first()
    state = (row[0] if row else {}) or {}

    if "startPageToken" not in state:
        start_token = await _get_start_page_token(tok)
        state["startPageToken"] = start_token
        import json as _json
        await session.execute(
            text("UPDATE app.connectors SET state = (:s)::jsonb, updated_at=NOW() WHERE tenant_id=:t AND provider='google_drive'"),
            {"s": _json.dumps(state), "t": tenant_id},
        )
        await session.commit()

    page_token = state.get("pageToken", state.get("startPageToken"))
    processed = deleted = skipped = 0
    t0 = time.perf_counter()

    while True:
        chunk = await _list_changes(tok, page_token, settings.sync_page_size)
        for ch in chunk.get("changes", []):
            f = ch.get("file")
            fid = ch.get("fileId")
            if not f or f.get("trashed", False):
                await session.execute(
                    text(
                        """
                        UPDATE app.doc_files
                        SET ingest_status='deleted'
                        WHERE tenant_id=:t AND source='google_drive' AND external_id=:fid
                        """
                    ),
                    {"t": tenant_id, "fid": fid},
                )
                deleted += 1
                continue

            mime = f.get("mimeType", "")
            if not _is_supported_mime(mime):
                skipped += 1
                continue

            data = await _download_file(tok, fid, mime)
            if len(data) > settings.max_file_bytes:
                skipped += 1
                continue

            # Compute whole-file checksum
            import hashlib

            sha256_hex = hashlib.sha256(data).hexdigest()

            # Check existing by external_id
            res2 = await session.execute(
                text(
                    """
                    SELECT id, checksum_sha256 FROM app.doc_files
                    WHERE tenant_id=:t AND source='google_drive' AND external_id=:fid
                    LIMIT 1
                    """
                ),
                {"t": tenant_id, "fid": fid},
            )
            row2 = res2.first()
            if row2 and (row2[1] == sha256_hex):
                skipped += 1
                continue

            # Insert or update doc_files
            if not row2:
                res3 = await session.execute(
                    text(
                        """
                        INSERT INTO app.doc_files (tenant_id, filename, mime_type, size_bytes, sha256, source, ingest_status, external_id, checksum_sha256)
                        VALUES (:t, :fn, :mt, :sz, :sha, 'google_drive', 'processing', :fid, :fsha)
                        RETURNING id
                        """
                    ),
                    {
                        "t": tenant_id,
                        "fn": f.get("name") or fid,
                        "mt": mime,
                        "sz": len(data),
                        "sha": sha256_hex,
                        "fid": fid,
                        "fsha": sha256_hex,
                    },
                )
                doc_file_id = str(res3.scalar_one())
            else:
                doc_file_id = str(row2[0])
                await session.execute(
                    text(
                        """
                        UPDATE app.doc_files
                        SET filename=:fn, mime_type=:mt, size_bytes=:sz, sha256=:sha, checksum_sha256=:fsha, ingest_status='processing'
                        WHERE id=:id
                        """
                    ),
                    {
                        "fn": f.get("name") or fid,
                        "mt": mime,
                        "sz": len(data),
                        "sha": sha256_hex,
                        "fsha": sha256_hex,
                        "id": doc_file_id,
                    },
                )

            # Extract text
            if mime == "application/pdf":
                from pathlib import Path
                import tempfile

                fd, tmp_path = tempfile.mkstemp(prefix="vectralq_gdrive_")
                import os

                try:
                    with os.fdopen(fd, "wb") as out:
                        out.write(data)
                    text_content = pdf_to_text(Path(tmp_path))
                finally:
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass
            else:
                text_content = txt_to_text(data)

            # Chunk and upsert chunks
            chunks = chunk_text(text_content, tenant_id=tenant_id, doc_file_id=doc_file_id)
            values = []
            for (ctext, tcount, cid) in chunks:
                safe_text = ctext.replace("\x00", "")
                values.append(
                    {
                        "tenant_id": tenant_id,
                        "doc_file_id": doc_file_id,
                        "chunk_id_sha1": cid,
                        "text": safe_text,
                        "token_count": tcount,
                    }
                )
            if values:
                await session.execute(
                    text(
                        """
                        INSERT INTO app.doc_chunks (tenant_id, doc_file_id, chunk_id_sha1, text, token_count)
                        VALUES (:tenant_id, :doc_file_id, :chunk_id_sha1, :text, :token_count)
                        ON CONFLICT DO NOTHING
                        """
                    ),
                    values,
                )
            await session.execute(
                text("UPDATE app.doc_files SET ingest_status='complete' WHERE id=:id"),
                {"id": doc_file_id},
            )
            await session.commit()
            processed += 1

        page_token = chunk.get("nextPageToken")
        if not page_token:
            new_token = chunk.get("newStartPageToken")
            if new_token:
                state["startPageToken"] = new_token
            state["pageToken"] = state.get("startPageToken")
            state["lastSyncAt"] = _now_utc().isoformat()
            import json as _json
            await session.execute(
                text("UPDATE app.connectors SET state = (:s)::jsonb, updated_at=NOW() WHERE tenant_id=:t AND provider='google_drive'"),
                {"s": _json.dumps(state), "t": tenant_id},
            )
            await session.commit()
            break
        else:
            state["pageToken"] = page_token
            import json as _json
            await session.execute(
                text("UPDATE app.connectors SET state = (:s)::jsonb, updated_at=NOW() WHERE tenant_id=:t AND provider='google_drive'"),
                {"s": _json.dumps(state), "t": tenant_id},
            )
            await session.commit()

    took_ms = int((time.perf_counter() - t0) * 1000)
    return {"processed": processed, "deleted": deleted, "skipped": skipped, "took_ms": took_ms}


