from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from app.api.deps import get_tenant_scoped_session, get_tenant_id
from app.services.extract import pdf_to_text, txt_to_text
from app.services.chunk import chunk_text
from app.utils.files import sha256_stream_and_save, MAX_UPLOAD_BYTES


router = APIRouter(prefix="/docs", tags=["docs"])
upload_router = APIRouter(tags=["upload"])  # exposes /api/upload


SUPPORTED_MIME = {"application/pdf", "text/plain"}


async def _handle_upload(
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_tenant_scoped_session),
    tenant_id: str = Depends(get_tenant_id),
) -> Any:
    if file.content_type not in SUPPORTED_MIME:
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Unsupported MIME type")

    try:
        sha256_hex, tmp_path, size_bytes = sha256_stream_and_save(file.file)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="File too large (max 20MB)")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Upload failed: {exc}") from exc

    # Dedup check
    res = await session.execute(text("SELECT id FROM app.doc_files WHERE sha256=:sha LIMIT 1"), {"sha": sha256_hex})
    row = res.first()
    if row:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        return {"doc_file_id": str(row[0]), "status": "deduped"}

    # Insert doc_files
    result = await session.execute(
        text(
            """
            INSERT INTO app.doc_files (tenant_id, filename, mime_type, size_bytes, sha256, source, ingest_status)
            VALUES (:tenant_id, :filename, :mime_type, :size_bytes, :sha256, 'upload', 'processing')
            RETURNING id
            """
        ),
        {
            "tenant_id": tenant_id,
            "filename": file.filename or "upload",
            "mime_type": file.content_type,
            "size_bytes": size_bytes,
            "sha256": sha256_hex,
        },
    )
    doc_file_id = str(result.scalar_one())

    # Extract text
    try:
        if file.content_type == "application/pdf":
            text_content = pdf_to_text(Path(tmp_path))
        else:
            data = Path(tmp_path).read_bytes()
            text_content = txt_to_text(data)
    except Exception as exc:  # noqa: BLE001
        # Mark failed
        await session.execute(text("UPDATE app.doc_files SET ingest_status='failed' WHERE id=:id"), {"id": doc_file_id})
        await session.commit()
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"Extraction failed: {exc}") from exc
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

    # Chunk
    chunks = chunk_text(text_content, tenant_id=tenant_id, doc_file_id=doc_file_id)
    if not chunks:
        await session.execute(text("UPDATE app.doc_files SET ingest_status='no_content' WHERE id=:id"), {"id": doc_file_id})
        await session.commit()
        return {"doc_file_id": doc_file_id, "chunks": 0}

    # Bulk insert
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
    await session.execute(text("UPDATE app.doc_files SET ingest_status='complete' WHERE id=:id"), {"id": doc_file_id})
    await session.commit()

    return {"doc_file_id": doc_file_id, "chunks": len(chunks)}


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_tenant_scoped_session),
    tenant_id: str = Depends(get_tenant_id),
) -> Any:
    return await _handle_upload(file=file, session=session, tenant_id=tenant_id)


@upload_router.post("/upload")
async def upload_document_root(
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_tenant_scoped_session),
    tenant_id: str = Depends(get_tenant_id),
) -> Any:
    return await _handle_upload(file=file, session=session, tenant_id=tenant_id)


@router.get("/{doc_file_id}")
async def get_doc_metadata(
    doc_file_id: str,
    session: AsyncSession = Depends(get_tenant_scoped_session),
) -> Any:
    res = await session.execute(
        text(
            """
            SELECT id, filename, mime_type, size_bytes, sha256, source, ingest_status, created_at
            FROM app.doc_files WHERE id=:id
            """
        ),
        {"id": doc_file_id},
    )
    row = res.first()
    if not row:
        raise HTTPException(status_code=404, detail="Not found")

    cnt_res = await session.execute(
        text("SELECT COUNT(*) FROM app.doc_chunks WHERE doc_file_id=:id"), {"id": doc_file_id}
    )
    count = int(cnt_res.scalar() or 0)

    return {
        "id": str(row[0]),
        "filename": row[1],
        "mime_type": row[2],
        "size_bytes": row[3],
        "sha256": row[4],
        "source": row[5],
        "ingest_status": row[6],
        "created_at": str(row[7]),
        "chunks": count,
    }


