from __future__ import annotations

import io
import subprocess
from pathlib import Path
from typing import Optional

from pdfminer.high_level import extract_text as pdfminer_extract_text


class ExtractionError(Exception):
    pass


def pdf_to_text(pdf_path: Path, timeout_seconds: int = 60) -> str:
    try:
        text = pdfminer_extract_text(str(pdf_path))
        if text is None:
            raise ExtractionError("Empty text extracted from PDF")
        return text
    except Exception as exc:  # noqa: BLE001
        raise ExtractionError(f"PDF extraction failed: {exc}") from exc


def txt_to_text(data: bytes) -> str:
    # Try common encodings, favor UTF-8 but handle UTF-16 LE/BE which is common on Windows
    for enc in ("utf-8", "utf-16", "utf-16-le", "utf-16-be", "latin-1"):
        try:
            text = data.decode(enc, errors="ignore")
            break
        except Exception:
            continue
    else:
        raise ExtractionError("TXT decode failed: unsupported encoding")

    # Remove any embedded NULs and normalize newlines
    text = text.replace("\x00", "")
    return text


