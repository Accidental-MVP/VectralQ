from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import BinaryIO, Tuple


MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB


def sha256_stream_and_save(fileobj: BinaryIO) -> Tuple[str, Path, int]:
    hasher = hashlib.sha256()
    total = 0
    fd, tmp_path_str = tempfile.mkstemp(prefix="vectralq_upload_")
    tmp_path = Path(tmp_path_str)
    with os.fdopen(fd, "wb") as out:
        while True:
            chunk = fileobj.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                out.close()
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass
                raise ValueError("file too large")
            hasher.update(chunk)
            out.write(chunk)
    return hasher.hexdigest(), tmp_path, total


