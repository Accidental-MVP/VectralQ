from __future__ import annotations

import base64
import json
from typing import Any

from cryptography.fernet import Fernet, InvalidToken

from app.core.config import get_settings


def _get_fernet() -> Fernet:
    key_b64 = get_settings().google_token_encryption_key
    if not key_b64:
        raise RuntimeError("GOOGLE_TOKEN_ENCRYPTION_KEY is not set")
    # Accept 32-byte raw base64 or fernet key directly
    try:
        # If it's a 32-byte base64, transform to Fernet key
        raw = base64.b64decode(key_b64)
        if len(raw) == 32:
            fkey = base64.urlsafe_b64encode(raw)
        else:
            # Assume already Fernet key
            fkey = key_b64.encode()
    except Exception:
        fkey = key_b64.encode()
    return Fernet(fkey)


def encrypt_json(obj: dict[str, Any]) -> str:
    f = _get_fernet()
    data = json.dumps(obj).encode("utf-8")
    return f.encrypt(data).decode("utf-8")


def decrypt_json(token: str) -> dict[str, Any]:
    f = _get_fernet()
    try:
        data = f.decrypt(token.encode("utf-8"))
        return json.loads(data.decode("utf-8"))
    except InvalidToken:
        raise RuntimeError("Invalid encryption key for stored token")


