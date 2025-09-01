from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict, List


@lru_cache(maxsize=1)
def get_tenant_hints() -> Dict[str, Any]:
    """
    Load optional tenant-level parsing hints from /config/tenant.yaml if present.
    Structure:
    units:
      currency: ["USD","CAD","€","£"]
      duration: ["days","business days","hours"]
    section_markers:
      bullets: ["•","-","*","→"]
    ticker_pattern: "^[A-Z]{1,5}$"
    """
    path = os.getenv("TENANT_CONFIG_PATH", "/config/tenant.yaml")
    data: Dict[str, Any] = {}
    try:
        if os.path.exists(path):
            import yaml  # type: ignore
            with open(path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
            if isinstance(raw, dict):
                data = raw
    except Exception:
        # Non-fatal; return defaults
        pass
    return data


