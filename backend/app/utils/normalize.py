from __future__ import annotations

import re
import unicodedata


def normalize_query(text: str) -> str:
    """Lowercase, strip accents, collapse whitespace; keep ascii-alnum and basic punctuation.
    Original text is not mutated by this helper; callers can keep both forms.
    """
    if not text:
        return ""
    # Lowercase and NFKD normalize to strip accents
    s = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s
