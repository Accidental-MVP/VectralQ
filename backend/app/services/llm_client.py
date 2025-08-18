from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, Optional

import httpx

from app.core.config import get_settings


SYSTEM_PROMPT = (
    "You are a retrieval-grounded assistant. Answer ONLY using the provided CONTEXT. "
    "If the answer is missing, reply exactly: I don’t have enough information. "
    "Always cite sources using the provided doc_file_id and chunk_id. "
    "Keep answers concise and accurate."
)


async def _call_openai_compatible(
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    timeout_ms: int,
    max_tokens: int,
    temperature: float,
) -> str:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    async with httpx.AsyncClient(timeout=timeout_ms / 1000) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return str(content)


def _fallback_generate(user_prompt: str, citations_hint: list[dict[str, str]]) -> str:
    # Deterministic fallback: stitch a short answer from available context headers
    facts = []
    for c in citations_hint[:2]:
        facts.append(f"From {c.get('doc_file_id')}#{c.get('chunk_id')}")
    answer = "; ".join(facts) if facts else "I don’t have enough information."
    result = {
        "answer": answer if facts else "I don’t have enough information.",
        "citations": citations_hint[:2],
        "confidence": 0.5 if facts else 0.0,
    }
    return json.dumps(result)


async def generate_answer(
    system_prompt: str,
    user_prompt: str,
    citations_hint: list[dict[str, str]] | None = None,
) -> str:
    settings = get_settings()
    base_url = settings.llm_base_url
    model = settings.llm_model or "gpt-oss-20b"
    timeout_ms = settings.llm_timeout_ms
    max_tokens = settings.llm_max_tokens
    temperature = settings.llm_temperature

    if not base_url:
        return _fallback_generate(user_prompt, citations_hint or [])

    try:
        text = await _call_openai_compatible(
            base_url=base_url,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            timeout_ms=timeout_ms,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return text
    except Exception:
        return _fallback_generate(user_prompt, citations_hint or [])


