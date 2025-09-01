from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, Optional
import re
import os

import httpx

from app.core.config import get_settings


SYSTEM_PROMPT = (
    "You are a retrieval-grounded assistant. Answer ONLY from the provided EVIDENCE spans. "
    "Every sentence in your answer MUST be supported by at least one evidence span tag like [D1:S3]. "
    "Return ONLY strict JSON with keys: answer (string), citations (array of {doc_file_id, chunk_id}), confidence (number). "
    "If evidence is insufficient or ambiguous, return exactly: {\"answer\":\"I don’t have enough information.\",\"citations\":[],\"confidence\":0.0}. No extra text."
)

LLM_FIRST_SYSTEM_PROMPT = (
    "You are a retrieval-grounded assistant. You must follow these rules: "
    "1) Answer ONLY from EVIDENCE. 2) Each sentence must include at least one evidence tag [Dx:Sy]. "
    "3) Output STRICT JSON only: {\\\"answer\\\": string, \\\"citations\\\": [{\\\"doc_file_id\\\": string, \\\"chunk_id\\\": string}], \\\"confidence\\\": number}. "
    "If evidence is missing or conflicting, respond exactly with the refusal JSON shown above."
)


async def _call_openai_compatible(
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    timeout_ms: int,
    max_tokens: int,
    temperature: float,
    stream: bool = False,
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
        "stream": stream,
    }
    # Optional: provider JSON mode, if supported
    try:
        if get_settings().answer_json_required:
            # OpenAI compatibility flag (some providers ignore)
            payload["response_format"] = {"type": "json_object"}
    except Exception:
        pass
    # Build headers (Authorization for OpenAI-compatible; optional org header)
    api_key = (get_settings().llm_api_key or os.getenv("OPENAI_API_KEY") or "").strip()
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    # Optional OpenAI headers for org/project scoping (esp. with project-scoped keys)
    org = os.getenv("OPENAI_ORG_ID") or os.getenv("OPENAI_ORG") or os.getenv("OPENAI_ORGANIZATION") or os.getenv("LLM_ORG")
    if org:
        headers["OpenAI-Organization"] = org.strip()
    proj = os.getenv("OPENAI_PROJECT_ID") or os.getenv("OPENAI_PROJECT") or os.getenv("LLM_PROJECT")
    if proj:
        headers["OpenAI-Project"] = proj.strip()

    async with httpx.AsyncClient(timeout=timeout_ms / 1000) as client:
        if not stream:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            try:
                content = data["choices"][0]["message"]["content"]
                return str(content)
            except Exception:
                # fallback if provider returns a different shape
                if isinstance(data, dict):
                    if "content" in data:
                        return str(data.get("content"))
                    if "text" in data:
                        return str(data.get("text"))
                return json.dumps({"answer": "I don’t have enough information.", "citations": [], "confidence": 0.0})
        # Streaming path: return concatenated final text
        text_chunks: list[str] = []
        async with client.stream("POST", url, json=payload, headers=headers) as sresp:
            sresp.raise_for_status()
            async for line in sresp.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                chunk = line[len("data: ") :].strip()
                if chunk == "[DONE]":
                    break
                try:
                    obj = json.loads(chunk)
                    delta = obj.get("choices", [{}])[0].get("delta", {}).get("content")
                    if delta:
                        text_chunks.append(delta)
                except Exception:
                    continue
        return "".join(text_chunks)


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


def _try_extract_json(raw_text: str) -> Optional[str]:
    # Try to extract a JSON object from text with potential code fences / preambles
    if not raw_text:
        return None
    s = raw_text.strip()
    # Strip common code-fence wrappers
    if s.startswith("```"):
        # remove first fence line
        s = s.split("\n", 1)[1] if "\n" in s else s
        # remove trailing fence
        if s.endswith("```"):
            s = s[: -3]
    s = s.strip()
    # Heuristic: find first '{' and last '}' and try to parse the slice
    l = s.find("{")
    r = s.rfind("}")
    if l != -1:
        # If we have an opening brace but no closing, take from l to end
        candidate = s[l : (r + 1) if (r != -1 and r > l) else len(s)]
        try:
            obj = json.loads(candidate)
            # Re-emit normalized JSON string
            return json.dumps(obj)
        except Exception:
            # Try balancing braces/brackets by appending missing closers
            try:
                open_curly = candidate.count("{")
                close_curly = candidate.count("}")
                open_square = candidate.count("[")
                close_square = candidate.count("]")
                fix = candidate
                # Close arrays first, then objects (simple heuristic)
                if close_square < open_square:
                    fix += "]" * (open_square - close_square)
                if close_curly < open_curly:
                    fix += "}" * (open_curly - close_curly)
                obj2 = json.loads(fix)
                return json.dumps(obj2)
            except Exception:
                pass
    # Try to locate JSON block using a regex for top-level braces (best-effort)
    try:
        matches = re.findall(r"\{[\s\S]*\}", s)
        for m in matches:
            try:
                obj = json.loads(m)
                return json.dumps(obj)
            except Exception:
                continue
    except Exception:
        pass
    return None


def _validate_or_fallback(raw_text: str, citations_hint: list[dict[str, str]]) -> str:
    # Ensure valid JSON with required keys; else fallback
    try:
        obj = json.loads(raw_text)
        if not isinstance(obj, dict):
            raise ValueError("not dict")
        # Coerce required fields
        answer = str(obj.get("answer") or "")
        citations = obj.get("citations")
        confidence = obj.get("confidence")
        if answer == "" or not isinstance(citations, list) or confidence is None:
            raise ValueError("missing keys")
        return json.dumps({
            "answer": answer,
            "citations": citations,
            "confidence": float(confidence) if isinstance(confidence, (int, float)) else 0.5,
        })
    except Exception as exc:
        # Attempt salvage from code-fences / embedded JSON and log diagnostics
        snippet = (raw_text or "").strip()[:180]
        has_fence = "```" in (raw_text or "")
        brace_balance = (raw_text or "").count("{") - (raw_text or "").count("}")
        extracted = _try_extract_json(raw_text or "")
        if extracted is not None:
            print({
                "event": "answer_invalid_json_salvaged",
                "raw_len": len(raw_text or ""),
                "snippet": snippet,
                "has_code_fence": has_fence,
                "brace_balance": brace_balance,
                "reason": str(type(exc).__name__),
            })
            return extracted
        print({
            "event": "answer_invalid_json",
            "raw_len": len(raw_text or ""),
            "snippet": snippet,
            "has_code_fence": has_fence,
            "brace_balance": brace_balance,
            "reason": str(type(exc).__name__),
        })
        return _fallback_generate("", citations_hint)


async def generate_answer(
    system_prompt: str,
    user_prompt: str,
    citations_hint: list[dict[str, str]] | None = None,
    stream: bool = False,
    max_tokens_override: int | None = None,
    temperature_override: float | None = None,
) -> str:
    settings = get_settings()
    base_url = settings.llm_base_url
    model = settings.llm_model or "gpt-oss-20b"
    timeout_ms = settings.llm_timeout_ms
    max_tokens = max_tokens_override if max_tokens_override is not None else settings.llm_max_tokens
    temperature = temperature_override if temperature_override is not None else settings.llm_temperature

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
            stream=stream,
        )
        return _validate_or_fallback(text, citations_hint or [])
    except Exception:
        return _fallback_generate(user_prompt, citations_hint or [])


async def generate_answer_stream(
    system_prompt: str,
    user_prompt: str,
    citations_hint: list[dict[str, str]] | None = None,
) -> Any:
    """
    Async generator yielding text deltas if provider supports streaming; falls back to single final chunk.
    """
    settings = get_settings()
    base_url = settings.llm_base_url
    model = settings.llm_model or "gpt-oss-20b"
    timeout_ms = settings.llm_timeout_ms
    # Prefer stream-specific cap when provided
    max_tokens = getattr(settings, "llm_stream_max_tokens", settings.llm_max_tokens)
    temperature = settings.llm_temperature

    if not base_url:
        yield _fallback_generate(user_prompt, citations_hint or [])
        return

    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }
    try:
        if settings.answer_json_required:
            payload["response_format"] = {"type": "json_object"}
    except Exception:
        pass

    # Build headers (Authorization for OpenAI-compatible; optional org header)
    api_key = (settings.llm_api_key or os.getenv("OPENAI_API_KEY") or "").strip()
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    # Optional OpenAI headers for org/project scoping (esp. with project-scoped keys)
    org = os.getenv("OPENAI_ORG_ID") or os.getenv("OPENAI_ORG") or os.getenv("OPENAI_ORGANIZATION") or os.getenv("LLM_ORG")
    if org:
        headers["OpenAI-Organization"] = org.strip()
    proj = os.getenv("OPENAI_PROJECT_ID") or os.getenv("OPENAI_PROJECT") or os.getenv("LLM_PROJECT")
    if proj:
        headers["OpenAI-Project"] = proj.strip()

    async with httpx.AsyncClient(timeout=timeout_ms / 1000) as client:
        try:
            async with client.stream("POST", url, json=payload, headers=headers) as sresp:
                sresp.raise_for_status()
                async for line in sresp.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    chunk = line[len("data: ") :].strip()
                    if chunk == "[DONE]":
                        break
                    try:
                        obj = json.loads(chunk)
                        delta = obj.get("choices", [{}])[0].get("delta", {}).get("content")
                        if delta:
                            yield delta
                    except Exception:
                        continue
        except Exception:
            # Fallback: single final JSON
            yield _fallback_generate(user_prompt, citations_hint or [])


