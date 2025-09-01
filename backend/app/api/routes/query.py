from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional
import asyncio
import os
import re

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from app.api.deps import get_tenant_scoped_session, get_tenant_id
from app.core.config import get_settings
from app.core.logging import request_id_var
from app.services.search import run_bm25, run_vector, fuse_candidates, run_phrase_lane
from app.services.context_packer import pack_context_from_texts
from app.services.llm_client import SYSTEM_PROMPT, generate_answer


router = APIRouter(prefix="/query", tags=["query"])


class QueryOptions(BaseModel):
    top_k: Optional[int] = 6
    max_tokens: Optional[int] = 600
    sources: Optional[List[str]] = Field(default=None, description="Limit retrieval to sources, e.g., ['google_drive']")


class QueryRequest(BaseModel):
    question: str
    options: Optional[QueryOptions] = None


def _compute_confidence(num_chunks: int, top_score: float) -> float:
    # Simple evidence-based calibration
    conf = 0.9
    if num_chunks <= 1:
        conf -= 0.2
    if top_score < 0.2:
        conf -= 0.1
    return max(0.0, min(0.95, conf))


def _truncate(s: str, max_len: int = 220) -> str:
    if len(s) <= max_len:
        return s
    return s[: max_len - 1].rstrip() + "…"


@router.post("")
@router.post("/")
async def query(
    body: QueryRequest,
    session: AsyncSession = Depends(get_tenant_scoped_session),
    tenant_id: str = Depends(get_tenant_id),
) -> Any:
    settings = get_settings()
    q = (body.question or "").strip()
    if not q:
        raise HTTPException(status_code=422, detail="question must be non-empty")
    t0 = time.perf_counter()
    req_id = request_id_var.get() or "-"
    try:
        print({
            "event": "query_start",
            "request_id": req_id,
            "tenant_id": tenant_id,
            "q": q,
        })
    except Exception:
        pass
    # Fast-path deterministic intents (sub-10 ms): name and investment_in
    Q_NAME_RX = re.compile(r"\b(what\s+is\s+my\s+name|who\s+am\s+i|what's\s+my\s+name)\b", re.I)
    # Broader investment intent: covers "investment in X", "what is my investment in X", "how much investment in X", "amount in X"
    Q_INV_RX = re.compile(
        r"\b(?:(?:what\s+is\s+(?:my|our)\s+)?(?:investment|invested(?:\s+amount)?|amount)|how\s+much(?:\s+is)?(?:\s+(?:my|our))?\s+(?:investment|invested(?:\s+amount)?))\s+(?:in|for|into)\s+(?P<ent>[A-Za-z]{2,20})\b",
        re.I,
    )
    REFUSAL_RX = re.compile(r"^(?:i\s+don[’']?t\s+have\s+(?:enough|sufficient)\s+information\.?|i\s+don[’']?t\s+know\.?|insufficient\s+information\.?)(?:\s*)$", re.I)
    async def _fastpath_name() -> Optional[dict]:
        rows = await session.execute(
            text(
                """
                WITH s AS (
                  SELECT doc_file_id, chunk_id_sha1, jsonb_array_elements(sentences) AS s
                  FROM app.doc_chunks
                  WHERE tenant_id = :tenant
                )
                SELECT (s.s->>'text') AS text, doc_file_id, chunk_id_sha1
                FROM s
                WHERE (s.s->>'text') ~* '(my\\s+name\\s+is|^i\\s+am)\\s+[A-Z][A-Za-z]+(\\s+[A-Z][A-Za-z]+)*'
                LIMIT 1;
                """
            ),
            {"tenant": tenant_id},
        )
        r = rows.first()
        if r:
            ans = (r[0] or "").strip()
            if ans:
                return {
                    "answer": ans,
                    "citations": [{"doc_file_id": str(r[1]), "chunk_id": str(r[2])}],
                    "confidence": 0.9,
                }
        return None

    async def _fastpath_invest(entity: str) -> Optional[dict]:
        rows = await session.execute(
            text(
                """
                WITH s AS (
                  SELECT doc_file_id, chunk_id_sha1, jsonb_array_elements(sentences) AS s
                  FROM app.doc_chunks
                  WHERE tenant_id = :tenant
                )
                SELECT (s.s->>'text') AS text, doc_file_id, chunk_id_sha1
                FROM s
                WHERE (s.s->>'text') ILIKE ('%' || :entity || '%')
                  AND ( (s.s->>'text') ~* '\\b(\\d[\\d,\\.]*)\\s*(USD|CAD|INR|EUR|\\$|€|£|₹)\\b'
                        OR (s.s->>'text') ~* '(->|→)' )
                ORDER BY length(s.s->>'text') ASC
                LIMIT 1;
                """
            ),
            {"tenant": tenant_id, "entity": entity},
        )
        r = rows.first()
        if r:
            ans = (r[0] or "").strip()
            if ans:
                return {
                    "answer": ans,
                    "citations": [{"doc_file_id": str(r[1]), "chunk_id": str(r[2])}],
                    "confidence": 0.85,
                }
        return None

    intent_name = bool(Q_NAME_RX.search(q))
    m_inv = Q_INV_RX.search(q)
    if intent_name:
        fast = await _fastpath_name()
        if fast:
            latency_ms = int((time.perf_counter() - t0) * 1000)
            fast["latency_ms"] = latency_ms
            print({"event": "fastpath", "kind": "name", "hit": True, "total_ms": latency_ms})
            return fast
    elif m_inv:
        entity = (m_inv.group("ent") or "").upper()
        if entity and len(entity) >= 3 and entity not in {"THE","AND","FOR","WITH","THIS","FROM","IN","USD","CAD","EUR","INR"}:
            fast = await _fastpath_invest(entity)
            if fast:
                # Always use tiny LLM for slot-fill; build 1–2 evidence lines containing the entity
                import json as _json
                # Fetch up to 2 shortest lines with the entity to prefer bullets over section totals
                rows_sf = await session.execute(
                    text(
                        """
                        WITH s AS (
                          SELECT doc_file_id, chunk_id_sha1, jsonb_array_elements(sentences) AS s
                          FROM app.doc_chunks
                          WHERE tenant_id = :tenant
                        )
                        SELECT (s.s->>'text') AS text, doc_file_id, chunk_id_sha1
                        FROM s
                        WHERE to_tsvector('simple', coalesce(s.s->>'text','')) @@ plainto_tsquery('simple', :entity)
                        ORDER BY length(s.s->>'text') ASC
                        LIMIT 2;
                        """
                    ),
                    {"tenant": tenant_id, "entity": entity},
                )
                lines = rows_sf.fetchall()
                evid: List[dict] = []
                # Prefer lines that contain both the entity token and an amount+currency
                import re as _re
                AMT_RX = _re.compile(r"(?i)\b(\d{1,3}(?:,\d{3})*|\d+)(?:\s*(USD|CAD|INR|EUR|\$|€|£|₹|usd|cad|inr|eur))\b")
                def _has_entity_and_amount(s: str) -> bool:
                    return (entity.lower() in (s or '').lower()) and bool(AMT_RX.search(s or ''))
                # First collect entity+amount lines
                preferred: List[dict] = []
                fallback_lines: List[dict] = []
                for r in lines:
                    t = (r[0] or '').strip()
                    if not t:
                        continue
                    item = {"text": t, "doc_file_id": str(r[1]), "chunk_id": str(r[2])}
                    if _has_entity_and_amount(t):
                        preferred.append(item)
                    else:
                        if entity.lower() in t.lower():
                            fallback_lines.append(item)
                evid = (preferred or fallback_lines)[:2]
                if not evid:
                    try:
                        print({
                            "event": "fastpath_invest_no_entity_evidence",
                            "request_id": request_id_var.get() or "-",
                            "entity": entity,
                        })
                    except Exception:
                        pass
                    return {
                        "answer": "I don’t have enough information.",
                        "citations": [],
                        "confidence": 0.0,
                        "latency_ms": int((time.perf_counter() - t0) * 1000),
                    }
                try:
                    print({
                        "event": "fastpath_invest_evidence",
                        "request_id": request_id_var.get() or "-",
                        "entity": entity,
                        "evidence": [
                            {"text": (e.get("text") or "")[:160], "doc_file_id": e.get("doc_file_id"), "chunk_id": e.get("chunk_id")}
                            for e in evid
                        ],
                    })
                except Exception:
                    pass
                if not evid:
                    # fall back to the original fast line
                    evid = [{"text": fast["answer"], "doc_file_id": fast["citations"][0]["doc_file_id"], "chunk_id": fast["citations"][0]["chunk_id"]}]
                prompt_obj = {"question": q, "spans": evid[:2]}
                user_prompt = _json.dumps(prompt_obj, ensure_ascii=False)
                t_llm = time.perf_counter()
                raw_text = await generate_answer(
                    SYSTEM_PROMPT,
                    user_prompt,
                    citations_hint=[{"doc_file_id": e["doc_file_id"], "chunk_id": e["chunk_id"]} for e in evid[:2]],
                    max_tokens_override=96,
                    temperature_override=0.0,
                )
                llm_ms = int((time.perf_counter() - t_llm) * 1000)
                out_answer = None
                out_cites = None
                try:
                    obj = json.loads(raw_text)
                    out_answer = obj.get("answer")
                    out_cites = obj.get("citations")
                except Exception:
                    if isinstance(raw_text, str) and raw_text.strip():
                        out_answer = raw_text.strip()
                try:
                    print({
                        "event": "fastpath_invest_llm_parsed",
                        "request_id": request_id_var.get() or "-",
                        "refusal": bool(REFUSAL_RX.match((out_answer or "").strip())),
                        "answer_snippet": (out_answer or "")[:120],
                        "cites_n": len(out_cites or []),
                        "llm_ms": llm_ms,
                    })
                except Exception:
                    pass
                # If the model refused, or returned fallback-like content, extract amount from the best evidence line
                _uuid_sha_re_fp = _re.compile(r"(?i)\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}#[0-9a-f]{40}\b")
                _from_uuid_sha_re_fp = _re.compile(r"(?i)^\s*from\s+[0-9a-f-]{36}#[0-9a-f]{40}")
                looks_fallback = bool(out_answer and ("#" in out_answer) and (_uuid_sha_re_fp.search(out_answer) or _from_uuid_sha_re_fp.search(out_answer)))
                if (not out_answer) or REFUSAL_RX.match((out_answer or "").strip()) or looks_fallback:
                    # Scan evidence for the best line that contains both entity and amount
                    best = None
                    for ev in evid:
                        txt_ev = (ev["text"] or '')
                        if (entity.lower() in txt_ev.lower()):
                            m_amt = AMT_RX.search(txt_ev)
                            if m_amt:
                                best = (ev, m_amt)
                                break
                    if best:
                        ev, m_amt = best
                        amt = m_amt.group(1).replace(",", "")
                        cur = m_amt.group(2).upper()
                        cur = {"$": "USD", "€": "EUR", "£": "GBP", "₹": "INR"}.get(cur, cur)
                        out_answer = f"Your investment in {entity} is {amt} {cur}."
                        out_cites = [{"doc_file_id": ev["doc_file_id"], "chunk_id": ev["chunk_id"]}]
                        try:
                            print({
                                "event": "fastpath_invest_fallback_synth",
                                "request_id": request_id_var.get() or "-",
                                "entity": entity,
                                "amount": amt,
                                "currency": cur,
                            })
                        except Exception:
                            pass
                if out_answer:
                    fast["answer"] = out_answer
                if out_cites:
                    fast["citations"] = out_cites
                print({"event": "fastpath", "kind": "investment_in", "hit": True, "llm_ms": llm_ms})
                latency_ms = int((time.perf_counter() - t0) * 1000)
                fast["latency_ms"] = latency_ms
                return fast

    opts = body.options or QueryOptions()
    top_k = int(opts.top_k or settings.context_max_chunks)
    top_k = max(1, min(top_k, 12))

    t0 = time.perf_counter()
    # Retrieval via existing services
    filters: Dict[str, Any] | None = None
    if opts.sources:
        filters = {"source": list(opts.sources)}
    # Optional phrase lane bonus
    phrase_map: Dict[str, float] = {}
    if settings.phrase_lane_enabled:
        phrase_map = await run_phrase_lane(session, q, settings.bm25_k or settings.search_bm25_limit, filters)

    # Run BM25 and vector in parallel (timed)
    t_bm = time.perf_counter()
    bm_task = asyncio.create_task(run_bm25(session, q, settings.bm25_k or settings.search_bm25_limit, filters))
    ve_task = asyncio.create_task(run_vector(session, q, settings.vec_k or settings.search_vector_limit, filters))
    bm25_results, vector_results = await asyncio.gather(bm_task, ve_task)
    bm25_ms = int((time.perf_counter() - t_bm) * 1000)
    # Apply phrase bonus pre-fusion
    if phrase_map:
        boost = get_settings().phrase_boost
        for c in bm25_results:
            if c.chunk_id_sha1 in phrase_map:
                c.phrase_bonus = boost
        for c in vector_results:
            if c.chunk_id_sha1 in phrase_map:
                c.phrase_bonus = boost
    t_fuse = time.perf_counter()
    fused_results = fuse_candidates(
        bm25_results,
        vector_results,
        mode=settings.search_fusion_mode,
        linear_lambda=settings.search_linear_lambda,
        rrf_k=settings.search_rrf_k,
    )
    fuse_ms = int((time.perf_counter() - t_fuse) * 1000)
    fused_results = fused_results[:top_k]

    # Build BM25 rank map for gating (position-based)
    bm25_rank_map: Dict[str, int] = {}
    for idx, c in enumerate(bm25_results, start=1):
        bm25_rank_map[c.chunk_id_sha1] = idx
    # Vector similarity map for top fused lookups
    vec_sim_map: Dict[str, float] = {c.chunk_id_sha1: float(c.semantic_score or 0.0) for c in vector_results}
    # Consider BM25 hit within a relaxed window (default 30) across top fused window (8)
    bm25_window = int(os.getenv("BM25_RANK_WINDOW", "30"))
    top_fused_window = int(os.getenv("FUSED_TOP_WINDOW", "8"))
    bm25_rank_ok = any(bm25_rank_map.get(r.chunk_id, 10**9) <= bm25_window for r in fused_results[:top_fused_window])
    # Strong vector signal gate for top fused candidate
    vec_strong_min = float(os.getenv("VEC_STRONG_MIN", "0.78"))
    top_vec_sim = vec_sim_map.get(fused_results[0].chunk_id, 0.0) if fused_results else 0.0
    vec_strong = fused_results and (top_vec_sim >= vec_strong_min)

    # Refusal / hallucination guard (configurable)
    min_results = max(1, settings.refusal_min_results)
    min_top = max(0.0, settings.refusal_min_top_score)
    # Optional: RRF-based gating – minimal RRF if configured (kept but default 0.0)
    min_rrf = max(0.0, getattr(settings, "min_rrf", 0.0))
    early_refuse = False
    reason = ""
    if len(fused_results) < min_results:
        early_refuse = True
        reason = "min_results"
    elif fused_results and fused_results[0].score < min_top:
        early_refuse = True
        reason = "min_top"
    # Temporary evidence gate: allow if either BM25 is present (within window) OR vector similarity is strong
    elif not (bm25_rank_ok or vec_strong):
        early_refuse = True
        reason = "no_evidence"
    if early_refuse:
        top_score = float(fused_results[0].score) if fused_results else 0.0
        req_id = request_id_var.get() or "-"
        print({
            "event": "refusal",
            "request_id": req_id,
            "refusal_reason": reason,
            "len": len(fused_results),
            "top_score": top_score,
            "min_results": min_results,
            "min_top": min_top,
            "bm25_rank_ok": bm25_rank_ok,
            "vec_strong": bool(vec_strong),
            "top_vec_sim": float(top_vec_sim),
            "bm25_window": bm25_window,
            "top_fused_window": top_fused_window,
        })
        return {
            "answer": "I don’t have enough information.",
            "citations": [],
            "confidence": 0.0,
            "latency_ms": int((time.perf_counter() - t0) * 1000),
        }

    # Span picker: prefilter sentences per chunk and pick top 1–2 sentences overall
    id_pairs = [(r.doc_file_id, r.chunk_id) for r in fused_results]
    # Fetch text and sentences for those chunks
    sentence_top = get_settings().sentence_prefilter_top
    doc_ids = [d for (d, _) in id_pairs]
    allowed = {(str(d), str(c)) for (d, c) in id_pairs}
    t_sent_fetch = time.perf_counter()
    rows = await session.execute(
            text(
                """
            SELECT doc_file_id, chunk_id_sha1, text, sentences
            FROM app.doc_chunks
            WHERE doc_file_id = ANY(:doc_ids)
                """
            ),
        {"doc_ids": doc_ids},
    )
    fetched = rows.fetchall()
    sent_fetch_ms = int((time.perf_counter() - t_sent_fetch) * 1000)
    # Prefilter: top N sentences by overlap+heuristics
    def _terms(s: str) -> set:
        return set([t.lower() for t in re.findall(r"\w+", s)])
    q_terms = _terms(q)
    candidates: List[tuple[str, str, str]] = []  # (doc_id, chunk_id, sentence/span)
    import json as _json
    t_sent_score = time.perf_counter()
    fetched_map: Dict[tuple[str, str], str] = {}
    sent_index_map: Dict[tuple[str, str, str], int] = {}
    chunk_sents_map: Dict[tuple[str, str], list] = {}
    # Heuristic cleaner to drop inline id-ish markers like "<uuid>#<sha1>"
    _uuid_sha_re = re.compile(r"(?i)\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}#[0-9a-f]{40}\b")
    _from_uuid_sha_re = re.compile(r"(?i)^\s*from\s+[0-9a-f-]{36}#[0-9a-f]{40}\s*$")
    def _clean_sentence_text(text_in: str) -> str:
        # Remove entire metadata lines like: "From <uuid>#<sha>"
        if _from_uuid_sha_re.match(text_in or ""):
            return ""
        # Strip any embedded <uuid>#<sha> tokens
        t = _uuid_sha_re.sub("", text_in or "")
        t = re.sub(r"\s+", " ", t).strip(" \t-:")
        # If leftover is just a dangling 'From' or starts with it without substance, drop
        tl = t.strip()
        if tl.lower().startswith("from"):
            rest = tl[4:].strip(" \t:-")
            if not rest or len(rest.split()) <= 2:
                return ""
        # If any hash markers remain, it's likely an identifier fragment
        if "#" in tl:
            return ""
        return t.strip()

    def _is_bad_answer(text_in: str) -> bool:
        t = (text_in or "").strip()
        if not t:
            return True
        if _from_uuid_sha_re.match(t):
            return True
        if _uuid_sha_re.search(t):
            return True
        # Very low signal
        if sum(ch.isalnum() for ch in t) < 4:
            return True
        return False

    def _best_line_from_chunk(query_text: str, chunk_text: str) -> str:
        if not chunk_text:
            return ""
        best_line = ""
        best_over = -1.0
        for raw_line in (chunk_text or "").splitlines():
            ln = _clean_sentence_text(raw_line)
            if not ln:
                continue
            over = 0.0
            try:
                over = len(q_terms & _terms(ln)) / max(1, len(q_terms))
            except Exception:
                over = 0.0
            if over > best_over:
                best_over = over
                best_line = ln
        if not best_line:
            # fallback to first non-empty cleaned line
            for raw_line in (chunk_text or "").splitlines():
                ln = _clean_sentence_text(raw_line)
                if ln:
                    best_line = ln
                    break
        return _truncate(best_line, 220)
    for doc_id, chunk_id, txt, sents in fetched:
        key = (str(doc_id), str(chunk_id))
        if key not in allowed:
            continue
        fetched_map[key] = txt or ""
        try:
            arr = _json.loads(sents or "[]")
        except Exception:
            arr = []
        chunk_sents_map[key] = arr
        # Heuristic scoring helpers (prefer declarative facts; allow bullets/lines)
        QUESTION_RX = re.compile(r'^\s*(what|who|when|where|why|how|do|does|did|is|are|was|were|can|could|should)\b', re.I)
        FACT_RX = re.compile(r'\b(is|are|was|were|has|have|includes|means|=)\b', re.I)
        HAS_FACTUAL = re.compile(r'\b([A-Z][a-z]+|[0-9]{2,}|[0-9]+(?:\.[0-9]+)?\s?(USD|CAD|%|days?))\b')
        def _sentence_bonus(s: str) -> float:
            bonus = 0.0
            if '?' in s or QUESTION_RX.search(s):
                bonus -= 0.25
            if FACT_RX.search(s):
                bonus += 0.20
            if HAS_FACTUAL.search(s):
                bonus += 0.10
            if len(s.strip()) < 12:
                bonus -= 0.10
            return bonus
        scored: List[tuple[float, str]] = []
        # Entity-first preference: if the query contains candidate symbols/words like ETON, boost spans with those exact tokens
        q_entities = [t for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9._-]+", q) if len(t) >= 3]
        for obj in arr[:200]:
            stxt = str(obj.get("text") or "").strip()
            if not stxt:
                continue
            stxt_clean = _clean_sentence_text(stxt)
            if not stxt_clean or sum(ch.isalnum() for ch in stxt_clean) < 4:
                # Skip id-only or degenerate fragments
                continue
            over = 0.0
            try:
                over = len(q_terms & _terms(stxt_clean)) / max(1, len(q_terms))
            except Exception:
                over = 0.0
            entity_bonus = 0.0
            for ent in q_entities[:4]:
                if ent.lower() in stxt_clean.lower():
                    entity_bonus += 0.25
            scored.append((over + _sentence_bonus(stxt_clean) + entity_bonus, stxt_clean))
            try:
                idx_val = int(obj.get("idx")) if obj.get("idx") is not None else None
            except Exception:
                idx_val = None
            if idx_val is not None:
                sent_index_map[(str(doc_id), str(chunk_id), stxt_clean)] = idx_val
        scored.sort(key=lambda x: x[0], reverse=True)
        for _, stxt in scored[:max(1, sentence_top)]:
            candidates.append((str(doc_id), str(chunk_id), stxt))

    # Pick top 1–2 sentences across chunks by overlap+heuristics
    def _score_sentence(s: str) -> float:
        try:
            j = len(q_terms & _terms(s)) / max(1, len(q_terms))
        except Exception:
            j = 0.0
        # Reuse local regex from above scope
        return j  # already included bonus per-chunk selection

    candidates.sort(key=lambda t: _score_sentence(t[2]), reverse=True)
    # QA adjacency: if top is a question, prefer the next sentence in same chunk
    if candidates:
        top_doc, top_chunk, top_s = candidates[0]
        if ('?' in top_s) or re.match(r'^\s*(what|who|when|where|why|how|do|does|did|is|are|was|were|can|could|should)\b', top_s, re.I):
            idx0 = sent_index_map.get((top_doc, top_chunk, top_s))
            if idx0 is not None:
                arr = chunk_sents_map.get((top_doc, top_chunk)) or []
                if idx0 + 1 < len(arr):
                    nxt = str((arr[idx0 + 1] or {}).get("text") or "").strip()
                    nxt = _clean_sentence_text(nxt)
                    if nxt:
                        candidates.insert(0, (top_doc, top_chunk, nxt))
    top_spans = candidates[:2] if candidates else []
    overlap_best = (len(q_terms & _terms(top_spans[0][2])) / max(1, len(q_terms))) if top_spans else 0.0
    sent_score_ms = int((time.perf_counter() - t_sent_score) * 1000)
    if not top_spans or overlap_best < get_settings().min_overlap:
        # Slot-fill fallback: if intent investment_in and synth enabled, pull entity lines and synthesize
        if m_inv and settings.synthesize_from_spans:
            entity_sf = (m_inv.group("ent") or "").upper()
            if entity_sf and len(entity_sf) >= 3 and entity_sf not in {"THE","AND","FOR","WITH","THIS","FROM","IN","USD","CAD","EUR","INR"}:
                rows_sf = await session.execute(
                    text(
                        """
                        WITH s AS (
                          SELECT doc_file_id, chunk_id_sha1, jsonb_array_elements(sentences) AS s
                          FROM app.doc_chunks
                          WHERE tenant_id = :tenant
                        )
                        SELECT (s.s->>'text') AS text, doc_file_id, chunk_id_sha1
                        FROM s
                        WHERE to_tsvector('simple', coalesce(s.s->>'text','')) @@ plainto_tsquery('simple', :entity)
                        ORDER BY length(s.s->>'text') ASC
                        LIMIT 3;
                        """
                    ),
                    {"tenant": tenant_id, "entity": entity_sf},
                )
                sf = rows_sf.fetchall()
                evid = [
                    {"text": r[0], "doc_file_id": str(r[1]), "chunk_id": str(r[2])}
                    for r in sf if (r[0] or '').strip() and entity_sf.lower() in (r[0] or '').lower()
                ]
                if not evid:
                    try:
                        print({
                            "event": "slotfill_no_entity_evidence",
                            "request_id": request_id_var.get() or "-",
                            "entity": entity_sf,
                        })
                    except Exception:
                        pass
                    return {
                        "answer": "I don’t have enough information.",
                        "citations": [],
                        "confidence": 0.0,
                        "latency_ms": int((time.perf_counter() - t0) * 1000),
                    }
                if evid:
                    import json as _json
                    prompt = _json.dumps({"question": q, "spans": evid[:2]}, ensure_ascii=False)
                    try:
                        print({
                            "event": "slotfill_synth_request",
                            "request_id": request_id_var.get() or "-",
                            "entity": entity_sf,
                            "evidence": [
                                {"text": (e.get("text") or "")[:160], "doc_file_id": e.get("doc_file_id"), "chunk_id": e.get("chunk_id")}
                                for e in evid[:2]
                            ],
                        })
                    except Exception:
                        pass
                    t_llm = time.perf_counter()
                    raw = await generate_answer(
                        SYSTEM_PROMPT,
                        prompt,
                        citations_hint=[{"doc_file_id": e["doc_file_id"], "chunk_id": e["chunk_id"]} for e in evid[:2]],
                        max_tokens_override=36,
                        temperature_override=0.0,
                    )
                    llm_ms = int((time.perf_counter() - t_llm) * 1000)
                    # Parse answer; if refusal, deterministically synthesize from entity+amount in evidence
                    ans = None
                    cites = None
                    try:
                        obj = json.loads(raw)
                        ans = obj.get("answer")
                        cites = obj.get("citations")
                    except Exception:
                        # ignore
                        pass
                    if not ans or REFUSAL_RX.match((ans or "").strip()):
                        import re as _re
                        AMT_RX = _re.compile(r"(?i)\\b(\\d{1,3}(?:,\\d{3})*|\\d+)(?:\\s*(USD|CAD|INR|EUR|\\$|€|£|₹|usd|cad|inr|eur))\\b")
                        best = None
                        for ev in evid:
                            tline = (ev["text"] or "")
                            if entity_sf.lower() in tline.lower():
                                m_amt = AMT_RX.search(tline)
                                if m_amt:
                                    best = (ev, m_amt)
                                    break
                        if best:
                            ev, m_amt = best
                            amt = m_amt.group(1).replace(",", "")
                            cur = m_amt.group(2).upper()
                            cur = {"$": "USD", "€": "EUR", "£": "GBP", "₹": "INR"}.get(cur, cur)
                            ans = f"Your investment in {entity_sf} is {amt} {cur}."
                            cites = [{"doc_file_id": ev["doc_file_id"], "chunk_id": ev["chunk_id"]}]
                    try:
                        print({
                            "event": "slotfill_synth_outcome",
                            "request_id": request_id_var.get() or "-",
                            "refusal": bool(REFUSAL_RX.match((ans or "").strip())),
                            "answer_snippet": (ans or "")[:120],
                            "cites_n": len(cites or []),
                            "llm_ms": llm_ms,
                        })
                    except Exception:
                        pass
                    if not ans:
                        # last resort extractive
                        ans = evid[0]["text"]
                        cites = [{"doc_file_id": evid[0]["doc_file_id"], "chunk_id": evid[0]["chunk_id"]}]
                    latency_ms = int((time.perf_counter() - t0) * 1000)
                    return {
                        "answer": ans,
                        "citations": cites,
                        "confidence": 0.85,
                        "latency_ms": latency_ms,
                    }
        # Deterministic extractive fallback: use top fused chunk's best line (no LLM)
        top = fused_results[0] if fused_results else None
        if top is None:
            return {
                "answer": "I don’t have enough information.",
                "citations": [],
                "confidence": 0.0,
                "latency_ms": int((time.perf_counter() - t0) * 1000),
            }
        key = (str(top.doc_file_id), str(top.chunk_id))
        chunk_txt = fetched_map.get(key, "")
        best_line = _best_line_from_chunk(q, chunk_txt)
        if not best_line:
            best_line = _truncate(chunk_txt.strip().splitlines()[0] if chunk_txt else "", 220)
        used_citations = [{"doc_file_id": top.doc_file_id, "chunk_id": top.chunk_id}]
        latency_ms = int((time.perf_counter() - t0) * 1000)
        confidence = _compute_confidence(len(used_citations), fused_results[0].score if fused_results else 0.0)
        req_id = request_id_var.get() or "-"
        try:
            await session.execute(
                text(
                    """
                    INSERT INTO app.query_logs (tenant_id, request_id, question, top_k, selected_chunks, model, generation_ms, confidence, refused, streaming)
                    VALUES (:tenant_id, :request_id, :question, :top_k, :selected_chunks, :model, :generation_ms, :confidence, :refused, false)
                    """
                ),
                {
                    "tenant_id": tenant_id,
                    "request_id": req_id,
                    "question": q,
                    "top_k": top_k,
                    "selected_chunks": len(used_citations),
                    "model": settings.llm_model or "fallback",
                    "generation_ms": latency_ms,
                    "confidence": confidence,
                    "refused": False,
                },
            )
            await session.commit()
            try:
                print({
                    "event": "telemetry_query_ok",
                    "request_id": req_id,
                    "bm25_ms": locals().get("bm25_ms", None),
                    "fuse_ms": locals().get("fuse_ms", None),
                    "sent_fetch_ms": locals().get("sent_fetch_ms", None),
                    "sent_score_ms": locals().get("sent_score_ms", None),
                    "llm_ms": 0,
                    "total_ms": latency_ms,
                    "bm25_n": len(bm25_results),
                    "vec_n": len(vector_results),
                    "fused_k": len(fused_results),
                    "spans": 0,
                })
            except Exception:
                pass
        except Exception as exc:
            print({"event": "telemetry_query_err", "error": str(exc)})
        return {
            "answer": best_line,
            "citations": used_citations,
            "confidence": confidence,
            "latency_ms": latency_ms,
            "coverage": 1.0 if used_citations else 0.0,
        }
    else:
        # Prepare final answer extractively (optional synth kept off for now)
        # Build concise answer from spans, trimming excess whitespace
        answer_text = "\n".join([s.strip() for _, _, s in top_spans if s and s.strip()])
        used_citations = [{"doc_file_id": d, "chunk_id": c} for (d, c, _) in top_spans]
        # Fast-path: bypass LLM when spans are present and synthesis disabled
        if not settings.synthesize_from_spans:
            latency_ms = int((time.perf_counter() - t0) * 1000)
            answer = answer_text.strip()
            # If span looks like an identifier/metadata, pick best natural line from the chunk text
            if _is_bad_answer(answer) and used_citations:
                key = (str(used_citations[0]["doc_file_id"]), str(used_citations[0]["chunk_id"]))
                chunk_txt = fetched_map.get(key, "")
                fallback_line = _best_line_from_chunk(q, chunk_txt)
                if fallback_line:
                    answer = fallback_line
            citations = used_citations
            confidence = _compute_confidence(len(used_citations), fused_results[0].score if fused_results else 0.0)
            # Minimal DB log
            req_id = request_id_var.get() or "-"
            try:
                await session.execute(
                    text(
                        """
                        INSERT INTO app.query_logs (tenant_id, request_id, question, top_k, selected_chunks, model, generation_ms, confidence, refused, streaming)
                        VALUES (:tenant_id, :request_id, :question, :top_k, :selected_chunks, :model, :generation_ms, :confidence, :refused, false)
                        """
                    ),
                    {
                        "tenant_id": tenant_id,
                        "request_id": req_id,
                        "question": q,
                        "top_k": top_k,
                        "selected_chunks": len(used_citations),
                        "model": settings.llm_model or "fallback",
                        "generation_ms": latency_ms,
                        "confidence": confidence,
                        "refused": False,
                    },
                )
                await session.commit()
                try:
                    print({
                        "event": "telemetry_query_ok",
                        "request_id": req_id,
                        "bm25_ms": locals().get("bm25_ms", None),
                        "fuse_ms": locals().get("fuse_ms", None),
                        "sent_fetch_ms": locals().get("sent_fetch_ms", None),
                        "sent_score_ms": locals().get("sent_score_ms", None),
                        "llm_ms": 0,
                        "total_ms": latency_ms,
                        "bm25_n": len(bm25_results),
                        "vec_n": len(vector_results),
                        "fused_k": len(fused_results),
                        "spans": len(used_citations),
                    })
                except Exception:
                    pass
            except Exception as exc:
                print({"event": "telemetry_query_err", "error": str(exc)})
            coverage = 1.0 if used_citations else 0.0
            return {
                "answer": answer,
                "citations": citations,
                "confidence": confidence,
                "latency_ms": latency_ms,
                "coverage": coverage,
            }
        # If we do synthesize, only do it when pronouns/POV need fixing; else return extractive
        NEEDS_REWRITE = re.compile(r"\b(my|our|your)\b", re.I)
        CURRENCY_RX = re.compile(r"\b(USD|CAD|INR|EUR|Rs\.?|₹|\$|€|£|%|\d{1,3}(,\d{3})*(\.\d+)?|\d+\s?(USD|CAD|INR|EUR|%) )\b", re.I)
        ARROW_RX = re.compile(r"(->|→)")
        def _should_micro_synth(question_text: str, span_text: str) -> bool:
            qt = (question_text or "")
            st = (span_text or "")
            # pronoun POV fixes
            if NEEDS_REWRITE.search(qt) or re.search(r"\bmy name is\b", st.lower()):
                return True
            # slot-fill heuristics: entity present in question and numeric/currency or bullet/arrow line in evidence
            entities = [t for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9._-]+", qt) if len(t) >= 3]
            if entities:
                has_ent = any(ent.lower() in st.lower() for ent in entities[:4])
                if has_ent and (CURRENCY_RX.search(st) or ARROW_RX.search(st) or any(w in qt.lower().split() for w in ["in", "for", "of"])):
                    return True
            return False

        # If synthesis enabled but not needed (no pronouns/entity rewrite), keep extractive fast-path
        if settings.synthesize_from_spans and not _should_micro_synth(q, top_spans[0][2]):
            latency_ms = int((time.perf_counter() - t0) * 1000)
            answer = answer_text.strip()
            if _is_bad_answer(answer) and used_citations:
                key = (str(used_citations[0]["doc_file_id"]), str(used_citations[0]["chunk_id"]))
                chunk_txt = fetched_map.get(key, "")
                fallback_line = _best_line_from_chunk(q, chunk_txt)
                if fallback_line:
                    answer = fallback_line
            citations = used_citations
            confidence = _compute_confidence(len(used_citations), fused_results[0].score if fused_results else 0.0)
            req_id = request_id_var.get() or "-"
            try:
                await session.execute(
            text(
                """
                        INSERT INTO app.query_logs (tenant_id, request_id, question, top_k, selected_chunks, model, generation_ms, confidence, refused, streaming)
                        VALUES (:tenant_id, :request_id, :question, :top_k, :selected_chunks, :model, :generation_ms, :confidence, :refused, false)
                """
            ),
                    {
                        "tenant_id": tenant_id,
                        "request_id": req_id,
                        "question": q,
                        "top_k": top_k,
                        "selected_chunks": len(used_citations),
                        "model": settings.llm_model or "fallback",
                        "generation_ms": latency_ms,
                        "confidence": confidence,
                        "refused": False,
                    },
                )
                await session.commit()
            except Exception:
                pass
            return {
                "answer": answer,
                "citations": citations,
                "confidence": confidence,
                "latency_ms": latency_ms,
                "coverage": 1.0 if used_citations else 0.0,
            }

        # Else: micro-synthesis with tiny JSON prompt using spans only
        import json as _json
        prompt_obj = {
            "question": q,
            "spans": [
                {"text": s, "doc_file_id": d, "chunk_id": c}
                for (d, c, s) in top_spans[:3]
            ],
        }
        user_prompt = _json.dumps(prompt_obj, ensure_ascii=False)


    # Generate
    t_llm = time.perf_counter()
    # Tiny synthesis: clamp tokens and temperature to keep it fast and deterministic
    raw_text = await generate_answer(
        SYSTEM_PROMPT,
        user_prompt,
        citations_hint=used_citations,
        max_tokens_override=min(64, get_settings().llm_max_tokens or 64),
        temperature_override=0.0,
    )
    llm_ms = int((time.perf_counter() - t_llm) * 1000)

    # Validate/normalize output
    answer: str
    citations: List[dict]
    confidence = _compute_confidence(len(used_citations), fused_results[0].score if fused_results else 0.0)
    parsed_ok = True
    try:
        obj = json.loads(raw_text)
        answer = str(obj.get("answer") or "I don’t have enough information.")
        citations = obj.get("citations")
        if citations is None:
            citations = used_citations
        # Enforce citations subset of used_citations
        allowed = {f"{c['doc_file_id']}#{c['chunk_id']}" for c in used_citations}
        clean: List[dict] = []
        for c in citations:
            if not isinstance(c, dict):
                continue
            key = f"{c.get('doc_file_id')}#{c.get('chunk_id')}"
            if key in allowed:
                clean.append({"doc_file_id": c.get("doc_file_id"), "chunk_id": c.get("chunk_id")})
        citations = clean
    except Exception:
        parsed_ok = False
        answer = raw_text.strip()
        if not answer:
            answer = "I don’t have enough information."
        citations = used_citations
    try:
        print({
            "event": "llm_answer_parsed",
            "request_id": request_id_var.get() or "-",
            "refusal": bool(REFUSAL_RX.match((answer or "").strip())),
            "answer_snippet": (answer or "")[:120],
            "cites_n": len(citations or []),
            "llm_ms": llm_ms,
        })
    except Exception:
        pass

    # If model explicitly refused
    if parsed_ok and REFUSAL_RX.match((answer or "").strip()):
        req_id = request_id_var.get() or "-"
        top_score = float(fused_results[0].score) if fused_results else 0.0
        print({
            "event": "refusal",
            "request_id": req_id,
            "refusal_reason": "llm_refusal",
            "len": len(citations or []),
            "top_score": top_score,
        })

    # Post-generation refusal: require citations subset non-empty unless explicit refusal
    if (not REFUSAL_RX.match((answer or "").strip()) and not citations):
        req_id = request_id_var.get() or "-"
        top_score = float(fused_results[0].score) if fused_results else 0.0
        print({
            "event": "refusal",
            "request_id": req_id,
            "refusal_reason": "no_citations",
            "len": 0,
            "top_score": top_score,
        })
        answer = "I don’t have enough information."
        confidence = 0.0

    latency_ms = int((time.perf_counter() - t0) * 1000)
    # Log
    req_id = request_id_var.get() or "-"
    try:
        await session.execute(
            text(
                """
                INSERT INTO app.query_logs (tenant_id, request_id, question, top_k, selected_chunks, model, generation_ms, confidence, refused, streaming)
                VALUES (:tenant_id, :request_id, :question, :top_k, :selected_chunks, :model, :generation_ms, :confidence, :refused, false)
                """
            ),
            {
                "tenant_id": tenant_id,
                "request_id": req_id,
                "question": q,
                "top_k": top_k,
                "selected_chunks": len(used_citations),
                "model": settings.llm_model or "fallback",
                "generation_ms": latency_ms,
                "confidence": confidence,
                "refused": bool(REFUSAL_RX.match((answer or "").strip())),
            },
        )
        await session.commit()
        print({
            "event": "telemetry_query_ok",
            "request_id": req_id,
            "bm25_ms": bm25_ms,
            "fuse_ms": fuse_ms,
            "sent_fetch_ms": sent_fetch_ms,
            "sent_score_ms": sent_score_ms,
            "llm_ms": llm_ms,
            "total_ms": latency_ms,
            "bm25_n": len(bm25_results),
            "vec_n": len(vector_results),
            "fused_k": len(fused_results),
            "spans": len(used_citations),
        })
    except Exception as exc:
        print({"event": "telemetry_query_err", "error": str(exc)})

    # Extractive fallback: only if model failed JSON and enabled in settings
    try_extractive = settings.refusal_enable_extractive
    if (not parsed_ok) and try_extractive:
        if fused_results:
            top = fused_results[0]
            # Use picked span if available
            if top_spans:
                answer = _truncate(top_spans[0][2], 220)
                citations = [
                    {"doc_file_id": top_spans[0][0], "chunk_id": top_spans[0][1]}
                ]
                confidence = 1.0

    # Stub coverage tokens metric
    coverage = 0.0
    if used_citations:
        cited = {f"{c['doc_file_id']}#{c['chunk_id']}" for c in citations}
        coverage = round(len(cited) / max(1, len(used_citations)), 3)

    return {
        "answer": answer,
        "citations": citations,
        "confidence": confidence,
        "latency_ms": latency_ms,
        "coverage": coverage,
    }


