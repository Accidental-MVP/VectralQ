from __future__ import annotations

import math

import pytest

from app.services.search import Candidate, fuse_candidates, build_snippet


def make_candidate(chunk_id: str, bm: float | None, sem: float | None, created_at: str = "2024-01-01T00:00:00Z") -> Candidate:
    return Candidate(
        id=chunk_id,
        tenant_id="t",
        doc_file_id="d",
        chunk_id_sha1=chunk_id,
        text="alpha beta gamma",
        token_count=100,
        source="upload",
        created_at=created_at,
        bm25_score=bm,
        semantic_score=sem,
    )


def test_linear_fusion_ordering():
    bm = [make_candidate("a", bm=0.9, sem=None), make_candidate("b", bm=0.7, sem=None)]
    ve = [make_candidate("b", bm=None, sem=0.8), make_candidate("c", bm=None, sem=0.6)]
    res = fuse_candidates(bm, ve, mode="linear", linear_lambda=0.6, rrf_k=60)
    ids = [c.chunk_id for c in res]
    assert ids[0] in ("b", "a")
    assert "c" in ids


def test_rrf_fusion_rank_sum():
    bm = [make_candidate("x", bm=1.0, sem=None), make_candidate("y", bm=0.5, sem=None)]
    ve = [make_candidate("y", bm=None, sem=1.0), make_candidate("z", bm=None, sem=0.9)]
    res = fuse_candidates(bm, ve, mode="rrf", linear_lambda=0.6, rrf_k=60)
    ids = [c.chunk_id for c in res]
    # Item present in both lists should rank highly
    assert ids[0] == "y"


def test_minmax_handles_identical_scores():
    bm = [make_candidate("a", bm=1.0, sem=None), make_candidate("b", bm=1.0, sem=None)]
    ve = []
    res = fuse_candidates(bm, ve, mode="linear", linear_lambda=0.6, rrf_k=60)
    # Scores equal should not crash; preserve order by tie-breakers
    assert {c.chunk_id for c in res} == {"a", "b"}


def test_snippet_semantic_fallback():
    c = make_candidate("a", bm=None, sem=0.2)
    s = build_snippet(c, ts_terms=None)
    assert "<b>" not in s


def test_snippet_with_headline():
    c = make_candidate("a", bm=0.5, sem=0.2)
    c.headline = "<b>alpha</b> beta"
    s = build_snippet(c, ts_terms="alpha")
    assert s.startswith("<b>alpha</b>")


