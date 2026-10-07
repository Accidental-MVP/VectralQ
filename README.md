# VectralQ

Enterprise search that answers questions with citations, built on Postgres rather than a
vector database. Multi-tenant, with retrieval that fuses lexical and semantic lanes and
reranks the result.

Started August 2025, selected for **McGill TechAccel** within a month, piloted, and
**discontinued in December 2025** on weak demand. The retrieval worked; the market did not.
The code is here because the engineering is still worth reading.

---

## Why Postgres and not a vector database

Everything lives in one Postgres instance: documents, chunks, embeddings (`pgvector`),
full-text indexes, tenancy and telemetry. One database means one backup story, one migration
path, and transactional consistency between a document and its embeddings — a document and
its vectors cannot drift apart, because they commit together.

## Retrieval

Three lanes run against each query, then fuse.

**Lexical** — Postgres full-text with `ts_rank_cd`. When `bm25_weighted` is on it ranks
against a weighted tsvector (`text_tsv_enw`) that gives title and heading text more weight
than body text, so a match in a heading counts for more than a match in a footnote.

**Semantic** — `pgvector` cosine distance (`<=>`), converted to a similarity on the
assumption of normalised embeddings.

**Phrase** — quoted or inferred phrases get their own lane, contributing a bonus into
fusion rather than being folded into the lexical score. An exact phrase hit is different
evidence from a bag-of-words hit and is scored as such.

**Fusion** — [Reciprocal Rank Fusion](https://dl.acm.org/doi/10.1145/1571941.1572114) by
default, with a configurable `rrf_k`, summing `1/(k + rank)` across the lanes a candidate
appears in and adding the phrase bonus. RRF is used because it combines *ranks* rather than
scores, so the lanes do not need calibrating against each other.

A linear mode exists as a fallback, with min-max normalisation. One deliberate detail there:
candidates missing from a lane are **not** zero-filled into the normalisation baseline.
Zero-filling makes the floor of each lane depend on which documents happened to miss it,
which quietly distorts every other score in the set.

**Reranking** — surviving candidates go through a cross-encoder
(`cross-encoder/ms-marco-MiniLM-L-6-v2`), run off the event loop in an executor so it cannot
block the request.

`test_search_fusion.py` covers the fusion step directly, because ranking bugs are silent —
nothing throws, the results are just slightly worse, forever.

## Tenancy

Isolation is enforced by **Postgres row-level security**, not by application filtering.
`apply_tenant_rls()` enables `FORCE ROW LEVEL SECURITY` on each table and installs a policy
matching the row's tenant against `current_setting('app.tenant_id')`, with `WITH CHECK` so
writes are constrained as well as reads.

The reason is blunt: application-layer filtering is one forgotten `WHERE` clause away from a
cross-tenant leak. With `FORCE RLS`, a query that forgets the tenant returns nothing rather
than returning someone else's documents. The failure mode is an empty result, not a breach.

## Answering

Retrieved chunks go through `context_packer` before reaching the model, which assembles the
context window and keeps the mapping back to source chunks so answers can cite. Queries are
served both as a single response and over **SSE** (`query_stream`), so answers render
token-by-token instead of after a wait.

A telemetry table records what each query actually did — lane counts, `fusion_mode` — because
"retrieval feels worse this week" is unfalsifiable without it.

## Stack

FastAPI · PostgreSQL + pgvector · SQLAlchemy async · Alembic (9 migrations) · Next.js ·
Docker Compose · Ollama or a hosted model

## Running it

```bash
docker compose up -d db backend
```

Works against a local model, so the whole stack runs with no external API dependency and no
per-token cost:

```bash
ollama pull llama3.1:8b-instruct-q4_K_M
# then point LLM_BASE_URL at the Ollama host in docker-compose.yml
```

```bash
# ingest, embed, query
curl -X POST -H "X-Tenant-ID: $TENANT" -F file=@doc.pdf  localhost:8000/api/docs/upload
curl -X POST -H "X-Tenant-ID: $TENANT"                   localhost:8000/api/embeddings/missing
curl -X POST -H "X-Tenant-ID: $TENANT" -H "Content-Type: application/json" \
     -d '{"question":"...","options":{"top_k":3}}'       localhost:8000/api/query/
```

`/api/search/debug` returns per-lane scores and ranks before fusion, which is the only
practical way to work out why a result placed where it did.

## What I would change

The cross-encoder is a stock checkpoint; a reranker fine-tuned on the actual corpus would
earn more than any further tuning of the fusion weights. Chunking is fixed-strategy and
should adapt to document structure. And `rrf_k` was chosen by reading rather than by
measurement — there is no evaluation set in this repository, which is the real gap. Without
one, every claim above about retrieval quality is an argument rather than a result.

---

Built by [Uday Parmar](https://uday-parmar.vercel.app) ·
[Write-up](https://uday-parmar.vercel.app/work/vectralq)
