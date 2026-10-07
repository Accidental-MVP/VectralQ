VectralQ backend (FastAPI)
--------------------------

Quick start (local, Windows + Docker)
=====================================

1) Start DB and backend:
   docker compose up -d db backend

2) Local LLM (Ollama on Windows, 8GB VRAM)
   - Install: winget install -e --id Ollama.Ollama
   - Run on port 11435 if 11434 is busy:
     $env:OLLAMA_HOST="127.0.0.1:11435"; ollama serve
   - Pull a compact instruct model:
     ollama pull llama3.1:8b-instruct-q4_K_M
   - In docker-compose.yml (backend env):
     LLM_BASE_URL=http://host.docker.internal:11435
     LLM_MODEL=llama3.1:8b-instruct-q4_K_M
     LLM_TIMEOUT_MS=30000
     LLM_MAX_TOKENS=500
     ANSWER_JSON_REQUIRED=true
   - Restart backend:
     docker compose up -d --build backend

3) Basic ingestion + search
   - Upload: see /api/docs/upload (multipart form)
   - Embed missing: POST /api/embeddings/missing
   - Search:
     curl.exe -s -i -L -X POST `
       -H "X-Tenant-ID: $TENANT" -H "Content-Type: application/json" `
       -d '{"query":"hello","top_k":5}' `
       http://localhost:8000/api/search

4) Query (non-streaming)
   curl.exe -s -i -L -X POST `
     -H "X-Tenant-ID: $TENANT" -H "Content-Type: application/json" `
     -d '{"question":"hello","options":{"top_k":3}}' `
     http://localhost:8000/api/query/

5) Query (streaming SSE)
   curl.exe -s -N -H "X-Tenant-ID: $TENANT" -H "Content-Type: application/json" `
     -d '{"question":"hello","options":{"top_k":3}}' `
     http://localhost:8000/api/query/stream

Telemetry
=========
- Tables: app.search_logs, app.query_logs
- Verify counts:
  docker compose exec db psql -U vectralq -d vectralq -c "select count(*) from app.search_logs; select count(*) from app.query_logs;"

Cloud overflow (optional)
=========================
- You can wire a cloud LLM by adding CLOUD_LLM_BASE_URL / CLOUD_LLM_MODEL and routing low-confidence or long questions upstream.
  Keep ANSWER_JSON_REQUIRED=true and scrub PII before sending off-box.
