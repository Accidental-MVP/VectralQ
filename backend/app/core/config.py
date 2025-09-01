import os
from functools import lru_cache


class Settings:
    app_name: str = "VectralQ API"
    environment: str = os.getenv("ENVIRONMENT", "local")
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"

    # Database
    database_url: str = os.getenv(
        "DATABASE_URL",
        # Async SQLAlchemy URL
        "postgresql+asyncpg://vectralq:vectralq@db:5432/vectralq",
    )

    # Security / Tenancy
    # For now, we pass tenant via header X-Tenant-ID. Future: JWT w/ tenant claims
    tenant_header: str = os.getenv("TENANT_HEADER", "X-Tenant-ID")

    # CORS
    cors_allowed_origins: str | None = os.getenv("CORS_ALLOWED_ORIGINS")

    # Embeddings
    embedding_model_name: str | None = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    embedding_device: str | None = os.getenv("EMBEDDING_DEVICE", "cpu")
    embedding_batch_size: int | None = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))

    # Search configuration
    search_top_k_default: int = int(os.getenv("SEARCH_TOP_K_DEFAULT", "24"))
    search_bm25_limit: int = int(os.getenv("SEARCH_BM25_LIMIT", "100"))
    search_vector_limit: int = int(os.getenv("SEARCH_VECTOR_LIMIT", "100"))
    search_fusion_mode: str = os.getenv("SEARCH_FUSION_MODE", "linear")
    search_linear_lambda: float = float(os.getenv("SEARCH_LINEAR_LAMBDA", "0.6"))
    search_rrf_k: int = int(os.getenv("SEARCH_RRF_K", "60"))
    search_enable_cross_encoder: bool = os.getenv("SEARCH_ENABLE_CROSS_ENCODER", "false").lower() == "true"
    cross_encoder_model: str | None = os.getenv("CROSS_ENCODER_MODEL")
    cross_encoder_top_n: int = int(os.getenv("CROSS_ENCODER_TOP_N", "20"))
    ce_sentence_top_n: int = int(os.getenv("CE_SENTENCE_TOP_N", "30"))
    debug_search: bool = os.getenv("DEBUG_SEARCH", "false").lower() == "true"

    # Weighted BM25 and phrase lane toggles
    bm25_weighted: bool = os.getenv("BM25_WEIGHTED", "false").lower() == "true"
    phrase_lane_enabled: bool = os.getenv("PHRASE_LANE_ENABLED", "false").lower() == "true"
    phrase_boost: float = float(os.getenv("PHRASE_BOOST", "0.12"))
    bm25_k: int = int(os.getenv("BM25_K", "80"))
    vec_k: int = int(os.getenv("VEC_K", "80"))

    # Sentence prefilter / spans
    sentence_prefilter_top: int = int(os.getenv("SENTENCE_PREFILTER_TOP", "3"))
    span_picker: bool = os.getenv("SPAN_PICKER", "false").lower() == "true"
    min_overlap: float = float(os.getenv("MIN_OVERLAP", "0.15"))
    synthesize_from_spans: bool = os.getenv("SYNTHESIZE_FROM_SPANS", "false").lower() == "true"

    # Context packing / generation
    context_max_chunks: int = int(os.getenv("CONTEXT_MAX_CHUNKS", "6"))
    context_token_limit: int = int(os.getenv("CONTEXT_TOKEN_LIMIT", "6000"))

    # LLM client
    llm_base_url: str | None = os.getenv("LLM_BASE_URL")
    llm_model: str | None = os.getenv("LLM_MODEL")
    llm_timeout_ms: int = int(os.getenv("LLM_TIMEOUT_MS", "8000"))
    llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "600"))
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    answer_json_required: bool = os.getenv("ANSWER_JSON_REQUIRED", "true").lower() == "true"
    llm_first: bool = os.getenv("LLM_FIRST", "false").lower() == "true"
    # API key for OpenAI-compatible providers
    llm_api_key: str | None = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")

    # Google Drive connector
    google_client_id: str | None = os.getenv("GOOGLE_CLIENT_ID")
    google_client_secret: str | None = os.getenv("GOOGLE_CLIENT_SECRET")
    google_redirect_uri: str | None = os.getenv("GOOGLE_REDIRECT_URI")
    google_drive_scopes: str = os.getenv(
        "GOOGLE_DRIVE_SCOPES", "https://www.googleapis.com/auth/drive.readonly"
    )
    google_token_encryption_key: str | None = os.getenv("GOOGLE_TOKEN_ENCRYPTION_KEY")
    sync_page_size: int = int(os.getenv("SYNC_PAGE_SIZE", "1000"))
    max_file_bytes: int = int(os.getenv("MAX_FILE_BYTES", str(20 * 1024 * 1024)))

    # Frontend URL for redirects after OAuth
    frontend_base_url: str = os.getenv("FRONTEND_BASE_URL", "http://localhost:3000")

    # Refusal / safety thresholds
    refusal_min_results: int = int(os.getenv("REFUSAL_MIN_RESULTS", "1"))
    refusal_min_top_score: float = float(os.getenv("REFUSAL_MIN_TOP_SCORE", "0.02"))
    refusal_enable_extractive: bool = os.getenv("REFUSAL_ENABLE_EXTRACTIVE", "true").lower() == "true"
    # Rank-fusion (RRF) acceptance threshold (telemetry-tuned)
    min_rrf: float = float(os.getenv("MIN_RRF", "0.0"))

    def validate(self) -> None:
        # Clamp and validate search settings
        if self.search_top_k_default <= 0:
            self.search_top_k_default = 10
        self.search_fusion_mode = (self.search_fusion_mode or "linear").lower()
        if self.search_fusion_mode not in {"linear", "rrf"}:
            self.search_fusion_mode = "linear"
        # Lambda bounds [0,1]
        if not (0.0 <= self.search_linear_lambda <= 1.0):
            self.search_linear_lambda = 0.6
        # Refusal thresholds sanity
        if self.refusal_min_results < 1:
            self.refusal_min_results = 1
        if self.refusal_min_top_score < 0.0:
            self.refusal_min_top_score = 0.0
        if self.min_rrf < 0.0:
            self.min_rrf = 0.0


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    s.validate()
    return s


