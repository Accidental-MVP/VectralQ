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
    debug_search: bool = os.getenv("DEBUG_SEARCH", "false").lower() == "true"

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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    s.validate()
    return s


