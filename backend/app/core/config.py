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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


