from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.health import router as health_router
from app.api.routes.docs import router as docs_router, upload_router as upload_root_router
from app.api.routes.embeddings import router as embeddings_router
from app.api.routes.search import router as search_router
from app.api.routes.search_debug import router as search_debug_router
from app.api.routes.query import router as query_router
from app.api.routes.query_stream import router as query_stream_router
from app.api.routes.integrations_google import router as google_router
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.middleware.request_context import RequestContextMiddleware
from app.services.embeddings import encode_texts
from app.services.llm_client import generate_answer, SYSTEM_PROMPT
from app.services.xrerank import _get_cross_encoder


def create_app() -> FastAPI:
    configure_logging()
    settings = get_settings()
    app = FastAPI(title="VectralQ API", version="0.1.0")

    # Tenant/request context middleware
    app.add_middleware(RequestContextMiddleware)

    # CORS: wildcard locally; restrict in prod unless overridden
    default_prod_origins = [
        "https://vectralq.com",
        "https://app.vectralq.com",
    ]
    if settings.cors_allowed_origins:
        allowed_origins = [o.strip() for o in settings.cors_allowed_origins.split(",") if o.strip()]
    else:
        allowed_origins = ["*"] if settings.environment == "local" else default_prod_origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
    )

    app.include_router(health_router, prefix="/api")
    app.include_router(docs_router, prefix="/api")
    app.include_router(upload_root_router, prefix="/api")
    app.include_router(embeddings_router, prefix="/api")
    app.include_router(search_router, prefix="/api")
    app.include_router(search_debug_router, prefix="/api")
    app.include_router(query_router, prefix="/api")
    app.include_router(query_stream_router, prefix="/api")
    app.include_router(google_router, prefix="/api")

    @app.on_event("startup")
    async def _warmup() -> None:
        try:
            # Warm embeddings model
            encode_texts(["warmup"])
        except Exception:
            pass
        try:
            # Warm LLM (providers may ignore). Tiny deterministic prompt
            await generate_answer(
                SYSTEM_PROMPT,
                "CONTEXT: warmup\nQUESTION: reply with {\"answer\":\"ok\",\"citations\":[],\"confidence\":1}",
                citations_hint=[],
                stream=False,
            )
        except Exception:
            pass
        try:
            if get_settings().search_enable_cross_encoder:
                _get_cross_encoder()
        except Exception:
            pass

    return app


app = create_app()


