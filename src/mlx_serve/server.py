"""FastAPI server for mlx-serve."""

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from mlx_serve import __version__
from mlx_serve.routers import embeddings_router, models_router, rerank_router

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="mlx-serve",
        description="MLX-based embedding and reranking server with OpenAI-compatible API",
        version=__version__,
    )

    # Include routers
    app.include_router(embeddings_router)
    app.include_router(rerank_router)
    app.include_router(models_router)

    # Global exception handler for OpenAI-compatible errors
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": str(exc),
                    "type": "server_error",
                    "code": "internal_error",
                }
            },
        )

    @app.get("/health")
    async def health_check() -> dict:
        """Health check endpoint."""
        return {"status": "healthy", "version": __version__}

    return app


app = create_app()
