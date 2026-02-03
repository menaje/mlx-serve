"""FastAPI server for mlx-serve."""

import logging
import signal
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from mlx_serve import __version__
from mlx_serve.routers import chat_router, embeddings_router, models_router, rerank_router, tokenize_router

logger = logging.getLogger(__name__)

# Track active requests for graceful shutdown
_active_requests: int = 0
_shutting_down: bool = False


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for startup and shutdown events."""
    import asyncio

    from mlx_serve.config import settings
    from mlx_serve.core.model_manager import model_manager

    shutdown_event = asyncio.Event()

    def handle_shutdown(signum: int, frame) -> None:
        """Handle SIGTERM/SIGINT for graceful shutdown."""
        global _shutting_down
        _shutting_down = True
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        shutdown_event.set()

    # Register signal handlers
    signal.signal(signal.SIGTERM, handle_shutdown)
    signal.signal(signal.SIGINT, handle_shutdown)

    # Preload models at startup
    if settings.preload_models:
        logger.info(f"Preloading models: {settings.preload_models}")
        results = model_manager.preload_models()
        success = sum(1 for v in results.values() if v)
        logger.info(f"Preloaded {success}/{len(results)} models")

    logger.info("mlx-serve server started")
    yield

    # Graceful shutdown: wait for active requests to complete
    if _shutting_down:
        timeout = 30  # seconds
        logger.info(f"Waiting for {_active_requests} active requests to complete (timeout: {timeout}s)...")

        for _ in range(timeout * 10):
            if _active_requests == 0:
                break
            await asyncio.sleep(0.1)
        else:
            logger.warning(f"Shutdown timeout reached with {_active_requests} requests still active")

    logger.info("mlx-serve server stopped")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="mlx-serve",
        description="MLX-based multimodal AI server with OpenAI-compatible API",
        version=__version__,
        lifespan=lifespan,
    )

    # Include routers
    app.include_router(chat_router)
    app.include_router(embeddings_router)
    app.include_router(rerank_router)
    app.include_router(models_router)
    app.include_router(tokenize_router)

    # Middleware to track active requests
    @app.middleware("http")
    async def track_requests(request: Request, call_next):
        global _active_requests
        _active_requests += 1
        try:
            response = await call_next(request)
            return response
        finally:
            _active_requests -= 1

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
        return {"status": "healthy", "version": __version__, "shutting_down": _shutting_down}

    # Add metrics endpoint if enabled
    from mlx_serve.config import settings

    if settings.metrics_enabled:
        from mlx_serve.core.metrics import MetricsMiddleware, get_metrics

        # Add metrics middleware
        app.add_middleware(MetricsMiddleware)

        @app.get("/metrics")
        async def metrics() -> Response:
            """Prometheus metrics endpoint."""
            return Response(
                content=get_metrics(),
                media_type="text/plain; version=0.0.4; charset=utf-8",
            )

        logger.info("Prometheus metrics enabled at /metrics")

    return app


app = create_app()
