"""FastAPI server for mlx-serve."""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from mlx_serve import __version__
from mlx_serve.config import settings
from mlx_serve.core.mlx_memory import get_mlx_memory_snapshot
from mlx_serve.core.process_title import apply_process_title
from mlx_serve.core.retrieval_workers import RetrievalWorkerSupervisor
from mlx_serve.core.runtime_topology import (
    get_retrieval_worker_kind,
    get_server_role,
    retrieval_worker_isolation_enabled,
)
from mlx_serve.core.system_guard import memory_monitor
from mlx_serve.routers import (
    audio_router,
    chat_router,
    embeddings_router,
    images_router,
    models_router,
    retrieval_proxy_router,
    rerank_router,
    tokenize_router,
)

logger = logging.getLogger(__name__)
apply_process_title()

# Track active requests for graceful shutdown
_active_requests: int = 0
_shutting_down: bool = False


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for startup and shutdown events."""
    import asyncio

    retrieval_supervisor: RetrievalWorkerSupervisor | None = None
    global _shutting_down
    _shutting_down = False

    memory_monitor.start()

    if retrieval_worker_isolation_enabled():
        retrieval_supervisor = RetrievalWorkerSupervisor()
        app.state.retrieval_worker_supervisor = retrieval_supervisor
        app.state.retrieval_worker_urls = retrieval_supervisor.start()
    elif settings.preload_models:
        from mlx_serve.core.model_manager import model_manager

        logger.info(f"Preloading models: {settings.preload_models}")
        results = model_manager.preload_models()
        success = sum(1 for v in results.values() if v)
        logger.info(f"Preloaded {success}/{len(results)} models")

    logger.info("mlx-serve server started")
    yield

    _shutting_down = True

    # Graceful shutdown: wait for active requests to complete
    timeout = 30  # seconds
    logger.info(
        "Waiting for %s active requests to complete (timeout: %ss)...",
        _active_requests,
        timeout,
    )

    for _ in range(timeout * 10):
        if _active_requests == 0:
            break
        await asyncio.sleep(0.1)
    else:
        logger.warning(
            "Shutdown timeout reached with %s requests still active",
            _active_requests,
        )

    if retrieval_supervisor is not None:
        retrieval_supervisor.stop()

    memory_monitor.stop()
    logger.info("mlx-serve server stopped")


def _include_routers(app: FastAPI) -> None:
    """Attach routers based on the current runtime topology."""
    role = get_server_role()
    worker_kind = get_retrieval_worker_kind()

    if role == "worker":
        if worker_kind == "embedding":
            app.include_router(embeddings_router)
            app.include_router(tokenize_router)
        elif worker_kind == "reranker":
            app.include_router(rerank_router)
            app.include_router(tokenize_router)
        else:
            raise RuntimeError("Worker process started without a valid retrieval worker kind")
        return

    app.include_router(audio_router)
    app.include_router(chat_router)
    if retrieval_worker_isolation_enabled():
        app.include_router(retrieval_proxy_router)
    else:
        app.include_router(embeddings_router)
        app.include_router(rerank_router)
    app.include_router(images_router)
    app.include_router(models_router)
    app.include_router(tokenize_router)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="mlx-serve",
        description="MLX-based multimodal AI server with OpenAI-compatible API",
        version=__version__,
        lifespan=lifespan,
    )
    app.state.retrieval_worker_urls = {}
    app.state.retrieval_worker_supervisor = None

    _include_routers(app)

    # Middleware to track active requests and log request bodies
    @app.middleware("http")
    async def track_requests(request: Request, call_next):
        global _active_requests
        _active_requests += 1

        # Full request body logging is opt-in because it duplicates large payloads in memory.
        if (
            settings.debug_log_chat_request_bodies
            and logger.isEnabledFor(logging.DEBUG)
            and request.url.path == "/v1/chat/completions"
            and request.method == "POST"
        ):
            try:
                import json
                from pathlib import Path

                body = await request.body()
                body_str = body.decode("utf-8")

                # Write to debug file
                debug_file = Path.home() / ".mlx-serve" / "logs" / "request_debug.log"
                debug_file.parent.mkdir(parents=True, exist_ok=True)
                with open(debug_file, "a") as f:
                    f.write(f"\n\n========== REQUEST at {request.url} ==========\n")
                    f.write(f"Body: {body_str}\n")
                    try:
                        body_json = json.loads(body_str)
                        f.write(f"Parsed JSON:\n{json.dumps(body_json, indent=2)}\n")
                    except json.JSONDecodeError:
                        pass
                    f.write("==================================\n\n")

                logger.debug("[REQUEST BODY] /v1/chat/completions captured")
                # Re-create request with body for downstream processing

                async def receive():
                    return {"type": "http.request", "body": body}

                request._receive = receive
            except Exception as e:
                logger.error(f"[REQUEST BODY] Failed to read body: {e}")

        try:
            response = await call_next(request)
            return response
        finally:
            _active_requests -= 1

    # Validation error handler to log request validation failures
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        # Log the full request body and validation errors
        body = None
        try:
            import json
            from pathlib import Path

            body = await request.body()
            body_str = body.decode("utf-8")

            # Write to debug file
            debug_file = Path.home() / ".mlx-serve" / "logs" / "validation_errors.log"
            debug_file.parent.mkdir(parents=True, exist_ok=True)
            with open(debug_file, "a") as f:
                f.write(f"\n\n========== VALIDATION ERROR at {request.url.path} ==========\n")
                f.write(f"Timestamp: {__import__('datetime').datetime.now()}\n")
                f.write(f"Request body: {body_str}\n")
                try:
                    body_json = json.loads(body_str)
                    f.write(f"Parsed JSON:\n{json.dumps(body_json, indent=2)}\n")
                except json.JSONDecodeError:
                    pass
                f.write(f"Validation errors:\n{json.dumps(exc.errors(), indent=2)}\n")
                f.write("==================================\n\n")

            logger.error(f"[VALIDATION ERROR] Request to {request.url.path}")
            logger.error(f"[VALIDATION ERROR] Request body: {body_str}")
            logger.error(f"[VALIDATION ERROR] Validation errors: {exc.errors()}")
        except Exception as e:
            logger.error(f"[VALIDATION ERROR] Could not read request body: {e}")
            logger.error(f"[VALIDATION ERROR] Validation errors: {exc.errors()}")

        return JSONResponse(
            status_code=422,
            content={"detail": exc.errors()},
        )

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
        memory = memory_monitor.health_payload()
        role = get_server_role()
        worker_kind = get_retrieval_worker_kind()
        status = "degraded" if memory["overloaded"] else "healthy"
        payload = {
            "status": status,
            "version": __version__,
            "role": role,
            "retrieval_worker_kind": worker_kind,
            "shutting_down": _shutting_down,
            "memory": memory,
        }
        if role == "worker":
            payload["mlx_memory"] = get_mlx_memory_snapshot()
        supervisor = getattr(app.state, "retrieval_worker_supervisor", None)
        if supervisor is not None:
            payload["retrieval_workers"] = supervisor.snapshot()
        return payload

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
