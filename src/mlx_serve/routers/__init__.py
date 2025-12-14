"""API routers for mlx-serve."""

from mlx_serve.routers.embeddings import router as embeddings_router
from mlx_serve.routers.models import router as models_router
from mlx_serve.routers.rerank import router as rerank_router

__all__ = ["embeddings_router", "models_router", "rerank_router"]
