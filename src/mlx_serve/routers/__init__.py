"""API routers for mlx-serve."""

from mlx_serve.routers.audio import router as audio_router
from mlx_serve.routers.chat import router as chat_router
from mlx_serve.routers.embeddings import router as embeddings_router
from mlx_serve.routers.images import router as images_router
from mlx_serve.routers.models import router as models_router
from mlx_serve.routers.rerank import router as rerank_router
from mlx_serve.routers.tokenize import router as tokenize_router

__all__ = [
    "audio_router",
    "chat_router",
    "embeddings_router",
    "images_router",
    "models_router",
    "rerank_router",
    "tokenize_router",
]
