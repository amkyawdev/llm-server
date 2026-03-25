"""API routes for LLM Server."""

from .chat import router as chat_router
from .embeddings import router as embeddings_router
from .models import router as models_router
from .health import router as health_router
from .admin import router as admin_router

__all__ = [
    "chat_router",
    "embeddings_router",
    "models_router",
    "health_router",
    "admin_router",
]