"""Model dependencies for API routes."""

from typing import Optional

from src.core.model_manager import get_model_manager
from src.core.inference import get_inference_engine


async def get_model_manager_dep():
    """Get model manager dependency."""
    return get_model_manager()


async def get_inference_engine_dep():
    """Get inference engine dependency."""
    return get_inference_engine()


async def get_current_model_name(model_name: Optional[str] = None) -> str:
    """Get current model name with fallback to default.
    
    Args:
        model_name: Optional model name override
        
    Returns:
        Model name string
    """
    from config import settings
    return model_name or settings.model_name