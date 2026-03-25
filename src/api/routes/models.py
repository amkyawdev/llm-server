"""Models API route - Handles model information and management."""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from src.api.dependencies.auth import get_current_user
from src.core.model_manager import get_model_manager

router = APIRouter(prefix="/models", tags=["models"])


class ModelInfo(BaseModel):
    """Model information."""

    id: str = Field(..., description="Model ID")
    object: str = "model"
    created: int = Field(..., description="Creation timestamp")
    owned_by: str = Field(..., description="Owner organization")
    permission: Optional[List[dict]] = Field(None, description="Model permissions")
    root_model: Optional[str] = Field(None, description="Root model ID")
    parent_model: Optional[str] = Field(None, description="Parent model ID")


class ModelList(BaseModel):
    """List of available models."""

    object: str = "list"
    data: List[ModelInfo] = Field(..., description="List of models")


@router.get("/", response_model=ModelList)
async def list_models():
    """List available models.
    
    OpenAI-compatible models endpoint.
    """
    # Return a list of known models
    models = [
        ModelInfo(
            id="llama-2-7b",
            created=1699481848,
            owned_by="meta",
        ),
        ModelInfo(
            id="llama-2-13b",
            created=1699481848,
            owned_by="meta",
        ),
        ModelInfo(
            id="mistral-7b",
            created=1700000000,
            owned_by="mistralai",
        ),
    ]
    
    return ModelList(object="list", data=models)


@router.get("/{model_id}", response_model=ModelInfo)
async def get_model_info(model_id: str):
    """Get information about a specific model."""
    known_models = {
        "llama-2-7b": {"created": 1699481848, "owned_by": "meta"},
        "llama-2-13b": {"created": 1699481848, "owned_by": "meta"},
        "mistral-7b": {"created": 1700000000, "owned_by": "mistralai"},
    }
    
    if model_id not in known_models:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    
    info = known_models[model_id]
    return ModelInfo(
        id=model_id,
        object="model",
        created=info["created"],
        owned_by=info["owned_by"],
    )


@router.post("/load")
async def load_model(
    model_name: str,
    quantization: Optional[str] = None,
    
):
    """Load a model into memory."""
    try:
        model_manager = get_model_manager()
        model = model_manager.load_model(model_name, quantization=quantization)
        
        return {
            "status": "loaded",
            "model": model_name,
            "quantization": quantization,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/unload")
async def unload_model(
    model_name: str,
    
):
    """Unload a model from memory."""
    try:
        model_manager = get_model_manager()
        model_manager.unload_model(model_name)
        
        return {"status": "unloaded", "model": model_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/loaded")
async def list_loaded_models(user: dict = Depends(get_current_user)):
    """List currently loaded models."""
    try:
        model_manager = get_model_manager()
        loaded = model_manager.list_loaded_models()
        
        return {
            "loaded_models": loaded,
            "count": len(loaded),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/info/{model_name}")
async def get_model_details(
    model_name: str,
    
):
    """Get detailed information about a model."""
    # Return model configuration
    return {
        "name": model_name,
        "max_length": 4096,
        "default_temperature": 0.7,
        "supported_features": [
            "text-generation",
            "chat",
            "embeddings",
        ],
    }