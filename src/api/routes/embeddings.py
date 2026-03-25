"""Embeddings API route - Handles embedding requests."""

from typing import List, Union
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from src.api.dependencies.auth import get_current_user
from src.services.embedding_service import EmbeddingService

router = APIRouter(prefix="/embeddings", tags=["embeddings"])


class EmbeddingRequest(BaseModel):
    """Embedding request payload."""

    input: Union[str, List[str]] = Field(
        ..., description="Text(s) to embed"
    )
    model: str = Field(..., description="Embedding model to use")
    encoding_format: str = Field("float", description="Encoding format")


class EmbeddingItem(BaseModel):
    """Single embedding result."""

    object: str = "embedding"
    embedding: List[float] = Field(..., description="Embedding vector")
    index: int = Field(..., description="Index of the input text")


class EmbeddingResponse(BaseModel):
    """Embedding response."""

    object: str = "list"
    model: str = Field(..., description="Model used")
    data: List[EmbeddingItem] = Field(..., description="List of embeddings")
    usage: dict = Field(..., description="Token usage information")


@router.post("/", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    """Create embeddings for text input(s).
    
    OpenAI-compatible embeddings endpoint.
    """
    try:
        service = EmbeddingService()
        
        # Handle single string or list of strings
        texts = request.input if isinstance(request.input, list) else [request.input]
        
        # Get embeddings
        result = await service.create_embeddings(
            texts=texts,
            model=request.model,
        )
        
        return EmbeddingResponse(
            object="list",
            model=request.model,
            data=[
                EmbeddingItem(
                    object="embedding",
                    embedding=emb,
                    index=i,
                )
                for i, emb in enumerate(result["embeddings"])
            ],
            usage={
                "prompt_tokens": result["tokens"],
                "total_tokens": result["tokens"],
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models")
async def list_embedding_models(
    
):
    """List available embedding models."""
    return {
        "object": "list",
        "data": [
            {
                "id": "bge-small-en-v1.5",
                "object": "model",
                "created": 1699481848,
                "owned_by": "BAAI",
                "embedding_dimension": 384,
            },
            {
                "id": "bge-base-en-v1.5",
                "object": "model",
                "created": 1699481848,
                "owned_by": "BAAI",
                "embedding_dimension": 768,
            },
        ],
    }


@router.post("/batch")
async def create_embeddings_batch(
    texts: List[str],
    model: str = "bge-small-en-v1.5",
    
):
    """Create embeddings for a batch of texts."""
    try:
        service = EmbeddingService()
        result = await service.create_embeddings(texts=texts, model=model)
        
        return {
            "embeddings": result["embeddings"],
            "tokens": result["tokens"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))