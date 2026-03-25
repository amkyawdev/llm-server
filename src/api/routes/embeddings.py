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
            # BGE series (BAAI)
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
            {
                "id": "bge-large-en-v1.5",
                "object": "model",
                "created": 1699481848,
                "owned_by": "BAAI",
                "embedding_dimension": 1024,
            },
            # BGE multilingual
            {
                "id": "bge-multilingual-gemma-xl",
                "object": "model",
                "created": 1711974800,
                "owned_by": "BAAI",
                "embedding_dimension": 1024,
            },
            # E5 series
            {
                "id": "e5-small-v2",
                "object": "model",
                "created": 1689292800,
                "owned_by": "microsoft",
                "embedding_dimension": 384,
            },
            {
                "id": "e5-base-v2",
                "object": "model",
                "created": 1689292800,
                "owned_by": "microsoft",
                "embedding_dimension": 768,
            },
            {
                "id": "e5-large-v2",
                "object": "model",
                "created": 1689292800,
                "owned_by": "microsoft",
                "embedding_dimension": 1024,
            },
            # sentence-transformers
            {
                "id": "all-MiniLM-L6-v2",
                "object": "model",
                "created": 1672531200,
                "owned_by": "sentence-transformers",
                "embedding_dimension": 384,
            },
            {
                "id": "all-mpnet-base-v2",
                "object": "model",
                "created": 1672531200,
                "owned_by": "sentence-transformers",
                "embedding_dimension": 768,
            },
            {
                "id": "all-MiniLM-L12-v2",
                "object": "model",
                "created": 1672531200,
                "owned_by": "sentence-transformers",
                "embedding_dimension": 384,
            },
            # OpenAI
            {
                "id": "text-embedding-3-small",
                "object": "model",
                "created": 1709596800,
                "owned_by": "openai",
                "embedding_dimension": 1536,
            },
            {
                "id": "text-embedding-3-large",
                "object": "model",
                "created": 1709596800,
                "owned_by": "openai",
                "embedding_dimension": 3072,
            },
            {
                "id": "text-embedding-ada-002",
                "object": "model",
                "created": 1672531200,
                "owned_by": "openai",
                "embedding_dimension": 1536,
            },
            # Cohere
            {
                "id": "embed-english-v3.0",
                "object": "model",
                "created": 1704067200,
                "owned_by": "cohere",
                "embedding_dimension": 1024,
            },
            {
                "id": "embed-multilingual-v3.0",
                "object": "model",
                "created": 1704067200,
                "owned_by": "cohere",
                "embedding_dimension": 1024,
            },
            # Google
            {
                "id": "gemini-embedding-001",
                "object": "model",
                "created": 1704067200,
                "owned_by": "google",
                "embedding_dimension": 768,
            },
            # Nvidia
            {
                "id": "nvidia-embed-qa-1",
                "object": "model",
                "created": 1711974800,
                "owned_by": "nvidia",
                "embedding_dimension": 1024,
            },
            # Voyage
            {
                "id": "voyage-01",
                "object": "model",
                "created": 1704067200,
                "owned_by": "voyageai",
                "embedding_dimension": 1024,
            },
            {
                "id": "voyage-multilingual-01",
                "object": "model",
                "created": 1709596800,
                "owned_by": "voyageai",
                "embedding_dimension": 1024,
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