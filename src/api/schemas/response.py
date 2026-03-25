"""Response schemas for API endpoints."""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class ChatChoice(BaseModel):
    """Chat completion choice."""

    index: int = Field(..., description="Choice index")
    message: Dict[str, str] = Field(
        ..., description="Assistant message"
    )
    finish_reason: Optional[str] = Field(
        None, description="Finish reason (stop, length)"
    )


class CompletionChoice(BaseModel):
    """Text completion choice."""

    text: str = Field(..., description="Generated text")
    index: int = Field(..., description="Choice index")
    finish_reason: Optional[str] = Field(
        None, description="Finish reason"
    )


class Usage(BaseModel):
    """Token usage information."""

    prompt_tokens: int = Field(..., description="Prompt tokens")
    completion_tokens: int = Field(..., description="Completion tokens")
    total_tokens: int = Field(..., description="Total tokens")


class ChatResponse(BaseModel):
    """Chat completion response."""

    id: str = Field(..., description="Response ID")
    object: str = Field(default="chat.completion", description="Object type")
    created: int = Field(..., description="Unix timestamp")
    model: str = Field(..., description="Model used")
    choices: List[ChatChoice] = Field(..., description="Completion choices")
    usage: Usage = Field(..., description="Token usage")


class CompletionResponse(BaseModel):
    """Text completion response."""

    id: str = Field(..., description="Response ID")
    object: str = Field(default="text_completion", description="Object type")
    created: int = Field(..., description="Unix timestamp")
    model: str = Field(..., description="Model used")
    choices: List[CompletionChoice] = Field(..., description="Completion choices")
    usage: Usage = Field(..., description="Token usage")


class EmbeddingData(BaseModel):
    """Embedding data item."""

    object: str = Field(default="embedding", description="Object type")
    embedding: List[float] = Field(..., description="Embedding vector")
    index: int = Field(..., description="Item index")


class EmbeddingResponse(BaseModel):
    """Embedding response."""

    object: str = Field(default="list", description="Object type")
    model: str = Field(..., description="Model used")
    data: List[EmbeddingData] = Field(..., description="List of embeddings")
    usage: Dict[str, int] = Field(..., description="Token usage")


class StreamChunk(BaseModel):
    """Streaming response chunk."""

    id: str = Field(..., description="Chunk ID")
    object: str = Field(default="chat.completion.chunk", description="Object type")
    created: int = Field(..., description="Unix timestamp")
    model: str = Field(..., description="Model used")
    choices: List[Dict[str, Any]] = Field(..., description="Choice deltas")


class ModelInfo(BaseModel):
    """Model information."""

    id: str = Field(..., description="Model ID")
    object: str = Field(default="model", description="Object type")
    created: int = Field(..., description="Creation timestamp")
    owned_by: str = Field(..., description="Owner")


class ModelList(BaseModel):
    """List of models."""

    object: str = Field(default="list", description="Object type")
    data: List[ModelInfo] = Field(..., description="List of models")


class HealthStatus(BaseModel):
    """Health status response."""

    status: str = Field(..., description="Health status")
    timestamp: int = Field(..., description="Unix timestamp")
    version: str = Field(default="0.1.0", description="Version")


class ErrorDetail(BaseModel):
    """Error detail."""

    message: str = Field(..., description="Error message")
    type: str = Field(..., description="Error type")
    param: Optional[str] = Field(None, description="Parameter that caused error")


class ErrorResponse(BaseModel):
    """Error response."""

    error: ErrorDetail = Field(..., description="Error details")


class ValidationErrorResponse(BaseModel):
    """Validation error response."""

    message: str = Field(..., description="Error message")
    errors: List[Dict[str, Any]] = Field(..., description="Validation errors")