"""Request schemas for API endpoints."""

from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, validator


class ChatMessage(BaseModel):
    """Chat message schema."""

    role: str = Field(..., description="Message role (system, user, assistant)")
    content: str = Field(..., description="Message content")
    name: Optional[str] = Field(None, description="Optional name for the message")


class ChatRequest(BaseModel):
    """Chat completion request schema."""

    messages: List[ChatMessage] = Field(
        ..., description="List of chat messages"
    )
    model: Optional[str] = Field(None, description="Model to use")
    temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="Sampling temperature"
    )
    top_p: float = Field(
        default=0.9, ge=0.0, le=1.0, description="Nucleus sampling parameter"
    )
    max_tokens: Optional[int] = Field(
        None, ge=1, le=8192, description="Maximum tokens to generate"
    )
    stream: bool = Field(
        default=False, description="Enable streaming response"
    )
    stop: Optional[List[str]] = Field(
        None, description="Stop sequences"
    )
    n: int = Field(
        default=1, ge=1, le=10, description="Number of completions"
    )


class CompletionRequest(BaseModel):
    """Text completion request schema."""

    prompt: str = Field(..., description="Input prompt")
    suffix: Optional[str] = Field(None, description="Suffix to append")
    max_tokens: int = Field(
        default=512, ge=1, le=8192, description="Max tokens to generate"
    )
    temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="Sampling temperature"
    )
    top_p: float = Field(
        default=0.9, ge=0.0, le=1.0, description="Nucleus sampling"
    )
    top_k: int = Field(
        default=50, ge=0, description="Top-k sampling"
    )
    repetition_penalty: float = Field(
        default=1.1, ge=1.0, le=2.0, description="Repetition penalty"
    )
    stream: bool = Field(
        default=False, description="Enable streaming"
    )
    stop: Optional[List[str]] = Field(
        None, description="Stop sequences"
    )
    model: Optional[str] = Field(None, description="Model to use")


class EmbeddingRequest(BaseModel):
    """Embedding request schema."""

    input: Union[str, List[str]] = Field(
        ..., description="Text(s) to embed"
    )
    model: str = Field(..., description="Embedding model to use")
    encoding_format: str = Field(
        default="float", description="Encoding format"
    )
    user: Optional[str] = Field(None, description="User identifier")


class ModelLoadRequest(BaseModel):
    """Model loading request schema."""

    model_name: str = Field(..., description="Name of the model to load")
    quantization: Optional[str] = Field(
        None, description="Quantization method (4bit, 8bit)"
    )
    device_map: str = Field(
        default="auto", description="Device mapping strategy"
    )
    trust_remote_code: bool = Field(
        default=False, description="Trust remote code"
    )


class TokenizeRequest(BaseModel):
    """Tokenization request schema."""

    text: Union[str, List[str]] = Field(
        ..., description="Text(s) to tokenize"
    )
    model: Optional[str] = Field(None, description="Model for tokenizer")
    add_special_tokens: bool = Field(
        default=True, description="Add special tokens"
    )
    max_length: Optional[int] = Field(
        None, description="Maximum length"
    )
    truncation: bool = Field(
        default=False, description="Enable truncation"
    )


class ChatHistoryRequest(BaseModel):
    """Chat history request schema."""

    session_id: str = Field(..., description="Session ID")
    limit: int = Field(
        default=50, ge=1, le=200, description="Maximum history items"
    )


class CacheInvalidateRequest(BaseModel):
    """Cache invalidation request schema."""

    pattern: Optional[str] = Field(
        None, description="Pattern to match for invalidation"
    )
    model: Optional[str] = Field(None, description="Specific model to clear")