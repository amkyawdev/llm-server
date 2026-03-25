"""Base model classes."""

from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
import torch
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Model configuration."""

    name: str = Field(..., description="Model name")
    path: str = Field(..., description="Model path or HuggingFace ID")
    max_length: int = Field(4096, description="Maximum sequence length")
    quantization: Optional[str] = Field(None, description="Quantization type")
    device: str = Field("cuda", description="Device to use")
    trust_remote_code: bool = Field(False, description="Trust remote code")


class BaseModel(ABC):
    """Abstract base class for models."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self._model = None
        self._tokenizer = None

    @abstractmethod
    def load(self) -> None:
        """Load the model."""
        pass

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt."""
        pass

    @abstractmethod
    def get_embeddings(self, text: str) -> list[float]:
        """Get embeddings for text."""
        pass

    def unload(self) -> None:
        """Unload the model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
            torch.cuda.empty_cache()

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None


class GenerationResult(BaseModel):
    """Generation result."""

    text: str = Field(..., description="Generated text")
    tokens: int = Field(..., description="Number of tokens generated")
    finish_reason: str = Field(..., description="Finish reason")


class EmbeddingResult(BaseModel):
    """Embedding result."""

    embeddings: list[list[float]] = Field(..., description="Embedding vectors")
    dimension: int = Field(..., description="Embedding dimension")