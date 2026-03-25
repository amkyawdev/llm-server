"""Error schemas for API responses."""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: Dict[str, Any] = Field(
        ..., description="Error details"
    )


class ValidationError(BaseModel):
    """Validation error detail."""

    loc: List[str] = Field(
        ..., description="Location of the error"
    )
    msg: str = Field(..., description="Error message")
    type: str = Field(..., description="Error type")


class ValidationErrorResponse(BaseModel):
    """Validation error response."""

    detail: List[ValidationError] = Field(
        ..., description="List of validation errors"
    )


class RateLimitError(BaseModel):
    """Rate limit error response."""

    error: Dict[str, Any] = Field(
        default={
            "message": "Rate limit exceeded",
            "type": "rate_limit_error",
            "param": None,
        },
        description="Rate limit error details"
    )


class AuthenticationError(BaseModel):
    """Authentication error response."""

    error: Dict[str, Any] = Field(
        default={
            "message": "Invalid authentication",
            "type": "authentication_error",
        },
        description="Authentication error details"
    )


class NotFoundError(BaseModel):
    """Not found error response."""

    error: Dict[str, Any] = Field(
        default={
            "message": "Resource not found",
            "type": "not_found_error",
        },
        description="Not found error details"
    )


class ModelLoadError(BaseModel):
    """Model loading error response."""

    error: Dict[str, Any] = Field(
        ..., description="Model loading error details"
    )


class InferenceError(BaseModel):
    """Inference error response."""

    error: Dict[str, Any] = Field(
        ..., description="Inference error details"
    )


class ServerError(BaseModel):
    """Internal server error response."""

    error: Dict[str, Any] = Field(
        default={
            "message": "Internal server error",
            "type": "server_error",
        },
        description="Server error details"
    )