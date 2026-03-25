"""Custom exceptions for LLM Server."""


class LLMError(Exception):
    """Base exception for LLM Server."""

    def __init__(self, message: str, code: str = "internal_error"):
        self.message = message
        self.code = code
        super().__init__(self.message)


class ModelNotFoundError(LLMError):
    """Exception raised when a model is not found."""

    def __init__(self, model_name: str):
        super().__init__(
            message=f"Model not found: {model_name}",
            code="model_not_found",
        )
        self.model_name = model_name


class ModelLoadError(LLMError):
    """Exception raised when model loading fails."""

    def __init__(self, model_name: str, reason: str):
        super().__init__(
            message=f"Failed to load model {model_name}: {reason}",
            code="model_load_error",
        )
        self.model_name = model_name


class InferenceError(LLMError):
    """Exception raised during inference."""

    def __init__(self, message: str):
        super().__init__(
            message=f"Inference error: {message}",
            code="inference_error",
        )


class RateLimitError(LLMError):
    """Exception raised when rate limit is exceeded."""

    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(
            message=message,
            code="rate_limit_exceeded",
        )


class AuthenticationError(LLMError):
    """Exception raised for authentication failures."""

    def __init__(self, message: str = "Authentication failed"):
        super().__init__(
            message=message,
            code="authentication_error",
        )


class ValidationError(LLMError):
    """Exception raised for validation failures."""

    def __init__(self, message: str):
        super().__init__(
            message=message,
            code="validation_error",
        )


class CacheError(LLMError):
    """Exception raised for cache-related errors."""

    def __init__(self, message: str):
        super().__init__(
            message=f"Cache error: {message}",
            code="cache_error",
        )


class ConfigurationError(LLMError):
    """Exception raised for configuration issues."""

    def __init__(self, message: str):
        super().__init__(
            message=f"Configuration error: {message}",
            code="configuration_error",
        )