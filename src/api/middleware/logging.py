"""Logging middleware."""

import time
import uuid
from typing import Optional
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from loguru import logger


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request/response logging."""

    def __init__(self, app, log_body: bool = False):
        super().__init__(app)
        self.log_body = log_body

    async def dispatch(self, request: Request, call_next):
        """Log request and response details."""
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id
        
        # Log request
        start_time = time.time()
        
        logger.info(
            f"[{request_id}] {request.method} {request.url.path} "
            f"from {request.client.host}"
        )
        
        # Process request
        try:
            response = await call_next(request)
            
            # Log response
            duration = time.time() - start_time
            logger.info(
                f"[{request_id}] {request.method} {request.url.path} "
                f"completed with {response.status_code} in {duration:.3f}s"
            )
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                f"[{request_id}] {request.method} {request.url.path} "
                f"failed after {duration:.3f}s: {str(e)}"
            )
            raise


class RequestContextLogger:
    """Context-aware logger that includes request information."""

    def __init__(self):
        self._context = {}

    def set_context(self, key: str, value: any) -> None:
        """Set context value."""
        self._context[key] = value

    def clear_context(self) -> None:
        """Clear all context."""
        self._context.clear()

    def _format_message(self, message: str) -> str:
        """Format message with context."""
        if self._context:
            context_str = " ".join(f"{k}={v}" for k, v in self._context.items())
            return f"[{context_str}] {message}"
        return message

    def info(self, message: str, **kwargs) -> None:
        """Log info with context."""
        logger.info(self._format_message(message), **kwargs)

    def warning(self, message: str, **kwargs) -> None:
        """Log warning with context."""
        logger.warning(self._format_message(message), **kwargs)

    def error(self, message: str, **kwargs) -> None:
        """Log error with context."""
        logger.error(self._format_message(message), **kwargs)

    def debug(self, message: str, **kwargs) -> None:
        """Log debug with context."""
        logger.debug(self._format_message(message), **kwargs)


# Global context logger
request_logger = RequestContextLogger()


def log_request(
    request: Request,
    message: str,
    level: str = "info",
) -> None:
    """Log request with context.
    
    Args:
        request: FastAPI request
        message: Log message
        level: Log level (debug, info, warning, error)
    """
    request_logger.set_context("method", request.method)
    request_logger.set_context("path", request.url.path)
    request_logger.set_context("client", request.client.host)
    
    if hasattr(request.state, "request_id"):
        request_logger.set_context("request_id", request.state.request_id)
    
    log_func = getattr(request_logger, level, request_logger.info)
    log_func(message)
    
    request_logger.clear_context()