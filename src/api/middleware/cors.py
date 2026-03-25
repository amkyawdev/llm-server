"""CORS (Cross-Origin Resource Sharing) middleware."""

from typing import List, Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware as FastAPICORSMiddleware

from config import settings


def setup_cors(app: FastAPI) -> None:
    """Setup CORS middleware for the application.
    
    Args:
        app: FastAPI application instance
    """
    if not settings.enable_cors:
        return
    
    # Parse origins
    origins = settings.cors_origins
    if origins == "*":
        allow_origins = ["*"]
    elif isinstance(origins, str):
        allow_origins = [o.strip() for o in origins.split(",")]
    else:
        allow_origins = origins
    
    app.add_middleware(
        FastAPICORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=[
            "X-Request-ID",
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining",
            "X-RateLimit-Reset",
        ],
        max_age=600,  # Cache preflight for 10 minutes
    )


class CORSConfig:
    """CORS configuration container."""

    def __init__(
        self,
        allow_origins: List[str] = None,
        allow_methods: List[str] = None,
        allow_headers: List[str] = None,
        allow_credentials: bool = True,
        expose_headers: List[str] = None,
        max_age: int = 600,
    ):
        self.allow_origins = allow_origins or ["*"]
        self.allow_methods = allow_methods or ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
        self.allow_headers = allow_headers or ["*"]
        self.allow_credentials = allow_credentials
        self.expose_headers = expose_headers or ["X-Request-ID"]
        self.max_age = max_age

    def to_dict(self) -> dict:
        """Convert to dictionary for FastAPI middleware."""
        return {
            "allow_origins": self.allow_origins,
            "allow_credentials": self.allow_credentials,
            "allow_methods": self.allow_methods,
            "allow_headers": self.allow_headers,
            "expose_headers": self.expose_headers,
            "max_age": self.max_age,
        }


# Default CORS config
default_cors_config = CORSConfig()