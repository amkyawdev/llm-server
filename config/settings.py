"""Application settings using Pydantic."""

from typing import List, Optional
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Server Configuration
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    workers: int = Field(default=4, description="Number of worker processes")
    log_level: str = Field(default="INFO", description="Logging level")

    # Model Configuration
    model_name: str = Field(default="llama-2-7b", description="Default model name")
    model_path: str = Field(
        default="/workspace/project/llm-server/data/models",
        description="Path to model files",
    )
    model_device: str = Field(default="cuda", description="Device to run model on")
    model_quantization: Optional[str] = Field(
        default="4bit", description="Quantization type"
    )
    model_max_length: int = Field(
        default=4096, description="Maximum sequence length"
    )

    # Authentication
    api_key: Optional[str] = Field(
        default=None, description="API key for authentication"
    )
    jwt_secret: str = Field(
        default="your-jwt-secret-here", description="JWT secret key"
    )
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    jwt_expiration_minutes: int = Field(
        default=60, description="JWT token expiration in minutes"
    )

    # Rate Limiting
    rate_limit_per_minute: int = Field(
        default=60, description="Rate limit per minute"
    )
    rate_limit_storage: str = Field(
        default="memory", description="Rate limit storage backend"
    )

    # Cache Configuration
    cache_enabled: bool = Field(default=True, description="Enable caching")
    cache_ttl_seconds: int = Field(
        default=3600, description="Cache TTL in seconds"
    )
    redis_url: str = Field(
        default="redis://localhost:6379/0", description="Redis connection URL"
    )
    redis_max_connections: int = Field(
        default=50, description="Maximum Redis connections"
    )

    # GPU Configuration
    cuda_visible_devices: str = Field(
        default="0", description="Comma-separated GPU device IDs"
    )
    gpu_memory_fraction: float = Field(
        default=0.9, description="GPU memory fraction to use"
    )
    enable_cpu_offload: bool = Field(
        default=False, description="Enable CPU offloading for large models"
    )

    # API Configuration
    enable_cors: bool = Field(default=True, description="Enable CORS")
    cors_origins: str = Field(default="*", description="CORS allowed origins")
    max_request_size: int = Field(
        default=10 * 1024 * 1024, description="Maximum request size in bytes"
    )

    # Monitoring
    enable_metrics: bool = Field(
        default=True, description="Enable Prometheus metrics"
    )
    metrics_port: int = Field(
        default=9090, description="Metrics export port"
    )


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()