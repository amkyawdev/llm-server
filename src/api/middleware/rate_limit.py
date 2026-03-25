"""Rate limiting middleware."""

from typing import Dict, Optional
import time
from collections import defaultdict
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from loguru import logger

from config import settings


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware using token bucket algorithm."""

    def __init__(self, app, requests_per_minute: int = None):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute or settings.rate_limit_per_minute
        self._buckets: Dict[str, dict] = defaultdict(self._create_bucket)
        self._cleanup_interval = 3600  # Cleanup old entries every hour
        self._last_cleanup = time.time()

    def _create_bucket(self) -> dict:
        """Create a new token bucket."""
        return {
            "tokens": self.requests_per_minute,
            "last_update": time.time(),
        }

    def _refill_bucket(self, bucket: dict) -> None:
        """Refill tokens based on time elapsed."""
        now = time.time()
        elapsed = now - bucket["last_update"]
        
        # Add tokens based on elapsed time
        tokens_to_add = (elapsed / 60) * self.requests_per_minute
        bucket["tokens"] = min(
            self.requests_per_minute,
            bucket["tokens"] + tokens_to_add,
        )
        bucket["last_update"] = now

    def _get_client_id(self, request: Request) -> str:
        """Get unique identifier for client."""
        # Try to get user ID if authenticated
        if hasattr(request.state, "user"):
            return f"user:{request.state.user.get('id', 'unknown')}"
        
        # Fall back to IP address
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return f"ip:{forwarded.split(',')[0].strip()}"
        
        return f"ip:{request.client.host}"

    async def dispatch(self, request: Request, call_next):
        """Process request with rate limiting."""
        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/health/live", "/health/ready"]:
            return await call_next(request)
        
        # Periodic cleanup
        now = time.time()
        if now - self._last_cleanup > self._cleanup_interval:
            self._cleanup_old_buckets()
            self._last_cleanup = now
        
        client_id = self._get_client_id(request)
        bucket = self._buckets[client_id]
        
        # Refill tokens
        self._refill_bucket(bucket)
        
        # Check if request is allowed
        if bucket["tokens"] >= 1:
            bucket["tokens"] -= 1
            response = await call_next(request)
            return response
        else:
            logger.warning(f"Rate limit exceeded for {client_id}")
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please try again later.",
                headers={
                    "Retry-After": "60",
                    "X-RateLimit-Limit": str(self.requests_per_minute),
                    "X-RateLimit-Remaining": "0",
                },
            )

    def _cleanup_old_buckets(self) -> None:
        """Remove inactive buckets to save memory."""
        now = time.time()
        inactive_threshold = 3600  # 1 hour
        
        to_remove = [
            client_id
            for client_id, bucket in self._buckets.items()
            if now - bucket["last_update"] > inactive_threshold
        ]
        
        for client_id in to_remove:
            del self._buckets[client_id]
        
        if to_remove:
            logger.debug(f"Cleaned up {len(to_remove)} inactive rate limit buckets")


class InMemoryRateLimiter:
    """Simple in-memory rate limiter for endpoints."""

    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: Dict[str, list] = defaultdict(list)

    def is_allowed(self, key: str) -> bool:
        """Check if request is allowed for given key."""
        now = time.time()
        cutoff = now - self.window_seconds
        
        # Clean old requests
        self._requests[key] = [t for t in self._requests[key] if t > cutoff]
        
        if len(self._requests[key]) < self.max_requests:
            self._requests[key].append(now)
            return True
        
        return False

    def get_remaining(self, key: str) -> int:
        """Get remaining requests for key."""
        now = time.time()
        cutoff = now - self.window_seconds
        
        recent_requests = [
            t for t in self._requests[key] if t > cutoff
        ]
        
        return max(0, self.max_requests - len(recent_requests))

    def reset(self, key: str) -> None:
        """Reset limit for key."""
        if key in self._requests:
            del self._requests[key]