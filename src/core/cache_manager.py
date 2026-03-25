"""Cache Manager - Handles caching for inference requests."""

import hashlib
import json
from typing import Optional, Any, Dict
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import pickle

import diskcache
from loguru import logger

from config import settings


@dataclass
class CacheEntry:
    """Cache entry metadata."""

    key: str
    created_at: datetime
    expires_at: Optional[datetime]
    hit_count: int = 0


class CacheManager:
    """Manages caching for inference requests."""

    def __init__(self):
        self._enabled = settings.cache_enabled
        self._cache: Optional[diskcache.Cache] = None

        if self._enabled:
            try:
                self._cache = diskcache.Cache(
                    settings.model_path + "/cache",
                    size_limit=10 * 1024 * 1024 * 1024,  # 10GB
                )
                logger.info("Cache initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize cache: {e}")
                self._enabled = False

    @property
    def is_enabled(self) -> bool:
        """Check if caching is enabled."""
        return self._enabled

    def _generate_key(self, data: Dict[str, Any]) -> str:
        """Generate a cache key from request data.

        Args:
            data: Request data

        Returns:
            Cache key string
        """
        # Create a deterministic string from the data
        serialized = json.dumps(data, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()

    def get(self, prompt: str, model_name: str, **kwargs) -> Optional[str]:
        """Get cached response.

        Args:
            prompt: Input prompt
            model_name: Model name
            **kwargs: Additional generation parameters

        Returns:
            Cached response or None
        """
        if not self._enabled or not self._cache:
            return None

        try:
            data = {
                "prompt": prompt,
                "model_name": model_name,
                **kwargs,
            }
            key = self._generate_key(data)

            cached = self._cache.get(key)
            if cached:
                logger.debug(f"Cache hit for key: {key[:16]}...")
                return cached

            return None

        except Exception as e:
            logger.warning(f"Cache get error: {e}")
            return None

    def set(
        self,
        prompt: str,
        model_name: str,
        response: str,
        ttl: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Store response in cache.

        Args:
            prompt: Input prompt
            model_name: Model name
            response: Generated response
            ttl: Time to live in seconds
            **kwargs: Additional generation parameters
        """
        if not self._enabled or not self._cache:
            return

        try:
            data = {
                "prompt": prompt,
                "model_name": model_name,
                **kwargs,
            }
            key = self._generate_key(data)

            ttl = ttl or settings.cache_ttl_seconds
            self._cache.set(key, response, expire=ttl)
            logger.debug(f"Cached response for key: {key[:16]}...")

        except Exception as e:
            logger.warning(f"Cache set error: {e}")

    def get_embeddings(
        self, text: str, model_name: str
    ) -> Optional[list[float]]:
        """Get cached embeddings.

        Args:
            text: Input text
            model_name: Model name

        Returns:
            Cached embeddings or None
        """
        if not self._enabled or not self._cache:
            return None

        try:
            data = {"text": text, "model_name": model_name}
            key = f"emb_{self._generate_key(data)}"

            cached = self._cache.get(key)
            if cached:
                logger.debug(f"Embedding cache hit for key: {key[:16]}...")
                return cached

            return None

        except Exception as e:
            logger.warning(f"Embedding cache get error: {e}")
            return None

    def set_embeddings(
        self, text: str, model_name: str, embeddings: list[float]
    ) -> None:
        """Store embeddings in cache.

        Args:
            text: Input text
            model_name: Model name
            embeddings: Embedding vectors
        """
        if not self._enabled or not self._cache:
            return

        try:
            data = {"text": text, "model_name": model_name}
            key = f"emb_{self._generate_key(data)}"

            self._cache.set(key, embeddings, expire=settings.cache_ttl_seconds)
            logger.debug(f"Cached embeddings for key: {key[:16]}...")

        except Exception as e:
            logger.warning(f"Embedding cache set error: {e}")

    def invalidate(self, pattern: Optional[str] = None) -> int:
        """Invalidate cache entries.

        Args:
            pattern: Optional key pattern to match

        Returns:
            Number of entries invalidated
        """
        if not self._enabled or not self._cache:
            return 0

        try:
            if pattern:
                count = 0
                for key in list(self._cache.iterkeys()):
                    if pattern in key:
                        del self._cache[key]
                        count += 1
                return count
            else:
                size = len(self._cache)
                self._cache.clear()
                logger.info(f"Cleared {size} cache entries")
                return size

        except Exception as e:
            logger.warning(f"Cache invalidation error: {e}")
            return 0

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        if not self._enabled or not self._cache:
            return {
                "enabled": False,
                "size": 0,
                "volume": 0,
            }

        try:
            return {
                "enabled": True,
                "size": len(self._cache),
                "volume": self._cache.volume(),
            }
        except Exception as e:
            logger.warning(f"Cache stats error: {e}")
            return {"enabled": True, "error": str(e)}

    def close(self) -> None:
        """Close the cache."""
        if self._cache:
            self._cache.close()
            logger.info("Cache closed")


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """Get the global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager