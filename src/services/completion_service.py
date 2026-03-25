"""Completion Service - Handles text completion requests."""

import time
import uuid
from typing import Optional, List, Dict, Any

from loguru import logger

from config import settings
from src.core.model_manager import get_model_manager
from src.core.inference import get_inference_engine, GenerationConfig
from src.core.cache_manager import get_cache_manager


class CompletionService:
    """Service for handling text completion requests."""

    def __init__(self):
        self.model_manager = get_model_manager()
        self.inference_engine = get_inference_engine()
        self.cache_manager = get_cache_manager()
        logger.info("CompletionService initialized")

    async def complete(
        self,
        prompt: str,
        model: Optional[str] = None,
        suffix: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        stream: bool = False,
        stop: Optional[List[str]] = None,
        echo: bool = False,
        stop_token_ids: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Generate a text completion.

        Args:
            prompt: Input prompt
            model: Model name
            suffix: Suffix to append after completion
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling
            top_k: Top-k sampling
            repetition_penalty: Repetition penalty
            stream: Enable streaming
            stop: Stop sequences
            echo: Include prompt in response
            stop_token_ids: Stop token IDs

        Returns:
            Response dictionary
        """
        model = model or settings.model_name

        # Build cache key
        cache_key_data = {
            "prompt": prompt,
            "suffix": suffix,
            "model": model,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "max_tokens": max_tokens,
        }

        # Check cache
        cached = self.cache_manager.get(
            str(cache_key_data), model, temperature=temperature
        )
        if cached:
            logger.info("Returning cached completion")
            return {
                "id": str(uuid.uuid4()),
                "created": int(time.time()),
                "model": model,
                "text": cached,
                "finish_reason": "cached",
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(cached.split()),
                "total_tokens": len(prompt.split()) + len(cached.split()),
            }

        # Load model if needed
        try:
            self.model_manager.load_model(
                model,
                quantization=settings.model_quantization,
            )
        except Exception as e:
            logger.warning(f"Failed to load model {model}: {e}")
            return {
                "id": str(uuid.uuid4()),
                "created": int(time.time()),
                "model": model,
                "text": "Model not loaded. Please configure a valid model.",
                "finish_reason": "error",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }

        # Build generation config
        config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_tokens,
            stop_strings=stop,
        )

        try:
            response = self.inference_engine.generate(
                prompt=prompt,
                model_name=model,
                config=config,
            )

            # Append suffix if provided
            if suffix:
                response = response + suffix

            # Cache response
            self.cache_manager.set(
                str(cache_key_data), model, response, temperature=temperature
            )

            return {
                "id": str(uuid.uuid4()),
                "created": int(time.time()),
                "model": model,
                "text": response,
                "finish_reason": "stop",
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(response.split()),
                "total_tokens": len(prompt.split()) + len(response.split()),
            }

        except Exception as e:
            logger.error(f"Completion error: {e}")
            return {
                "id": str(uuid.uuid4()),
                "created": int(time.time()),
                "model": model,
                "text": f"Error: {str(e)}",
                "finish_reason": "error",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }

    def stream_complete(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        stop: Optional[List[str]] = None,
    ):
        """Stream completion response.

        Args:
            prompt: Input prompt
            model: Model name
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling
            top_k: Top-k sampling
            repetition_penalty: Repetition penalty
            stop: Stop sequences

        Yields:
            Stream chunks
        """
        model = model or settings.model_name

        config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_tokens,
            stop_strings=stop,
        )

        try:
            for chunk in self.inference_engine.generate_streaming(
                prompt=prompt,
                model_name=model,
                config=config,
            ):
                yield f"data: {chunk}\n\n"
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"data: Error: {str(e)}\n\n"

        yield "data: [DONE]\n\n"


# Singleton instance
_completion_service: Optional[CompletionService] = None


def get_completion_service() -> CompletionService:
    """Get the completion service singleton."""
    global _completion_service
    if _completion_service is None:
        _completion_service = CompletionService()
    return _completion_service