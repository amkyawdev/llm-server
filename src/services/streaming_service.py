"""Streaming Service - Handles streaming responses."""

import json
import time
import uuid
from typing import Optional, List, Dict, Any, Generator, AsyncGenerator

from loguru import logger

from config import settings
from src.core.inference import get_inference_engine, GenerationConfig


class StreamingService:
    """Service for handling streaming responses."""

    def __init__(self):
        self.inference_engine = get_inference_engine()
        logger.info("StreamingService initialized")

    def stream_chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
    ) -> Generator[str, None, None]:
        """Stream chat completion response.

        Args:
            messages: List of chat messages
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stop: Stop sequences

        Yields:
            SSE data chunks
        """
        model = model or settings.model_name
        prompt = self._messages_to_prompt(messages)

        config = GenerationConfig(
            temperature=temperature,
            max_new_tokens=max_tokens or 512,
            stop_strings=stop,
        )

        chunk_id = f"chatcmpl-{str(uuid.uuid4())[:8]}"
        created = int(time.time())

        try:
            for i, token in enumerate(
                self.inference_engine.generate_streaming(
                    prompt=prompt,
                    model_name=model,
                    config=config,
                )
            ):
                chunk = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": token},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk)}\n\n"

            # Send final chunk
            final_chunk = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            error_chunk = {
                "error": {
                    "message": str(e),
                    "type": "streaming_error",
                }
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"

        yield "data: [DONE]\n\n"

    def stream_completion(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        stop: Optional[List[str]] = None,
    ) -> Generator[str, None, None]:
        """Stream text completion response.

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
            SSE data chunks
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

        chunk_id = f"cmpl-{str(uuid.uuid4())[:8]}"
        created = int(time.time())

        try:
            for i, token in enumerate(
                self.inference_engine.generate_streaming(
                    prompt=prompt,
                    model_name=model,
                    config=config,
                )
            ):
                chunk = {
                    "id": chunk_id,
                    "object": "text_completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "text": token,
                            "index": 0,
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk)}\n\n"

            # Send final chunk
            final_chunk = {
                "id": chunk_id,
                "object": "text_completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "text": "",
                        "index": 0,
                        "finish_reason": "stop",
                    }
                ],
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            error_chunk = {
                "error": {
                    "message": str(e),
                    "type": "streaming_error",
                }
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"

        yield "data: [DONE]\n\n"

    async def stream_chat_async(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
    ) -> AsyncGenerator[str, None]:
        """Async version of stream_chat.

        Args:
            messages: List of chat messages
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stop: Stop sequences

        Yields:
            SSE data chunks
        """
        # Convert to synchronous generator and yield
        for chunk in self.stream_chat(messages, model, temperature, max_tokens, stop):
            yield chunk

    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert chat messages to a prompt string.

        Args:
            messages: List of chat messages

        Returns:
            Formatted prompt string
        """
        prompt_parts = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                prompt_parts.append(f"System: {content}\n")
            elif role == "user":
                prompt_parts.append(f"User: {content}\n")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}\n")

        prompt_parts.append("Assistant:")
        return "\n".join(prompt_parts)


# Singleton instance
_streaming_service: Optional[StreamingService] = None


def get_streaming_service() -> StreamingService:
    """Get the streaming service singleton."""
    global _streaming_service
    if _streaming_service is None:
        _streaming_service = StreamingService()
    return _streaming_service