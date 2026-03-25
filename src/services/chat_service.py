"""Chat Service - Handles chat interactions."""

import time
import uuid
from typing import Optional, List, Dict, Any

from loguru import logger

from config import settings
from src.core.model_manager import get_model_manager
from src.core.inference import InferenceEngine, GenerationConfig, get_inference_engine
from src.core.cache_manager import get_cache_manager


class ChatService:
    """Service for handling chat interactions."""

    def __init__(self):
        self.model_manager = get_model_manager()
        self.inference_engine = get_inference_engine()
        self.cache_manager = get_cache_manager()
        logger.info("ChatService initialized")

    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """Process a chat request.
        
        Args:
            messages: List of chat messages
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stop: Stop sequences
            use_cache: Whether to use caching
            
        Returns:
            Response dictionary
        """
        model = model or settings.model_name
        
        # Convert messages to prompt
        prompt = self._messages_to_prompt(messages)
        
        # Check cache
        if use_cache:
            cached = self.cache_manager.get(prompt, model, temperature=temperature)
            if cached:
                logger.info("Returning cached response")
                return {
                    "id": str(uuid.uuid4()),
                    "created": int(time.time()),
                    "model": model,
                    "content": cached,
                    "finish_reason": "cached",
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                }
        
        # Load model if not loaded
        try:
            self.model_manager.load_model(
                model,
                quantization=settings.model_quantization,
            )
        except Exception as e:
            logger.warning(f"Failed to load model {model}: {e}")
            # Use dummy response for testing without model
            return {
                "id": str(uuid.uuid4()),
                "created": int(time.time()),
                "model": model,
                "content": "Model not loaded. Please configure a valid model.",
                "finish_reason": "error",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
        
        # Generate response
        config = GenerationConfig(
            temperature=temperature,
            max_new_tokens=max_tokens or 512,
            stop_strings=stop,
        )
        
        try:
            response = self.inference_engine.generate(
                prompt=prompt,
                model_name=model,
                config=config,
            )
            
            # Cache response
            if use_cache:
                self.cache_manager.set(prompt, model, response, temperature=temperature)
            
            return {
                "id": str(uuid.uuid4()),
                "created": int(time.time()),
                "model": model,
                "content": response,
                "finish_reason": "stop",
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(response.split()),
                "total_tokens": len(prompt.split()) + len(response.split()),
            }
            
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return {
                "id": str(uuid.uuid4()),
                "created": int(time.time()),
                "model": model,
                "content": f"Error: {str(e)}",
                "finish_reason": "error",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }

    async def create_completion(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        stop: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Create a text completion.
        
        Args:
            prompt: Input prompt
            model: Model name
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling
            top_k: Top-k sampling
            repetition_penalty: Repetition penalty
            stop: Stop sequences
            
        Returns:
            Response dictionary
        """
        model = model or settings.model_name
        
        # Check cache
        cached = self.cache_manager.get(prompt, model, temperature=temperature)
        if cached:
            return {
                "id": str(uuid.uuid4()),
                "created": int(time.time()),
                "model": model,
                "text": cached,
                "finish_reason": "cached",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
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
            
            # Cache response
            self.cache_manager.set(prompt, model, response, temperature=temperature)
            
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

    def stream_chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
    ):
        """Stream chat response.
        
        Args:
            messages: List of chat messages
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stop: Stop sequences
            
        Yields:
            Stream chunks
        """
        model = model or settings.model_name
        prompt = self._messages_to_prompt(messages)
        
        config = GenerationConfig(
            temperature=temperature,
            max_new_tokens=max_tokens or 512,
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
_chat_service: Optional[ChatService] = None


def get_chat_service() -> ChatService:
    """Get the chat service singleton."""
    global _chat_service
    if _chat_service is None:
        _chat_service = ChatService()
    return _chat_service