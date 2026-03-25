"""Inference Engine - Handles text generation and inference."""

import asyncio
from typing import Optional, Dict, Any, Generator, AsyncGenerator
from dataclasses import dataclass

import torch
from loguru import logger

from config import settings


@dataclass
class GenerationConfig:
    """Configuration for text generation."""

    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    max_new_tokens: int = 512
    do_sample: bool = True
    num_beams: int = 1
    stop_strings: Optional[list] = None


@dataclass
class EmbeddingConfig:
    """Configuration for embeddings."""

    normalize: bool = True
    pooling_strategy: str = "mean"


class InferenceEngine:
    """Handles model inference for text generation and embeddings."""

    def __init__(self, model_manager):
        self.model_manager = model_manager
        self._default_config = GenerationConfig()

    def generate(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ) -> str:
        """Generate text from a prompt.

        Args:
            prompt: Input prompt
            model_name: Optional model name override
            config: Generation configuration

        Returns:
            Generated text
        """
        config = config or self._default_config
        model = self.model_manager.get_model(model_name)
        tokenizer = self.model_manager.get_tokenizer(
            model_name or settings.model_name
        )

        logger.debug(f"Generating text for prompt: {prompt[:50]}...")

        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.model_manager.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                repetition_penalty=config.repetition_penalty,
                max_new_tokens=config.max_new_tokens,
                do_sample=config.do_sample,
                num_beams=config.num_beams,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove input prompt from output if present
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()

        return generated_text

    def generate_streaming(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ) -> Generator[str, None, None]:
        """Generate text with streaming output.

        Args:
            prompt: Input prompt
            model_name: Optional model name override
            config: Generation configuration

        Yields:
            Generated text chunks
        """
        config = config or self._default_config
        model = self.model_manager.get_model(model_name)
        tokenizer = self.model_manager.get_tokenizer(
            model_name or settings.model_name
        )

        logger.debug(f"Generating streaming text for prompt: {prompt[:50]}...")

        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.model_manager.device) for k, v in inputs.items()}

        # Generate with streaming
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                repetition_penalty=config.repetition_penalty,
                max_new_tokens=config.max_new_tokens,
                do_sample=config.do_sample,
                num_beams=config.num_beams,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )

        # Stream the output token by token
        generated_ids = outputs.sequences[0]
        for token_id in generated_ids:
            token = tokenizer.decode(token_id, skip_special_tokens=True)
            if token:
                yield token

    async def generate_async(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ) -> str:
        """Generate text asynchronously.

        Args:
            prompt: Input prompt
            model_name: Optional model name override
            config: Generation configuration

        Returns:
            Generated text
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.generate, prompt, model_name, config
        )

    def generate_embeddings(
        self,
        texts: list[str],
        model_name: Optional[str] = None,
        config: Optional[EmbeddingConfig] = None,
    ) -> list[list[float]]:
        """Generate embeddings for texts.

        Args:
            texts: List of texts to embed
            model_name: Optional model name override
            config: Embedding configuration

        Returns:
            List of embedding vectors
        """
        config = config or EmbeddingConfig()
        model = self.model_manager.get_model(model_name)
        tokenizer = self.model_manager.get_tokenizer(
            model_name or settings.model_name
        )

        logger.debug(f"Generating embeddings for {len(texts)} texts")

        # Tokenize
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=settings.model_max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.model_manager.device) for k, v in encoded.items()}

        # Generate embeddings
        with torch.no_grad():
            outputs = model(**encoded)

        # Apply pooling strategy
        if config.pooling_strategy == "mean":
            attention_mask = encoded["attention_mask"]
            hidden_states = outputs.last_hidden_state

            # Mean pooling
            mask_expanded = (
                attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            )
            sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            embeddings = sum_embeddings / sum_mask
        elif config.pooling_strategy == "cls":
            embeddings = outputs.last_hidden_state[:, 0]
        else:
            embeddings = outputs.last_hidden_state.mean(dim=1)

        # Normalize if configured
        if config.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().tolist()


# Global inference engine instance
_inference_engine: Optional[InferenceEngine] = None


def get_inference_engine() -> InferenceEngine:
    """Get the global inference engine instance."""
    global _inference_engine
    if _inference_engine is None:
        from .model_manager import get_model_manager

        _inference_engine = InferenceEngine(get_model_manager())
    return _inference_engine