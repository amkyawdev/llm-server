"""LLM Loader - Handles loading LLM models from various sources."""

import os
from typing import Optional, Dict, Any
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModel,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from loguru import logger

from config import settings


class LLMLoader:
    """Handles loading LLM models from HuggingFace or local paths."""

    def __init__(self):
        self._loaded_models: Dict[str, Any] = {}

    def load_model(
        self,
        model_name: str,
        model_path: Optional[str] = None,
        quantization: Optional[str] = None,
        device_map: str = "auto",
        trust_remote_code: bool = False,
    ) -> Any:
        """Load a causal language model.

        Args:
            model_name: Model name or HuggingFace ID
            model_path: Optional local path
            quantization: Quantization type (4bit, 8bit)
            device_map: Device mapping strategy
            trust_remote_code: Trust remote code

        Returns:
            Loaded model
        """
        cache_key = f"{model_name}_{quantization or 'fp16'}"

        if cache_key in self._loaded_models:
            logger.info(f"Using cached model: {cache_key}")
            return self._loaded_models[cache_key]

        logger.info(f"Loading model: {model_name}")

        model_path = model_path or settings.model_path

        # Build loading kwargs
        load_kwargs = {
            "device_map": device_map,
            "trust_remote_code": trust_remote_code,
            "torch_dtype": torch.float16,
        }

        # Add quantization config
        if quantization == "4bit":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif quantization == "8bit":
            load_kwargs["load_in_8bit"] = True

        try:
            # Check if local path exists
            if os.path.isdir(model_path):
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    **load_kwargs,
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    **load_kwargs,
                )

            self._loaded_models[cache_key] = model
            logger.info(f"Model loaded successfully: {cache_key}")

            return model

        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise

    def load_embedding_model(
        self,
        model_name: str,
        model_path: Optional[str] = None,
        device_map: str = "auto",
    ) -> Any:
        """Load an embedding model.

        Args:
            model_name: Model name or HuggingFace ID
            model_path: Optional local path
            device_map: Device mapping strategy

        Returns:
            Loaded model
        """
        cache_key = f"embedding_{model_name}"

        if cache_key in self._loaded_models:
            return self._loaded_models[cache_key]

        logger.info(f"Loading embedding model: {model_name}")

        model_path = model_path or settings.model_path

        try:
            if os.path.isdir(model_path):
                model = AutoModel.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    device_map=device_map,
                )
            else:
                model = AutoModel.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map=device_map,
                )

            self._loaded_models[cache_key] = model
            logger.info(f"Embedding model loaded: {cache_key}")

            return model

        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    def load_tokenizer(
        self,
        model_name: str,
        trust_remote_code: bool = False,
    ) -> Any:
        """Load a tokenizer.

        Args:
            model_name: Model name
            trust_remote_code: Trust remote code

        Returns:
            Tokenizer
        """
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code,
            )

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            return tokenizer

        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise

    def unload_model(self, model_name: str) -> None:
        """Unload a model.

        Args:
            model_name: Model name to unload
        """
        if model_name in self._loaded_models:
            del self._loaded_models[model_name]
            torch.cuda.empty_cache()
            logger.info(f"Model unloaded: {model_name}")

    def clear(self) -> None:
        """Clear all loaded models."""
        self._loaded_models.clear()
        torch.cuda.empty_cache()
        logger.info("All models cleared")


# Global loader instance
_llm_loader: Optional[LLMLoader] = None


def get_llm_loader() -> LLMLoader:
    """Get the LLM loader singleton."""
    global _llm_loader
    if _llm_loader is None:
        _llm_loader = LLMLoader()
    return _llm_loader