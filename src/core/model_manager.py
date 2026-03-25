"""Model Manager - Handles loading and managing LLM models."""

import os
from typing import Optional, Dict, Any, List
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer
from loguru import logger

from config import settings


class ModelManager:
    """Manages loading, caching, and accessing LLM models."""

    def __init__(self):
        self._models: Dict[str, Any] = {}
        self._tokenizers: Dict[str, Any] = {}
        self._current_model_name: Optional[str] = None
        self._device = settings.model_device if torch.cuda.is_available() else "cpu"
        logger.info(f"ModelManager initialized with device: {self._device}")

    @property
    def device(self) -> str:
        """Get the current device."""
        return self._device

    def load_model(
        self,
        model_name: str,
        model_path: Optional[str] = None,
        quantization: Optional[str] = None,
        device_map: str = "auto",
        trust_remote_code: bool = False,
    ) -> Any:
        """Load a model into memory.

        Args:
            model_name: Name of the model to load
            model_path: Optional path to local model
            quantization: Quantization method (4bit, 8bit, or None)
            device_map: Device mapping strategy
            trust_remote_code: Whether to trust remote code

        Returns:
            Loaded model
        """
        cache_key = f"{model_name}_{quantization or 'fp16'}"

        if cache_key in self._models:
            logger.info(f"Model {cache_key} already loaded, returning cached version")
            return self._models[cache_key]

        logger.info(f"Loading model: {model_name}")

        try:
            model_path = model_path or settings.model_path

            # Determine loading kwargs based on quantization
            load_kwargs = {
                "device_map": device_map,
                "trust_remote_code": trust_remote_code,
                "torch_dtype": torch.float16,
            }

            if quantization == "4bit":
                load_kwargs["load_in_4bit"] = True
                load_kwargs["bnb_4bit_compute_dtype"] = torch.float16
                load_kwargs["bnb_4bit_use_double_quant"] = True
                load_kwargs["bnb_4bit_quant_type"] = "nf4"
            elif quantization == "8bit":
                load_kwargs["load_in_8bit"] = True

            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                model_path if os.path.isdir(model_path) else model_name,
                **load_kwargs,
            )

            self._models[cache_key] = model
            self._current_model_name = cache_key
            logger.info(f"Model {model_name} loaded successfully")

            return model

        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise

    def load_embedding_model(
        self,
        model_name: str,
        model_path: Optional[str] = None,
    ) -> Any:
        """Load an embedding model.

        Args:
            model_name: Name of the embedding model
            model_path: Optional path to local model

        Returns:
            Loaded embedding model
        """
        cache_key = f"embedding_{model_name}"

        if cache_key in self._models:
            logger.info(f"Embedding model {cache_key} already loaded")
            return self._models[cache_key]

        logger.info(f"Loading embedding model: {model_name}")

        try:
            model_path = model_path or settings.model_path

            model = AutoModel.from_pretrained(
                model_path if os.path.isdir(model_path) else model_name,
                torch_dtype=torch.float16,
                device_map="auto",
            )

            self._models[cache_key] = model
            logger.info(f"Embedding model {model_name} loaded successfully")

            return model

        except Exception as e:
            logger.error(f"Failed to load embedding model {model_name}: {e}")
            raise

    def get_tokenizer(self, model_name: str) -> Any:
        """Get or load tokenizer for a model.

        Args:
            model_name: Name of the model

        Returns:
            Tokenizer instance
        """
        if model_name in self._tokenizers:
            return self._tokenizers[model_name]

        logger.info(f"Loading tokenizer for: {model_name}")

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=False,
            )

            # Add padding token if not present
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            self._tokenizers[model_name] = tokenizer
            return tokenizer

        except Exception as e:
            logger.error(f"Failed to load tokenizer for {model_name}: {e}")
            raise

    def get_model(self, model_name: Optional[str] = None) -> Any:
        """Get a loaded model.

        Args:
            model_name: Name of the model to retrieve

        Returns:
            The model instance
        """
        if model_name is None:
            model_name = self._current_model_name

        if model_name and model_name in self._models:
            return self._models[model_name]

        raise ValueError(f"Model {model_name} not loaded")

    def unload_model(self, model_name: str) -> None:
        """Unload a model from memory.

        Args:
            model_name: Name of the model to unload
        """
        if model_name in self._models:
            del self._models[model_name]
            logger.info(f"Model {model_name} unloaded")
            torch.cuda.empty_cache()

    def list_loaded_models(self) -> List[str]:
        """List all currently loaded models.

        Returns:
            List of loaded model names
        """
        return list(self._models.keys())

    def clear(self) -> None:
        """Clear all loaded models and tokenizers."""
        self._models.clear()
        self._tokenizers.clear()
        self._current_model_name = None
        torch.cuda.empty_cache()
        logger.info("All models cleared from memory")


# Global model manager instance
_model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """Get the global model manager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager