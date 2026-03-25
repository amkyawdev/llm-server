"""Quantizer - Handles model quantization."""

from typing import Optional, Dict, Any
import torch
from transformers import BitsAndBytesConfig
from loguru import logger


class Quantizer:
    """Handles model quantization for efficient inference."""

    QUANTIZATION_CONFIGS = {
        "4bit": {
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": torch.float16,
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_quant_type": "nf4",
        },
        "8bit": {
            "load_in_8bit": True,
        },
        "fp8": {
            # FP8 quantization (experimental)
            "load_in_8bit": False,
        },
    }

    def __init__(self):
        logger.info("Quantizer initialized")

    def get_quantization_config(
        self,
        quantization: str,
        compute_dtype: torch.dtype = torch.float16,
    ) -> Optional[BitsAndBytesConfig]:
        """Get quantization config for a model.

        Args:
            quantization: Quantization type (4bit, 8bit)
            compute_dtype: Compute dtype

        Returns:
            BitsAndBytesConfig or None
        """
        if quantization not in self.QUANTIZATION_CONFIGS:
            logger.warning(f"Unknown quantization type: {quantization}")
            return None

        config_dict = self.QUANTIZATION_CONFIGS[quantization].copy()
        
        if quantization == "4bit":
            config_dict["bnb_4bit_compute_dtype"] = compute_dtype

        return BitsAndBytesConfig(**config_dict)

    def is_quantization_available(self, quantization: str) -> bool:
        """Check if quantization is available.

        Args:
            quantization: Quantization type

        Returns:
            True if available
        """
        if quantization not in self.QUANTIZATION_CONFIGS:
            return False

        # Check if bitsandbytes is available
        try:
            import bitsandbytes
            return True
        except ImportError:
            logger.warning("bitsandbytes not installed")
            return False

    def get_memory_estimate(
        self,
        model_params: int,
        quantization: str,
    ) -> int:
        """Estimate memory usage for a quantized model.

        Args:
            model_params: Number of model parameters
            quantization: Quantization type

        Returns:
            Estimated memory in bytes
        """
        bits_per_param = {
            "fp16": 16,
            "fp32": 32,
            "4bit": 4,
            "8bit": 8,
        }

        bits = bits_per_param.get(quantization, 16)
        bytes_per_param = bits / 8

        # Add overhead for quantization metadata
        overhead_factor = 1.1 if quantization in ["4bit", "8bit"] else 1.0

        return int(model_params * bytes_per_param * overhead_factor)


# Global quantizer instance
_quantizer: Optional[Quantizer] = None


def get_quantizer() -> Quantizer:
    """Get the quantizer singleton."""
    global _quantizer
    if _quantizer is None:
        _quantizer = Quantizer()
    return _quantizer