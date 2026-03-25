"""Tokenizer Manager - Handles tokenization operations."""

from typing import Optional, List, Dict, Any
import torch
from transformers import AutoTokenizer, PreTrainedTokenizer
from loguru import logger

from config import settings


class TokenizerManager:
    """Manages tokenizers for LLM models."""

    def __init__(self):
        self._tokenizers: Dict[str, PreTrainedTokenizer] = {}
        logger.info("TokenizerManager initialized")

    def load_tokenizer(
        self,
        model_name: str,
        cache_dir: Optional[str] = None,
        trust_remote_code: bool = False,
    ) -> PreTrainedTokenizer:
        """Load a tokenizer.

        Args:
            model_name: Name of the model
            cache_dir: Optional cache directory
            trust_remote_code: Whether to trust remote code

        Returns:
            Loaded tokenizer
        """
        if model_name in self._tokenizers:
            logger.debug(f"Using cached tokenizer for {model_name}")
            return self._tokenizers[model_name]

        logger.info(f"Loading tokenizer for: {model_name}")

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
        )

        # Configure tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if tokenizer.chat_template is None:
            # Set default chat template
            tokenizer.chat_template = "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{% set role = 'system' %}{% else %}{% set role = message['role'] %}{% endif %}{{ '<|' + role + '|>\n' + message['content'] + '<|end_of_text|>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>\n' }}{% endif %}"

        self._tokenizers[model_name] = tokenizer
        logger.info(f"Tokenizer loaded for {model_name}")

        return tokenizer

    def encode(
        self,
        text: str,
        model_name: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        truncation: bool = False,
    ) -> List[int]:
        """Encode text to token IDs.

        Args:
            text: Text to encode
            model_name: Model name for tokenizer
            add_special_tokens: Whether to add special tokens
            max_length: Maximum length
            truncation: Whether to truncate

        Returns:
            List of token IDs
        """
        tokenizer = self.get_tokenizer(model_name)
        return tokenizer.encode(
            text,
            add_special_tokens=add_special_tokens,
            max_length=max_length or settings.model_max_length,
            truncation=truncation,
        )

    def decode(
        self,
        token_ids: List[int],
        model_name: str,
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode token IDs to text.

        Args:
            token_ids: Token IDs to decode
            model_name: Model name for tokenizer
            skip_special_tokens: Whether to skip special tokens

        Returns:
            Decoded text
        """
        tokenizer = self.get_tokenizer(model_name)
        return tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def get_tokenizer(self, model_name: str) -> PreTrainedTokenizer:
        """Get a loaded tokenizer.

        Args:
            model_name: Name of the model

        Returns:
            Tokenizer instance
        """
        if model_name not in self._tokenizers:
            return self.load_tokenizer(model_name)
        return self._tokenizers[model_name]

    def get_token_count(self, text: str, model_name: str) -> int:
        """Get the number of tokens in text.

        Args:
            text: Text to count
            model_name: Model name for tokenizer

        Returns:
            Number of tokens
        """
        tokenizer = self.get_tokenizer(model_name)
        return len(tokenizer.encode(text))

    def batch_encode(
        self,
        texts: List[str],
        model_name: str,
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Batch encode texts.

        Args:
            texts: List of texts to encode
            model_name: Model name for tokenizer
            max_length: Maximum length
            padding: Whether to pad
            truncation: Whether to truncate

        Returns:
            Dictionary with input_ids and attention_mask
        """
        tokenizer = self.get_tokenizer(model_name)
        return tokenizer(
            texts,
            max_length=max_length or settings.model_max_length,
            padding=padding,
            truncation=truncation,
            return_tensors="pt",
        )

    def clear(self) -> None:
        """Clear all loaded tokenizers."""
        self._tokenizers.clear()
        logger.info("All tokenizers cleared")


# Global tokenizer manager instance
_tokenizer_manager: Optional[TokenizerManager] = None


def get_tokenizer_manager() -> TokenizerManager:
    """Get the global tokenizer manager instance."""
    global _tokenizer_manager
    if _tokenizer_manager is None:
        _tokenizer_manager = TokenizerManager()
    return _tokenizer_manager