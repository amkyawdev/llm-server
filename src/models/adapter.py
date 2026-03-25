"""Model Adapter - Adapts different model types for unified access."""

from typing import Optional, List, Dict, Any
import torch
from loguru import logger

from config import settings


class ModelAdapter:
    """Unified adapter for different model types."""

    def __init__(self):
        self._adapters: Dict[str, Any] = {}
        logger.info("ModelAdapter initialized")

    def adapt_model(self, model: Any, model_type: str) -> Any:
        """Adapt a model for unified access.

        Args:
            model: The model to adapt
            model_type: Type of model (causal_lm, embedding, etc.)

        Returns:
            Adapted model
        """
        if model_type == "causal_lm":
            return self._adapt_causal_lm(model)
        elif model_type == "embedding":
            return self._adapt_embedding_model(model)
        else:
            logger.warning(f"Unknown model type: {model_type}")
            return model

    def _adapt_causal_lm(self, model: Any) -> Any:
        """Adapt a causal language model.

        Args:
            model: Causal LM model

        Returns:
            Adapted model
        """
        # Add custom generate method if not present
        if not hasattr(model, "custom_generate"):
            original_generate = model.generate

            def custom_generate(*args, **kwargs):
                # Add default parameters
                if "max_new_tokens" not in kwargs:
                    kwargs["max_new_tokens"] = 512
                if "temperature" not in kwargs:
                    kwargs["temperature"] = 0.7
                return original_generate(*args, **kwargs)

            model.custom_generate = custom_generate

        return model

    def _adapt_embedding_model(self, model: Any) -> Any:
        """Adapt an embedding model.

        Args:
            model: Embedding model

        Returns:
            Adapted model
        """
        # Add pooling method if not present
        if not hasattr(model, "get_embeddings"):

            def get_embeddings(texts: List[str], pooling: str = "mean") -> List[List[float]]:
                # Tokenize
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(
                    model.config._name_or_path
                )

                inputs = tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )

                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model(**inputs)

                # Apply pooling
                if pooling == "mean":
                    attention_mask = inputs["attention_mask"]
                    hidden_states = outputs.last_hidden_state

                    mask_expanded = (
                        attention_mask.unsqueeze(-1)
                        .expand(hidden_states.size())
                        .float()
                    )
                    sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
                    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                    embeddings = sum_embeddings / sum_mask

                elif pooling == "cls":
                    embeddings = outputs.last_hidden_state[:, 0]
                else:
                    embeddings = outputs.last_hidden_state.mean(dim=1)

                # Normalize
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                return embeddings.cpu().tolist()

            model.get_embeddings = get_embeddings

        return model

    def get_model_info(self, model: Any) -> Dict[str, Any]:
        """Get information about a model.

        Args:
            model: Model to inspect

        Returns:
            Model information dictionary
        """
        info = {
            "type": type(model).__name__,
            "device": str(next(model.parameters()).device)
            if hasattr(model, "parameters")
            else "unknown",
            "dtype": str(next(model.parameters()).dtype)
            if hasattr(model, "parameters")
            else "unknown",
        }

        if hasattr(model, "config"):
            info["model_name"] = getattr(model.config, "model_type", "unknown")
            info["max_length"] = getattr(model.config, "max_position_embeddings", 0)

        return info


# Global adapter instance
_model_adapter: Optional[ModelAdapter] = None


def get_model_adapter() -> ModelAdapter:
    """Get the model adapter singleton."""
    global _model_adapter
    if _model_adapter is None:
        _model_adapter = ModelAdapter()
    return _model_adapter