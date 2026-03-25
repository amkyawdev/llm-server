"""Download models script."""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Dict
from huggingface_hub import snapshot_download
from loguru import logger


# Model configurations
LLM_MODELS: Dict[str, str] = {
    "llama-2-7b": "meta-llama/Llama-2-7b-hf",
    "llama-2-13b": "meta-llama/Llama-2-13b-hf",
    "llama-3-8b": "meta-llama/Llama-3-8b-Instruct",
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "mistral-7b": "mistralai/Mistral-7B-v0.1",
    "mistral-7b-instruct": "mistralai/Mistral-7B-Instruct-v0.2",
    "mistral-small": "mistralai/Mistral-Small-Instruct-2409",
    "gemma-2b": "google/gemma-2b",
    "gemma-7b": "google/gemma-7b",
    "phi-3-mini-4k": "microsoft/Phi-3-mini-4k-instruct",
    "phi-3.5-mini": "microsoft/Phi-3.5-mini-instruct",
    "qwen-2-7b": "Qwen/Qwen2-7B-Instruct",
    "qwen-2-1.5b": "Qwen/Qwen2-1.5B-Instruct",
    "falcon-7b": "tiiuae/falcon-7b",
    "openchat-7b": "openchat/openchat-7b",
    "stablelm-3b": "stabilityai/stablelm-3b-4k",
}

EMBEDDING_MODELS: Dict[str, str] = {
    "bge-small-en-v1.5": "BAAI/bge-small-en-v1.5",
    "bge-base-en-v1.5": "BAAI/bge-base-en-v1.5",
    "bge-large-en-v1.5": "BAAI/bge-large-en-v1.5",
    "e5-small-v2": "intfloat/e5-small-v2",
    "e5-base-v2": "intfloat/e5-base-v2",
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
}


def download_model(model_name: str, output_dir: str, token: Optional[str] = None) -> bool:
    """Download a model from HuggingFace.
    
    Args:
        model_name: Model name (key from LLM_MODELS or EMBEDDING_MODELS)
        output_dir: Output directory
        token: HuggingFace access token
        
    Returns:
        True if successful, False otherwise
    """
    # Check if it's an LLM model
    if model_name in LLM_MODELS:
        repo_id = LLM_MODELS[model_name]
    elif model_name in EMBEDDING_MODELS:
        repo_id = EMBEDDING_MODELS[model_name]
    else:
        logger.error(f"Unknown model: {model_name}")
        return False
    
    logger.info(f"Downloading model: {model_name} from {repo_id}")
    
    output_path = Path(output_dir) / model_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(output_path),
            local_dir_use_symlinks=False,
            token=token,
        )
        logger.info(f"Model downloaded to: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        return False


def download_all(output_dir: str, token: Optional[str] = None) -> None:
    """Download all available models.
    
    Args:
        output_dir: Output directory
        token: HuggingFace access token
    """
    print("\n" + "="*60)
    print("Downloading all LLM models...")
    print("="*60)
    
    for model_name in LLM_MODELS:
        print(f"\n--- {model_name} ---")
        download_model(model_name, output_dir, token)
    
    print("\n" + "="*60)
    print("Downloading all Embedding models...")
    print("="*60)
    
    for model_name in EMBEDDING_MODELS:
        print(f"\n--- {model_name} ---")
        download_model(model_name, output_dir, token)


def list_models() -> None:
    """List all available models."""
    print("\n=== Available LLM Models ===")
    for name, repo in LLM_MODELS.items():
        print(f"  {name}: {repo}")
    
    print("\n=== Available Embedding Models ===")
    for name, repo in EMBEDDING_MODELS.items():
        print(f"  {name}: {repo}")


def main():
    parser = argparse.ArgumentParser(description="Download LLM and embedding models")
    parser.add_argument(
        "--model",
        type=str,
        help="Model name to download (use --list to see available models)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/models",
        help="Output directory",
    )
    parser.add_argument(
        "--token",
        type=str,
        help="HuggingFace access token (required for gated models)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all available models",
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_models()
        return
    
    if args.all:
        download_all(args.output_dir, args.token)
        return
    
    if args.model:
        success = download_model(args.model, args.output_dir, args.token)
        sys.exit(0 if success else 1)
    
    parser.print_help()


if __name__ == "__main__":
    main()