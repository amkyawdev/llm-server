"""Download models script."""

import argparse
import os
from pathlib import Path
from huggingface_hub import snapshot_download
from loguru import logger


def download_model(model_name: str, output_dir: str) -> None:
    """Download a model from HuggingFace.
    
    Args:
        model_name: Model name or path
        output_dir: Output directory
    """
    logger.info(f"Downloading model: {model_name}")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        snapshot_download(
            repo_id=model_name,
            local_dir=str(output_path / model_name),
            local_dir_use_symlinks=False,
        )
        logger.info(f"Model downloaded to: {output_path / model_name}")
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Download LLM models")
    parser.add_argument(
        "--model",
        type=str,
        default="llama-2-7b",
        help="Model name to download",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/models",
        help="Output directory",
    )
    
    args = parser.parse_args()
    
    download_model(args.model, args.output_dir)


if __name__ == "__main__":
    main()