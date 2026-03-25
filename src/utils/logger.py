"""Logging configuration and utilities."""

import sys
from pathlib import Path
from typing import Optional
import logging

from loguru import logger

from config import settings


# Default format
DEFAULT_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)


def setup_logger(
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    rotation: str = "10 MB",
    retention: str = "7 days",
) -> None:
    """Setup application logging.

    Args:
        log_level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
        rotation: Log rotation size
        retention: Log retention period
    """
    log_level = log_level or settings.log_level

    # Remove default handler
    logger.remove()

    # Console handler
    logger.add(
        sys.stdout,
        format=DEFAULT_FORMAT,
        level=log_level,
        colorize=True,
    )

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            log_file,
            format=DEFAULT_FORMAT,
            level=log_level,
            rotation=rotation,
            retention=retention,
            compression="zip",
        )

    # Set default logger
    logger.configure(
        handlers=[
            {"sink": sys.stdout, "level": log_level},
        ]
    )

    logger.info(f"Logger initialized at {log_level} level")


def get_logger(name: str):
    """Get a named logger.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logger.bind(name=name)