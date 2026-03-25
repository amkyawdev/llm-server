"""Validation utilities."""

import re
from typing import Optional


def validate_prompt(prompt: str, max_length: Optional[int] = None) -> bool:
    """Validate a prompt string.
    
    Args:
        prompt: Prompt to validate
        max_length: Optional maximum length
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If prompt is invalid
    """
    if not prompt:
        raise ValueError("Prompt cannot be empty")
    
    if not isinstance(prompt, str):
        raise ValueError("Prompt must be a string")
    
    if max_length and len(prompt) > max_length:
        raise ValueError(f"Prompt exceeds maximum length of {max_length}")
    
    return True


def validate_model_name(model_name: str) -> bool:
    """Validate a model name.
    
    Args:
        model_name: Model name to validate
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If model name is invalid
    """
    if not model_name:
        raise ValueError("Model name cannot be empty")
    
    # Allow alphanumeric, hyphens, and underscores
    if not re.match(r"^[a-zA-Z0-9_-]+$", model_name):
        raise ValueError("Model name contains invalid characters")
    
    return True


def validate_temperature(temperature: float) -> bool:
    """Validate temperature parameter.
    
    Args:
        temperature: Temperature value
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If temperature is invalid
    """
    if not 0.0 <= temperature <= 2.0:
        raise ValueError("Temperature must be between 0.0 and 2.0")
    
    return True


def validate_max_tokens(max_tokens: int) -> bool:
    """Validate max tokens parameter.
    
    Args:
        max_tokens: Max tokens value
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If max_tokens is invalid
    """
    if max_tokens < 1:
        raise ValueError("max_tokens must be at least 1")
    
    if max_tokens > 8192:
        raise ValueError("max_tokens cannot exceed 8192")
    
    return True


def validate_api_key(api_key: str) -> bool:
    """Validate API key format.
    
    Args:
        api_key: API key to validate
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If API key is invalid
    """
    if not api_key:
        raise ValueError("API key cannot be empty")
    
    if len(api_key) < 16:
        raise ValueError("API key is too short")
    
    return True