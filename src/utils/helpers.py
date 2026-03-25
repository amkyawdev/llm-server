"""Helper utilities."""

import uuid
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def generate_id(prefix: str = "") -> str:
    """Generate a unique ID.
    
    Args:
        prefix: Optional prefix for the ID
        
    Returns:
        Unique ID string
    """
    unique_id = str(uuid.uuid4())[:8]
    if prefix:
        return f"{prefix}-{unique_id}"
    return unique_id


def timestamp_now() -> int:
    """Get current Unix timestamp.
    
    Returns:
        Unix timestamp
    """
    return int(time.time())


def datetime_now_iso() -> str:
    """Get current datetime in ISO format.
    
    Returns:
        ISO format datetime string
    """
    return datetime.now(timezone.utc).isoformat()


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to a maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def safe_dict_get(d: Dict[str, Any], key: str, default: Any = None) -> Any:
    """Safely get a value from a dictionary.
    
    Args:
        d: Dictionary
        key: Key to get
        default: Default value if key not found
        
    Returns:
        Value or default
    """
    return d.get(key, default)


def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple dictionaries.
    
    Args:
        *dicts: Dictionaries to merge
        
    Returns:
        Merged dictionary
    """
    result = {}
    for d in dicts:
        result.update(d)
    return result


def format_bytes(size: int) -> str:
    """Format bytes to human-readable string.
    
    Args:
        size: Size in bytes
        
    Returns:
        Formatted string
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"


def parse_bool(value: str) -> bool:
    """Parse boolean from string.
    
    Args:
        value: String value
        
    Returns:
        Boolean
    """
    return value.lower() in ("true", "1", "yes", "on")


def chunks(lst: list, n: int):
    """Yield successive n-sized chunks from list.
    
    Args:
        lst: List to chunk
        n: Chunk size
        
    Yields:
        List chunks
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]