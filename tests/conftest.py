"""Pytest configuration and fixtures."""

import pytest
import asyncio
from typing import Generator

from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create an event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_client():
    """Create a test client."""
    from src.main import app
    
    with TestClient(app) as client:
        yield client


@pytest.fixture
def mock_model_manager():
    """Create a mock model manager."""
    from unittest.mock import MagicMock
    
    manager = MagicMock()
    manager.list_loaded_models.return_value = []
    manager.get_model.return_value = None
    return manager


@pytest.fixture
def sample_prompt() -> str:
    """Sample prompt for testing."""
    return "Once upon a time"


@pytest.fixture
def sample_messages() -> list:
    """Sample chat messages for testing."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
    ]