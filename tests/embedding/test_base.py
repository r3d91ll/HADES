"""Tests for the embedding base module.

This module tests the core functionality of the embedding registry
and adapter protocol.
"""

import sys
import os
import pytest
from typing import List, cast
import asyncio
from unittest.mock import MagicMock, AsyncMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.embedding.base import (
    EmbeddingAdapter,
    register_adapter,
    get_adapter,
    EmbeddingVector,
)


class MockEmbeddingAdapter:
    """Mock embedding adapter for testing."""
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.embed_called = False
        self.embed_single_called = False
    
    async def embed(self, texts: List[str], **kwargs) -> List[EmbeddingVector]:
        """Mock implementation of embed method."""
        self.embed_called = True
        return [[0.1, 0.2, 0.3] for _ in texts]
    
    async def embed_single(self, text: str, **kwargs) -> EmbeddingVector:
        """Mock implementation of embed_single method."""
        self.embed_single_called = True
        return [0.1, 0.2, 0.3]


@pytest.fixture
def reset_registry():
    """Reset the adapter registry before and after tests."""
    from src.embedding.base import _adapter_registry
    
    # Store original registry
    original_registry = _adapter_registry.copy()
    
    # Clear registry for tests
    _adapter_registry.clear()
    
    yield
    
    # Restore original registry
    _adapter_registry.clear()
    _adapter_registry.update(original_registry)


@pytest.mark.asyncio
async def test_register_and_get_adapter(reset_registry):
    """Test registering and retrieving adapters."""
    # Register mock adapter
    register_adapter("mock", MockEmbeddingAdapter)
    
    # Get adapter instance
    adapter = get_adapter("mock", param1="value1")
    
    # Check adapter type
    assert isinstance(adapter, MockEmbeddingAdapter)
    
    # Check constructor arguments were passed
    assert adapter.kwargs["param1"] == "value1"


@pytest.mark.asyncio
async def test_adapter_embed_method(reset_registry):
    """Test the embed method of adapters."""
    # Register mock adapter
    register_adapter("mock", MockEmbeddingAdapter)
    
    # Get adapter instance
    adapter = get_adapter("mock")
    
    # Call embed method
    texts = ["text1", "text2", "text3"]
    results = await adapter.embed(texts)
    
    # Check results
    assert adapter.embed_called
    assert len(results) == len(texts)
    assert all(isinstance(emb, list) for emb in results)
    assert all(len(emb) == 3 for emb in results)


@pytest.mark.asyncio
async def test_adapter_embed_single_method(reset_registry):
    """Test the embed_single method of adapters."""
    # Register mock adapter
    register_adapter("mock", MockEmbeddingAdapter)
    
    # Get adapter instance
    adapter = get_adapter("mock")
    
    # Call embed_single method
    result = await adapter.embed_single("text")
    
    # Check result
    assert adapter.embed_single_called
    assert isinstance(result, list)
    assert len(result) == 3


@pytest.mark.asyncio
async def test_get_nonexistent_adapter(reset_registry):
    """Test getting an adapter that doesn't exist."""
    with pytest.raises(KeyError):
        get_adapter("nonexistent")


@pytest.mark.asyncio
async def test_register_duplicate_adapter(reset_registry):
    """Test registering an adapter with a name that's already taken."""
    # Register first adapter
    register_adapter("test", MockEmbeddingAdapter)
    
    # Register second adapter with same name (should work but log a warning)
    register_adapter("test", MockEmbeddingAdapter)
    
    # Get adapter and ensure it's the second one
    adapter = get_adapter("test")
    assert isinstance(adapter, MockEmbeddingAdapter)


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
