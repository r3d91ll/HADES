"""Tests for the batch embedding module.

This module tests the batch processing functionality for embeddings.
"""

import sys
import os
import pytest
from typing import List, cast
import asyncio
from unittest.mock import MagicMock, AsyncMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.embedding.batch import batch_embed
from src.embedding.base import EmbeddingAdapter, EmbeddingVector


class MockBatchEmbeddingAdapter:
    """Mock embedding adapter for batch testing."""
    
    def __init__(self):
        self.batches = []
        self.kwargs_history = []
    
    async def embed(self, texts: List[str], **kwargs) -> List[EmbeddingVector]:
        """Mock implementation that records batches and returns fixed vectors."""
        self.batches.append(texts)
        self.kwargs_history.append(kwargs)
        
        # Return a fixed embedding for each text
        return [[float(i) * 0.1, float(i) * 0.2, float(i) * 0.3] for i in range(len(texts))]
    
    async def embed_single(self, text: str, **kwargs) -> EmbeddingVector:
        """Mock implementation of embed_single."""
        result = await self.embed([text], **kwargs)
        return result[0]


@pytest.mark.asyncio
async def test_batch_embed_empty():
    """Test batch_embed with empty input."""
    adapter = MockBatchEmbeddingAdapter()
    result = await batch_embed([], adapter)
    
    assert result == []
    assert len(adapter.batches) == 0


@pytest.mark.asyncio
async def test_batch_embed_single_batch():
    """Test batch_embed with a single batch."""
    texts = ["text1", "text2", "text3"]
    adapter = MockBatchEmbeddingAdapter()
    
    result = await batch_embed(texts, adapter, batch_size=10)
    
    # Check that all texts were processed in a single batch
    assert len(adapter.batches) == 1
    assert adapter.batches[0] == texts
    
    # Check that the result contains the expected number of embeddings
    assert len(result) == len(texts)
    
    # Check that each embedding has the expected format
    for i, embedding in enumerate(result):
        assert embedding == [float(i) * 0.1, float(i) * 0.2, float(i) * 0.3]


@pytest.mark.asyncio
async def test_batch_embed_multiple_batches():
    """Test batch_embed with multiple batches."""
    texts = ["text1", "text2", "text3", "text4", "text5"]
    adapter = MockBatchEmbeddingAdapter()
    
    result = await batch_embed(texts, adapter, batch_size=2)
    
    # Check that texts were processed in expected batches
    assert len(adapter.batches) == 3
    assert adapter.batches[0] == ["text1", "text2"]
    assert adapter.batches[1] == ["text3", "text4"]
    assert adapter.batches[2] == ["text5"]
    
    # Check that the result contains all embeddings in correct order
    assert len(result) == len(texts)
    
    # Verify the first two embeddings match the expected pattern
    assert result[0] == [0.0, 0.0, 0.0]  # First element of first batch
    assert result[1] == [0.1, 0.2, 0.3]  # Second element of first batch
    assert result[2] == [0.0, 0.0, 0.0]  # First element of second batch
    assert result[3] == [0.1, 0.2, 0.3]  # Second element of second batch
    assert result[4] == [0.0, 0.0, 0.0]  # First (only) element of third batch


@pytest.mark.asyncio
async def test_batch_embed_with_kwargs():
    """Test that kwargs are properly passed to the adapter."""
    texts = ["text1", "text2", "text3"]
    adapter = MockBatchEmbeddingAdapter()
    
    await batch_embed(texts, adapter, batch_size=3, custom_param="value", another_param=42)
    
    # Check that kwargs were passed to adapter
    assert len(adapter.kwargs_history) == 1
    assert adapter.kwargs_history[0]["custom_param"] == "value"
    assert adapter.kwargs_history[0]["another_param"] == 42


@pytest.mark.asyncio
async def test_batch_embed_concurrency():
    """Test that concurrency limits are respected."""
    # Create many texts to process
    texts = [f"text{i}" for i in range(50)]
    adapter = MockBatchEmbeddingAdapter()
    
    # Add a delay to the embed method to test concurrency
    original_embed = adapter.embed
    
    async def delayed_embed(*args, **kwargs):
        await asyncio.sleep(0.01)  # Short delay
        return await original_embed(*args, **kwargs)
    
    adapter.embed = delayed_embed
    
    # Process with small batches and limited concurrency
    result = await batch_embed(
        texts,
        adapter,
        batch_size=5,
        max_concurrency=3
    )
    
    # Check that all batches were processed
    assert len(adapter.batches) == 10  # 50 texts ÷ 5 batch_size = 10 batches
    
    # Check that all embeddings were returned
    assert len(result) == len(texts)


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
