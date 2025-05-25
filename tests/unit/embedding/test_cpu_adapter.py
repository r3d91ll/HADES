"""
Unit tests for the CPU embedding adapter.
"""
import asyncio
import os
from typing import List, Dict, Any

import pytest
import numpy as np

from src.embedding.adapters.cpu_adapter import CPUEmbeddingAdapter
from src.types.common import EmbeddingVector


class MockSentenceTransformer:
    """Mock SentenceTransformer class for testing."""
    
    def __init__(self, model_name, device="cpu"):
        self.model_name = model_name
        self.device = device
        self.dimensions = 384  # Standard size for mini models
    
    def encode(
        self, 
        texts: List[str], 
        batch_size: int = 32, 
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True
    ) -> np.ndarray:
        """Generate mock embeddings for testing."""
        # Create deterministic embeddings based on text content
        embeddings = []
        for text in texts:
            # Simple deterministic embedding based on text content
            text_hash = sum(ord(c) for c in text) % 100
            # Create a vector where each element is influenced by text_hash
            vector = np.array([
                (i * text_hash / 100) % 1.0 for i in range(self.dimensions)
            ], dtype=np.float32)
            
            # Normalize if requested
            if normalize_embeddings:
                vector = vector / np.linalg.norm(vector)
                
            embeddings.append(vector)
        
        return np.array(embeddings)


@pytest.fixture
def mock_sentence_transformers(monkeypatch):
    """Patch the SentenceTransformer import to use our mock."""
    class MockSentenceTransformersModule:
        SentenceTransformer = MockSentenceTransformer
    
    # Create a mock module that will be imported
    monkeypatch.setattr("src.embedding.adapters.cpu_adapter.SentenceTransformer", 
                      MockSentenceTransformer)


@pytest.fixture
def cpu_adapter(mock_sentence_transformers):
    """Create a CPU adapter instance for testing."""
    return CPUEmbeddingAdapter(
        model_name="test-model",
        max_length=256,
        batch_size=16
    )


def test_init(cpu_adapter):
    """Test the initialization parameters are stored correctly."""
    assert cpu_adapter.model_name == "test-model"
    assert cpu_adapter.max_length == 256
    assert cpu_adapter.batch_size == 16
    assert cpu_adapter._model is None


def test_model_lazy_loading(cpu_adapter):
    """Test that the model is lazily loaded."""
    # Model shouldn't be loaded yet
    assert cpu_adapter._model is None
    
    # Accessing the model property should trigger loading
    model = cpu_adapter.model
    assert model is not None
    assert isinstance(model, MockSentenceTransformer)
    assert model.model_name == "test-model"
    assert model.device == "cpu"


@pytest.mark.asyncio
async def test_embed_single(cpu_adapter):
    """Test embedding a single text string."""
    # Test embedding a single string
    text = "This is a test sentence."
    embedding = await cpu_adapter.embed_single(text)
    
    # Verify the result is a valid embedding vector
    assert isinstance(embedding, list)
    assert len(embedding) == 384  # Standard size for mini models
    assert all(isinstance(x, float) for x in embedding)


@pytest.mark.asyncio
async def test_embed_batch(cpu_adapter):
    """Test embedding a batch of text strings."""
    texts = [
        "This is the first test sentence.",
        "This is the second test sentence.",
        "This is the third test sentence.",
    ]
    
    embeddings = await cpu_adapter.embed(texts)
    
    # Verify the results
    assert isinstance(embeddings, list)
    assert len(embeddings) == 3
    assert all(isinstance(emb, list) for emb in embeddings)
    assert all(len(emb) == 384 for emb in embeddings)
    
    # Embeddings should be different for different texts
    # Convert to numpy arrays for easy comparison
    emb_arrays = [np.array(emb) for emb in embeddings]
    assert not np.allclose(emb_arrays[0], emb_arrays[1])
    assert not np.allclose(emb_arrays[1], emb_arrays[2])


@pytest.mark.asyncio
async def test_embed_empty_list(cpu_adapter):
    """Test embedding an empty list of texts."""
    embeddings = await cpu_adapter.embed([])
    
    # Should return an empty list
    assert isinstance(embeddings, list)
    assert len(embeddings) == 0


@pytest.mark.asyncio
async def test_embed_with_options(cpu_adapter):
    """Test embedding with additional options."""
    text = "This is a test sentence."
    
    # Test with custom batch size
    options = {"batch_size": 8}
    embedding = await cpu_adapter.embed_single(text, **options)
    
    # Should still produce a valid embedding
    assert isinstance(embedding, list)
    assert len(embedding) == 384


@pytest.mark.asyncio
async def test_embedding_error_handling(monkeypatch, cpu_adapter):
    """Test error handling during embedding."""
    # Create a model that raises an exception
    def mock_encode(*args, **kwargs):
        raise RuntimeError("Test error")
    
    # Patch the encode method to raise an error
    monkeypatch.setattr(cpu_adapter.model, "encode", mock_encode)
    
    # Test error handling in embed method
    with pytest.raises(RuntimeError):
        await cpu_adapter.embed(["Test text"])
    
    # Test error handling in embed_single method
    with pytest.raises(RuntimeError):
        await cpu_adapter.embed_single("Test text")
