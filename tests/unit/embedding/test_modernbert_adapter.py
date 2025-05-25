"""
Unit tests for the ModernBERT embedding adapter.

This module tests the ModernBERT adapter functionality with mocked Haystack server.
"""

import unittest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from src.embedding.adapters.modernbert_adapter import ModernBERTEmbeddingAdapter


@pytest.mark.asyncio
class TestModernBERTAdapter:
    """Test the ModernBERT embedding adapter."""
    
    def setup_method(self):
        """Set up the test environment."""
        # Create client mock
        self.client_mock = MagicMock()
        self.client_mock.request = AsyncMock()
        
        # Create engine mock
        self.engine_mock = MagicMock()
        self.engine_mock.get_status.return_value = {"running": True}
        self.engine_mock.client = self.client_mock
        
        # Create patcher for HaystackModelEngine
        self.engine_patcher = patch(
            'src.embedding.adapters.modernbert_adapter.HaystackModelEngine',
            return_value=self.engine_mock
        )
        self.engine_mock_instance = self.engine_patcher.start()
        
        # Create the adapter instance
        self.adapter = ModernBERTEmbeddingAdapter(
            model_name="answerdotai/ModernBERT-base",
            pooling_strategy="cls",
            normalize_embeddings=True
        )
        
        # Explicitly set the engine to our mock to bypass property logic
        self.adapter._engine = self.engine_mock
        
        # Set model as loaded to avoid additional calls
        self.adapter._model_loaded = True
    
    def teardown_method(self):
        """Clean up after the test."""
        self.engine_patcher.stop()
    
    def test_initialization(self):
        """Test adapter initialization with default and custom parameters."""
        # Test default initialization
        adapter_default = ModernBERTEmbeddingAdapter()
        assert adapter_default.model_name == "answerdotai/ModernBERT-base"
        assert adapter_default.max_length == 8192
        assert adapter_default.pooling_strategy == "cls"
        assert adapter_default.normalize_embeddings is True
        
        # Test custom initialization
        adapter_custom = ModernBERTEmbeddingAdapter(
            model_name="custom/model",
            max_length=4096,
            pooling_strategy="mean",
            normalize_embeddings=False
        )
        assert adapter_custom.model_name == "custom/model"
        assert adapter_custom.max_length == 4096
        assert adapter_custom.pooling_strategy == "mean"
        assert adapter_custom.normalize_embeddings is False
    
    async def test_ensure_model_loaded(self):
        """Test model loading through Haystack engine."""
        # Reset model loaded flag
        self.adapter._model_loaded = False
        
        # Call method
        await self.adapter._ensure_model_loaded()
        
        # Check that load_model was called with correct model name
        self.engine_mock.load_model.assert_called_once_with(self.adapter.model_name)
        assert self.adapter._model_loaded is True
    
    async def test_get_embeddings_from_model(self):
        """Test embedding generation with mock Haystack server."""
        # Mock response from Haystack server
        mock_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_response = {
            "result": {
                "embeddings": mock_embeddings,
                "dimensions": 3,
                "model_id": self.adapter.model_name
            }
        }
        # Set up the mock to return our response
        self.client_mock.request.reset_mock()
        self.client_mock.request.return_value = mock_response
        
        # Test with sample texts
        texts = ["This is a test", "Another test"]
        result = await self.adapter._get_embeddings_from_model(texts)
        
        # Verify result
        assert result == mock_embeddings
        
        # Check request parameters
        self.client_mock.request.assert_called_once()
        call_args = self.client_mock.request.call_args[0][0]
        assert call_args["action"] == "embed"
        assert call_args["model_id"] == self.adapter.model_name
        assert call_args["texts"] == texts
        assert call_args["pooling"] == self.adapter.pooling_strategy
        assert call_args["normalize"] == self.adapter.normalize_embeddings
    
    async def test_get_embeddings_error_handling(self):
        """Test error handling in embedding generation."""
        # Mock error response
        self.client_mock.request.reset_mock()
        self.client_mock.request.return_value = {"error": "Test error"}
        
        # Test error handling
        with pytest.raises(RuntimeError) as excinfo:
            await self.adapter._get_embeddings_from_model(["Test"])
        
        assert "Haystack server error: Test error" in str(excinfo.value)
    
    async def test_embed_single(self):
        """Test embedding a single text."""
        # Mock the embed method
        mock_embeddings = [[0.1, 0.2, 0.3]]
        self.adapter.embed = AsyncMock(return_value=mock_embeddings)
        
        # Test with a single text
        result = await self.adapter.embed_single("Test text")
        
        # Verify result
        assert result == mock_embeddings[0]
        self.adapter.embed.assert_called_once_with(["Test text"])
    
    async def test_empty_text_handling(self):
        """Test handling of empty text lists."""
        # Reset mock to track calls
        self.client_mock.request.reset_mock()
        
        # Test with empty text list
        result = await self.adapter.embed([])
        
        # Should return empty list without calling server
        assert result == []
        assert not self.client_mock.request.called


if __name__ == "__main__":
    unittest.main()
