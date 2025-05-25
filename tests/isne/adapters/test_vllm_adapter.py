"""
Tests for the VLLMAdapter class.

These tests verify that the VLLMAdapter properly communicates with the vLLM server
and provides embeddings for text content.
"""

import pytest
import unittest.mock as mock
import json
import numpy as np
from typing import Dict, List, Any

from src.model_engine.adapters.vllm_adapter import VLLMAdapter


class TestVLLMAdapter:
    """Test suite for VLLMAdapter."""

    @pytest.fixture
    def mock_requests(self):
        """Provide a mock for the requests module."""
        with mock.patch("src.model_engine.adapters.vllm_adapter.requests") as mock_requests:
            # Mock successful server response for health check
            mock_response = mock.MagicMock()
            mock_response.status_code = 200
            mock_requests.get.return_value = mock_response
            
            # For embeddings endpoint
            mock_embed_response = mock.MagicMock()
            mock_embed_response.status_code = 200
            mock_embed_response.json.return_value = {
                "object": "list",
                "data": [
                    {
                        "object": "embedding",
                        "embedding": [0.1, 0.2, 0.3, 0.4, 0.5] * 200,  # 1000-dim vector
                        "index": 0
                    },
                    {
                        "object": "embedding",
                        "embedding": [0.2, 0.3, 0.4, 0.5, 0.6] * 200,  # 1000-dim vector
                        "index": 1
                    }
                ],
                "model": "BAAI/bge-large-en-v1.5",
                "usage": {
                    "prompt_tokens": 20,
                    "total_tokens": 20
                }
            }
            mock_requests.post.return_value = mock_embed_response
            
            yield mock_requests
    
    def test_initialization(self, mock_requests):
        """Test that the adapter initializes correctly."""
        adapter = VLLMAdapter(
            model_name="BAAI/bge-large-en-v1.5",
            server_url="http://localhost:8000",
            batch_size=32
        )
        
        # Check that the server health was checked
        mock_requests.get.assert_called_once_with(
            "http://localhost:8000/v1/models", 
            timeout=60
        )
        
        assert adapter.model_name == "BAAI/bge-large-en-v1.5"
        assert adapter.server_url == "http://localhost:8000"
        assert adapter.batch_size == 32
    
    def test_get_embeddings(self, mock_requests):
        """Test getting embeddings for text."""
        adapter = VLLMAdapter(
            model_name="BAAI/bge-large-en-v1.5",
            server_url="http://localhost:8000"
        )
        
        texts = ["This is a test", "Another test document"]
        embeddings = adapter.get_embeddings(texts)
        
        # Check request to the server
        mock_requests.post.assert_called_once()
        # Get the actual call arguments
        args, kwargs = mock_requests.post.call_args
        
        # Verify the endpoint
        assert args[0] == "http://localhost:8000/v1/embeddings"
        
        # Verify the request JSON payload
        request_data = json.loads(kwargs["data"])
        assert request_data["model"] == "BAAI/bge-large-en-v1.5"
        assert request_data["input"] == texts
        
        # Verify the response processing
        assert len(embeddings) == 2
        assert len(embeddings[0]) == 1000  # Our mock returns 1000-dim vectors
        assert len(embeddings[1]) == 1000
        
        # Verify normalization (if enabled)
        if adapter.normalize_embeddings:
            # Check vectors are normalized (L2 norm ≈ 1.0)
            norm1 = np.linalg.norm(embeddings[0])
            norm2 = np.linalg.norm(embeddings[1])
            assert 0.99 <= norm1 <= 1.01
            assert 0.99 <= norm2 <= 1.01
    
    @pytest.mark.parametrize("batch_size", [1, 2, 5])
    def test_embedding_batching(self, mock_requests, batch_size):
        """Test that large input is properly batched."""
        adapter = VLLMAdapter(
            model_name="BAAI/bge-large-en-v1.5",
            server_url="http://localhost:8000",
            batch_size=batch_size
        )
        
        # Create more texts than the batch size
        texts = [f"Test document {i}" for i in range(10)]
        embeddings = adapter.get_embeddings(texts)
        
        # Calculate expected number of batches
        expected_batches = (len(texts) + batch_size - 1) // batch_size
        
        # Check the server was called the right number of times
        assert mock_requests.post.call_count == expected_batches
