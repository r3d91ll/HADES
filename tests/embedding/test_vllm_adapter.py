"""Tests for the vLLM embedding adapter.

This module tests the vLLM adapter implementation for embedding generation.
"""

import sys
import os
import pytest
from typing import List, Dict, Any, cast
import asyncio
import json
from unittest.mock import MagicMock, AsyncMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.embedding.adapters.vllm_adapter import VLLMEmbeddingAdapter
from src.types.common import EmbeddingVector


@pytest.fixture
def mock_aiohttp_response():
    """Create a mock aiohttp response."""
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={
        "data": [
            {"embedding": [0.1, 0.2, 0.3]},
            {"embedding": [0.4, 0.5, 0.6]},
        ]
    })
    return mock_response


@pytest.fixture
def mock_session():
    """Create a mock aiohttp ClientSession."""
    mock = AsyncMock()
    mock.__aenter__.return_value = mock
    return mock


@pytest.mark.asyncio
async def test_vllm_adapter_initialization():
    """Test initializing the vLLM adapter."""
    # Test with default parameters
    adapter = VLLMEmbeddingAdapter()
    assert adapter.model_alias == "default"
    assert adapter.base_url is None
    assert adapter.timeout == 60.0
    
    # Test with custom parameters
    adapter = VLLMEmbeddingAdapter(
        model_alias="custom-model",
        base_url="http://localhost:8000",
        timeout=30.0,
        dimension=1536
    )
    assert adapter.model_alias == "custom-model"
    assert adapter.base_url == "http://localhost:8000"
    assert adapter.timeout == 30.0
    assert adapter.additional_params["dimension"] == 1536


@pytest.mark.asyncio
@patch("src.embedding.adapters.vllm_adapter.get_vllm_base_url")
async def test_get_base_url_with_explicit_url(mock_get_base_url):
    """Test getting base URL when explicitly provided."""
    mock_get_base_url.return_value = "http://auto-detected:8000"
    
    adapter = VLLMEmbeddingAdapter(base_url="http://explicit:8000")
    base_url = await adapter._get_base_url()
    
    assert base_url == "http://explicit:8000"
    # Should not call the auto-detection
    mock_get_base_url.assert_not_called()


@pytest.mark.asyncio
@patch("src.embedding.adapters.vllm_adapter.get_vllm_base_url")
async def test_get_base_url_with_auto_detection(mock_get_base_url):
    """Test getting base URL via auto-detection."""
    mock_get_base_url.return_value = "http://auto-detected:8000"
    
    adapter = VLLMEmbeddingAdapter()
    base_url = await adapter._get_base_url()
    
    assert base_url == "http://auto-detected:8000"
    mock_get_base_url.assert_called_once_with("default")


@pytest.mark.asyncio
@patch("src.embedding.adapters.vllm_adapter.get_vllm_base_url")
@patch("aiohttp.ClientSession.post")
async def test_embed_with_successful_response(mock_post, mock_get_base_url, mock_aiohttp_response):
    """Test embedding with a successful API response."""
    # Configure mocks
    mock_get_base_url.return_value = "http://localhost:8000"
    mock_post.return_value.__aenter__.return_value = mock_aiohttp_response
    
    # Initialize adapter and call embed
    adapter = VLLMEmbeddingAdapter()
    embeddings = await adapter.embed(["text1", "text2"])
    
    # Check results
    assert len(embeddings) == 2
    assert embeddings[0] == [0.1, 0.2, 0.3]
    assert embeddings[1] == [0.4, 0.5, 0.6]
    
    # Check API call
    mock_post.assert_called_once()
    args, kwargs = mock_post.call_args
    assert args[0] == "http://localhost:8000/v1/embeddings"
    assert kwargs["json"]["input"] == ["text1", "text2"]
    assert kwargs["json"]["model"] == "default"


@pytest.mark.asyncio
@patch("src.embedding.adapters.vllm_adapter.get_vllm_base_url")
@patch("aiohttp.ClientSession.post")
async def test_embed_with_empty_input(mock_post, mock_get_base_url):
    """Test embedding with empty input."""
    mock_get_base_url.return_value = "http://localhost:8000"
    
    adapter = VLLMEmbeddingAdapter()
    embeddings = await adapter.embed([])
    
    assert embeddings == []
    mock_post.assert_not_called()


@pytest.mark.asyncio
@patch("src.embedding.adapters.vllm_adapter.get_vllm_base_url")
@patch("aiohttp.ClientSession.post")
async def test_embed_with_api_error(mock_post, mock_get_base_url):
    """Test embedding with API error response."""
    # Configure mocks
    mock_get_base_url.return_value = "http://localhost:8000"
    
    error_response = AsyncMock()
    error_response.status = 400
    error_response.text = AsyncMock(return_value="Bad request")
    
    mock_post.return_value.__aenter__.return_value = error_response
    
    # Initialize adapter and test error handling
    adapter = VLLMEmbeddingAdapter()
    
    with pytest.raises(RuntimeError) as excinfo:
        await adapter.embed(["text1"])
    
    assert "vLLM API request failed with status 400" in str(excinfo.value)


@pytest.mark.asyncio
@patch("src.embedding.adapters.vllm_adapter.get_vllm_base_url")
@patch("aiohttp.ClientSession.post")
async def test_embed_with_timeout(mock_post, mock_get_base_url):
    """Test embedding with request timeout."""
    # Configure mocks
    mock_get_base_url.return_value = "http://localhost:8000"
    mock_post.side_effect = asyncio.TimeoutError()
    
    # Initialize adapter and test timeout handling
    adapter = VLLMEmbeddingAdapter(timeout=2.0)
    
    with pytest.raises(RuntimeError) as excinfo:
        await adapter.embed(["text1"])
    
    assert "vLLM API request timed out after 2.0 seconds" in str(excinfo.value)


@pytest.mark.asyncio
@patch("src.embedding.adapters.vllm_adapter.VLLMEmbeddingAdapter.embed")
async def test_embed_single(mock_embed):
    """Test embed_single method properly calls embed with a single text."""
    # Configure mock
    mock_embed.return_value = [[0.1, 0.2, 0.3]]
    
    # Call the method
    adapter = VLLMEmbeddingAdapter()
    embedding = await adapter.embed_single("test text")
    
    # Check results
    assert embedding == [0.1, 0.2, 0.3]
    mock_embed.assert_called_once_with(["test text"])


@pytest.mark.asyncio
@patch("src.embedding.adapters.vllm_adapter.VLLMEmbeddingAdapter.embed")
async def test_embed_single_with_empty_result(mock_embed):
    """Test embed_single method handling empty result from embed."""
    # Configure mock
    mock_embed.return_value = []
    
    # Call the method and check error handling
    adapter = VLLMEmbeddingAdapter()
    
    with pytest.raises(RuntimeError) as excinfo:
        await adapter.embed_single("test text")
    
    assert "vLLM API returned empty results" in str(excinfo.value)


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
