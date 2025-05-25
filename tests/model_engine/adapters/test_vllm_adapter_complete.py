"""
Comprehensive tests for the VLLMAdapter's completion and chat capabilities.

This test suite focuses on the completion and chat functionality of the VLLMAdapter,
which was not covered in the basic adapter tests.
"""

import pytest
import unittest.mock as mock
import json
import numpy as np
import asyncio
import requests
from typing import Dict, List, Any

from src.model_engine.adapters.vllm_adapter import VLLMAdapter


class TestVLLMAdapterComplete:
    """Test suite for VLLMAdapter completion and chat functionality."""

    @pytest.fixture
    def mock_requests(self):
        """Provide a mock for the requests module."""
        with mock.patch("src.model_engine.adapters.vllm_adapter.requests") as mock_requests:
            # Mock successful server response for health check
            mock_response = mock.MagicMock()
            mock_response.status_code = 200
            mock_requests.get.return_value = mock_response
            yield mock_requests
    
    @pytest.fixture
    def mock_aiohttp_session(self):
        """Mock aiohttp.ClientSession for async tests."""
        with mock.patch("src.model_engine.adapters.vllm_adapter.aiohttp.ClientSession") as mock_session:
            # Create mock for the session
            mock_client_session = mock.MagicMock()
            mock_session.return_value.__aenter__.return_value = mock_client_session
            
            # Create mock for the response
            mock_response = mock.MagicMock()
            mock_response.status = 200
            mock_response.__aenter__.return_value = mock_response
            
            # Set up completion response
            mock_completion_data = {
                "id": "cmpl-123",
                "object": "text_completion",
                "created": 1677858242,
                "model": "test-model",
                "choices": [
                    {
                        "text": "This is a test completion.",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 7,
                    "total_tokens": 12
                }
            }
            mock_response.json.return_value = mock_completion_data
            
            # Make the session post method return the mock response
            mock_client_session.post.return_value = mock_response
            
            yield mock_client_session
    
    @pytest.fixture
    def adapter(self, mock_requests):
        """Create a VLLMAdapter instance with mocked server."""
        adapter = VLLMAdapter(
            model_name="test-model",
            server_url="http://localhost:8000",
            normalize_embeddings=True
        )
        return adapter
    
    @pytest.mark.asyncio
    async def test_complete_async(self, adapter, mock_aiohttp_session):
        """Test the asynchronous completion method."""
        prompt = "This is a test prompt."
        
        # Setup mock response
        mock_response = mock.MagicMock()
        mock_response.status = 200
        mock_response.__aenter__.return_value = mock_response
        mock_response.json = mock.AsyncMock()
        mock_response.json.return_value = {
            "choices": [{"text": "This is a test completion."}]
        }
        mock_aiohttp_session.post.return_value = mock_response
        
        # Call the method
        completion = await adapter.complete_async(prompt)
        
        # Check that the correct endpoint was called
        mock_aiohttp_session.post.assert_called_once()
        args, kwargs = mock_aiohttp_session.post.call_args
        
        # Verify the endpoint
        assert args[0] == "http://localhost:8000/v1/completions"
        
        # Verify the request payload
        assert kwargs["json"]["model"] == "test-model"
        assert kwargs["json"]["prompt"] == prompt
        
        # Verify the response processing
        assert completion == "This is a test completion."
    
    def test_complete(self, adapter, mock_aiohttp_session):
        """Test the synchronous completion method."""
        prompt = "This is a test prompt."
        
        # We need to patch asyncio.run since we're calling it in a synchronous context
        with mock.patch("src.model_engine.adapters.vllm_adapter.asyncio.run") as mock_run:
            mock_run.return_value = "This is a test completion."
            completion = adapter.complete(prompt)
        
        # Verify that asyncio.run was called
        mock_run.assert_called_once()
        
        # Verify the result
        assert completion == "This is a test completion."
    
    @pytest.mark.asyncio
    async def test_chat_complete_async(self, adapter, mock_aiohttp_session):
        """Test the asynchronous chat completion method."""
        # Set up chat completion mock response
        chat_response = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677858242,
            "model": "test-model",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "This is a test chat response."
                    },
                    "index": 0,
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 15,
                "total_tokens": 25
            }
        }
        
        # Setup mock response
        mock_response = mock.MagicMock()
        mock_response.status = 200
        mock_response.__aenter__.return_value = mock_response
        mock_response.json = mock.AsyncMock()
        mock_response.json.return_value = chat_response
        mock_aiohttp_session.post.return_value = mock_response
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"}
        ]
        
        chat_result = await adapter.chat_complete_async(messages)
        
        # Check that the correct endpoint was called
        mock_aiohttp_session.post.assert_called_once()
        args, kwargs = mock_aiohttp_session.post.call_args
        
        # Verify the endpoint
        assert args[0] == "http://localhost:8000/v1/chat/completions"
        
        # Verify the request payload
        assert kwargs["json"]["model"] == "test-model"
        assert kwargs["json"]["messages"] == messages
        
        # Verify the response
        assert chat_result == chat_response
    
    def test_chat_complete(self, adapter, mock_aiohttp_session):
        """Test the synchronous chat completion method."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"}
        ]
        
        # We need to patch asyncio.run since we're calling it in a synchronous context
        with mock.patch("src.model_engine.adapters.vllm_adapter.asyncio.run") as mock_run:
            mock_chat_response = {
                "id": "chatcmpl-123",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "This is a test chat response."
                        }
                    }
                ]
            }
            mock_run.return_value = mock_chat_response
            chat_result = adapter.chat_complete(messages)
        
        # Verify that asyncio.run was called
        mock_run.assert_called_once()
        
        # Verify the result
        assert chat_result == mock_chat_response
    
    @pytest.mark.asyncio
    async def test_complete_async_error_handling(self, adapter, mock_aiohttp_session):
        """Test error handling in async completion."""
        # Set up error response
        mock_error_response = mock.MagicMock()
        mock_error_response.status = 400
        mock_error_response.__aenter__.return_value = mock_error_response
        mock_error_response.text = mock.AsyncMock()
        mock_error_response.text.return_value = '{"error": "Invalid request"}'
        
        mock_aiohttp_session.post.return_value = mock_error_response
        
        # Test error handling
        with pytest.raises(RuntimeError):
            await adapter.complete_async("This will fail")
    
    @pytest.mark.asyncio
    async def test_chat_complete_async_error_handling(self, adapter, mock_aiohttp_session):
        """Test error handling in async chat completion."""
        # Set up error response
        mock_error_response = mock.MagicMock()
        mock_error_response.status = 400
        mock_error_response.__aenter__.return_value = mock_error_response
        mock_error_response.text = mock.AsyncMock()
        mock_error_response.text.return_value = '{"error": "Invalid chat request"}'
        
        mock_aiohttp_session.post.return_value = mock_error_response
        
        # Test error handling
        with pytest.raises(RuntimeError):
            messages = [{"role": "user", "content": "This will fail"}]
            await adapter.chat_complete_async(messages)
    
    def test_adapter_custom_parameters(self, mock_requests):
        """Test that adapter accepts and uses custom parameters."""
        # Create adapter with non-default parameters
        adapter = VLLMAdapter(
            model_name="custom-model",
            server_url="http://custom-server:9000",
            batch_size=16,
            device="cpu",
            normalize_embeddings=False,
            max_retries=5,
            timeout=30,
            use_openai_api=False
        )
        
        # Verify parameters were set correctly
        assert adapter.model_name == "custom-model"
        assert adapter.server_url == "http://custom-server:9000"
        assert adapter.batch_size == 16
        assert adapter.device == "cpu"
        assert adapter.normalize_embeddings is False
        assert adapter.max_retries == 5
        assert adapter.timeout == 30
        assert adapter.use_openai_api is False
        
        # Verify non-OpenAI endpoints are configured
        assert adapter.embedding_endpoint == "http://custom-server:9000/v1/embeddings"
        assert adapter.completion_endpoint == "http://custom-server:9000/generate"
        assert adapter.chat_endpoint == "http://custom-server:9000/generate"
        assert adapter.health_endpoint == "http://custom-server:9000/health"
    
    def test_server_url_normalization(self, mock_requests):
        """Test that trailing slashes are removed from server URLs."""
        adapter = VLLMAdapter(
            model_name="test-model",
            server_url="http://localhost:8000/"  # Note trailing slash
        )
        
        # Verify trailing slash was removed
        assert adapter.server_url == "http://localhost:8000"
        assert adapter.embedding_endpoint == "http://localhost:8000/v1/embeddings"
    
    def test_server_unavailable(self):
        """Test behavior when server is not available."""
        # Mock requests.get to raise an exception
        with mock.patch("src.model_engine.adapters.vllm_adapter.requests.get") as mock_get:
            mock_get.side_effect = requests.exceptions.ConnectionError("Connection refused")
            
            adapter = VLLMAdapter(
                model_name="test-model",
                server_url="http://localhost:8000"
            )
            
            # Check that is_available returns False
            assert adapter.is_available is False
            # It's called once during initialization and once during is_available check
            assert mock_get.call_count == 2
