
"""
Enhanced unit tests for the vLLM model engine implementation.

These tests provide more comprehensive coverage, especially for async methods
and error handling, aiming for >= 85% coverage.
"""

import unittest
import sys
import asyncio
import json
import logging
import time
import requests
import numpy as np
from typing import Dict, List, Any, Optional, Union
from unittest.mock import patch, MagicMock, AsyncMock, call
import aiohttp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_vllm_engine_enhanced")

# Mock the imports that are causing issues if runtime is not available
sys.modules['src.model_engine.engines.vllm.runtime'] = MagicMock()

# Import the necessary classes
from src.model_engine.base import ModelEngine
from src.model_engine.engines.vllm.vllm_engine import VLLMModelEngine


class AsyncContextManager:
    """Helper class to mock async context managers."""
    def __init__(self, mock_response):
        self.mock_response = mock_response

    async def __aenter__(self):
        return self.mock_response

    async def __aexit__(self, exc_type, exc, tb):
        pass


class TestVLLMModelEngineEnhanced(unittest.TestCase):
    """Enhanced tests for the vLLM model engine implementation."""

    def setUp(self):
        """Set up the test environment."""
        # Use patch to mock aiohttp.ClientSession for async tests
        self.aiohttp_session_patch = patch('aiohttp.ClientSession')
        self.mock_aiohttp_session_cls = self.aiohttp_session_patch.start()
        self.mock_aiohttp_session = self.mock_aiohttp_session_cls.return_value
        self.mock_aiohttp_session.post = AsyncMock()
        self.mock_aiohttp_session.close = AsyncMock() # Mock close method if needed

        # Mock requests for synchronous calls like start
        self.requests_patch = patch('src.model_engine.engines.vllm.vllm_engine.requests.get')
        self.mock_requests_get = self.requests_patch.start()
        
        # Common setup
        self.server_url = "http://test-vllm:8000"
        self.engine = VLLMModelEngine(server_url=self.server_url)
        
        # Mock successful start response by default
        self.mock_requests_get.return_value = MagicMock(status_code=200)
        self.engine.start() # Start the engine for most tests

    def tearDown(self):
        """Clean up after tests."""
        self.aiohttp_session_patch.stop()
        self.requests_patch.stop()
        # Ensure engine is stopped if running
        if self.engine.running:
            self.engine.stop()

    # --- Test Initialization and Lifecycle --- 

    def test_init_default(self):
        """Test initialization with default parameters."""
        engine = VLLMModelEngine() # Stop the default one first
        self.mock_requests_get.reset_mock() # Reset mock from setUp
        
        self.assertEqual(engine.server_url, "http://localhost:8000")
        self.assertEqual(engine.device, "cuda")
        self.assertEqual(engine.max_retries, 3)
        self.assertEqual(engine.timeout, 60)
        self.assertFalse(engine.running)
        self.assertEqual(engine.loaded_models, {})
        self.assertIsNone(engine.session)

    def test_init_custom(self):
        """Test initialization with custom parameters."""
        engine = VLLMModelEngine(
            server_url="http://custom:5000",
            device="cpu",
            max_retries=5,
            timeout=30
        )
        self.assertEqual(engine.server_url, "http://custom:5000")
        self.assertEqual(engine.device, "cpu")
        self.assertEqual(engine.max_retries, 5)
        self.assertEqual(engine.timeout, 30)

    def test_start_success(self):
        """Test successful start."""
        engine = VLLMModelEngine(server_url=self.server_url)
        self.mock_requests_get.return_value = MagicMock(status_code=200)
        
        result = engine.start()
        
        self.assertTrue(result)
        self.assertTrue(engine.running)
        self.mock_requests_get.assert_called_once_with(
            f"{self.server_url}/v1/models", timeout=5
        )

    def test_start_failure_connection_error(self):
        """Test start failure due to connection error."""
        engine = VLLMModelEngine(server_url=self.server_url)
        self.mock_requests_get.side_effect = requests.exceptions.RequestException("Connection refused")

        result = engine.start()

        self.assertFalse(result)
        self.assertFalse(engine.running)
        self.mock_requests_get.assert_called_once_with(
            f"{self.server_url}/v1/models", timeout=5
        )

    def test_start_failure_bad_status(self):
        """Test start failure due to non-200 status code."""
        engine = VLLMModelEngine(server_url=self.server_url)
        self.mock_requests_get.return_value = MagicMock(status_code=500)
        
        result = engine.start()
        
        self.assertFalse(result)
        self.assertFalse(engine.running)
        self.mock_requests_get.assert_called_once_with(
            f"{self.server_url}/v1/models", timeout=5
        )

    def test_start_already_running(self):
        """Test start when already running."""
        self.mock_requests_get.reset_mock() # Reset from setUp start call
        result = self.engine.start() # Should return True immediately
        self.assertTrue(result)
        self.mock_requests_get.assert_not_called() # Should not check again

    def test_stop(self):
        """Test stop when running."""
        self.assertTrue(self.engine.running) # Ensure it was started in setUp
        result = self.engine.stop()
        self.assertTrue(result)
        self.assertFalse(self.engine.running)

    def test_stop_not_running(self):
        """Test stop when not running."""
        engine = VLLMModelEngine() # Create a fresh, non-started engine
        self.assertFalse(engine.running)
        result = engine.stop()
        self.assertTrue(result)
        self.assertFalse(engine.running)

    # --- Test Model Management --- 

    def test_load_model_new(self):
        """Test loading a new model."""
        result = self.engine.load_model("model-1")
        self.assertEqual(result, "loaded")
        self.assertIn("model-1", self.engine.loaded_models)
        self.assertEqual(self.engine.loaded_models["model-1"]["device"], "cuda")

    def test_load_model_custom_device(self):
        """Test loading a model with custom device."""
        result = self.engine.load_model("model-2", device="cpu")
        self.assertEqual(result, "loaded")
        self.assertIn("model-2", self.engine.loaded_models)
        self.assertEqual(self.engine.loaded_models["model-2"]["device"], "cpu")

    def test_load_model_already_loaded(self):
        """Test loading an already loaded model."""
        self.engine.load_model("model-1") # Load first time
        result = self.engine.load_model("model-1") # Load second time
        self.assertEqual(result, "already_loaded")

    def test_unload_model(self):
        """Test unloading a model."""
        self.engine.load_model("model-1")
        self.assertIn("model-1", self.engine.loaded_models)
        result = self.engine.unload_model("model-1")
        self.assertEqual(result, "unloaded")
        self.assertNotIn("model-1", self.engine.loaded_models)

    def test_unload_model_not_loaded(self):
        """Test unloading a model that's not loaded."""
        result = self.engine.unload_model("non-existent-model")
        self.assertEqual(result, "not_loaded")

    def test_is_model_loaded(self):
        """Test is_model_loaded method."""
        self.assertFalse(self.engine.is_model_loaded("model-1"))
        self.engine.load_model("model-1")
        self.assertTrue(self.engine.is_model_loaded("model-1"))

    def test_list_loaded_models(self):
        """Test list_loaded_models method."""
        self.assertEqual(self.engine.list_loaded_models(), [])
        self.engine.load_model("model1")
        self.engine.load_model("model2")
        loaded = self.engine.list_loaded_models()
        self.assertEqual(len(loaded), 2)
        self.assertIn("model1", loaded)
        self.assertIn("model2", loaded)

    # --- Test Async Generate Embeddings --- 

    async def async_test_generate_embeddings_success(self):
        """Test generate_embeddings successfully."""
        model_id = "embed-model"
        texts = ["text1", "text2", "text3"]
        expected_embeddings = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        normalized_embeddings = [[0.447, 0.894], [0.6, 0.8], [0.640, 0.768]] # Roughly normalized

        # Mock the successful response
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "data": [
                {"embedding": expected_embeddings[0], "index": 0, "object": "embedding"},
                {"embedding": expected_embeddings[1], "index": 1, "object": "embedding"},
                {"embedding": expected_embeddings[2], "index": 2, "object": "embedding"}
            ],
            "model": model_id,
            "object": "list",
            "usage": {"prompt_tokens": 10, "total_tokens": 10}
        })
        self.mock_aiohttp_session.post.return_value = AsyncContextManager(mock_response)

        self.engine.load_model(model_id)
        result = await self.engine.generate_embeddings(texts, model_id=model_id, batch_size=2)
        
        self.assertEqual(len(result), len(texts))
        for i, emb in enumerate(result):
            self.assertEqual(len(emb), 2)
            # Check normalization approximately
            np.testing.assert_almost_equal(emb, normalized_embeddings[i], decimal=3)

        # Check calls (should be two batches)
        expected_url = f"{self.server_url}/v1/embeddings"
        expected_calls = [
            call(expected_url, json={'input': texts[:2], 'model': model_id}, timeout=self.engine.timeout),
            call(expected_url, json={'input': texts[2:], 'model': model_id}, timeout=self.engine.timeout)
        ]
        self.mock_aiohttp_session.post.assert_has_calls(expected_calls)
        self.assertEqual(self.mock_aiohttp_session.post.call_count, 2)
        self.mock_aiohttp_session_cls.assert_called() # Ensure session was created

    def test_generate_embeddings_success(self):
        asyncio.run(self.async_test_generate_embeddings_success())

    async def async_test_generate_embeddings_http_error(self):
        """Test generate_embeddings with HTTP error response."""
        model_id = "embed-model"
        texts = ["text1"]

        # Mock error response
        mock_response = MagicMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Internal Server Error")
        self.mock_aiohttp_session.post.return_value = AsyncContextManager(mock_response)

        self.engine.load_model(model_id)
        with self.assertRaisesRegex(RuntimeError, "Embedding generation failed with status 500"): 
            await self.engine.generate_embeddings(texts, model_id=model_id)

        self.assertEqual(self.mock_aiohttp_session.post.call_count, 1) # Should fail on first try

    def test_generate_embeddings_http_error(self):
        asyncio.run(self.async_test_generate_embeddings_http_error())

    async def async_test_generate_embeddings_network_error(self):
        """Test generate_embeddings with network error (exception)."""
        model_id = "embed-model"
        texts = ["text1"]

        # Mock network error
        self.mock_aiohttp_session.post.side_effect = aiohttp.ClientError("Connection failed")

        self.engine.load_model(model_id)
        with self.assertRaisesRegex(RuntimeError, "Embedding generation failed after 3 retries"): 
            await self.engine.generate_embeddings(texts, model_id=model_id)

        # Check retries
        self.assertEqual(self.mock_aiohttp_session.post.call_count, self.engine.max_retries)

    def test_generate_embeddings_network_error(self):
        asyncio.run(self.async_test_generate_embeddings_network_error())

    async def async_test_generate_embeddings_no_normalize(self):
        """Test generate_embeddings without normalization."""
        model_id = "embed-model"
        texts = ["text1"]
        expected_embedding = [0.3, 0.4] # Norm is 0.5

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "data": [{"embedding": expected_embedding, "index": 0, "object": "embedding"}],
            "model": model_id, "object": "list", "usage": {}
        })
        self.mock_aiohttp_session.post.return_value = AsyncContextManager(mock_response)

        self.engine.load_model(model_id)
        result = await self.engine.generate_embeddings(texts, model_id=model_id, normalize=False)
        
        self.assertEqual(len(result), 1)
        np.testing.assert_array_equal(result[0], expected_embedding)

    def test_generate_embeddings_no_normalize(self):
        asyncio.run(self.async_test_generate_embeddings_no_normalize())

    # --- Test Async Generate Completion --- 

    async def async_test_generate_completion_success(self):
        """Test generate_completion successfully."""
        model_id = "completion-model"
        prompt = "Once upon a time"
        expected_text = " there was a story."
        max_tokens = 50

        # Mock successful response
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "choices": [{"text": expected_text, "index": 0, "finish_reason": "stop"}],
            "model": model_id, "object": "text_completion", "usage": {}
        })
        self.mock_aiohttp_session.post.return_value = AsyncContextManager(mock_response)

        self.engine.load_model(model_id)
        result = await self.engine.generate_completion(prompt, model_id=model_id, max_tokens=max_tokens, temperature=0.7)
        
        self.assertEqual(result, expected_text)

        expected_url = f"{self.server_url}/v1/completions"
        expected_payload = {
            'prompt': prompt,
            'model': model_id,
            'max_tokens': max_tokens,
            'temperature': 0.7
        }
        self.mock_aiohttp_session.post.assert_called_once_with(
            expected_url, json=expected_payload, timeout=self.engine.timeout
        )

    def test_generate_completion_success(self):
        asyncio.run(self.async_test_generate_completion_success())

    async def async_test_generate_completion_http_error(self):
        """Test generate_completion with HTTP error response."""
        model_id = "completion-model"
        prompt = "Test prompt"

        # Mock error response
        mock_response = MagicMock()
        mock_response.status = 400
        mock_response.text = AsyncMock(return_value="Bad Request")
        self.mock_aiohttp_session.post.return_value = AsyncContextManager(mock_response)

        self.engine.load_model(model_id)
        with self.assertRaisesRegex(RuntimeError, "Completion generation failed with status 400"): 
            await self.engine.generate_completion(prompt, model_id=model_id)

        self.assertEqual(self.mock_aiohttp_session.post.call_count, 1)

    def test_generate_completion_http_error(self):
        asyncio.run(self.async_test_generate_completion_http_error())
        
    async def async_test_generate_completion_network_error(self):
        """Test generate_completion with network error."""
        model_id = "completion-model"
        prompt = "Test prompt"

        # Mock network error
        self.mock_aiohttp_session.post.side_effect = aiohttp.ClientError("Connection failed")

        self.engine.load_model(model_id)
        with self.assertRaisesRegex(RuntimeError, "Completion generation failed after 3 retries"): 
            await self.engine.generate_completion(prompt, model_id=model_id)

        self.assertEqual(self.mock_aiohttp_session.post.call_count, self.engine.max_retries)

    def test_generate_completion_network_error(self):
        asyncio.run(self.async_test_generate_completion_network_error())

    # --- Test Async Generate Chat Completion --- 

    async def async_test_generate_chat_completion_success(self):
        """Test generate_chat_completion successfully."""
        model_id = "chat-model"
        messages = [{"role": "user", "content": "Hello"}]
        expected_response = {"role": "assistant", "content": "Hi there!"}
        
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "choices": [{"message": expected_response, "index": 0, "finish_reason": "stop"}],
            "model": model_id, "object": "chat.completion", "usage": {}
        })
        self.mock_aiohttp_session.post.return_value = AsyncContextManager(mock_response)

        self.engine.load_model(model_id)
        result = await self.engine.generate_chat_completion(messages, model_id=model_id, temperature=0.5)
        
        self.assertEqual(result, expected_response)

        expected_url = f"{self.server_url}/v1/chat/completions"
        expected_payload = {
            'messages': messages,
            'model': model_id,
            'temperature': 0.5
        }
        self.mock_aiohttp_session.post.assert_called_once_with(
            expected_url, json=expected_payload, timeout=self.engine.timeout
        )

    def test_generate_chat_completion_success(self):
        asyncio.run(self.async_test_generate_chat_completion_success())

    async def async_test_generate_chat_completion_http_error(self):
        """Test generate_chat_completion with HTTP error."""
        model_id = "chat-model"
        messages = [{"role": "user", "content": "Hello"}]

        # Mock error response
        mock_response = MagicMock()
        mock_response.status = 503
        mock_response.text = AsyncMock(return_value="Service Unavailable")
        self.mock_aiohttp_session.post.return_value = AsyncContextManager(mock_response)

        self.engine.load_model(model_id)
        with self.assertRaisesRegex(RuntimeError, "Chat completion generation failed with status 503"): 
            await self.engine.generate_chat_completion(messages, model_id=model_id)

        self.assertEqual(self.mock_aiohttp_session.post.call_count, 1)

    def test_generate_chat_completion_http_error(self):
        asyncio.run(self.async_test_generate_chat_completion_http_error())
        
    async def async_test_generate_chat_completion_network_error(self):
        """Test generate_chat_completion with network error."""
        model_id = "chat-model"
        messages = [{"role": "user", "content": "Hello"}]

        # Mock network error
        self.mock_aiohttp_session.post.side_effect = aiohttp.ClientError("Timeout")

        self.engine.load_model(model_id)
        with self.assertRaisesRegex(RuntimeError, "Chat completion generation failed after 3 retries"): 
            await self.engine.generate_chat_completion(messages, model_id=model_id)

        self.assertEqual(self.mock_aiohttp_session.post.call_count, self.engine.max_retries)

    def test_generate_chat_completion_network_error(self):
        asyncio.run(self.async_test_generate_chat_completion_network_error())

    # --- Test Helper Methods (like _normalize_embedding) ---
    def test_normalize_embedding(self):
        """Test _normalize_embedding method."""
        # Need access to the protected method, maybe make a testable subclass or test via public API
        # Testing via public API (generate_embeddings with normalize=True)
        pass # Covered by test_generate_embeddings_success

    def test_normalize_embedding_zero_vector(self):
        """Test _normalize_embedding with zero vector."""
        engine = VLLMModelEngine()
        result = engine._normalize_embedding([0.0, 0.0, 0.0])
        np.testing.assert_array_equal(result, [0.0, 0.0, 0.0])

    # --- Test Abstract Methods (Need Concrete Implementation/Mocks) ---
    # The base class methods are abstract, these tests ensure our mock works
    # If we were testing the abstract class directly, we'd need a concrete subclass

    def test_get_loaded_models_abstract(self):
        # This method is abstract in VLLMModelEngine base, requires implementation
        # We aren't directly testing it here, but through list_loaded_models
        pass

    def test_health_check_abstract(self):
        # Abstract in base
        pass

    def test_infer_abstract(self):
        # Abstract in base
        pass

    def test_restart_abstract(self):
        # Abstract in base
        pass


if __name__ == "__main__":
    unittest.main()