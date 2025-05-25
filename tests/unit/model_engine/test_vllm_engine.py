"""
Unit tests for the vLLM model engine implementation.

These tests focus on the public API of the VLLMModelEngine class
to achieve the required 85% test coverage, following the project's standard 
testing protocol.
"""

import unittest
import asyncio
import json
import logging
import time
from typing import Dict, List, Any, Optional, Union
from unittest.mock import patch, MagicMock, AsyncMock, call

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_vllm_engine")

# Import the model engine
from src.model_engine.engines.vllm import VLLMModelEngine


class MockResponse:
    """Mock HTTP response for testing."""
    
    def __init__(self, status=200, json_data=None, text_data=None, raise_error=False):
        self.status = status
        self._json_data = json_data or {}
        self._text_data = text_data or ""
        self._raise = raise_error
    
    async def json(self):
        """Return mock JSON data."""
        if self._raise:
            raise ValueError("Mock JSON error")
        return self._json_data
    
    async def text(self):
        """Return mock text data."""
        if self._raise:
            raise ValueError("Mock text error")
        return self._text_data
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        return False
    
    async def __aenter__(self):
        """Async context manager enter."""
        return self


class MockClientSession:
    """Mock aiohttp ClientSession for testing."""
    
    def __init__(self, responses=None):
        self.responses = responses or []
        self.calls = []
    
    async def __aenter__(self):
        """Async context manager enter."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        return False
    
    def set_responses(self, responses):
        """Set the mock responses."""
        self.responses = responses
    
    async def post(self, url, json=None, timeout=None):
        """Mock post request."""
        self.calls.append(("post", url, json, timeout))
        
        if not self.responses:
            return MockResponse(status=200, json_data={"choices": [{"text": "mock response"}]})
        
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class TestVLLMModelEngine(unittest.TestCase):
    """Test the vLLM model engine implementation."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create patches
        self.requests_patcher = patch('src.model_engine.engines.vllm.vllm_engine.requests')
        self.aiohttp_patcher = patch('src.model_engine.engines.vllm.vllm_engine.aiohttp')
        
        # Start patches
        self.mock_requests = self.requests_patcher.start()
        self.mock_aiohttp = self.aiohttp_patcher.start()
        
        # Configure mock client session
        self.mock_client_session = MockClientSession()
        self.mock_aiohttp.ClientSession.return_value = self.mock_client_session
    
    def tearDown(self):
        """Clean up after tests."""
        # Stop patches
        self.requests_patcher.stop()
        self.aiohttp_patcher.stop()
    
    def test_init(self):
        """Test engine initialization."""
        engine = VLLMModelEngine()
        self.assertIsNotNone(engine)
        self.assertEqual(engine.server_url, "http://localhost:8000")
        self.assertEqual(engine.device, "cuda")
        self.assertEqual(engine.max_retries, 3)
        self.assertEqual(engine.timeout, 60)
        self.assertFalse(engine.running)
        self.assertEqual(engine.loaded_models, {})
    
    def test_init_custom_params(self):
        """Test engine initialization with custom parameters."""
        engine = VLLMModelEngine(
            server_url="http://vllm-server:8080",
            device="cpu",
            max_retries=5,
            timeout=30
        )
        self.assertEqual(engine.server_url, "http://vllm-server:8080")
        self.assertEqual(engine.device, "cpu")
        self.assertEqual(engine.max_retries, 5)
        self.assertEqual(engine.timeout, 30)
    
    def test_start_already_running(self):
        """Test starting the engine when it's already running."""
        engine = VLLMModelEngine()
        engine.running = True
        result = engine.start()
        self.assertTrue(result)
        self.assertTrue(engine.running)
    
    def test_start_server_available(self):
        """Test starting the engine when the server is available."""
        # Configure mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        self.mock_requests.get.return_value = mock_response
        
        # Start the engine
        engine = VLLMModelEngine()
        result = engine.start()
        
        # Verify the result
        self.assertTrue(result)
        self.assertTrue(engine.running)
        
        # Verify correct calls were made
        self.mock_requests.get.assert_called_once_with(
            "http://localhost:8000/v1/models",
            timeout=5
        )
    
    def test_start_server_unavailable(self):
        """Test starting the engine when the server is unavailable."""
        # Configure mock to raise an exception
        self.mock_requests.get.side_effect = ConnectionError("Connection refused")
        
        # Start the engine
        engine = VLLMModelEngine()
        result = engine.start()
        
        # Verify the result
        self.assertFalse(result)
        self.assertFalse(engine.running)
    
    def test_stop_not_running(self):
        """Test stopping the engine when it's not running."""
        engine = VLLMModelEngine()
        engine.running = False
        result = engine.stop()
        self.assertTrue(result)
        self.assertFalse(engine.running)
    
    def test_stop_running(self):
        """Test stopping the engine when it's running."""
        engine = VLLMModelEngine()
        engine.running = True
        result = engine.stop()
        self.assertTrue(result)
        self.assertFalse(engine.running)
    
    def test_load_model_new(self):
        """Test loading a new model."""
        engine = VLLMModelEngine()
        result = engine.load_model("test-model")
        
        # Verify the result
        self.assertEqual(result, "loaded")
        self.assertIn("test-model", engine.loaded_models)
        self.assertEqual(engine.loaded_models["test-model"]["device"], "cuda")
    
    def test_load_model_new_custom_device(self):
        """Test loading a new model with a custom device."""
        engine = VLLMModelEngine()
        result = engine.load_model("test-model", device="cpu")
        
        # Verify the result
        self.assertEqual(result, "loaded")
        self.assertIn("test-model", engine.loaded_models)
        self.assertEqual(engine.loaded_models["test-model"]["device"], "cpu")
    
    def test_load_model_already_loaded(self):
        """Test loading a model that's already loaded."""
        engine = VLLMModelEngine()
        engine.loaded_models["test-model"] = {"device": "cuda", "loaded_at": time.time()}
        
        result = engine.load_model("test-model")
        
        # Verify the result
        self.assertEqual(result, "already_loaded")
    
    def test_unload_model_loaded(self):
        """Test unloading a loaded model."""
        engine = VLLMModelEngine()
        engine.loaded_models["test-model"] = {"device": "cuda", "loaded_at": time.time()}
        
        result = engine.unload_model("test-model")
        
        # Verify the result
        self.assertEqual(result, "unloaded")
        self.assertNotIn("test-model", engine.loaded_models)
    
    def test_unload_model_not_loaded(self):
        """Test unloading a model that's not loaded."""
        engine = VLLMModelEngine()
        
        result = engine.unload_model("test-model")
        
        # Verify the result
        self.assertEqual(result, "not_loaded")
    
    def test_is_model_loaded(self):
        """Test checking if a model is loaded."""
        engine = VLLMModelEngine()
        engine.loaded_models["test-model"] = {"device": "cuda", "loaded_at": time.time()}
        
        # Check loaded model
        self.assertTrue(engine.is_model_loaded("test-model"))
        
        # Check non-loaded model
        self.assertFalse(engine.is_model_loaded("other-model"))
    
    def test_list_loaded_models(self):
        """Test listing loaded models."""
        engine = VLLMModelEngine()
        engine.loaded_models["model1"] = {"device": "cuda", "loaded_at": time.time()}
        engine.loaded_models["model2"] = {"device": "cpu", "loaded_at": time.time()}
        
        models = engine.list_loaded_models()
        
        self.assertEqual(len(models), 2)
        self.assertIn("model1", models)
        self.assertIn("model2", models)
    
    def test_normalize_embedding(self):
        """Test embedding normalization."""
        engine = VLLMModelEngine()
        
        # Test with a simple vector
        embedding = [1.0, 0.0, 0.0]
        normalized = engine._normalize_embedding(embedding)
        self.assertEqual(normalized, [1.0, 0.0, 0.0])
        
        # Test with a more complex vector
        embedding = [1.0, 1.0, 1.0]
        normalized = engine._normalize_embedding(embedding)
        expected = [1.0 / 1.732, 1.0 / 1.732, 1.0 / 1.732]  # 1.732 ~= sqrt(3)
        
        # Check each value with tolerance
        for i in range(len(normalized)):
            self.assertAlmostEqual(normalized[i], expected[i], places=4)
    
    async def async_test_generate_embeddings_success(self):
        """Test generating embeddings with a successful response."""
        engine = VLLMModelEngine()
        
        # Configure mock response
        mock_response = MockResponse(
            status=200,
            json_data={
                "data": [
                    {"embedding": [0.1, 0.2, 0.3]},
                    {"embedding": [0.4, 0.5, 0.6]}
                ]
            }
        )
        self.mock_client_session.set_responses([mock_response])
        
        # Generate embeddings
        texts = ["text1", "text2"]
        embeddings = await engine.generate_embeddings(texts, model_id="test-model")
        
        # Verify the result
        self.assertEqual(len(embeddings), 2)
        
        # Verify normalization was applied
        norm1 = sum(e**2 for e in embeddings[0])**0.5
        norm2 = sum(e**2 for e in embeddings[1])**0.5
        self.assertAlmostEqual(norm1, 1.0, places=4)
        self.assertAlmostEqual(norm2, 1.0, places=4)
        
        # Verify correct API call was made
        self.assertEqual(len(self.mock_client_session.calls), 1)
        call_name, url, payload, timeout = self.mock_client_session.calls[0]
        self.assertEqual(call_name, "post")
        self.assertEqual(url, "http://localhost:8000/v1/embeddings")
        self.assertEqual(payload["input"], texts)
        self.assertEqual(payload["model"], "test-model")
    
    async def async_test_generate_embeddings_error_retry(self):
        """Test generating embeddings with an error and retry."""
        engine = VLLMModelEngine()
        engine.max_retries = 2
        
        # Configure mock responses - first fails, second succeeds
        error_response = Exception("Connection error")
        success_response = MockResponse(
            status=200,
            json_data={
                "data": [
                    {"embedding": [0.1, 0.2, 0.3]}
                ]
            }
        )
        self.mock_client_session.set_responses([error_response, success_response])
        
        # Generate embeddings
        texts = ["text1"]
        embeddings = await engine.generate_embeddings(texts, model_id="test-model")
        
        # Verify the result
        self.assertEqual(len(embeddings), 1)
        
        # Verify two API calls were made (one failure, one success)
        self.assertEqual(len(self.mock_client_session.calls), 2)
    
    async def async_test_generate_embeddings_max_retries_exceeded(self):
        """Test generating embeddings with max retries exceeded."""
        engine = VLLMModelEngine()
        engine.max_retries = 2
        
        # Configure mock responses - all fail
        error_response = Exception("Connection error")
        self.mock_client_session.set_responses([error_response, error_response])
        
        # Generate embeddings - should raise RuntimeError
        texts = ["text1"]
        with self.assertRaises(RuntimeError):
            await engine.generate_embeddings(texts, model_id="test-model")
    
    async def async_test_generate_completion_success(self):
        """Test generating completion with a successful response."""
        engine = VLLMModelEngine()
        
        # Configure mock response
        mock_response = MockResponse(
            status=200,
            json_data={
                "choices": [
                    {"text": "Generated text"}
                ]
            }
        )
        self.mock_client_session.set_responses([mock_response])
        
        # Generate completion
        result = await engine.generate_completion("Prompt text", model_id="test-model")
        
        # Verify the result
        self.assertEqual(result, "Generated text")
        
        # Verify correct API call was made
        self.assertEqual(len(self.mock_client_session.calls), 1)
        call_name, url, payload, timeout = self.mock_client_session.calls[0]
        self.assertEqual(call_name, "post")
        self.assertEqual(url, "http://localhost:8000/v1/completions")
        self.assertEqual(payload["prompt"], "Prompt text")
        self.assertEqual(payload["model"], "test-model")
    
    async def async_test_generate_chat_completion_success(self):
        """Test generating chat completion with a successful response."""
        engine = VLLMModelEngine()
        
        # Configure mock response
        mock_response = MockResponse(
            status=200,
            json_data={
                "choices": [
                    {"message": {"content": "AI response"}}
                ]
            }
        )
        self.mock_client_session.set_responses([mock_response])
        
        # Generate chat completion
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"}
        ]
        result = await engine.generate_chat_completion(messages, model_id="test-model")
        
        # Verify correct API call was made
        self.assertEqual(len(self.mock_client_session.calls), 1)
        call_name, url, payload, timeout = self.mock_client_session.calls[0]
        self.assertEqual(call_name, "post")
        self.assertEqual(url, "http://localhost:8000/v1/chat/completions")
        self.assertEqual(payload["messages"], messages)
        self.assertEqual(payload["model"], "test-model")
    
    # Helper method to run async tests
    def _run_async_test(self, coro):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)
    
    def test_generate_embeddings(self):
        """Test wrapper for async_test_generate_embeddings_success."""
        self._run_async_test(self.async_test_generate_embeddings_success())
    
    def test_generate_embeddings_retry(self):
        """Test wrapper for async_test_generate_embeddings_error_retry."""
        self._run_async_test(self.async_test_generate_embeddings_error_retry())
    
    def test_generate_embeddings_max_retries(self):
        """Test wrapper for async_test_generate_embeddings_max_retries_exceeded."""
        self._run_async_test(self.async_test_generate_embeddings_max_retries_exceeded())
    
    def test_generate_completion(self):
        """Test wrapper for async_test_generate_completion_success."""
        self._run_async_test(self.async_test_generate_completion_success())
    
    def test_generate_chat_completion(self):
        """Test wrapper for async_test_generate_chat_completion_success."""
        self._run_async_test(self.async_test_generate_chat_completion_success())


if __name__ == "__main__":
    unittest.main()
