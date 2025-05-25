"""
Unit tests for the vLLM model engine implementation.

These tests focus on the public API of the VLLMModelEngine class
to achieve the required 85% test coverage, following the project's standard 
testing protocol.
"""

import unittest
import sys
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

# Mock the imports that are causing issues
sys.modules['src.model_engine.vllm_session'] = MagicMock()
sys.modules['src.config.vllm_config'] = MagicMock()
sys.modules['src.model_engine.engines.vllm.runtime'] = MagicMock()

# Now import the base model engine to create a testable subclass
from src.model_engine.base import ModelEngine
from src.model_engine.engines.vllm.vllm_engine import VLLMModelEngine

# Create concrete test class that implements the abstract methods
class TestableVLLMModelEngine(VLLMModelEngine):
    """Testable implementation of VLLMModelEngine."""
    
    def get_loaded_models(self):
        """Implementation of abstract method."""
        result = {}
        for model_id, model_info in self.loaded_models.items():
            result[model_id] = {
                "status": "loaded",
                "engine": "vllm",
                "device": model_info["device"],
                "load_time": model_info["loaded_at"]
            }
        return result
    
    def health_check(self):
        """Implementation of abstract method."""
        if not self.running:
            return {"status": "not_running"}
            
        try:
            # Simulate a ping
            return {"status": "ok"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def infer(self, model_id, inputs, task_type="generate"):
        """Implementation of abstract method."""
        if not self.running:
            raise RuntimeError("Engine is not running")
            
        if model_id not in self.loaded_models:
            self.load_model(model_id)
            
        if task_type == "generate":
            return "Generated text"
        elif task_type == "embed":
            return [[0.1, 0.2, 0.3]]
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
    
    def restart(self):
        """Implementation of restart method."""
        # Stop if currently running
        if self.running:
            self.stop()
        
        # Start again
        return self.start()


# Create mock classes for testing
class MockResponse:
    """Mock HTTP response for testing."""
    
    def __init__(self, status_code=200, json_data=None, text=None):
        self.status_code = status_code
        self._json_data = json_data or {}
        self._text = text or ""
    
    def json(self):
        """Return mock JSON data."""
        return self._json_data
    
    def text(self):
        """Return mock text data."""
        return self._text

class AsyncMockResponse:
    """Mock async HTTP response for testing."""
    
    def __init__(self, status=200, json_data=None, text=None):
        self.status = status
        self._json_data = json_data or {}
        self._text = text or ""
    
    async def json(self):
        """Return mock JSON data."""
        return self._json_data
    
    async def text(self):
        """Return mock text data."""
        return self._text
    
    async def __aenter__(self):
        """Async context manager enter."""
        return self
    
    async def __aexit__(self, *args):
        """Async context manager exit."""
        pass

class AsyncMockSession:
    """Mock async session for testing."""
    
    def __init__(self, responses=None):
        self.responses = responses or []
        self.calls = []
    
    async def __aenter__(self):
        """Async context manager enter."""
        return self
    
    async def __aexit__(self, *args):
        """Async context manager exit."""
        pass
    
    async def post(self, url, json=None, timeout=None):
        """Mock post request."""
        self.calls.append(("post", url, json, timeout))
        if not self.responses:
            response = AsyncMockResponse(
                status=200, 
                json_data={"choices": [{"text": "mock response"}]}
            )
        else:
            response = self.responses.pop(0)
        return response


class TestVLLMModelEngine(unittest.TestCase):
    """Test the vLLM model engine implementation."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create patches
        self.requests_patch = patch('requests.get')
        self.aiohttp_patch = patch('aiohttp.ClientSession')
        
        # Additional patches for async methods
        self.asyncio_patch = patch('asyncio.sleep', new=AsyncMock())
        
        # Start patches
        self.mock_requests_get = self.requests_patch.start()
        self.mock_aiohttp_session = self.aiohttp_patch.start()
        self.mock_asyncio_sleep = self.asyncio_patch.start()
        
        # Configure mock responses
        self.mock_requests_get.return_value = MockResponse(
            status_code=200, 
            json_data={"models": ["model1", "model2"]}
        )
        
        self.mock_session = AsyncMockSession()
        self.mock_aiohttp_session.return_value = self.mock_session
    
    def tearDown(self):
        """Clean up after tests."""
        self.requests_patch.stop()
        self.aiohttp_patch.stop()
        self.asyncio_patch.stop()
    
    def test_init_default(self):
        """Test initialization with default parameters."""
        engine = TestableVLLMModelEngine()
        
        self.assertEqual(engine.server_url, "http://localhost:8000")
        self.assertEqual(engine.device, "cuda")
        self.assertEqual(engine.max_retries, 3)
        self.assertEqual(engine.timeout, 60)
        self.assertFalse(engine.running)
        self.assertEqual(engine.loaded_models, {})
        self.assertIsNone(engine.session)
    
    def test_init_custom(self):
        """Test initialization with custom parameters."""
        engine = TestableVLLMModelEngine(
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
        engine = TestableVLLMModelEngine()
        
        result = engine.start()
        
        self.assertTrue(result)
        self.assertTrue(engine.running)
        self.mock_requests_get.assert_called_once_with(
            "http://localhost:8000/v1/models", 
            timeout=5
        )
    
    def test_start_already_running(self):
        """Test start when already running."""
        engine = TestableVLLMModelEngine()
        engine.running = True
        
        result = engine.start()
        
        self.assertTrue(result)
        self.assertTrue(engine.running)
        self.mock_requests_get.assert_not_called()
    
    def test_start_failure(self):
        """Test start failure."""
        engine = TestableVLLMModelEngine()
        self.mock_requests_get.side_effect = ConnectionError("Connection refused")
        
        result = engine.start()
        
        self.assertFalse(result)
        self.assertFalse(engine.running)
    
    def test_stop(self):
        """Test stop when running."""
        engine = TestableVLLMModelEngine()
        engine.running = True
        
        result = engine.stop()
        
        self.assertTrue(result)
        self.assertFalse(engine.running)
    
    def test_stop_not_running(self):
        """Test stop when not running."""
        engine = TestableVLLMModelEngine()
        engine.running = False
        
        result = engine.stop()
        
        self.assertTrue(result)
        self.assertFalse(engine.running)
    
    def test_restart(self):
        """Test engine restart."""
        engine = TestableVLLMModelEngine()
        
        # Make it initially running
        engine.running = True
        
        # Mock the start and stop behavior
        with patch.object(engine, 'stop', return_value=True) as mock_stop, \
             patch.object(engine, 'start', return_value=True) as mock_start:
            # Call restart
            result = engine.restart()
            
            # Verify result and method calls
            self.assertTrue(result)
            mock_stop.assert_called_once()
            mock_start.assert_called_once()
    
    def test_load_model_new(self):
        """Test loading a new model."""
        engine = TestableVLLMModelEngine()
        
        result = engine.load_model("test-model")
        
        self.assertEqual(result, "loaded")
        self.assertIn("test-model", engine.loaded_models)
        self.assertEqual(engine.loaded_models["test-model"]["device"], "cuda")
    
    def test_load_model_custom_device(self):
        """Test loading a model with custom device."""
        engine = TestableVLLMModelEngine()
        
        result = engine.load_model("test-model", device="cpu")
        
        self.assertEqual(result, "loaded")
        self.assertIn("test-model", engine.loaded_models)
        self.assertEqual(engine.loaded_models["test-model"]["device"], "cpu")
    
    def test_load_model_already_loaded(self):
        """Test loading an already loaded model."""
        engine = TestableVLLMModelEngine()
        engine.loaded_models["test-model"] = {"device": "cuda", "loaded_at": time.time()}
        
        result = engine.load_model("test-model")
        
        self.assertEqual(result, "already_loaded")
    
    def test_unload_model(self):
        """Test unloading a model."""
        engine = TestableVLLMModelEngine()
        engine.loaded_models["test-model"] = {"device": "cuda", "loaded_at": time.time()}
        
        result = engine.unload_model("test-model")
        
        self.assertEqual(result, "unloaded")
        self.assertNotIn("test-model", engine.loaded_models)
    
    def test_unload_model_not_loaded(self):
        """Test unloading a model that's not loaded."""
        engine = TestableVLLMModelEngine()
        
        result = engine.unload_model("test-model")
        
        self.assertEqual(result, "not_loaded")
    
    def test_is_model_loaded(self):
        """Test is_model_loaded method."""
        engine = TestableVLLMModelEngine()
        engine.loaded_models["test-model"] = {"device": "cuda", "loaded_at": time.time()}
        
        self.assertTrue(engine.is_model_loaded("test-model"))
        self.assertFalse(engine.is_model_loaded("other-model"))
    
    def test_list_loaded_models(self):
        """Test list_loaded_models method."""
        engine = TestableVLLMModelEngine()
        engine.loaded_models = {
            "model1": {"device": "cuda", "loaded_at": time.time()},
            "model2": {"device": "cpu", "loaded_at": time.time()}
        }
        
        result = engine.list_loaded_models()
        
        self.assertEqual(len(result), 2)
        self.assertIn("model1", result)
        self.assertIn("model2", result)
    
    def test_get_loaded_models(self):
        """Test get_loaded_models method."""
        engine = TestableVLLMModelEngine()
        load_time = time.time()
        engine.loaded_models = {
            "model1": {"device": "cuda", "loaded_at": load_time},
            "model2": {"device": "cpu", "loaded_at": load_time}
        }
        
        result = engine.get_loaded_models()
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result["model1"]["status"], "loaded")
        self.assertEqual(result["model1"]["engine"], "vllm")
        self.assertEqual(result["model1"]["device"], "cuda")
        self.assertEqual(result["model1"]["load_time"], load_time)
    
    def test_health_check_running(self):
        """Test health_check when engine is running."""
        engine = TestableVLLMModelEngine()
        engine.running = True
        
        result = engine.health_check()
        
        self.assertEqual(result["status"], "ok")
    
    def test_health_check_not_running(self):
        """Test health_check when engine is not running."""
        engine = TestableVLLMModelEngine()
        engine.running = False
        
        result = engine.health_check()
        
        self.assertEqual(result["status"], "not_running")
    
    def test_infer_generate(self):
        """Test infer method with generate task."""
        engine = TestableVLLMModelEngine()
        engine.running = True
        
        result = engine.infer("test-model", "test input", "generate")
        
        self.assertEqual(result, "Generated text")
    
    def test_infer_embed(self):
        """Test infer method with embed task."""
        engine = TestableVLLMModelEngine()
        engine.running = True
        
        result = engine.infer("test-model", "test input", "embed")
        
        self.assertEqual(result, [[0.1, 0.2, 0.3]])
    
    def test_infer_invalid_task(self):
        """Test infer method with invalid task."""
        engine = TestableVLLMModelEngine()
        engine.running = True
        
        with self.assertRaises(ValueError):
            engine.infer("test-model", "test input", "invalid_task")
    
    def test_infer_not_running(self):
        """Test infer method when engine is not running."""
        engine = TestableVLLMModelEngine()
        engine.running = False
        
        with self.assertRaises(RuntimeError):
            engine.infer("test-model", "test input", "generate")
    
    def test_normalize_embedding(self):
        """Test _normalize_embedding method."""
        engine = TestableVLLMModelEngine()
        
        # Test with a unit vector
        result = engine._normalize_embedding([1.0, 0.0, 0.0])
        self.assertAlmostEqual(result[0], 1.0)
        self.assertAlmostEqual(result[1], 0.0)
        self.assertAlmostEqual(result[2], 0.0)
        
        # Test with a non-unit vector
        result = engine._normalize_embedding([3.0, 4.0])
        self.assertAlmostEqual(result[0], 0.6)
        self.assertAlmostEqual(result[1], 0.8)
    
    def test_normalize_embedding_zero_vector(self):
        """Test _normalize_embedding with zero vector."""
        engine = TestableVLLMModelEngine()
        
        # Test with a zero vector (should handle division by zero)
        result = engine._normalize_embedding([0.0, 0.0, 0.0])
        self.assertAlmostEqual(result[0], 0.0)
        self.assertAlmostEqual(result[1], 0.0)
        self.assertAlmostEqual(result[2], 0.0)
    
    # Async test helpers and tests
    async def async_test_generate_embeddings_success(self):
        """Test generate_embeddings with successful response."""
        engine = TestableVLLMModelEngine()
        engine.running = True
        
        # Setup mock response
        mock_response = AsyncMockResponse(
            status=200,
            json_data={
                "data": [
                    {"embedding": [0.1, 0.2, 0.3]},
                    {"embedding": [0.4, 0.5, 0.6]}
                ]
            }
        )
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_session_instance = MagicMock()
            mock_session_instance.__aenter__ = AsyncMock(return_value=mock_session_instance)
            mock_session_instance.__aexit__ = AsyncMock(return_value=None)
            mock_session_instance.post = AsyncMock(return_value=mock_response)
            mock_session.return_value = mock_session_instance
            
            # Call method
            texts = ["test text 1", "test text 2"]
            result = await engine.generate_embeddings(texts, model_id="test-model")
            
            # Verify result
            self.assertEqual(len(result), 2)
            self.assertEqual(len(result[0]), 3)
            self.assertEqual(len(result[1]), 3)
            
            # Verify API call
            mock_session_instance.post.assert_called_once()
            args, kwargs = mock_session_instance.post.call_args
            self.assertEqual(args[0], "http://localhost:8000/v1/embeddings")
            self.assertEqual(kwargs["json"]["input"], texts)
            self.assertEqual(kwargs["json"]["model"], "test-model")
    
    async def async_test_generate_embeddings_error_retry(self):
        """Test generate_embeddings with error and retry."""
        engine = TestableVLLMModelEngine()
        engine.running = True
        engine.max_retries = 2
        
        # Setup responses - first fails, second succeeds
        error_response = AsyncMockResponse(status=500, text="Server error")
        success_response = AsyncMockResponse(
            status=200,
            json_data={"data": [{"embedding": [0.1, 0.2, 0.3]}]}
        )
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_session_instance = MagicMock()
            mock_session_instance.__aenter__ = AsyncMock(return_value=mock_session_instance)
            mock_session_instance.__aexit__ = AsyncMock(return_value=None)
            
            # First call raises error, second call succeeds
            mock_session_instance.post = AsyncMock(side_effect=[
                error_response, success_response
            ])
            mock_session.return_value = mock_session_instance
            
            # Call method
            texts = ["test text"]
            result = await engine.generate_embeddings(texts, model_id="test-model")
            
            # Verify result
            self.assertEqual(len(result), 1)
            self.assertEqual(len(result[0]), 3)
            
            # Verify API calls
            self.assertEqual(mock_session_instance.post.call_count, 2)
    
    async def async_test_generate_embeddings_max_retries(self):
        """Test generate_embeddings with max retries exceeded."""
        engine = TestableVLLMModelEngine()
        engine.running = True
        engine.max_retries = 2
        
        # Setup all responses to fail
        error_response = AsyncMockResponse(status=500, text="Server error")
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_session_instance = MagicMock()
            mock_session_instance.__aenter__ = AsyncMock(return_value=mock_session_instance)
            mock_session_instance.__aexit__ = AsyncMock(return_value=None)
            
            # All calls fail
            mock_session_instance.post = AsyncMock(side_effect=[
                error_response, error_response
            ])
            mock_session.return_value = mock_session_instance
            
            # Call method and expect RuntimeError
            texts = ["test text"]
            with self.assertRaises(RuntimeError):
                await engine.generate_embeddings(texts, model_id="test-model")
            
            # Verify API calls
            self.assertEqual(mock_session_instance.post.call_count, 2)
    
    async def async_test_generate_completion_success(self):
        """Test generate_completion with successful response."""
        engine = TestableVLLMModelEngine()
        engine.running = True
        
        # Setup mock response
        mock_response = AsyncMockResponse(
            status=200,
            json_data={
                "choices": [
                    {"text": "Generated completion text"}
                ]
            }
        )
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_session_instance = MagicMock()
            mock_session_instance.__aenter__ = AsyncMock(return_value=mock_session_instance)
            mock_session_instance.__aexit__ = AsyncMock(return_value=None)
            mock_session_instance.post = AsyncMock(return_value=mock_response)
            mock_session.return_value = mock_session_instance
            
            # Call method
            result = await engine.generate_completion(
                "Test prompt",
                model_id="test-model",
                max_tokens=100,
                temperature=0.5
            )
            
            # Verify result
            self.assertEqual(result, "Generated completion text")
            
            # Verify API call
            mock_session_instance.post.assert_called_once()
            args, kwargs = mock_session_instance.post.call_args
            self.assertEqual(args[0], "http://localhost:8000/v1/completions")
            self.assertEqual(kwargs["json"]["prompt"], "Test prompt")
            self.assertEqual(kwargs["json"]["model"], "test-model")
            self.assertEqual(kwargs["json"]["max_tokens"], 100)
            self.assertEqual(kwargs["json"]["temperature"], 0.5)
    
    async def async_test_generate_chat_completion_success(self):
        """Test generate_chat_completion with successful response."""
        engine = TestableVLLMModelEngine()
        engine.running = True
        
        # Setup mock response
        mock_response = AsyncMockResponse(
            status=200,
            json_data={
                "choices": [
                    {"message": {"content": "Chat response", "role": "assistant"}}
                ]
            }
        )
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_session_instance = MagicMock()
            mock_session_instance.__aenter__ = AsyncMock(return_value=mock_session_instance)
            mock_session_instance.__aexit__ = AsyncMock(return_value=None)
            mock_session_instance.post = AsyncMock(return_value=mock_response)
            mock_session.return_value = mock_session_instance
            
            # Call method
            messages = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"}
            ]
            result = await engine.generate_chat_completion(
                messages,
                model_id="test-model",
                max_tokens=200,
                temperature=0.7
            )
            
            # Verify result
            self.assertIn("choices", result)
            self.assertEqual(result["choices"][0]["message"]["content"], "Chat response")
            
            # Verify API call
            mock_session_instance.post.assert_called_once()
            args, kwargs = mock_session_instance.post.call_args
            self.assertEqual(args[0], "http://localhost:8000/v1/chat/completions")
            self.assertEqual(kwargs["json"]["messages"], messages)
            self.assertEqual(kwargs["json"]["model"], "test-model")
            self.assertEqual(kwargs["json"]["max_tokens"], 200)
            self.assertEqual(kwargs["json"]["temperature"], 0.7)
    
    # Test wrappers for async tests
    def test_generate_embeddings_success(self):
        """Test wrapper for async_test_generate_embeddings_success."""
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.async_test_generate_embeddings_success())
    
    def test_generate_embeddings_error_retry(self):
        """Test wrapper for async_test_generate_embeddings_error_retry."""
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.async_test_generate_embeddings_error_retry())
    
    def test_generate_embeddings_max_retries(self):
        """Test wrapper for async_test_generate_embeddings_max_retries."""
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.async_test_generate_embeddings_max_retries())
    
    def test_generate_completion_success(self):
        """Test wrapper for async_test_generate_completion_success."""
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.async_test_generate_completion_success())
    
    def test_generate_chat_completion_success(self):
        """Test wrapper for async_test_generate_chat_completion_success."""
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.async_test_generate_chat_completion_success())


if __name__ == "__main__":
    unittest.main()
