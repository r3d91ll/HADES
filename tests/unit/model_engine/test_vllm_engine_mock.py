"""
Unit tests for the vLLM model engine implementation.

These tests focus on the public API of the VLLMModelEngine class
to achieve the required 85% test coverage, following the project's standard 
testing protocol.
"""

import unittest
import sys
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
sys.modules['src.model_engine.engines.vllm.runtime'] = MagicMock()

# Now import the engine class
from src.model_engine.engines.vllm.vllm_engine import VLLMModelEngine

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
        
        # Start patches
        self.mock_requests_get = self.requests_patch.start()
        self.mock_aiohttp_session = self.aiohttp_patch.start()
        
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
    
    def test_init_default(self):
        """Test initialization with default parameters."""
        engine = VLLMModelEngine()
        
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
        engine = VLLMModelEngine()
        
        result = engine.start()
        
        self.assertTrue(result)
        self.assertTrue(engine.running)
        self.mock_requests_get.assert_called_once_with(
            "http://localhost:8000/v1/models", 
            timeout=5
        )
    
    def test_start_already_running(self):
        """Test start when already running."""
        engine = VLLMModelEngine()
        engine.running = True
        
        result = engine.start()
        
        self.assertTrue(result)
        self.assertTrue(engine.running)
        self.mock_requests_get.assert_not_called()
    
    def test_start_failure(self):
        """Test start failure."""
        engine = VLLMModelEngine()
        self.mock_requests_get.side_effect = ConnectionError("Connection refused")
        
        result = engine.start()
        
        self.assertFalse(result)
        self.assertFalse(engine.running)
    
    def test_stop(self):
        """Test stop when running."""
        engine = VLLMModelEngine()
        engine.running = True
        
        result = engine.stop()
        
        self.assertTrue(result)
        self.assertFalse(engine.running)
    
    def test_stop_not_running(self):
        """Test stop when not running."""
        engine = VLLMModelEngine()
        engine.running = False
        
        result = engine.stop()
        
        self.assertTrue(result)
        self.assertFalse(engine.running)
    
    def test_load_model_new(self):
        """Test loading a new model."""
        engine = VLLMModelEngine()
        
        result = engine.load_model("test-model")
        
        self.assertEqual(result, "loaded")
        self.assertIn("test-model", engine.loaded_models)
        self.assertEqual(engine.loaded_models["test-model"]["device"], "cuda")
    
    def test_load_model_custom_device(self):
        """Test loading a model with custom device."""
        engine = VLLMModelEngine()
        
        result = engine.load_model("test-model", device="cpu")
        
        self.assertEqual(result, "loaded")
        self.assertIn("test-model", engine.loaded_models)
        self.assertEqual(engine.loaded_models["test-model"]["device"], "cpu")
    
    def test_load_model_already_loaded(self):
        """Test loading an already loaded model."""
        engine = VLLMModelEngine()
        engine.loaded_models["test-model"] = {"device": "cuda", "loaded_at": time.time()}
        
        result = engine.load_model("test-model")
        
        self.assertEqual(result, "already_loaded")
    
    def test_unload_model(self):
        """Test unloading a model."""
        engine = VLLMModelEngine()
        engine.loaded_models["test-model"] = {"device": "cuda", "loaded_at": time.time()}
        
        result = engine.unload_model("test-model")
        
        self.assertEqual(result, "unloaded")
        self.assertNotIn("test-model", engine.loaded_models)
    
    def test_unload_model_not_loaded(self):
        """Test unloading a model that's not loaded."""
        engine = VLLMModelEngine()
        
        result = engine.unload_model("test-model")
        
        self.assertEqual(result, "not_loaded")
    
    def test_is_model_loaded(self):
        """Test is_model_loaded method."""
        engine = VLLMModelEngine()
        engine.loaded_models["test-model"] = {"device": "cuda", "loaded_at": time.time()}
        
        self.assertTrue(engine.is_model_loaded("test-model"))
        self.assertFalse(engine.is_model_loaded("other-model"))
    
    def test_list_loaded_models(self):
        """Test list_loaded_models method."""
        engine = VLLMModelEngine()
        engine.loaded_models = {
            "model1": {"device": "cuda", "loaded_at": time.time()},
            "model2": {"device": "cpu", "loaded_at": time.time()}
        }
        
        result = engine.list_loaded_models()
        
        self.assertEqual(len(result), 2)
        self.assertIn("model1", result)
        self.assertIn("model2", result)
    
    def test_normalize_embedding(self):
        """Test _normalize_embedding method."""
        engine = VLLMModelEngine()
        
        # Test with a unit vector
        result = engine._normalize_embedding([1.0, 0.0, 0.0])
        self.assertAlmostEqual(result[0], 1.0)
        self.assertAlmostEqual(result[1], 0.0)
        self.assertAlmostEqual(result[2], 0.0)
        
        # Test with a non-unit vector
        result = engine._normalize_embedding([3.0, 4.0])
        self.assertAlmostEqual(result[0], 0.6)
        self.assertAlmostEqual(result[1], 0.8)


if __name__ == "__main__":
    unittest.main()
