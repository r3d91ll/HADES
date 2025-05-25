"""
Unit tests for model engine implementations.

This module provides comprehensive tests for the ModelEngine
implementations including HaystackModelEngine and VLLMModelEngine.
"""

import unittest
import logging
import os
from typing import Optional
from unittest.mock import patch, MagicMock

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_model_engines")

# Import the model engine base class
from src.model_engine.base import ModelEngine

# Import the Haystack model engine
from src.model_engine.engines.haystack import HaystackModelEngine

# Use try/except to handle potential import errors with VLLMModelEngine
try:
    from src.model_engine.engines.vllm import VLLMModelEngine
    VLLM_AVAILABLE = True
except ImportError:
    logger.warning("VLLMModelEngine could not be imported, some tests will be skipped")
    VLLM_AVAILABLE = False
    # Create a mock class for testing
    class VLLMModelEngine:
        """Mock VLLMModelEngine for testing."""
        def __init__(self, *args, **kwargs):
            pass


class TestModelEngineInterface(unittest.TestCase):
    """Test the ModelEngine interface implementation requirements."""
    
    def test_interface_methods(self):
        """Test that both engines implement all required interface methods."""
        # Get all abstract methods from ModelEngine
        import inspect
        abstract_methods = [
            name for name, method in inspect.getmembers(ModelEngine, predicate=inspect.isfunction)
            if not name.startswith('_')
        ]
        
        # Verify HaystackModelEngine implements all methods
        haystack_methods = [
            name for name, method in inspect.getmembers(HaystackModelEngine, predicate=inspect.isfunction)
            if not name.startswith('_')
        ]
        for method in abstract_methods:
            self.assertIn(method, haystack_methods, 
                         f"HaystackModelEngine missing required method: {method}")
        
        # Verify VLLMModelEngine implements all methods
        vllm_methods = [
            name for name, method in inspect.getmembers(VLLMModelEngine, predicate=inspect.isfunction)
            if not name.startswith('_')
        ]
        for method in abstract_methods:
            self.assertIn(method, vllm_methods, 
                         f"VLLMModelEngine missing required method: {method}")


class TestHaystackModelEngine(unittest.TestCase):
    """Test the Haystack model engine implementation."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create a mock server path to avoid socket issues during tests
        self.test_socket_path = "/tmp/test_haystack_socket"
        self.original_socket_path = os.environ.get("HADES_MODEL_MGR_SOCKET")
        os.environ["HADES_MODEL_MGR_SOCKET"] = self.test_socket_path
    
    def tearDown(self):
        """Clean up after tests."""
        # Restore original environment variable
        if self.original_socket_path:
            os.environ["HADES_MODEL_MGR_SOCKET"] = self.original_socket_path
        else:
            if "HADES_MODEL_MGR_SOCKET" in os.environ:
                del os.environ["HADES_MODEL_MGR_SOCKET"]
    
    @patch('src.model_engine.engines.haystack.runtime.ModelClient')
    def test_engine_initialization(self, mock_client):
        """Test that the engine initializes correctly."""
        # Set up the mock client
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        
        # Create engine instance
        engine = HaystackModelEngine()
        
        # Verify engine is initialized but not running
        self.assertIsNotNone(engine)
        self.assertFalse(engine.running)
        self.assertIsNone(engine.client)
    
    @patch('src.model_engine.engines.haystack.runtime.ModelClient')
    def test_start_engine(self, mock_client):
        """Test starting the engine."""
        # Set up the mock client
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        
        # Mock successful server start
        mock_client_instance.ping.return_value = True
        
        # Create and start engine
        engine = HaystackModelEngine()
        result = engine.start()
        
        # Verify engine is started
        self.assertTrue(result)
        self.assertTrue(engine.running)
        self.assertIsNotNone(engine.client)
    
    @patch('src.model_engine.engines.haystack.runtime.ModelClient')
    def test_load_model(self, mock_client):
        """Test loading a model."""
        # Set up the mock client
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        
        # Mock successful server start and model loading
        mock_client_instance.ping.return_value = True
        mock_client_instance.load_model.return_value = {"success": True, "model_id": "test-model"}
        
        # Create and start engine
        engine = HaystackModelEngine()
        engine.start()
        
        # Test loading a model
        result = engine.load_model("test-model", device="cpu")
        
        # Verify model was loaded
        self.assertTrue(result)
        mock_client_instance.load_model.assert_called_once()
    
    @patch('src.model_engine.engines.haystack.runtime.ModelClient')
    def test_run_model(self, mock_client):
        """Test running a model for inference."""
        # Set up the mock client
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        
        # Mock successful server start and model operations
        mock_client_instance.ping.return_value = True
        mock_client_instance.run_model.return_value = {"result": "test output"}
        
        # Create and start engine
        engine = HaystackModelEngine()
        engine.start()
        
        # Test running a model
        result = engine.run_model("test-model", {"text": "test input"})
        
        # Verify model was run
        self.assertEqual(result, {"result": "test output"})
        mock_client_instance.run_model.assert_called_once()
    
    @patch('src.model_engine.engines.haystack.runtime.ModelClient')
    def test_stop_engine(self, mock_client):
        """Test stopping the engine."""
        # Set up the mock client
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        
        # Mock successful server start
        mock_client_instance.ping.return_value = True
        
        # Create and start engine
        engine = HaystackModelEngine()
        engine.start()
        
        # Stop the engine
        result = engine.stop()
        
        # Verify engine is stopped
        self.assertTrue(result)
        self.assertFalse(engine.running)
        self.assertIsNone(engine.client)


@unittest.skipIf(not VLLM_AVAILABLE, "VLLM is not available")
class TestVLLMModelEngine(unittest.TestCase):
    """Test the vLLM model engine implementation."""
    
    def setUp(self):
        """Set up the test environment."""
        # Use multiple patches to mock all dependencies
        self.patcher1 = patch('src.model_engine.engines.vllm.vllm_engine.LLM', autospec=True)
        self.patcher2 = patch('src.model_engine.engines.vllm.vllm_engine.utils', autospec=True)
        
        # Start patchers
        self.mock_llm = self.patcher1.start()
        self.mock_utils = self.patcher2.start()
        
        # Set up mock instance
        self.mock_llm_instance = MagicMock()
        self.mock_llm.return_value = self.mock_llm_instance
    
    def tearDown(self):
        """Clean up after tests."""
        # Stop patchers
        self.patcher1.stop()
        self.patcher2.stop()
    
    def test_engine_initialization(self):
        """Test that the engine initializes correctly."""
        # Create engine instance
        engine = VLLMModelEngine()
        
        # Verify engine is initialized but not running
        self.assertIsNotNone(engine)
        self.assertFalse(engine.is_running())
    
    def test_start_engine(self):
        """Test starting the engine."""
        # Create and start engine
        engine = VLLMModelEngine()
        result = engine.start()
        
        # Verify engine is started
        self.assertTrue(result)
        self.assertTrue(engine.is_running())
    
    def test_load_model(self):
        """Test loading a model."""
        # Create and start engine
        engine = VLLMModelEngine()
        engine.start()
        
        # Test loading a model
        result = engine.load_model("test-model", device="cuda")
        
        # Verify model was loaded
        self.assertTrue(result)
    
    def test_run_model(self):
        """Test running a model for inference."""
        # Mock successful response
        self.mock_llm_instance.generate.return_value = MagicMock(
            outputs=[MagicMock(text="test output")]
        )
        
        # Create and start engine
        engine = VLLMModelEngine()
        engine.start()
        engine.load_model("test-model")
        
        # Test running a model
        result = engine.run_model("test-model", {"prompt": "test input"})
        
        # Verify model was run
        self.assertIn("result", result)
        self.mock_llm_instance.generate.assert_called_once()
    
    def test_stop_engine(self):
        """Test stopping the engine."""
        # Create and start engine
        engine = VLLMModelEngine()
        engine.start()
        
        # Stop the engine
        result = engine.stop()
        
        # Verify engine is stopped
        self.assertTrue(result)
        self.assertFalse(engine.is_running())


if __name__ == "__main__":
    unittest.main()
