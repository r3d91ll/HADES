"""
Unit tests for the Haystack model engine with mocked dependencies.

This module provides mock-based tests for the HaystackModelEngine core functionality
without triggering actual PyTorch imports that cause conflicts.
"""

import unittest
import os
import sys
import logging
import tempfile
from typing import Dict, Any, List
from unittest.mock import patch, MagicMock, call, mock_open

# Mock PyTorch and other problematic imports before importing the actual module
sys.modules['torch'] = MagicMock()
sys.modules['transformers'] = MagicMock()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_haystack_core")


class MockModelClient:
    """Mock implementation of ModelClient to avoid socket operations."""
    
    def __init__(self, socket_path=None):
        self.socket_path = socket_path or "/tmp/mock_socket"
        self.loaded_models = {}
        self.calls = []
    
    def ping(self):
        """Mock ping method."""
        self.calls.append(("ping",))
        return "pong"
    
    def load(self, model_id, device=None):
        """Mock load method."""
        self.calls.append(("load", model_id, device))
        self.loaded_models[model_id] = {"device": device or "cpu"}
        return "loaded"
    
    def unload(self, model_id):
        """Mock unload method."""
        self.calls.append(("unload", model_id))
        if model_id in self.loaded_models:
            del self.loaded_models[model_id]
        return "unloaded"
    
    def info(self):
        """Mock info method."""
        self.calls.append(("info",))
        # Return model_id to timestamp mapping
        return {k: "2025-05-16T12:00:00Z" for k in self.loaded_models.keys()}
    
    def debug(self):
        """Mock debug method."""
        self.calls.append(("debug",))
        return {"status": "ok", "loaded_models": self.loaded_models}
    
    def shutdown(self):
        """Mock shutdown method."""
        self.calls.append(("shutdown",))
        return "shutdown"
        
    def _request(self, payload):
        """Mock _request method to simulate successful communication."""
        method = payload.get("method")
        params = payload.get("params", {})
        
        if method == "ping":
            return "pong"
        elif method == "load":
            model_id = params.get("model_id")
            device = params.get("device")
            return self.load(model_id, device)
        elif method == "unload":
            model_id = params.get("model_id")
            return self.unload(model_id)
        elif method == "info":
            return self.info()
        elif method == "debug":
            return self.debug()
        elif method == "shutdown":
            return self.shutdown()
        else:
            # Default response for other methods
            return {"status": "ok", "method": method, "params": params}


# We need to patch at the module level first
patch_runtime = patch('src.model_engine.engines.haystack.runtime.ModelClient', MockModelClient)
patch_runtime.start()

# Create a set of patches we'll use in our tests
def mock_ensure_server(socket_path=None):
    """Mock implementation to avoid actual server startup."""
    return None

# Create a mock socket object to avoid actual socket connections
class MockSocket:
    def __init__(self, *args, **kwargs):
        pass
        
    def connect(self, *args, **kwargs):
        pass
        
    def sendall(self, data):
        pass
        
    def recv(self, bufsize):
        # Return a mock response that looks like a valid JSON-RPC response
        import json
        return json.dumps({"result": "pong", "id": 1}).encode('utf-8')
        
    def close(self):
        pass
        
    # Support for context manager protocol
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

# Create our patches    
_ensure_server_patch = patch('src.model_engine.engines.haystack.runtime._ensure_server', 
                           side_effect=mock_ensure_server)
_socket_patch = patch('socket.socket', return_value=MockSocket())

# Now we can import the module under test with our mocks in place
from src.model_engine.base import ModelEngine
from src.model_engine.engines.haystack import HaystackModelEngine


class TestHaystackModelEngineCore(unittest.TestCase):
    """Test the core functionality of the Haystack model engine with mocked dependencies."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create a mock server path
        self.test_socket_path = os.path.join(tempfile.gettempdir(), "test_haystack_socket")
        
        # Create engine with mocked client
        self.mock_client = MockModelClient(self.test_socket_path)
        
        # Apply the _ensure_server patch in this test class
        self.ensure_server_patch = _ensure_server_patch
        self.ensure_server_patch.start()
        self.addCleanup(self.ensure_server_patch.stop)
        
        # Apply socket patch to avoid actual socket connections
        self.socket_patch = _socket_patch
        self.socket_patch.start()
        self.addCleanup(self.socket_patch.stop)
        
        # Patch the ModelClient's _request method to use our mock implementation
        self.request_patch = patch('src.model_engine.engines.haystack.runtime.ModelClient._request', 
                                  side_effect=self.mock_client._request)
        self.request_patch.start()
        self.addCleanup(self.request_patch.stop)
        
        # Create engine
        self.engine = HaystackModelEngine(socket_path=self.test_socket_path)
        self.engine.client = self.mock_client
        self.engine.running = True
    
    def tearDown(self):
        """Clean up after tests."""
        self.engine = None
        self.mock_client = None
    
    def test_engine_initialization(self):
        """Test basic initialization of the engine."""
        engine = HaystackModelEngine(socket_path=self.test_socket_path)
        self.assertEqual(engine.socket_path, self.test_socket_path)
        self.assertFalse(engine.running)
        self.assertIsNone(engine.client)
    
    def test_start_engine(self):
        """Test starting the engine."""
        # For this test, manually mock the communication to simulate success
        with patch('src.model_engine.engines.haystack.runtime.ModelClient.ping', return_value="pong"):
            engine = HaystackModelEngine(socket_path=self.test_socket_path)
            
            # Start should create a client and mark as running
            success = engine.start()
            self.assertTrue(success)
            self.assertTrue(engine.running)
            self.assertIsNotNone(engine.client)
    
    def test_stop_engine(self):
        """Test stopping the engine."""
        # Engine is already started in setUp
        self.assertTrue(self.engine.running)
        
        # Stop the engine
        success = self.engine.stop()
        self.assertTrue(success)
        self.assertFalse(self.engine.running)
        
        # Verify shutdown was called
        self.assertIn(("shutdown",), self.mock_client.calls)
    
    def test_running_status(self):
        """Test checking if the engine is running."""
        # Engine is already running in setUp
        self.assertTrue(self.engine.running)
        
        # Stop the engine and check again
        self.engine.running = False
        self.engine.client = None
        self.assertFalse(self.engine.running)
    
    def test_get_loaded_models(self):
        """Test getting loaded models."""
        # Mock some models
        self.mock_client.loaded_models = {
            "model1": {"device": "cpu"},
            "model2": {"device": "cuda:0"}
        }
        
        # Get the loaded models
        models = self.engine.get_loaded_models()
        self.assertIsInstance(models, dict)
        self.assertEqual(set(models.keys()), {"model1", "model2"})
        
        # Verify call to info
        self.assertIn(("info",), self.mock_client.calls)
    
    def test_load_model(self):
        """Test loading a model."""
        # Load a model
        model_id = "test-embedding-model"
        result = self.engine.load_model(model_id)
        
        # Verify the call to load
        self.assertIn(("load", model_id, None), self.mock_client.calls)
        self.assertEqual(result, "loaded")
    
    def test_unload_model(self):
        """Test unloading a model."""
        # Load a model first
        model_id = "test-model-to-unload"
        self.mock_client.loaded_models[model_id] = {"device": "cpu"}
        
        # Unload the model
        result = self.engine.unload_model(model_id)
        
        # Verify the result and client call
        self.assertEqual(result, "unloaded")
        self.assertIn(("unload", model_id), self.mock_client.calls)
    
    def test_health_check(self):
        """Test health check."""
        # Engine is running
        health = self.engine.health_check()
        self.assertEqual(health["status"], "ok")
        
        # Engine not running
        self.engine.running = False
        health = self.engine.health_check()
        self.assertEqual(health["status"], "not_running")
        
    def test_restart(self):
        """Test restarting the engine."""
        # For restart test, fully mock both stop and start
        with patch.object(HaystackModelEngine, 'stop', return_value=True) as mock_stop, \
             patch.object(HaystackModelEngine, 'start', return_value=True) as mock_start:
            
            # Engine is already running in setUp
            self.assertTrue(self.engine.running)
            
            # Restart the engine
            success = self.engine.restart()
            self.assertTrue(success)
            
            # Verify stop and start were called
            mock_stop.assert_called_once()
            mock_start.assert_called_once()
    
    def test_status(self):
        """Test getting engine status."""
        # Set up some loaded models
        self.mock_client.loaded_models = {
            "model1": {"device": "cpu"},
            "model2": {"device": "cuda:0"}
        }
        
        # Get status with running engine
        status = self.engine.status()
        self.assertTrue(status["running"])
        self.assertTrue(status["healthy"])
        self.assertEqual(status["model_count"], 2)
        
        # Test with non-running engine
        self.engine.running = False
        status = self.engine.status()
        self.assertFalse(status["running"])
        
    def test_context_manager(self):
        """Test using the engine as a context manager."""
        # For context manager test, mock the start and stop methods
        with patch.object(HaystackModelEngine, 'start') as mock_start, \
             patch.object(HaystackModelEngine, 'stop') as mock_stop:
            # Set up the mocks
            mock_start.return_value = True
            mock_stop.return_value = True
            
            # Create a new engine
            engine = HaystackModelEngine(socket_path=self.test_socket_path)
            self.assertFalse(engine.running)
            
            # Using as context manager
            with engine as cm:
                # Manually set running since we're mocking start
                engine.running = True
                self.assertTrue(engine.running)
                
            # Verify start and stop were called
            mock_start.assert_called_once()
            mock_stop.assert_called_once()
    
    def test_infer_not_implemented(self):
        """Test that infer raises NotImplementedError."""
        model_id = "test-model"
        with self.assertRaises(NotImplementedError):
            self.engine.infer(model_id, "test input", "generate")
            
    def test_exception_handling(self):
        """Test exception handling in various methods."""
        # Test load_model with client error
        def mock_load_error(*args, **kwargs):
            raise RuntimeError("Mock load error")
            
        self.mock_client.load = mock_load_error
        with self.assertRaises(RuntimeError):
            self.engine.load_model("error-model")
        
        # Test stop with error
        def mock_shutdown_error(*args, **kwargs):
            raise RuntimeError("Mock shutdown error")
            
        self.mock_client.shutdown = mock_shutdown_error
        # Should return False but not propagate the error
        self.assertFalse(self.engine.stop())


class TestRuntimeUtils(unittest.TestCase):
    """Test utility functions in the runtime module."""
    
    @patch('os.path.exists')
    def test_ensure_server_already_running(self, mock_exists):
        """Test ensuring server when it's already running."""
        # Mock that the socket already exists
        mock_exists.return_value = True
        
        # Import the function directly
        from src.model_engine.engines.haystack.runtime import _ensure_server
        
        # Call the function
        _ensure_server("/tmp/test_socket")
        
        # Verify the socket was checked
        mock_exists.assert_called_once_with("/tmp/test_socket")
    
    @patch('os.path.exists')
    @patch('subprocess.Popen')
    def test_ensure_server_not_running(self, mock_popen, mock_exists):
        """Test ensuring server when it's not already running."""
        # Mock that the socket doesn't exist initially, then exists after startup
        mock_exists.side_effect = [False, True]
        
        # Mock the subprocess
        mock_process = MagicMock()
        mock_process.poll.return_value = None  # Process is still running
        mock_popen.return_value = mock_process
        
        # Import the function directly
        from src.model_engine.engines.haystack.runtime import _ensure_server
        
        # Call the function
        _ensure_server("/tmp/test_socket")
        
        # Verify the socket was checked twice
        self.assertEqual(mock_exists.call_count, 2)
        
        # Verify subprocess was used to start the server
        self.assertTrue(mock_popen.called)
        args, kwargs = mock_popen.call_args
        self.assertIn("python", args[0][0])
        self.assertIn("src.model_engine.engines.haystack.runtime.server", args[0][2])


if __name__ == "__main__":
    unittest.main()
