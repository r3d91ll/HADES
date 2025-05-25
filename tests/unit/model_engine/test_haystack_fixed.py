"""
Unit tests for the Haystack model engine implementation.

These tests focus on the public API of the HaystackModelEngine class
to achieve the required 85% test coverage, following the project's standard 
testing protocol.
"""

import unittest
import os
import sys
import json
import socket
import logging
import tempfile
from typing import Dict, Any, List, Optional, Union
from unittest.mock import patch, MagicMock, mock_open, call

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_haystack_engine")

# Import the model engine
from src.model_engine.engines.haystack import HaystackModelEngine
from src.model_engine.engines.haystack.runtime import ModelClient


class MockClient:
    """Mock implementation of the ModelClient for testing."""
    
    def __init__(self, *args, **kwargs):
        self.calls = []
    
    def ping(self):
        self.calls.append(("ping", {}))
        return "pong"
    
    def shutdown(self):
        self.calls.append(("shutdown", {}))
        return True
    
    def load(self, model_id, device=None):
        self.calls.append(("load", {"model_id": model_id, "device": device}))
        return "loaded"
    
    def unload(self, model_id):
        self.calls.append(("unload", {"model_id": model_id}))
        return "unloaded"
    
    def info(self):
        self.calls.append(("info", {}))
        return {"model1": 1620000000, "model2": 1630000000}
    
    def run(self, model_id, inputs):
        self.calls.append(("run", {"model_id": model_id, "inputs": inputs}))
        return {"output": "test output"}


class TestHaystackModelEngine(unittest.TestCase):
    """Test the Haystack model engine implementation."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create patches
        self.client_patcher = patch('src.model_engine.engines.haystack.ModelClient', 
                                   spec=ModelClient)
        
        # Start patches
        self.mock_client_class = self.client_patcher.start()
        
        # Configure mock
        self.mock_client = MockClient()
        self.mock_client_class.return_value = self.mock_client
    
    def tearDown(self):
        """Clean up after tests."""
        # Stop patches
        self.client_patcher.stop()
    
    def test_engine_initialization(self):
        """Test that the engine can be created."""
        engine = HaystackModelEngine()
        self.assertIsNotNone(engine)
        self.assertFalse(engine.running)
        self.assertIsNone(engine.client)
    
    def test_engine_start_success(self):
        """Test successful engine startup."""
        # Create and start engine
        engine = HaystackModelEngine()
        result = engine.start()
        
        # Verify engine is started
        self.assertTrue(result)
        self.assertTrue(engine.running)
        self.assertIsNotNone(engine.client)
        
        # Verify ping was called
        self.assertEqual(self.mock_client.calls[0][0], "ping")
    
    def test_engine_start_ping_failure(self):
        """Test engine startup with ping failure."""
        # Mock failed ping
        self.mock_client.ping = lambda: "failed"
        
        # Create and start engine
        engine = HaystackModelEngine()
        result = engine.start()
        
        # Verify engine failed to start
        self.assertFalse(result)
        self.assertFalse(engine.running)
    
    def test_engine_stop(self):
        """Test stopping the engine."""
        # Create and start engine
        engine = HaystackModelEngine()
        engine.start()
        
        # Verify engine is running
        self.assertTrue(engine.running)
        
        # Stop the engine
        result = engine.stop()
        
        # Verify engine is stopped
        self.assertTrue(result)
        self.assertFalse(engine.running)
        self.assertIsNone(engine.client)
        
        # Verify shutdown was called
        self.assertEqual(self.mock_client.calls[1][0], "shutdown")
    
    def test_engine_stop_not_running(self):
        """Test stopping an engine that isn't running."""
        # Create engine that isn't running
        engine = HaystackModelEngine()
        self.assertFalse(engine.running)
        
        # Stop the engine
        result = engine.stop()
        
        # Should return True even though it wasn't running
        self.assertTrue(result)
        self.assertFalse(engine.running)
    
    def test_restart(self):
        """Test restarting the engine."""
        # Create engine
        engine = HaystackModelEngine()
        
        # Restart the engine
        result = engine.restart()
        
        # Verify restart was successful
        self.assertTrue(result)
        self.assertTrue(engine.running)
    
    def test_status_running(self):
        """Test getting status when engine is running."""
        # Create and start engine
        engine = HaystackModelEngine()
        engine.start()
        
        # Get status
        status = engine.status()
        
        # Verify status
        self.assertEqual(status["running"], True)
        self.assertEqual(status["healthy"], True)
        self.assertEqual(status["model_count"], 2)
        
        # Verify correct calls were made
        self.assertEqual(self.mock_client.calls[1][0], "ping")  # Second call after start() ping
        self.assertEqual(self.mock_client.calls[2][0], "info")
    
    def test_status_not_running(self):
        """Test getting status when engine is not running."""
        # Create engine without starting
        engine = HaystackModelEngine()
        
        # Get status
        status = engine.status()
        
        # Verify status
        self.assertEqual(status["running"], False)
        self.assertNotIn("healthy", status)
    
    def test_context_manager(self):
        """Test using the engine as a context manager."""
        # Use engine as context manager
        with HaystackModelEngine() as engine:
            # Verify engine is running
            self.assertTrue(engine.running)
            self.assertIsNotNone(engine.client)
        
        # Verify engine is stopped after exiting context
        self.assertFalse(engine.running)
        self.assertIsNone(engine.client)
        
        # Verify correct calls were made
        self.assertEqual(self.mock_client.calls[0][0], "ping")  # First call in __enter__
        self.assertEqual(self.mock_client.calls[1][0], "shutdown")  # Second call in __exit__
    
    def test_load_model_success(self):
        """Test loading a model successfully."""
        # Create and start engine
        engine = HaystackModelEngine()
        engine.start()
        
        # Load model
        result = engine.load_model("test-model", device="cpu")
        
        # Verify model was loaded
        self.assertEqual(result, "loaded")
        
        # Verify correct calls were made
        self.assertEqual(self.mock_client.calls[1][0], "load")
        self.assertEqual(self.mock_client.calls[1][1]["model_id"], "test-model")
        self.assertEqual(self.mock_client.calls[1][1]["device"], "cpu")
    
    def test_load_model_auto_start(self):
        """Test loading a model when engine is not running but auto-starts."""
        # Create engine without starting
        engine = HaystackModelEngine()
        
        # Load model
        result = engine.load_model("test-model")
        
        # Verify model was loaded
        self.assertEqual(result, "loaded")
        
        # Verify engine was started automatically
        self.assertTrue(engine.running)
        
        # Verify correct calls were made
        self.assertEqual(self.mock_client.calls[0][0], "ping")  # Auto-start ping
        self.assertEqual(self.mock_client.calls[1][0], "load")
    
    def test_load_model_auto_start_fails(self):
        """Test loading a model when engine is not running and auto-start fails."""
        # Create engine without starting
        engine = HaystackModelEngine()
        
        # Mock failed ping
        self.mock_client.ping = lambda: "failed"
        
        # Load model, expecting RuntimeError
        with self.assertRaises(RuntimeError):
            engine.load_model("test-model")
    
    def test_unload_model_success(self):
        """Test unloading a model successfully."""
        # Create and start engine
        engine = HaystackModelEngine()
        engine.start()
        
        # Unload model
        result = engine.unload_model("test-model")
        
        # Verify model was unloaded
        self.assertEqual(result, "unloaded")
        
        # Verify correct calls were made
        self.assertEqual(self.mock_client.calls[1][0], "unload")
        self.assertEqual(self.mock_client.calls[1][1]["model_id"], "test-model")
    
    def test_unload_model_not_running(self):
        """Test unloading a model when engine is not running."""
        # Create engine without starting
        engine = HaystackModelEngine()
        
        # Unload model, expecting RuntimeError
        with self.assertRaises(RuntimeError):
            engine.unload_model("test-model")
    
    def test_infer_not_implemented(self):
        """Test the infer method."""
        # Create and start engine
        engine = HaystackModelEngine()
        engine.start()
        
        # Call infer, expecting NotImplementedError
        with self.assertRaises(NotImplementedError):
            engine.infer("test-model", "test input", "generate")
    
    def test_infer_auto_start_fails(self):
        """Test inference when engine is not running and auto-start fails."""
        # Create engine without starting
        engine = HaystackModelEngine()
        
        # Mock failed ping
        self.mock_client.ping = lambda: "failed"
        
        # Call infer, expecting RuntimeError
        with self.assertRaises(RuntimeError):
            engine.infer("test-model", "test input", "generate")
    
    def test_get_loaded_models_success(self):
        """Test getting loaded models successfully."""
        # Create and start engine
        engine = HaystackModelEngine()
        engine.start()
        
        # Get loaded models
        result = engine.get_loaded_models()
        
        # Verify result
        self.assertEqual(len(result), 2)
        self.assertIn("model1", result)
        self.assertIn("model2", result)
        self.assertEqual(result["model1"]["status"], "loaded")
        self.assertEqual(result["model1"]["engine"], "haystack")
        self.assertEqual(result["model1"]["load_time"], 1620000000)
        
        # Verify correct calls were made
        self.assertEqual(self.mock_client.calls[1][0], "info")
    
    def test_get_loaded_models_not_running(self):
        """Test getting loaded models when engine is not running."""
        # Create engine without starting
        engine = HaystackModelEngine()
        
        # Get loaded models, expecting RuntimeError
        with self.assertRaises(RuntimeError):
            engine.get_loaded_models()
    
    def test_health_check_running(self):
        """Test health check when engine is running."""
        # Create and start engine
        engine = HaystackModelEngine()
        engine.start()
        
        # Check health
        health = engine.health_check()
        
        # Verify health
        self.assertEqual(health["status"], "ok")
        
        # Verify correct calls were made
        self.assertEqual(self.mock_client.calls[1][0], "ping")
    
    def test_health_check_not_running(self):
        """Test health check when engine is not running."""
        # Create engine without starting
        engine = HaystackModelEngine()
        
        # Check health
        health = engine.health_check()
        
        # Verify health
        self.assertEqual(health["status"], "not_running")
    
    def test_health_check_error(self):
        """Test health check when there's an error."""
        # Create and start engine
        engine = HaystackModelEngine()
        engine.start()
        
        # Mock ping to raise exception
        original_ping = self.mock_client.ping
        
        def mock_ping_with_error():
            self.mock_client.calls.append(("ping", {}))
            raise Exception("Test error")
        
        self.mock_client.ping = mock_ping_with_error
        
        # Check health
        health = engine.health_check()
        
        # Verify health
        self.assertEqual(health["status"], "error")
        self.assertEqual(health["error"], "Test error")
        
        # Restore original ping
        self.mock_client.ping = original_ping
    
    def test_full_client_api(self):
        """Test several client API calls in sequence."""
        # Create and start engine
        engine = HaystackModelEngine()
        engine.start()
        
        # Load model
        load_result = engine.load_model("test-model")
        
        # Get loaded models
        models = engine.get_loaded_models()
        
        # Check health
        health = engine.health_check()
        
        # Stop engine
        stop_result = engine.stop()
        
        # Verify results
        self.assertEqual(load_result, "loaded")
        self.assertEqual(len(models), 2)
        self.assertEqual(health["status"], "ok")
        self.assertTrue(stop_result)
        
        # Verify sequence of calls
        call_sequence = [call[0] for call in self.mock_client.calls]
        expected_sequence = ["ping", "load", "info", "ping", "shutdown"]
        self.assertEqual(call_sequence, expected_sequence)


# A separate test class for testing the low-level runtime components
class TestModelClientFunctions(unittest.TestCase):
    """Test the ModelClient class and related functions."""
    
    @patch('src.model_engine.engines.haystack.runtime._ensure_server')
    @patch('socket.socket')
    def test_client_initialization(self, mock_socket, mock_ensure_server):
        """Test client initialization."""
        # Configure mocks
        mock_ensure_server.return_value = True
        mock_socket_instance = MagicMock()
        mock_socket.return_value = mock_socket_instance
        
        # Create client
        client = ModelClient()
        
        # Verify socket was created
        mock_socket.assert_called_once()
    
    @patch('src.model_engine.engines.haystack.runtime._ensure_server')
    @patch('socket.socket')
    def test_client_custom_socket_path(self, mock_socket, mock_ensure_server):
        """Test client initialization with custom socket path."""
        # Configure mocks
        mock_ensure_server.return_value = True
        mock_socket_instance = MagicMock()
        mock_socket.return_value = mock_socket_instance
        
        # Create client with custom path
        custom_path = "/tmp/custom_socket_path"
        client = ModelClient(socket_path=custom_path)
        
        # Verify socket was created
        mock_socket.assert_called_once()
    
    @patch('src.model_engine.engines.haystack.runtime._ensure_server')
    @patch('socket.socket')
    def test_client_ensure_server_called(self, mock_socket, mock_ensure_server):
        """Test ensure_server is called during client initialization."""
        # Configure mocks
        mock_ensure_server.return_value = True
        mock_socket_instance = MagicMock()
        mock_socket.return_value = mock_socket_instance
        
        # Create client
        client = ModelClient()
        
        # Verify ensure_server was called
        mock_ensure_server.assert_called_once()
    
    @patch('src.model_engine.engines.haystack.runtime._ensure_server')
    @patch('socket.socket')
    def test_client_ping(self, mock_socket, mock_ensure_server):
        """Test client ping method."""
        # Configure mocks
        mock_ensure_server.return_value = True
        mock_socket_instance = MagicMock()
        mock_socket.return_value = mock_socket_instance
        mock_socket_instance.recv.return_value = b'{"jsonrpc": "2.0", "result": "pong", "id": 1}'
        
        # Create client
        client = ModelClient()
        
        # Ping the server
        result = client.ping()
        
        # Verify ping worked
        self.assertEqual(result, "pong")
        mock_socket_instance.sendall.assert_called_once()


if __name__ == "__main__":
    unittest.main()
