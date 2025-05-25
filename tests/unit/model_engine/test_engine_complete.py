"""
Unit tests for the model engine module.

This test suite covers both the HaystackModelEngine class and the supporting 
runtime components to achieve the required 85% test coverage, following the
project's standard testing protocol.
"""

import unittest
import os
import sys
import json
import socket
import logging
import tempfile
import time
from typing import Dict, Any, List, Optional, Union
from unittest.mock import patch, MagicMock, mock_open, call

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_engine_complete")

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


class TestHaystackEngine(unittest.TestCase):
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
    

class TestRuntimeModule(unittest.TestCase):
    """Test the runtime module components."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create a temporary socket path
        self.temp_dir = tempfile.mkdtemp()
        self.socket_path = os.path.join(self.temp_dir, "test_socket")
        
        # Save original environment variable
        self.original_socket_path = os.environ.get("HADES_MODEL_MGR_SOCKET")
        os.environ["HADES_MODEL_MGR_SOCKET"] = self.socket_path
        
        # Create patches
        self.socket_patcher = patch('src.model_engine.engines.haystack.runtime.server.socket.socket')
        self.subprocess_patcher = patch('src.model_engine.engines.haystack.runtime.subprocess.Popen')
        self.os_path_exists_patcher = patch('src.model_engine.engines.haystack.runtime.os.path.exists')
        self.time_sleep_patcher = patch('src.model_engine.engines.haystack.runtime.time.sleep')
        
        # Start patches
        self.mock_socket = self.socket_patcher.start()
        self.mock_subprocess = self.subprocess_patcher.start()
        self.mock_os_path_exists = self.os_path_exists_patcher.start()
        self.mock_time_sleep = self.time_sleep_patcher.start()
    
    def tearDown(self):
        """Clean up after tests."""
        # Stop patches
        self.socket_patcher.stop()
        self.subprocess_patcher.stop()
        self.os_path_exists_patcher.stop()
        self.time_sleep_patcher.stop()
        
        # Restore original environment variable
        if self.original_socket_path:
            os.environ["HADES_MODEL_MGR_SOCKET"] = self.original_socket_path
        else:
            if "HADES_MODEL_MGR_SOCKET" in os.environ:
                del os.environ["HADES_MODEL_MGR_SOCKET"]
        
        # Remove temporary directory
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('src.model_engine.engines.haystack.runtime._is_server_running')
    def test_ensure_server_already_running(self, mock_is_running):
        """Test _ensure_server when server is already running."""
        from src.model_engine.engines.haystack.runtime import _ensure_server
        
        # Configure mock
        mock_is_running.return_value = True
        
        # Call ensure_server
        result = _ensure_server()
        
        # Verify result
        self.assertTrue(result)
        mock_is_running.assert_called_once()
        self.mock_subprocess.assert_not_called()
    
    @patch('src.model_engine.engines.haystack.runtime._is_server_running')
    def test_ensure_server_starts_successfully(self, mock_is_running):
        """Test _ensure_server when server starts successfully."""
        from src.model_engine.engines.haystack.runtime import _ensure_server
        
        # Configure mocks
        mock_is_running.side_effect = [False, True]  # Not running, then running after start
        
        # Mock process
        mock_process = MagicMock()
        self.mock_subprocess.return_value = mock_process
        
        # Call ensure_server
        result = _ensure_server()
        
        # Verify result
        self.assertTrue(result)
        self.assertEqual(mock_is_running.call_count, 2)
        self.mock_subprocess.assert_called_once()
        self.mock_time_sleep.assert_called_once()
    
    @patch('src.model_engine.engines.haystack.runtime._is_server_running')
    def test_ensure_server_fails_to_start(self, mock_is_running):
        """Test _ensure_server when server fails to start."""
        from src.model_engine.engines.haystack.runtime import _ensure_server
        
        # Configure mocks
        mock_is_running.return_value = False  # Never running
        
        # Mock process
        mock_process = MagicMock()
        self.mock_subprocess.return_value = mock_process
        
        # Call ensure_server
        result = _ensure_server(max_retries=2)
        
        # Verify result
        self.assertFalse(result)
        self.assertEqual(mock_is_running.call_count, 3)  # Initial check + 2 retries
        self.mock_subprocess.assert_called_once()
        self.assertEqual(self.mock_time_sleep.call_count, 2)
    
    def test_is_server_running_true(self):
        """Test _is_server_running when server is running."""
        from src.model_engine.engines.haystack.runtime import _is_server_running
        
        # Configure mock
        self.mock_os_path_exists.return_value = True
        
        # Call is_server_running
        result = _is_server_running()
        
        # Verify result
        self.assertTrue(result)
        self.mock_os_path_exists.assert_called_once_with(self.socket_path)
    
    def test_is_server_running_false(self):
        """Test _is_server_running when server is not running."""
        from src.model_engine.engines.haystack.runtime import _is_server_running
        
        # Configure mock
        self.mock_os_path_exists.return_value = False
        
        # Call is_server_running
        result = _is_server_running()
        
        # Verify result
        self.assertFalse(result)
        self.mock_os_path_exists.assert_called_once_with(self.socket_path)
    
    def test_model_client_init(self):
        """Test ModelClient initialization."""
        # Setup mock socket
        mock_socket_instance = MagicMock()
        self.mock_socket.return_value = mock_socket_instance
        
        # Configure path exists to simulate server already running
        self.mock_os_path_exists.return_value = True
        
        # Prevent actual socket connection
        mock_socket_instance.connect = MagicMock()
        
        # Import here to ensure mocks are applied
        from src.model_engine.engines.haystack.runtime import ModelClient
        
        # Create client
        client = ModelClient(socket_path=self.socket_path)
        
        # Verify socket was created
        self.mock_socket.assert_called()
    
    def test_model_client_rpc_request(self):
        """Test ModelClient._request method."""
        # Setup mock socket
        mock_socket_instance = MagicMock()
        self.mock_socket.return_value = mock_socket_instance
        
        # Configure path exists to simulate server already running
        self.mock_os_path_exists.return_value = True
        
        # Prepare mock response
        response_json = {"jsonrpc": "2.0", "result": "test_response", "id": 1}
        mock_socket_instance.recv.return_value = json.dumps(response_json).encode()
        
        # Import here to ensure mocks are applied
        from src.model_engine.engines.haystack.runtime import ModelClient
        
        # Create client and prevent actual socket connection
        client = ModelClient(socket_path=self.socket_path)
        client.socket.connect = MagicMock()
        
        # Test protected _request method directly (for coverage)
        result = client._request({"action": "test"})
        
        # Verify result
        self.assertEqual(result, response_json)
        mock_socket_instance.sendall.assert_called_once()
    
    def test_model_client_ping(self):
        """Test ModelClient.ping method."""
        # Setup mock socket
        mock_socket_instance = MagicMock()
        self.mock_socket.return_value = mock_socket_instance
        
        # Configure path exists to simulate server already running
        self.mock_os_path_exists.return_value = True
        
        # Prepare mock response
        response_json = {"jsonrpc": "2.0", "result": "pong", "id": 1}
        mock_socket_instance.recv.return_value = json.dumps(response_json).encode()
        
        # Import here to ensure mocks are applied
        from src.model_engine.engines.haystack.runtime import ModelClient
        
        # Create client and prevent actual socket connection
        client = ModelClient(socket_path=self.socket_path)
        client.socket.connect = MagicMock()
        
        # Test ping method
        result = client.ping()
        
        # Verify result
        self.assertEqual(result, "pong")
        mock_socket_instance.sendall.assert_called_once()
    
    def test_model_client_load(self):
        """Test ModelClient.load method."""
        # Setup mock socket
        mock_socket_instance = MagicMock()
        self.mock_socket.return_value = mock_socket_instance
        
        # Configure path exists to simulate server already running
        self.mock_os_path_exists.return_value = True
        
        # Prepare mock response
        response_json = {"jsonrpc": "2.0", "result": "loaded", "id": 1}
        mock_socket_instance.recv.return_value = json.dumps(response_json).encode()
        
        # Import here to ensure mocks are applied
        from src.model_engine.engines.haystack.runtime import ModelClient
        
        # Create client and prevent actual socket connection
        client = ModelClient(socket_path=self.socket_path)
        client.socket.connect = MagicMock()
        
        # Test load method
        result = client.load("test-model", device="cpu")
        
        # Verify result
        self.assertEqual(result, "loaded")
        mock_socket_instance.sendall.assert_called_once()


if __name__ == "__main__":
    unittest.main()
