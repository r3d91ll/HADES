"""
Unit tests for the Haystack model engine implementation.

This module provides comprehensive tests for the HaystackModelEngine
with a target of at least 85% test coverage.
"""

import unittest
import os
import logging
import tempfile
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_haystack_engine")

# Import the model engine
from src.model_engine.engines.haystack import HaystackModelEngine
from src.model_engine.engines.haystack.runtime import ModelClient, _ensure_server as ensure_server

# Implementation of missing functions needed for tests
def _is_server_running(socket_path: str) -> bool:
    """Check if the model server is running by testing if the socket exists.
    
    Args:
        socket_path: Path to the socket file
        
    Returns:
        True if the socket exists, False otherwise
    """
    return os.path.exists(socket_path)


class TestHaystackModelEngineCore(unittest.TestCase):
    """Test the core functionality of the Haystack model engine."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create a mock server path to avoid socket issues during tests
        self.test_socket_path = os.path.join(tempfile.gettempdir(), "test_haystack_socket")
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
        
        # Remove socket file if it exists
        if os.path.exists(self.test_socket_path):
            os.remove(self.test_socket_path)
    
    def test_engine_creation(self):
        """Test that the engine can be created."""
        engine = HaystackModelEngine()
        self.assertIsNotNone(engine)
        self.assertFalse(engine.running)
        self.assertIsNone(engine.client)
    
    @patch('src.model_engine.engines.haystack.runtime.ModelClient')
    @patch('src.model_engine.engines.haystack.runtime.ensure_server')
    def test_engine_start_success(self, mock_ensure_server, mock_client_class):
        """Test successful engine startup."""
        # Configure mocks
        mock_client_instance = MagicMock()
        mock_client_class.return_value = mock_client_instance
        mock_client_instance.ping.return_value = True
        mock_ensure_server.return_value = True
        
        # Create and start engine
        engine = HaystackModelEngine()
        result = engine.start()
        
        # Verify engine is started
        self.assertTrue(result)
        self.assertTrue(engine.running)
        self.assertIsNotNone(engine.client)
        mock_ensure_server.assert_called_once()
        mock_client_instance.ping.assert_called_once()
    
    @patch('src.model_engine.engines.haystack.runtime.ModelClient')
    @patch('src.model_engine.engines.haystack.runtime.ensure_server')
    def test_engine_start_failure_server(self, mock_ensure_server, mock_client_class):
        """Test engine startup failure due to server not starting."""
        # Configure mocks
        mock_ensure_server.return_value = False
        
        # Create and start engine
        engine = HaystackModelEngine()
        result = engine.start()
        
        # Verify engine failed to start
        self.assertFalse(result)
        self.assertFalse(engine.running)
        self.assertIsNone(engine.client)
        mock_ensure_server.assert_called_once()
    
    @patch('src.model_engine.engines.haystack.runtime.ModelClient')
    @patch('src.model_engine.engines.haystack.runtime.ensure_server')
    def test_engine_start_failure_ping(self, mock_ensure_server, mock_client_class):
        """Test engine startup failure due to ping failure."""
        # Configure mocks
        mock_client_instance = MagicMock()
        mock_client_class.return_value = mock_client_instance
        mock_client_instance.ping.return_value = False
        mock_ensure_server.return_value = True
        
        # Create and start engine
        engine = HaystackModelEngine()
        result = engine.start()
        
        # Verify engine failed to start
        self.assertFalse(result)
        self.assertFalse(engine.running)
        # Client should still be created even if ping fails
        self.assertIsNotNone(engine.client)
        mock_ensure_server.assert_called_once()
        mock_client_instance.ping.assert_called_once()
    
    @patch('src.model_engine.engines.haystack.runtime.ModelClient')
    @patch('src.model_engine.engines.haystack.runtime.ensure_server')
    def test_engine_start_exception(self, mock_ensure_server, mock_client_class):
        """Test engine startup handling exceptions."""
        # Configure mocks
        mock_client_instance = MagicMock()
        mock_client_class.return_value = mock_client_instance
        mock_client_instance.ping.side_effect = Exception("Test exception")
        mock_ensure_server.return_value = True
        
        # Create and start engine
        engine = HaystackModelEngine()
        result = engine.start()
        
        # Verify engine failed to start
        self.assertFalse(result)
        self.assertFalse(engine.running)
    
    @patch('src.model_engine.engines.haystack.runtime.ModelClient')
    def test_engine_stop(self, mock_client_class):
        """Test stopping the engine."""
        # Configure mocks
        mock_client_instance = MagicMock()
        mock_client_class.return_value = mock_client_instance
        
        # Create engine and manually set it as running
        engine = HaystackModelEngine()
        engine.running = True
        engine.client = mock_client_instance
        
        # Stop the engine
        result = engine.stop()
        
        # Verify engine is stopped
        self.assertTrue(result)
        self.assertFalse(engine.running)
        self.assertIsNone(engine.client)
    
    @patch('src.model_engine.engines.haystack.runtime.ModelClient')
    def test_engine_stop_not_running(self, mock_client_class):
        """Test stopping an engine that isn't running."""
        # Create engine that isn't running
        engine = HaystackModelEngine()
        self.assertFalse(engine.running)
        
        # Stop the engine
        result = engine.stop()
        
        # Should return True even though it wasn't running
        self.assertTrue(result)
        self.assertFalse(engine.running)
    
    @patch('src.model_engine.engines.haystack.runtime.ModelClient')
    def test_load_model_success(self, mock_client_class):
        """Test successfully loading a model."""
        # Configure mocks
        mock_client_instance = MagicMock()
        mock_client_class.return_value = mock_client_instance
        mock_client_instance.load_model.return_value = {"success": True}
        
        # Create engine and manually set it as running
        engine = HaystackModelEngine()
        engine.running = True
        engine.client = mock_client_instance
        
        # Load a model
        result = engine.load_model("test-model", device="cpu")
        
        # Verify model was loaded
        self.assertTrue(result)
        mock_client_instance.load_model.assert_called_once_with(
            "test-model", device="cpu", force_reload=False
        )
    
    @patch('src.model_engine.engines.haystack.runtime.ModelClient')
    def test_load_model_failure(self, mock_client_class):
        """Test handling model loading failure."""
        # Configure mocks
        mock_client_instance = MagicMock()
        mock_client_class.return_value = mock_client_instance
        mock_client_instance.load_model.return_value = {"success": False, "error": "Test error"}
        
        # Create engine and manually set it as running
        engine = HaystackModelEngine()
        engine.running = True
        engine.client = mock_client_instance
        
        # Load a model
        result = engine.load_model("test-model")
        
        # Verify model failed to load
        self.assertFalse(result)
    
    @patch('src.model_engine.engines.haystack.runtime.ModelClient')
    def test_load_model_not_running(self, mock_client_class):
        """Test loading a model when engine isn't running."""
        # Create engine that isn't running
        engine = HaystackModelEngine()
        self.assertFalse(engine.running)
        
        # Load a model
        result = engine.load_model("test-model")
        
        # Should fail because engine isn't running
        self.assertFalse(result)
    
    @patch('src.model_engine.engines.haystack.runtime.ModelClient')
    def test_load_model_exception(self, mock_client_class):
        """Test handling exceptions when loading a model."""
        # Configure mocks
        mock_client_instance = MagicMock()
        mock_client_class.return_value = mock_client_instance
        mock_client_instance.load_model.side_effect = Exception("Test exception")
        
        # Create engine and manually set it as running
        engine = HaystackModelEngine()
        engine.running = True
        engine.client = mock_client_instance
        
        # Load a model
        result = engine.load_model("test-model")
        
        # Should fail due to exception
        self.assertFalse(result)
    
    @patch('src.model_engine.engines.haystack.runtime.ModelClient')
    def test_unload_model_success(self, mock_client_class):
        """Test successfully unloading a model."""
        # Configure mocks
        mock_client_instance = MagicMock()
        mock_client_class.return_value = mock_client_instance
        mock_client_instance.unload_model.return_value = {"success": True}
        
        # Create engine and manually set it as running
        engine = HaystackModelEngine()
        engine.running = True
        engine.client = mock_client_instance
        
        # Unload a model
        result = engine.unload_model("test-model")
        
        # Verify model was unloaded
        self.assertTrue(result)
        mock_client_instance.unload_model.assert_called_once_with("test-model")
    
    @patch('src.model_engine.engines.haystack.runtime.ModelClient')
    def test_run_model_success(self, mock_client_class):
        """Test successfully running a model."""
        # Configure mocks
        mock_client_instance = MagicMock()
        mock_client_class.return_value = mock_client_instance
        mock_client_instance.run_model.return_value = {"result": "test-output"}
        
        # Create engine and manually set it as running
        engine = HaystackModelEngine()
        engine.running = True
        engine.client = mock_client_instance
        
        # Run a model
        inputs = {"text": "test-input"}
        result = engine.run_model("test-model", inputs)
        
        # Verify model was run
        self.assertEqual(result, {"result": "test-output"})
        mock_client_instance.run_model.assert_called_once_with("test-model", inputs)
    
    @patch('src.model_engine.engines.haystack.runtime.ModelClient')
    def test_run_model_not_running(self, mock_client_class):
        """Test running a model when engine isn't running."""
        # Create engine that isn't running
        engine = HaystackModelEngine()
        self.assertFalse(engine.running)
        
        # Run a model
        result = engine.run_model("test-model", {"text": "test-input"})
        
        # Should return empty dict because engine isn't running
        self.assertEqual(result, {})
    
    @patch('src.model_engine.engines.haystack.runtime.ModelClient')
    def test_run_model_exception(self, mock_client_class):
        """Test handling exceptions when running a model."""
        # Configure mocks
        mock_client_instance = MagicMock()
        mock_client_class.return_value = mock_client_instance
        mock_client_instance.run_model.side_effect = Exception("Test exception")
        
        # Create engine and manually set it as running
        engine = HaystackModelEngine()
        engine.running = True
        engine.client = mock_client_instance
        
        # Run a model
        result = engine.run_model("test-model", {"text": "test-input"})
        
        # Should return empty dict due to exception
        self.assertEqual(result, {})


class TestHaystackRuntimeFunctions(unittest.TestCase):
    """Test the runtime functions of the Haystack model engine."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create a mock server path to avoid socket issues during tests
        self.test_socket_path = os.path.join(tempfile.gettempdir(), "test_haystack_socket")
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
    
    @patch('os.path.exists')
    def test_is_server_running_true(self, mock_exists):
        """Test checking if server is running when it is."""
        # Configure mock
        mock_exists.return_value = True
        
        # Check if server is running
        result = _is_server_running()
        
        # Server should be running
        self.assertTrue(result)
        mock_exists.assert_called_once_with(self.test_socket_path)
    
    @patch('os.path.exists')
    def test_is_server_running_false(self, mock_exists):
        """Test checking if server is running when it isn't."""
        # Configure mock
        mock_exists.return_value = False
        
        # Check if server is running
        result = _is_server_running()
        
        # Server should not be running
        self.assertFalse(result)
        mock_exists.assert_called_once_with(self.test_socket_path)
    
    @patch('os.path.exists')
    def test_is_server_running_no_socket(self, mock_exists):
        """Test checking if server is running with no socket path."""
        # Configure mock
        mock_exists.return_value = False
        
        # Remove socket path from environment
        if "HADES_MODEL_MGR_SOCKET" in os.environ:
            del os.environ["HADES_MODEL_MGR_SOCKET"]
        
        # Check if server is running
        result = _is_server_running()
        
        # Server should not be running
        self.assertFalse(result)
    
    @patch('src.model_engine.engines.haystack.runtime._is_server_running')
    @patch('subprocess.Popen')
    @patch('sys.executable', '/usr/bin/python')
    def test_ensure_server_not_running(self, mock_popen, mock_is_running):
        """Test ensuring server is running when it isn't."""
        # Configure mocks
        mock_is_running.return_value = False
        mock_process = MagicMock()
        mock_popen.return_value = mock_process
        
        # Mock the builtins.open function for handling imports
        with patch('builtins.open', mock_open(read_data="")) as mock_file:
            # Ensure server is running
            result = ensure_server()
            
            # Server should be started
            self.assertTrue(result)
            mock_is_running.assert_called_once()
            mock_popen.assert_called_once()
    
    @patch('src.model_engine.engines.haystack.runtime._is_server_running')
    @patch('subprocess.Popen')
    def test_ensure_server_already_running(self, mock_popen, mock_is_running):
        """Test ensuring server is running when it already is."""
        # Configure mocks
        mock_is_running.return_value = True
        
        # Ensure server is running
        result = ensure_server()
        
        # Server should already be running
        self.assertTrue(result)
        mock_is_running.assert_called_once()
        mock_popen.assert_not_called()
    
    @patch('src.model_engine.engines.haystack.runtime._is_server_running')
    @patch('subprocess.Popen')
    @patch('time.sleep')
    def test_ensure_server_start_failure(self, mock_sleep, mock_popen, mock_is_running):
        """Test failure to start server."""
        # Configure mocks
        # First call returns False (not running), subsequent calls still return False (failed to start)
        mock_is_running.side_effect = [False, False, False, False]
        mock_process = MagicMock()
        mock_popen.return_value = mock_process
        
        # Mock the builtins.open function for handling imports
        with patch('builtins.open', mock_open(read_data="")) as mock_file:
            # Ensure server is running
            result = ensure_server()
            
            # Server should fail to start
            self.assertFalse(result)
            self.assertEqual(mock_is_running.call_count, 4)  # Initial check + 3 retries
            mock_popen.assert_called_once()
            self.assertEqual(mock_sleep.call_count, 3)  # 3 retries
    
    @patch('src.model_engine.engines.haystack.runtime._is_server_running')
    @patch('subprocess.Popen')
    @patch('time.sleep')
    def test_ensure_server_eventual_success(self, mock_sleep, mock_popen, mock_is_running):
        """Test server eventually starting after retries."""
        # Configure mocks
        # First call returns False (not running), second call returns True (started successfully)
        mock_is_running.side_effect = [False, True]
        mock_process = MagicMock()
        mock_popen.return_value = mock_process
        
        # Mock the builtins.open function for handling imports
        with patch('builtins.open', mock_open(read_data="")) as mock_file:
            # Ensure server is running
            result = ensure_server()
            
            # Server should eventually start
            self.assertTrue(result)
            self.assertEqual(mock_is_running.call_count, 2)  # Initial check + 1 successful retry
            mock_popen.assert_called_once()
            mock_sleep.assert_called_once()  # 1 retry


class TestModelClient(unittest.TestCase):
    """Test the ModelClient class."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create a mock server path to avoid socket issues during tests
        self.test_socket_path = os.path.join(tempfile.gettempdir(), "test_haystack_socket")
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
    
    @patch('socket.socket')
    def test_client_initialization(self, mock_socket):
        """Test client initialization."""
        # Configure mock
        mock_socket_instance = MagicMock()
        mock_socket.return_value = mock_socket_instance
        
        # Create client
        client = ModelClient()
        
        # Verify client is initialized
        self.assertIsNotNone(client)
        mock_socket.assert_called_once()
    
    @patch('socket.socket')
    def test_client_initialization_custom_path(self, mock_socket):
        """Test client initialization with custom socket path."""
        # Configure mock
        mock_socket_instance = MagicMock()
        mock_socket.return_value = mock_socket_instance
        
        # Create client with custom path
        custom_path = "/tmp/custom_socket_path"
        client = ModelClient(socket_path=custom_path)
        
        # Verify client is initialized
        self.assertIsNotNone(client)
        mock_socket.assert_called_once()
    
    @patch('socket.socket')
    def test_client_ping(self, mock_socket):
        """Test client ping method."""
        # Configure mock
        mock_socket_instance = MagicMock()
        mock_socket.return_value = mock_socket_instance
        mock_socket_instance.recv.return_value = b'{"jsonrpc": "2.0", "result": true, "id": 1}'
        
        # Create client
        client = ModelClient()
        
        # Ping the server
        result = client.ping()
        
        # Verify ping worked
        self.assertTrue(result)
        mock_socket_instance.sendall.assert_called_once()
    
    @patch('socket.socket')
    def test_client_load_model(self, mock_socket):
        """Test client load_model method."""
        # Configure mock
        mock_socket_instance = MagicMock()
        mock_socket.return_value = mock_socket_instance
        mock_socket_instance.recv.return_value = b'{"jsonrpc": "2.0", "result": {"success": true}, "id": 1}'
        
        # Create client
        client = ModelClient()
        
        # Load a model
        result = client.load_model("test-model", device="cpu")
        
        # Verify model was loaded
        self.assertEqual(result, {"success": True})
        mock_socket_instance.sendall.assert_called_once()
    
    @patch('socket.socket')
    def test_client_unload_model(self, mock_socket):
        """Test client unload_model method."""
        # Configure mock
        mock_socket_instance = MagicMock()
        mock_socket.return_value = mock_socket_instance
        mock_socket_instance.recv.return_value = b'{"jsonrpc": "2.0", "result": {"success": true}, "id": 1}'
        
        # Create client
        client = ModelClient()
        
        # Unload a model
        result = client.unload_model("test-model")
        
        # Verify model was unloaded
        self.assertEqual(result, {"success": True})
        mock_socket_instance.sendall.assert_called_once()
    
    @patch('socket.socket')
    def test_client_run_model(self, mock_socket):
        """Test client run_model method."""
        # Configure mock
        mock_socket_instance = MagicMock()
        mock_socket.return_value = mock_socket_instance
        mock_socket_instance.recv.return_value = b'{"jsonrpc": "2.0", "result": {"output": "test-output"}, "id": 1}'
        
        # Create client
        client = ModelClient()
        
        # Run a model
        result = client.run_model("test-model", {"text": "test-input"})
        
        # Verify model was run
        self.assertEqual(result, {"output": "test-output"})
        mock_socket_instance.sendall.assert_called_once()
    
    @patch('socket.socket')
    def test_client_list_models(self, mock_socket):
        """Test client list_models method."""
        # Configure mock
        mock_socket_instance = MagicMock()
        mock_socket.return_value = mock_socket_instance
        mock_socket_instance.recv.return_value = b'{"jsonrpc": "2.0", "result": ["model1", "model2"], "id": 1}'
        
        # Create client
        client = ModelClient()
        
        # List models
        result = client.list_models()
        
        # Verify models were listed
        self.assertEqual(result, ["model1", "model2"])
        mock_socket_instance.sendall.assert_called_once()
    
    @patch('socket.socket')
    def test_client_exception_handling(self, mock_socket):
        """Test client exception handling."""
        # Configure mock
        mock_socket_instance = MagicMock()
        mock_socket.return_value = mock_socket_instance
        mock_socket_instance.sendall.side_effect = Exception("Test exception")
        
        # Create client
        client = ModelClient()
        
        # Ping the server (should handle exception)
        result = client.ping()
        
        # Verify ping failed
        self.assertFalse(result)
        mock_socket_instance.sendall.assert_called_once()


if __name__ == "__main__":
    unittest.main()
