"""
Basic unit tests for the Haystack ModelClient implementation.

These tests focus on the core functionality of the client without threading concerns.
"""

import unittest
import os
import json
import socket
import tempfile
from unittest.mock import patch, MagicMock, mock_open

from src.model_engine.engines.haystack.runtime import ModelClient


class TestHaystackModelClient(unittest.TestCase):
    """Test basic functionality of the Haystack ModelClient."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        
        # Mock socket to avoid actual network communication
        self.socket_patcher = patch('socket.socket')
        self.mock_socket_class = self.socket_patcher.start()
        self.mock_socket = MagicMock()
        self.mock_socket_class.return_value.__enter__.return_value = self.mock_socket
        
        # Set up _ensure_server patch
        self.ensure_server_patcher = patch('src.model_engine.engines.haystack.runtime._ensure_server')
        self.mock_ensure_server = self.ensure_server_patcher.start()
        
        # Create a temporary socket path
        self.socket_path = os.path.join(self.test_dir, "test_socket")
        
        # Create the client with our mocked dependencies
        self.client = ModelClient(socket_path=self.socket_path)
        
    def tearDown(self):
        """Clean up after tests."""
        # Stop all patches
        self.socket_patcher.stop()
        self.ensure_server_patcher.stop()
        
        # Clean up temporary directory
        import shutil
        shutil.rmtree(self.test_dir)
    
    def setup_response(self, response_data):
        """Helper to set up the socket mock to return the given response."""
        if isinstance(response_data, dict):
            response_bytes = json.dumps(response_data).encode()
        else:
            response_bytes = response_data
        self.mock_socket.recv.return_value = response_bytes
    
    def test_initialization(self):
        """Test client initialization."""
        # Verify socket path is stored correctly
        self.assertEqual(self.client.socket_path, self.socket_path)
        
        # Verify ensure_server was called
        self.mock_ensure_server.assert_called_once_with(self.socket_path)
    
    def test_initialization_default_path(self):
        """Test client initialization with default socket path."""
        # Use the default path value from the module
        from src.model_engine.engines.haystack.runtime import _DEFAULT_SOCKET_PATH
        
        # Create a client with default path
        with patch('src.model_engine.engines.haystack.runtime._ensure_server') as mock_ensure:
            client = ModelClient()
            
            # Verify the default path was used
            self.assertEqual(client.socket_path, _DEFAULT_SOCKET_PATH)
            mock_ensure.assert_called_once_with(_DEFAULT_SOCKET_PATH)
    
    def test_ping(self):
        """Test the ping method."""
        # Set up mock response
        self.setup_response({"result": "pong"})
        
        # Call the method
        result = self.client.ping()
        
        # Verify the request was sent correctly
        self.mock_socket.sendall.assert_called_once()
        sent_data = json.loads(self.mock_socket.sendall.call_args[0][0].decode())
        self.assertEqual(sent_data, {"action": "ping"})
        
        # Verify the response was processed correctly
        self.assertEqual(result, "pong")
    
    def test_load(self):
        """Test the load method."""
        # Set up mock response
        self.setup_response({"result": "loaded"})
        
        # Call the method
        result = self.client.load("test-model", device="cpu")
        
        # Verify the request was sent correctly
        self.mock_socket.sendall.assert_called_once()
        sent_data = json.loads(self.mock_socket.sendall.call_args[0][0].decode())
        self.assertEqual(sent_data, {"action": "load", "model_id": "test-model", "device": "cpu"})
        
        # Verify the response was processed correctly
        self.assertEqual(result, "loaded")
    
    def test_load_without_device(self):
        """Test the load method without specifying a device."""
        # Set up mock response
        self.setup_response({"result": "loaded"})
        
        # Call the method without device
        result = self.client.load("test-model")
        
        # Verify the request was sent correctly
        self.mock_socket.sendall.assert_called_once()
        sent_data = json.loads(self.mock_socket.sendall.call_args[0][0].decode())
        
        # Device should be None when not specified
        self.assertEqual(sent_data, {"action": "load", "model_id": "test-model", "device": None})
        
        # Verify the response was processed correctly
        self.assertEqual(result, "loaded")
    
    def test_unload(self):
        """Test the unload method."""
        # Set up mock response
        self.setup_response({"result": "unloaded"})
        
        # Call the method
        result = self.client.unload("test-model")
        
        # Verify the request was sent correctly
        self.mock_socket.sendall.assert_called_once()
        sent_data = json.loads(self.mock_socket.sendall.call_args[0][0].decode())
        self.assertEqual(sent_data, {"action": "unload", "model_id": "test-model"})
        
        # Verify the response was processed correctly
        self.assertEqual(result, "unloaded")
    
    def test_info(self):
        """Test the info method."""
        # Set up mock response with model info
        mock_info = {"model1": 1620000000.0, "model2": 1620000001.0}
        self.setup_response({"result": mock_info})
        
        # Call the method
        result = self.client.info()
        
        # Verify the request was sent correctly
        self.mock_socket.sendall.assert_called_once()
        sent_data = json.loads(self.mock_socket.sendall.call_args[0][0].decode())
        self.assertEqual(sent_data, {"action": "info"})
        
        # Verify the response was processed correctly
        self.assertEqual(result, mock_info)
    
    def test_debug(self):
        """Test the debug method."""
        # Set up mock response with debug info
        mock_debug = {
            "cache_size": 2,
            "max_cache_size": 3,
            "keys": ["model1", "model2"],
            "debug": "LRUCache with 2/3 models loaded"
        }
        self.setup_response({"result": mock_debug})
        
        # Call the method
        result = self.client.debug()
        
        # Verify the request was sent correctly
        self.mock_socket.sendall.assert_called_once()
        sent_data = json.loads(self.mock_socket.sendall.call_args[0][0].decode())
        self.assertEqual(sent_data, {"action": "debug"})
        
        # Verify the response was processed correctly
        self.assertEqual(result, mock_debug)
    
    def test_shutdown(self):
        """Test the shutdown method."""
        # Set up mock response
        self.setup_response({"result": "shutdown_initiated"})
        
        # Call the method
        result = self.client.shutdown()
        
        # Verify the request was sent correctly
        self.mock_socket.sendall.assert_called_once()
        sent_data = json.loads(self.mock_socket.sendall.call_args[0][0].decode())
        self.assertEqual(sent_data, {"action": "shutdown"})
        
        # Verify the response was processed correctly
        self.assertEqual(result, "shutdown_initiated")
    
    def test_request_error_handling(self):
        """Test error handling in the _request method."""
        # Set up socket to raise an exception
        self.mock_socket.sendall.side_effect = socket.error("Connection refused")
        
        # Call a method that uses _request and check that it raises RuntimeError
        with self.assertRaises(RuntimeError) as context:
            self.client.ping()
        
        # Verify the error contains the original exception info
        self.assertIn("Connection refused", str(context.exception))
    
    def test_response_error_handling(self):
        """Test handling of error responses from the server."""
        # Set up mock response with an error
        self.setup_response({"error": "Model not found"})
        
        # Calling info should pass through the error response
        result = self.client._request({"action": "info"})
        self.assertEqual(result, {"error": "Model not found"})
        
        # Using the public method should check for errors
        with self.assertRaises(RuntimeError) as context:
            self.client.info()
        
        # Verify the error contains the server error message
        self.assertIn("Model not found", str(context.exception))


if __name__ == "__main__":
    unittest.main()
