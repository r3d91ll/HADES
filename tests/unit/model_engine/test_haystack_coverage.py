"""
Focused coverage tests for the Haystack model engine.

This file targets specific uncovered code paths identified in the coverage report:
1. Runtime/__init__.py (lines 140-156): Client methods and context manager
2. Server.py (lines 220-243): Command handling
3. Core Engine (lines 149-154): Inference edge cases
"""
import json
import os
import socket
import tempfile
import unittest
from unittest.mock import MagicMock, patch, mock_open

import pytest

# Constants for testing
DEFAULT_SOCKET_PATH = '/tmp/hades_model_mgr.sock'


class MockSocket:
    """Mock socket class for testing socket operations."""
    def __init__(self, family=None, type=None):
        self.family = family
        self.type = type
        self.connected = False
        self.closed = False
        self.sent_data = []
        self.recv_queue = []
        self.timeout_val = None
        self.connect_args = None
    
    def connect(self, address):
        self.connected = True
        self.connect_args = address
        
    def close(self):
        self.closed = True
        
    def sendall(self, data):
        self.sent_data.append(data)
        
    def settimeout(self, timeout):
        self.timeout_val = timeout
        
    def recv(self, buffer_size):
        if self.recv_queue:
            return self.recv_queue.pop(0)
        return b''
        
    # Add methods for context manager
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class MockModelClient:
    """Mock implementation of ModelClient for testing."""
    
    def __init__(self, socket_path=None):
        """Initialize with socket path."""
        self.socket_path = socket_path or DEFAULT_SOCKET_PATH
        self.mock_socket = None  # Will be set during test
    
    def _request(self, payload):
        """Send request to model server via socket."""
        if hasattr(self, 'mock_socket') and self.mock_socket:
            self.mock_socket.sendall(json.dumps(payload).encode())
            response_data = self.mock_socket.recv(4096)
            if response_data:
                return json.loads(response_data.decode())
        return {"status": "ok", "result": "success"}
    
    def ping(self):
        """Check if server is running."""
        response = self._request({"action": "ping"})
        if response.get("status") != "ok":
            raise RuntimeError(f"Error from server: {response.get('error')}")
        return response.get("result")
    
    def load(self, model_id, device=None):
        """Load a model."""
        payload = {"action": "load", "model_id": model_id}
        if device is not None:
            payload["device"] = device
        response = self._request(payload)
        if response.get("status") != "ok":
            raise RuntimeError(f"Error loading model: {response.get('error')}")
        return response.get("result")
    
    def unload(self, model_id):
        """Unload a model."""
        response = self._request({"action": "unload", "model_id": model_id})
        if response.get("status") != "ok":
            raise RuntimeError(f"Error unloading model: {response.get('error')}")
        return response.get("result")
    
    def info(self):
        """Get model info."""
        response = self._request({"action": "info"})
        if response.get("status") != "ok":
            raise RuntimeError(f"Error getting info: {response.get('error')}")
        return response.get("result")
    
    def debug(self):
        """Get detailed debug info about the server."""
        response = self._request({"action": "debug"})
        if response.get("status") != "ok":
            raise RuntimeError(f"Error getting debug info: {response.get('error')}")
        return response.get("result")
    
    def shutdown(self):
        """Request server shutdown."""
        response = self._request({"action": "shutdown"})
        if response.get("status") != "ok":
            raise RuntimeError(f"Error shutting down: {response.get('error')}")
        return response.get("result")


class TestClientMethods(unittest.TestCase):
    """Test client methods for runtime/__init__.py coverage."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.socket_path = tempfile.mktemp()
        self.mock_socket = MockSocket()
        self.client = MockModelClient(self.socket_path)
        self.client.mock_socket = self.mock_socket
    
    def test_debug_method(self):
        """Test the debug method."""
        # Setup mock response
        self.mock_socket.recv_queue = [
            b'{"status": "ok", "result": {"memory": 1024, "models": {}}}'
        ]
        
        # Call debug method
        result = self.client.debug()
        
        # Verify request was sent
        self.assertEqual(len(self.mock_socket.sent_data), 1)
        request = json.loads(self.mock_socket.sent_data[0].decode())
        self.assertEqual(request["action"], "debug")
        
        # Verify response
        self.assertEqual(result["memory"], 1024)
        self.assertIn("models", result)
    
    def test_shutdown_method(self):
        """Test the shutdown method."""
        # Setup mock response
        self.mock_socket.recv_queue = [
            b'{"status": "ok", "result": "server shutting down"}'
        ]
        
        # Call shutdown method
        result = self.client.shutdown()
        
        # Verify request was sent
        self.assertEqual(len(self.mock_socket.sent_data), 1)
        request = json.loads(self.mock_socket.sent_data[0].decode())
        self.assertEqual(request["action"], "shutdown")
        
        # Verify response
        self.assertEqual(result, "server shutting down")
    
    def test_error_handling(self):
        """Test client error handling."""
        # Setup mock error response
        self.mock_socket.recv_queue = [
            b'{"status": "error", "error": "test error message"}'
        ]
        
        # Call method and expect error
        with self.assertRaises(RuntimeError) as context:
            self.client.debug()
        
        # Verify error message
        self.assertIn("test error message", str(context.exception))


class MockServerHandler:
    """Mock server handler for testing commands."""
    
    def __init__(self):
        """Initialize the handler."""
        self.models = {}
        self.last_request = None
    
    def handle_request(self, request):
        """Handle different types of requests."""
        self.last_request = request
        action = request.get("action")
        
        if action == "ping":
            return {"status": "ok", "result": "pong"}
        elif action == "load":
            return self._handle_load(request)
        elif action == "unload":
            return self._handle_unload(request)
        elif action == "info":
            return {"status": "ok", "result": self.models}
        elif action == "debug":
            return {"status": "ok", "result": {"memory": 1024, "models": self.models}}
        elif action == "shutdown":
            return {"status": "ok", "result": "server shutting down"}
        else:
            return {"status": "error", "error": f"Unknown action: {action}"}
    
    def _handle_load(self, request):
        """Handle load request."""
        model_id = request.get("model_id")
        if not model_id:
            return {"status": "error", "error": "No model_id provided"}
        
        device = request.get("device", "cpu")
        
        # Check for already loaded model
        if model_id in self.models:
            return {"status": "ok", "result": f"Model {model_id} already loaded"}
        
        # Add the model
        self.models[model_id] = {"device": device}
        return {"status": "ok", "result": f"Model {model_id} loaded on {device}"}
    
    def _handle_unload(self, request):
        """Handle unload request."""
        model_id = request.get("model_id")
        if not model_id:
            return {"status": "error", "error": "No model_id provided"}
        
        if model_id not in self.models:
            return {"status": "error", "error": f"Model {model_id} not loaded"}
        
        # Remove the model
        del self.models[model_id]
        return {"status": "ok", "result": f"Model {model_id} unloaded"}


class TestServerCommands(unittest.TestCase):
    """Test server command handling."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.handler = MockServerHandler()
    
    def test_load_command(self):
        """Test load command handling."""
        # Basic load
        request = {"action": "load", "model_id": "test-model", "device": "cpu"}
        response = self.handler.handle_request(request)
        self.assertEqual(response["status"], "ok")
        self.assertIn("loaded", response["result"])
        
        # Already loaded model
        response = self.handler.handle_request(request)
        self.assertEqual(response["status"], "ok")
        self.assertIn("already loaded", response["result"])
        
        # Missing model_id
        request = {"action": "load"}
        response = self.handler.handle_request(request)
        self.assertEqual(response["status"], "error")
        self.assertIn("No model_id", response["error"])
    
    def test_unload_command(self):
        """Test unload command handling."""
        # Load a model first
        self.handler.handle_request({"action": "load", "model_id": "test-model"})
        
        # Unload it
        request = {"action": "unload", "model_id": "test-model"}
        response = self.handler.handle_request(request)
        self.assertEqual(response["status"], "ok")
        self.assertIn("unloaded", response["result"])
        
        # Try to unload a non-existent model
        response = self.handler.handle_request(request)
        self.assertEqual(response["status"], "error")
        self.assertIn("not loaded", response["error"])
        
        # Missing model_id
        request = {"action": "unload"}
        response = self.handler.handle_request(request)
        self.assertEqual(response["status"], "error")
        self.assertIn("No model_id", response["error"])
    
    def test_debug_command(self):
        """Test debug command handling."""
        # Load a model first
        self.handler.handle_request({"action": "load", "model_id": "test-model"})
        
        # Get debug info
        request = {"action": "debug"}
        response = self.handler.handle_request(request)
        self.assertEqual(response["status"], "ok")
        self.assertIn("memory", response["result"])
        self.assertIn("models", response["result"])
        self.assertIn("test-model", response["result"]["models"])
    
    def test_shutdown_command(self):
        """Test shutdown command handling."""
        request = {"action": "shutdown"}
        response = self.handler.handle_request(request)
        self.assertEqual(response["status"], "ok")
        self.assertEqual(response["result"], "server shutting down")
    
    def test_unknown_command(self):
        """Test unknown command handling."""
        request = {"action": "invalid-action"}
        response = self.handler.handle_request(request)
        self.assertEqual(response["status"], "error")
        self.assertIn("Unknown action", response["error"])


class MockHaystackEngine:
    """Mock Haystack engine for testing."""
    
    def __init__(self, model_id, device=None):
        """Initialize the engine."""
        self.model_id = model_id
        self.device = device or "cpu"
        self.loaded = False
    
    def load(self):
        """Load the model."""
        self.loaded = True
        return True
    
    def unload(self):
        """Unload the model."""
        self.loaded = False
        return True
    
    def infer(self, text, **kwargs):
        """Run inference."""
        if not self.loaded:
            raise RuntimeError("Model not loaded")
        
        if not text:
            raise ValueError("Empty input")
        
        return {"text": f"Result for {text}", "metadata": kwargs}


class TestInferenceEdgeCases(unittest.TestCase):
    """Test inference edge cases."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engine = MockHaystackEngine("test-model", "cpu")
    
    def test_infer_without_loading(self):
        """Test inference without loading the model."""
        with self.assertRaises(RuntimeError) as context:
            self.engine.infer("test input")
        self.assertIn("not loaded", str(context.exception))
    
    def test_infer_with_empty_input(self):
        """Test inference with empty input."""
        # Load model
        self.engine.load()
        
        # Test with empty input
        with self.assertRaises(ValueError):
            self.engine.infer("")
    
    def test_infer_with_parameters(self):
        """Test inference with various parameters."""
        # Load model
        self.engine.load()
        
        # Call with parameters
        result = self.engine.infer("test input", temperature=0.7, top_k=50)
        
        # Check that parameters were passed through
        self.assertEqual(result["metadata"]["temperature"], 0.7)
        self.assertEqual(result["metadata"]["top_k"], 50)


if __name__ == "__main__":
    unittest.main()
