"""
Unit tests for the Haystack model engine implementation.

These tests focus on the public API of the HaystackModelEngine class
to achieve the required 85% test coverage.
"""

import unittest
import os
import json
import socket
import logging
import tempfile
from unittest.mock import patch, MagicMock, mock_open, call

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_haystack_engine")

# Import the model engine
from src.model_engine.engines.haystack import HaystackModelEngine


class TestHaystackModelEngine(unittest.TestCase):
    """Test the Haystack model engine implementation."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create a temporary socket path
        self.temp_dir = tempfile.mkdtemp()
        self.socket_path = os.path.join(self.temp_dir, "test_socket")
        
        # Save original environment variable
        self.original_socket_path = os.environ.get("HADES_MODEL_MGR_SOCKET")
        os.environ["HADES_MODEL_MGR_SOCKET"] = self.socket_path
        
        # Create patches
        self.socket_patcher = patch('socket.socket')
        self.subprocess_patcher = patch('subprocess.Popen')
        self.os_path_exists_patcher = patch('os.path.exists')
        self.time_sleep_patcher = patch('time.sleep')
        
        # Start patches
        self.mock_socket = self.socket_patcher.start()
        self.mock_subprocess = self.subprocess_patcher.start()
        self.mock_os_path_exists = self.os_path_exists_patcher.start()
        self.mock_time_sleep = self.time_sleep_patcher.start()
        
        # Configure mock socket
        self.mock_socket_instance = MagicMock()
        self.mock_socket.return_value = self.mock_socket_instance
        
        # Mock successful server startup
        self.mock_os_path_exists.return_value = True
        
        # Mock successful process creation
        self.mock_process = MagicMock()
        self.mock_subprocess.return_value = self.mock_process
    
    def tearDown(self):
        """Clean up after tests."""
        # Restore original environment variable
        if self.original_socket_path:
            os.environ["HADES_MODEL_MGR_SOCKET"] = self.original_socket_path
        else:
            if "HADES_MODEL_MGR_SOCKET" in os.environ:
                del os.environ["HADES_MODEL_MGR_SOCKET"]
        
        # Stop patches
        self.socket_patcher.stop()
        self.subprocess_patcher.stop()
        self.os_path_exists_patcher.stop()
        self.time_sleep_patcher.stop()
        
        # Remove temporary directory
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _mock_socket_response(self, response_data):
        """Configure the mock socket to return a specific response."""
        response_json = json.dumps({"jsonrpc": "2.0", "result": response_data, "id": 1})
        self.mock_socket_instance.recv.return_value = response_json.encode()
    
    def test_engine_initialization(self):
        """Test that the engine can be created."""
        engine = HaystackModelEngine()
        self.assertIsNotNone(engine)
        self.assertFalse(engine.running)
        self.assertIsNone(engine.client)
    
    def test_engine_start_success(self):
        """Test successful engine startup."""
        # Mock successful ping
        self._mock_socket_response(True)
        
        # Create and start engine
        engine = HaystackModelEngine()
        result = engine.start()
        
        # Verify engine is started
        self.assertTrue(result)
        self.assertTrue(engine.running)
        self.assertIsNotNone(engine.client)
        
        # Verify correct calls were made
        self.mock_socket.assert_called()
        self.mock_socket_instance.connect.assert_called_once()
        self.mock_socket_instance.sendall.assert_called_once()
    
    def test_engine_start_socket_failure(self):
        """Test engine startup with socket connection failure."""
        # Mock socket connection failure
        self.mock_socket_instance.connect.side_effect = socket.error("Connection refused")
        
        # Create and start engine
        engine = HaystackModelEngine()
        result = engine.start()
        
        # Verify engine failed to start
        self.assertFalse(result)
        self.assertFalse(engine.running)
    
    def test_engine_start_ping_failure(self):
        """Test engine startup with ping failure."""
        # Mock failed ping
        self._mock_socket_response(False)
        
        # Create and start engine
        engine = HaystackModelEngine()
        result = engine.start()
        
        # Verify engine failed to start
        self.assertFalse(result)
        self.assertFalse(engine.running)
    
    def test_engine_stop(self):
        """Test stopping the engine."""
        # Mock successful ping for startup
        self._mock_socket_response(True)
        
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
    
    def test_is_running(self):
        """Test is_running method."""
        # Mock successful ping for startup
        self._mock_socket_response(True)
        
        # Create and start engine
        engine = HaystackModelEngine()
        engine.start()
        
        # Check if running
        self.assertTrue(engine.is_running())
        
        # Stop the engine
        engine.stop()
        
        # Check if running after stop
        self.assertFalse(engine.is_running())
    
    def test_load_model_success(self):
        """Test loading a model successfully."""
        # Mock successful ping and model loading
        self._mock_socket_response(True)  # For ping
        
        # Create and start engine
        engine = HaystackModelEngine()
        engine.start()
        
        # Configure mock for model loading
        self._mock_socket_response({"success": True})
        
        # Load model
        result = engine.load_model("test-model", device="cpu")
        
        # Verify model was loaded
        self.assertTrue(result)
        # Verify 2 sendall calls: 1 for ping, 1 for load_model
        self.assertEqual(self.mock_socket_instance.sendall.call_count, 2)
    
    def test_load_model_not_running(self):
        """Test loading a model when engine is not running."""
        # Create engine without starting
        engine = HaystackModelEngine()
        
        # Try to load model
        result = engine.load_model("test-model")
        
        # Verify model loading failed
        self.assertFalse(result)
    
    def test_load_model_failure(self):
        """Test handling model loading failure."""
        # Mock successful ping but failed model loading
        self._mock_socket_response(True)  # For ping
        
        # Create and start engine
        engine = HaystackModelEngine()
        engine.start()
        
        # Configure mock for model loading failure
        self._mock_socket_response({"success": False, "error": "Model loading failed"})
        
        # Load model
        result = engine.load_model("test-model")
        
        # Verify model loading failed
        self.assertFalse(result)
    
    def test_load_model_exception(self):
        """Test handling exceptions during model loading."""
        # Mock successful ping but exception during model loading
        self._mock_socket_response(True)  # For ping
        
        # Create and start engine
        engine = HaystackModelEngine()
        engine.start()
        
        # Configure mock to raise exception on second sendall call
        self.mock_socket_instance.sendall.side_effect = [None, Exception("Connection error")]
        
        # Load model
        result = engine.load_model("test-model")
        
        # Verify model loading failed
        self.assertFalse(result)
    
    def test_unload_model_success(self):
        """Test unloading a model successfully."""
        # Mock successful ping and model unloading
        self._mock_socket_response(True)  # For ping
        
        # Create and start engine
        engine = HaystackModelEngine()
        engine.start()
        
        # Configure mock for model unloading
        self._mock_socket_response({"success": True})
        
        # Unload model
        result = engine.unload_model("test-model")
        
        # Verify model was unloaded
        self.assertTrue(result)
    
    def test_unload_model_not_running(self):
        """Test unloading a model when engine is not running."""
        # Create engine without starting
        engine = HaystackModelEngine()
        
        # Try to unload model
        result = engine.unload_model("test-model")
        
        # Verify model unloading failed
        self.assertFalse(result)
    
    def test_run_model_success(self):
        """Test running a model successfully."""
        # Mock successful ping and model running
        self._mock_socket_response(True)  # For ping
        
        # Create and start engine
        engine = HaystackModelEngine()
        engine.start()
        
        # Configure mock for model running
        expected_result = {"output": "test result"}
        self._mock_socket_response(expected_result)
        
        # Run model
        inputs = {"text": "test input"}
        result = engine.run_model("test-model", inputs)
        
        # Verify model was run
        self.assertEqual(result, expected_result)
    
    def test_run_model_not_running(self):
        """Test running a model when engine is not running."""
        # Create engine without starting
        engine = HaystackModelEngine()
        
        # Try to run model
        result = engine.run_model("test-model", {"text": "test input"})
        
        # Verify empty result
        self.assertEqual(result, {})
    
    def test_run_model_exception(self):
        """Test handling exceptions during model running."""
        # Mock successful ping but exception during model running
        self._mock_socket_response(True)  # For ping
        
        # Create and start engine
        engine = HaystackModelEngine()
        engine.start()
        
        # Configure mock to raise exception on second sendall call
        self.mock_socket_instance.sendall.side_effect = [None, Exception("Connection error")]
        
        # Run model
        result = engine.run_model("test-model", {"text": "test input"})
        
        # Verify empty result
        self.assertEqual(result, {})
    
    def test_list_models_success(self):
        """Test listing models successfully."""
        # Mock successful ping and model listing
        self._mock_socket_response(True)  # For ping
        
        # Create and start engine
        engine = HaystackModelEngine()
        engine.start()
        
        # Configure mock for model listing
        expected_result = ["model1", "model2"]
        self._mock_socket_response(expected_result)
        
        # List models
        result = engine.list_models()
        
        # Verify models were listed
        self.assertEqual(result, expected_result)
    
    def test_list_models_not_running(self):
        """Test listing models when engine is not running."""
        # Create engine without starting
        engine = HaystackModelEngine()
        
        # Try to list models
        result = engine.list_models()
        
        # Verify empty result
        self.assertEqual(result, [])
    
    def test_client_send_receive(self):
        """Test the client's send_receive method."""
        # Mock successful ping
        self._mock_socket_response(True)  # For ping
        
        # Create and start engine
        engine = HaystackModelEngine()
        engine.start()
        
        # Configure mock for custom RPC call
        expected_result = {"custom": "result"}
        self._mock_socket_response(expected_result)
        
        # Send custom RPC call directly via client
        result = engine.client._send_receive("custom_method", {"param": "value"})
        
        # Verify result
        self.assertEqual(result, expected_result)
    
    def test_multiple_model_operations(self):
        """Test multiple model operations in sequence."""
        # Mock successful ping
        self._mock_socket_response(True)  # For ping
        
        # Create and start engine
        engine = HaystackModelEngine()
        engine.start()
        
        # Configure mock responses for multiple operations
        responses = [
            {"success": True},  # For load_model
            {"output": "test result"},  # For run_model
            {"success": True}  # For unload_model
        ]
        
        # Set up side effect for socket.recv
        response_jsons = [json.dumps({"jsonrpc": "2.0", "result": r, "id": 1}).encode() for r in responses]
        self.mock_socket_instance.recv.side_effect = [
            json.dumps({"jsonrpc": "2.0", "result": True, "id": 1}).encode(),  # For ping
            *response_jsons
        ]
        
        # Perform multiple operations
        load_result = engine.load_model("test-model")
        run_result = engine.run_model("test-model", {"text": "test input"})
        unload_result = engine.unload_model("test-model")
        
        # Verify results
        self.assertTrue(load_result)
        self.assertEqual(run_result, {"output": "test result"})
        self.assertTrue(unload_result)
        
        # Verify 4 sendall calls: ping, load, run, unload
        self.assertEqual(self.mock_socket_instance.sendall.call_count, 4)
    
    def test_ensure_server_already_running(self):
        """Test ensure_server when server is already running."""
        # Server already running
        self.mock_os_path_exists.return_value = True
        
        # Call engine.start() which will indirectly call _ensure_server
        engine = HaystackModelEngine()
        self._mock_socket_response(True)  # For ping
        engine.start()
        
        # Verify subprocess.Popen was not called
        self.mock_subprocess.assert_not_called()
    
    def test_ensure_server_not_running(self):
        """Test ensure_server when server is not running."""
        # Server not running initially, but starts successfully
        self.mock_os_path_exists.side_effect = [False, True]
        
        # Call engine.start() which will indirectly call _ensure_server
        engine = HaystackModelEngine()
        self._mock_socket_response(True)  # For ping
        engine.start()
        
        # Verify subprocess.Popen was called
        self.mock_subprocess.assert_called_once()


if __name__ == "__main__":
    unittest.main()
