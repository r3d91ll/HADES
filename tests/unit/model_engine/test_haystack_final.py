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
        self.should_raise = False
        self.raised_exception = None
    
    def ping(self):
        if self.should_raise:
            if self.raised_exception:
                raise self.raised_exception
            raise Exception("Mock ping exception")
        self.calls.append(("ping", {}))
        return "pong"
    
    def shutdown(self):
        if self.should_raise:
            raise Exception("Mock shutdown exception")
        self.calls.append(("shutdown", {}))
        return True
    
    def load(self, model_id, device=None):
        if self.should_raise:
            raise Exception(f"Mock load exception for {model_id}")
        self.calls.append(("load", {"model_id": model_id, "device": device}))
        return "loaded"
    
    def unload(self, model_id):
        if self.should_raise:
            raise Exception(f"Mock unload exception for {model_id}")
        self.calls.append(("unload", {"model_id": model_id}))
        return "unloaded"
    
    def info(self):
        if self.should_raise:
            raise Exception("Mock info exception")
        self.calls.append(("info", {}))
        return {"model1": 1620000000, "model2": 1630000000}
    
    def run(self, model_id, inputs):
        if self.should_raise:
            raise Exception(f"Mock run exception for {model_id}")
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
    
    def test_engine_start_client_creation_failure(self):
        """Test engine startup with client creation failure."""
        # Make client initialization raise an exception
        self.mock_client_class.side_effect = Exception("Failed to create client")
        
        # Create and start engine
        engine = HaystackModelEngine()
        result = engine.start()
        
        # Verify engine failed to start
        self.assertFalse(result)
        self.assertFalse(engine.running)
        
        # Reset mock for other tests
        self.mock_client_class.side_effect = None
    
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
        
        # Reset mock for other tests
        self.mock_client = MockClient()
        self.mock_client_class.return_value = self.mock_client
    
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
    
    def test_restart_with_stop_error(self):
        """Test restarting the engine when stop has an error but start succeeds."""
        # First start the engine to get it running
        engine = HaystackModelEngine()
        engine.start()
        self.assertTrue(engine.running)
        
        # Setup a mock that will fail on shutdown but succeed on ping for restart
        original_shutdown = self.mock_client.shutdown
        self.mock_client.shutdown = lambda: False
        
        # Restart should still succeed even if stop fails
        result = engine.restart()
        
        # Verify restart was successful
        self.assertTrue(result)
        self.assertTrue(engine.running)
        
        # Reset mock for other tests
        self.mock_client = MockClient()
        self.mock_client_class.return_value = self.mock_client
    
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
    
    def test_context_manager_with_exception(self):
        """Test using the engine as a context manager with an exception."""
        try:
            with HaystackModelEngine() as engine:
                # Verify engine is running
                self.assertTrue(engine.running)
                self.assertIsNotNone(engine.client)
                # Raise an exception
                raise ValueError("Test exception")
        except ValueError:
            pass
        
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
        
        # Reset mock client for other tests
        self.mock_client = MockClient()
        self.mock_client_class.return_value = self.mock_client
    
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
        
        # Reset mock client for other tests
        self.mock_client = MockClient()
        self.mock_client_class.return_value = self.mock_client
    
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
        
        # Make ping raise an exception
        self.mock_client.should_raise = True
        self.mock_client.raised_exception = Exception("Test error")
        
        # Check health
        health = engine.health_check()
        
        # Verify health
        self.assertEqual(health["status"], "error")
        self.assertEqual(health["error"], "Test error")
        
        # Reset mock client for other tests
        self.mock_client.should_raise = False
        self.mock_client.raised_exception = None
    
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
    
    def test_engine_error_handling(self):
        """Test engine's error handling during various operations."""
        # Create and start engine
        engine = HaystackModelEngine()
        engine.start()
        
        # Make client raise exceptions
        self.mock_client.should_raise = True
        
        # Test various error scenarios
        with self.assertRaises(Exception):
            engine.load_model("test-model")
        
        with self.assertRaises(Exception):
            engine.unload_model("test-model")
        
        with self.assertRaises(Exception):
            engine.get_loaded_models()
        
        # Reset mock client
        self.mock_client.should_raise = False


if __name__ == "__main__":
    unittest.main()
