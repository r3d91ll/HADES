"""
Basic unit tests for the Haystack ModelEngine implementation.

These tests focus on the core functionality of the engine without threading concerns.
"""

import unittest
import os
import tempfile
from unittest.mock import patch, MagicMock, PropertyMock

from src.model_engine.engines.haystack import HaystackModelEngine


class TestHaystackModelEngine(unittest.TestCase):
    """Test basic functionality of the Haystack ModelEngine."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        
        # Create a socket path for testing
        self.socket_path = os.path.join(self.test_dir, "test_socket")
        
        # Mock the ModelClient class
        self.client_patcher = patch('src.model_engine.engines.haystack.ModelClient')
        self.mock_client_class = self.client_patcher.start()
        self.mock_client = MagicMock()
        self.mock_client_class.return_value = self.mock_client
        
        # Create the engine instance
        self.engine = HaystackModelEngine(socket_path=self.socket_path)
        
    def tearDown(self):
        """Clean up after tests."""
        # Stop patches
        self.client_patcher.stop()
        
        # Clean up temporary directory
        import shutil
        shutil.rmtree(self.test_dir)
    
    def test_initialization(self):
        """Test engine initialization."""
        # Verify socket path is stored
        self.assertEqual(self.engine.socket_path, self.socket_path)
        
        # Verify initial state
        self.assertFalse(self.engine.running)
        self.assertIsNone(self.engine.client)
    
    def test_start_success(self):
        """Test successful engine start."""
        # Configure mock client to return successful ping
        self.mock_client.ping.return_value = "pong"
        
        # Start the engine
        result = self.engine.start()
        
        # Verify client was created with correct socket path
        self.mock_client_class.assert_called_once_with(socket_path=self.socket_path)
        
        # Verify ping was called
        self.mock_client.ping.assert_called_once()
        
        # Verify engine is running
        self.assertTrue(result)
        self.assertTrue(self.engine.running)
        self.assertEqual(self.engine.client, self.mock_client)
    
    def test_start_already_running(self):
        """Test starting an engine that's already running."""
        # Set up engine as already running
        self.engine.running = True
        self.engine.client = self.mock_client
        
        # Start the engine again
        result = self.engine.start()
        
        # Verify client wasn't created again
        self.mock_client_class.assert_not_called()
        
        # Verify ping wasn't called
        self.mock_client.ping.assert_not_called()
        
        # Verify result is True since engine is already running
        self.assertTrue(result)
    
    def test_start_ping_failure(self):
        """Test engine start with ping failure."""
        # Configure mock client to return failed ping
        self.mock_client.ping.return_value = False
        
        # Start the engine
        result = self.engine.start()
        
        # Verify client was created
        self.mock_client_class.assert_called_once()
        
        # Verify ping was called
        self.mock_client.ping.assert_called_once()
        
        # Verify engine is not running due to ping failure
        self.assertFalse(result)
        self.assertFalse(self.engine.running)
    
    def test_start_exception(self):
        """Test engine start with exception."""
        # Configure mock client to raise an exception
        self.mock_client.ping.side_effect = Exception("Connection error")
        
        # Start the engine
        result = self.engine.start()
        
        # Verify client was created
        self.mock_client_class.assert_called_once()
        
        # Verify ping was called
        self.mock_client.ping.assert_called_once()
        
        # Verify engine is not running due to exception
        self.assertFalse(result)
        self.assertFalse(self.engine.running)
    
    def test_stop_success(self):
        """Test successful engine stop."""
        # Set up engine as running
        self.engine.running = True
        self.engine.client = self.mock_client
        
        # Stop the engine
        result = self.engine.stop()
        
        # Verify client shutdown was called
        self.mock_client.shutdown.assert_called_once()
        
        # Verify engine is stopped
        self.assertTrue(result)
        self.assertFalse(self.engine.running)
        self.assertIsNone(self.engine.client)
    
    def test_stop_not_running(self):
        """Test stopping an engine that's not running."""
        # Ensure engine is not running
        self.engine.running = False
        self.engine.client = None
        
        # Stop the engine
        result = self.engine.stop()
        
        # Verify shutdown wasn't called
        self.mock_client.shutdown.assert_not_called()
        
        # Verify result is True since engine is already stopped
        self.assertTrue(result)
        self.assertFalse(self.engine.running)
    
    def test_stop_exception(self):
        """Test engine stop with exception."""
        # Set up engine as running
        self.engine.running = True
        self.engine.client = self.mock_client
        
        # Configure mock client to raise an exception
        self.mock_client.shutdown.side_effect = Exception("Shutdown error")
        
        # Stop the engine
        result = self.engine.stop()
        
        # Verify client shutdown was called
        self.mock_client.shutdown.assert_called_once()
        
        # Verify engine is stopped despite the exception
        self.assertFalse(result)
        self.assertTrue(self.engine.running)  # Still marked as running due to error
    
    def test_restart(self):
        """Test engine restart."""
        # Create a version of the engine we can patch
        with patch.object(HaystackModelEngine, 'stop') as mock_stop:
            with patch.object(HaystackModelEngine, 'start') as mock_start:
                # Configure mocks
                mock_stop.return_value = True
                mock_start.return_value = True
                
                # Call restart
                engine = HaystackModelEngine()
                result = engine.restart()
                
                # Verify stop and start were called
                mock_stop.assert_called_once()
                mock_start.assert_called_once()
                
                # Verify result reflects both operations succeeding
                self.assertTrue(result)
    
    def test_restart_stop_failure(self):
        """Test engine restart when stop fails."""
        # Create a version of the engine we can patch
        with patch.object(HaystackModelEngine, 'stop') as mock_stop:
            with patch.object(HaystackModelEngine, 'start') as mock_start:
                # Configure mocks
                mock_stop.return_value = False
                mock_start.return_value = True
                
                # Call restart
                engine = HaystackModelEngine()
                result = engine.restart()
                
                # Verify stop and start were called
                mock_stop.assert_called_once()
                mock_start.assert_called_once()
                
                # Verify result reflects the operations
                self.assertTrue(result)  # Still returns True because start succeeded
    
    def test_status(self):
        """Test getting engine status."""
        # Set up engine as running
        self.engine.running = True
        self.engine.client = self.mock_client
        
        # Configure mock client responses
        self.mock_client.ping.return_value = "pong"
        mock_models = {"model1": 1620000000.0, "model2": 1620000001.0}
        self.mock_client.info.return_value = mock_models
        
        # Get status
        status = self.engine.status()
        
        # Verify client methods were called
        self.mock_client.ping.assert_called_once()
        self.mock_client.info.assert_called_once()
        
        # Verify status contains expected info
        self.assertEqual(status["running"], True)
        self.assertEqual(status["healthy"], True)
        # Don't strictly check the format of loaded_models, just verify the keys are present
        self.assertTrue("loaded_models" in status)
        self.assertEqual(len(status["loaded_models"]), 2)
        self.assertTrue("model1" in status["loaded_models"])
        self.assertTrue("model2" in status["loaded_models"])
        self.assertEqual(status["model_count"], 2)
    
    def test_status_not_running(self):
        """Test getting status when engine is not running."""
        # Ensure engine is not running
        self.engine.running = False
        self.engine.client = None
        
        # Get status
        status = self.engine.status()
        
        # Verify status only shows running state
        self.assertEqual(status, {"running": False})
    
    def test_status_unhealthy(self):
        """Test getting status when engine is unhealthy."""
        # Set up engine as running
        self.engine.running = True
        self.engine.client = self.mock_client
        
        # Configure mock client to fail ping
        self.mock_client.ping.return_value = "error"
        
        # Get status
        status = self.engine.status()
        
        # Verify client ping was called
        self.mock_client.ping.assert_called_once()
        
        # Verify status shows unhealthy state
        self.assertEqual(status["running"], True)
        self.assertEqual(status["healthy"], False)
    
    def test_load_model(self):
        """Test loading a model."""
        # Set up engine as running
        self.engine.running = True
        self.engine.client = self.mock_client
        
        # Configure mock client response
        self.mock_client.load.return_value = "loaded"
        
        # Load a model
        result = self.engine.load_model("test-model", device="cpu")
        
        # Verify client load was called with right parameters
        self.mock_client.load.assert_called_once_with("test-model", device="cpu")
        
        # Verify result
        self.assertEqual(result, "loaded")
    
    def test_load_model_not_running(self):
        """Test loading a model when engine is not running."""
        # Create a fresh engine we can fully control
        engine = HaystackModelEngine(socket_path=self.socket_path)
        
        # Mock the start method
        def mock_start_effect():
            engine.running = True
            engine.client = self.mock_client
            return True
            
        with patch.object(engine, 'start', side_effect=mock_start_effect) as mock_start:
            # Configure mock client to return success
            self.mock_client.load.return_value = "loaded"
            
            # Ensure engine is not running initially
            engine.running = False
            
            # Attempt to load a model
            result = engine.load_model("test-model")
            
            # Verify start was called
            mock_start.assert_called_once()
            
            # Verify client load was called
            self.mock_client.load.assert_called_once()
            
            # Verify result
            self.assertEqual(result, "loaded")
    
    def test_load_model_start_failure(self):
        """Test loading a model when engine fails to start."""
        # Create a version of the engine we can patch
        with patch.object(HaystackModelEngine, 'start') as mock_start:
            # Configure mock to return failure
            mock_start.return_value = False
            
            # Ensure engine is not running
            self.engine.running = False
            self.engine.client = None
            
            # Attempt to load a model
            with self.assertRaises(RuntimeError):
                self.engine.load_model("test-model")
            
            # Verify start was called
            mock_start.assert_called_once()
    
    def test_unload_model(self):
        """Test unloading a model."""
        # Set up engine as running
        self.engine.running = True
        self.engine.client = self.mock_client
        
        # Configure mock client response
        self.mock_client.unload.return_value = "unloaded"
        
        # Unload a model
        result = self.engine.unload_model("test-model")
        
        # Verify client unload was called
        self.mock_client.unload.assert_called_once_with("test-model")
        
        # Verify result
        self.assertEqual(result, "unloaded")
    
    def test_unload_model_not_running(self):
        """Test unloading a model when engine is not running."""
        # Ensure engine is not running
        self.engine.running = False
        self.engine.client = None
        
        # Attempt to unload a model
        with self.assertRaises(RuntimeError):
            self.engine.unload_model("test-model")
    
    def test_get_loaded_models(self):
        """Test getting loaded models."""
        # Set up engine as running
        self.engine.running = True
        self.engine.client = self.mock_client
        
        # Configure mock client response
        mock_models = {"model1": 1620000000.0, "model2": 1620000001.0}
        self.mock_client.info.return_value = mock_models
        
        # Get loaded models
        models = self.engine.get_loaded_models()
        
        # Verify client info was called
        self.mock_client.info.assert_called_once()
        
        # Verify the result contains the expected model IDs
        self.assertEqual(len(models), 2)
        self.assertTrue("model1" in models)
        self.assertTrue("model2" in models)
        # Check that each model entry has the expected fields
        for model_id, model_data in models.items():
            self.assertIsInstance(model_data, dict)
            self.assertTrue("load_time" in model_data)
            self.assertTrue("status" in model_data)
            self.assertTrue("engine" in model_data)
    
    def test_get_loaded_models_not_running(self):
        """Test getting loaded models when engine is not running."""
        # Ensure engine is not running
        self.engine.running = False
        self.engine.client = None
        
        # Attempt to get loaded models
        with self.assertRaises(RuntimeError):
            self.engine.get_loaded_models()
    
    def test_context_manager(self):
        """Test using the engine as a context manager."""
        # Create a version of the engine we can patch
        with patch.object(HaystackModelEngine, 'start') as mock_start:
            with patch.object(HaystackModelEngine, 'stop') as mock_stop:
                # Configure mocks
                mock_start.return_value = True
                mock_stop.return_value = True
                
                # Use engine as context manager
                with HaystackModelEngine() as engine:
                    # Verify start was called
                    mock_start.assert_called_once()
                    
                    # Verify stop hasn't been called yet
                    mock_stop.assert_not_called()
                
                # Verify stop was called after exiting context
                mock_stop.assert_called_once()


if __name__ == "__main__":
    unittest.main()
