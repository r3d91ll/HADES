"""
Unit tests for vLLM session manager.

These tests verify the functionality of the vLLM process manager
and session context managers.
"""

import os
import pytest
import time
from unittest.mock import patch, MagicMock

from src.types.vllm_types import ModelMode, VLLMProcessInfo
from src.model_engine.vllm_session import (
    VLLMProcessManager,
    VLLMSessionContext,
    AsyncVLLMSessionContext,
    get_vllm_manager
)


@pytest.fixture
def mock_subprocess():
    """Mock subprocess.Popen."""
    with patch('subprocess.Popen') as mock_popen:
        # Set up the mock process
        mock_process = MagicMock()
        mock_process.poll.return_value = None  # Process is running
        mock_process.returncode = None
        mock_process.stdout = MagicMock()
        mock_process.stderr = MagicMock()
        
        # Make Popen return our mock process
        mock_popen.return_value = mock_process
        
        yield mock_popen, mock_process


@pytest.fixture
def mock_requests():
    """Mock requests for API health checks."""
    with patch('requests.get') as mock_get:
        # Set up mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        yield mock_get


@pytest.fixture
def mock_config():
    """Create mock configuration."""
    with patch('src.config.vllm_config.VLLMConfig') as mock_config_class:
        # Set up mock configuration
        mock_config = MagicMock()
        
        # Add server configuration
        mock_config.server = MagicMock()
        mock_config.server.host = "localhost"
        mock_config.server.port = 8000
        
        # Add model configuration
        mock_inference_model = MagicMock()
        mock_inference_model.port = 8001
        
        mock_ingestion_model = MagicMock()
        mock_ingestion_model.port = 8002
        
        # Mock models dict
        mock_config.inference_models = {"model1": mock_inference_model}
        mock_config.ingestion_models = {"model2": mock_ingestion_model}
        
        # Make load_from_yaml return our mock config
        mock_config_class.load_from_yaml.return_value = mock_config
        
        yield mock_config


@pytest.fixture
def mock_vllm_cmd():
    """Mock the make_vllm_command function."""
    with patch('src.config.vllm_config.make_vllm_command') as mock_cmd:
        # Make the function return a simple command
        mock_cmd.return_value = ["/test/vllm", "serve", "test-model"]
        yield mock_cmd


def test_process_manager_init():
    """Test initialization of the process manager."""
    with patch('src.config.vllm_config.VLLMConfig') as mock_config_class:
        # Configure the mock
        mock_config = MagicMock()
        mock_config_class.load_from_yaml.return_value = mock_config
        
        # Create the manager
        manager = VLLMProcessManager(config_path="/test/config.yaml", vllm_executable="/test/vllm")
        
        # Check initialization
        assert manager.config_path == "/test/config.yaml"
        assert manager.vllm_executable == "/test/vllm"
        assert manager.processes == {}
        assert manager.config == mock_config
        
        # Verify config was loaded
        mock_config_class.load_from_yaml.assert_called_once_with("/test/config.yaml")


def test_start_model(mock_subprocess, mock_requests, mock_config, mock_vllm_cmd):
    """Test starting a model."""
    mock_popen, mock_process = mock_subprocess
    
    # Create the manager
    manager = VLLMProcessManager(vllm_executable="/test/vllm")
    
    # Start a model
    process_info = manager.start_model("model1", ModelMode.INFERENCE)
    
    # Check that the process was started correctly
    assert process_info is not None
    assert process_info['model_alias'] == "model1"
    assert process_info['mode'] == ModelMode.INFERENCE
    assert process_info['server_url'] == "http://localhost:8001"
    assert process_info['process'] == mock_process
    
    # Check that the command was generated
    mock_vllm_cmd.assert_called_once_with(
        model_alias="model1",
        mode=ModelMode.INFERENCE,
        yaml_path=None,
        vllm_executable="/test/vllm"
    )
    
    # Check that Popen was called
    mock_popen.assert_called_once()
    
    # Check that the process was stored
    assert "inference_model1" in manager.processes
    assert manager.processes["inference_model1"] == process_info


def test_stop_model(mock_subprocess, mock_config):
    """Test stopping a model."""
    mock_popen, mock_process = mock_subprocess
    
    # Create the manager
    manager = VLLMProcessManager()
    
    # Add a mock process
    process_info = VLLMProcessInfo(
        process=mock_process,
        model_alias="model1",
        mode=ModelMode.INFERENCE,
        server_url="http://localhost:8001",
        start_time=time.time()
    )
    manager.processes["inference_model1"] = process_info
    
    # Stop the model
    result = manager.stop_model("model1", ModelMode.INFERENCE)
    
    # Check that the process was terminated
    assert result is True
    mock_process.terminate.assert_called_once()
    mock_process.wait.assert_called_once()
    
    # Check that the process was removed
    assert "inference_model1" not in manager.processes


def test_stop_all(mock_subprocess, mock_config):
    """Test stopping all models."""
    mock_popen, mock_process = mock_subprocess
    
    # Create the manager
    manager = VLLMProcessManager()
    
    # Add two mock processes
    process_info1 = VLLMProcessInfo(
        process=mock_process,
        model_alias="model1",
        mode=ModelMode.INFERENCE,
        server_url="http://localhost:8001",
        start_time=time.time()
    )
    manager.processes["inference_model1"] = process_info1
    
    process_info2 = VLLMProcessInfo(
        process=mock_process,
        model_alias="model2",
        mode=ModelMode.INGESTION,
        server_url="http://localhost:8002",
        start_time=time.time()
    )
    manager.processes["ingestion_model2"] = process_info2
    
    # Stop all models
    manager.stop_all()
    
    # Check that terminate was called for both processes
    assert mock_process.terminate.call_count == 2
    
    # Check that all processes were removed
    assert len(manager.processes) == 0


def test_context_manager(mock_subprocess, mock_requests, mock_config, mock_vllm_cmd):
    """Test the context manager interface."""
    mock_popen, mock_process = mock_subprocess
    
    # Create and use the context manager
    with VLLMSessionContext("model1", ModelMode.INFERENCE) as session:
        # Check that the model was started
        assert session.server_url == "http://localhost:8001"
        assert session.base_url == "http://localhost:8001"
        
        # Check that the process was stored in the manager
        assert "inference_model1" in session.manager.processes
    
    # Check that the process was terminated upon exit
    mock_process.terminate.assert_called_once()


def test_get_vllm_manager():
    """Test the global manager singleton."""
    with patch('src.model_engine.vllm_session.VLLMProcessManager') as mock_manager_class:
        # Call get_vllm_manager twice
        manager1 = get_vllm_manager()
        manager2 = get_vllm_manager()
        
        # Check that the manager was only created once
        mock_manager_class.assert_called_once()
        
        # Check that the same instance was returned both times
        assert manager1 == manager2


def test_process_already_running(mock_subprocess, mock_config):
    """Test starting a model that's already running."""
    mock_popen, mock_process = mock_subprocess
    
    # Create the manager
    manager = VLLMProcessManager()
    
    # Add a mock process
    process_info = VLLMProcessInfo(
        process=mock_process,
        model_alias="model1",
        mode=ModelMode.INFERENCE,
        server_url="http://localhost:8001",
        start_time=time.time()
    )
    manager.processes["inference_model1"] = process_info
    
    # Try to start the same model again
    with patch('src.config.vllm_config.make_vllm_command') as mock_cmd:
        new_process_info = manager.start_model("model1", ModelMode.INFERENCE)
    
    # Check that no new process was started
    mock_cmd.assert_not_called()
    assert new_process_info == process_info


def test_wait_for_model_ready_timeout(mock_subprocess, mock_config):
    """Test waiting for a model to be ready with timeout."""
    mock_popen, mock_process = mock_subprocess
    
    # Create the manager
    manager = VLLMProcessManager()
    
    # Create a process info
    process_info = VLLMProcessInfo(
        process=mock_process,
        model_alias="model1",
        mode=ModelMode.INFERENCE,
        server_url="http://localhost:8001",
        start_time=time.time()
    )
    
    # Mock requests.get to always raise an exception (server not ready)
    with patch('requests.get', side_effect=Exception("Connection error")):
        # Wait for the model to be ready with a short timeout
        result = manager._wait_for_model_ready(process_info, timeout=0.1)
    
    # Check that the wait timed out
    assert result is False


def test_process_died_during_wait(mock_subprocess, mock_config):
    """Test handling a process that dies during wait."""
    mock_popen, mock_process = mock_subprocess
    
    # Make the process appear to have died
    mock_process.poll.return_value = 1
    mock_process.returncode = 1
    
    # Create the manager
    manager = VLLMProcessManager()
    
    # Create a process info
    process_info = VLLMProcessInfo(
        process=mock_process,
        model_alias="model1",
        mode=ModelMode.INFERENCE,
        server_url="http://localhost:8001",
        start_time=time.time()
    )
    
    # Wait for the model to be ready
    result = manager._wait_for_model_ready(process_info)
    
    # Check that the wait detected that the process died
    assert result is False


@pytest.mark.asyncio
async def test_async_context_manager(mock_subprocess, mock_requests, mock_config, mock_vllm_cmd):
    """Test the async context manager interface."""
    mock_popen, mock_process = mock_subprocess
    
    # Create and use the async context manager
    async with AsyncVLLMSessionContext("model1", ModelMode.INFERENCE) as session:
        # Check that the model was started
        assert session.server_url == "http://localhost:8001"
        assert session.base_url == "http://localhost:8001"
        
        # Check that the process was stored in the manager
        assert "inference_model1" in session.manager.processes
    
    # Check that the process was terminated upon exit
    mock_process.terminate.assert_called_once()
