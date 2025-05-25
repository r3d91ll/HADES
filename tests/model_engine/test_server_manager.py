"""
Comprehensive tests for the ServerManager class.

This test suite focuses on testing the ServerManager functionality,
including server lifecycle (start, stop, health check) and configuration management.
"""

import pytest
import unittest.mock as mock
import asyncio
import json
import os
import signal
import subprocess
from pathlib import Path
from typing import Dict, Any

import requests
import psutil

from src.model_engine.server_manager import ServerManager
from src.config.model_config import ModelConfig


class TestServerManager:
    """Test suite for ServerManager functionality."""
    
    @pytest.fixture
    def mock_config(self):
        """Mock ModelConfig to provide test configuration."""
        with mock.patch('src.model_engine.server_manager.ModelConfig') as mock_config_class:
            # Create a mock instance
            mock_config = mock.MagicMock()
            
            # Setup the server config
            mock_config.server = mock.MagicMock(
                tensor_parallel_size=1,
                gpu_memory_utilization=0.8
            )
            
            # Setup the get_model_config method
            model_config = mock.MagicMock(
                model_id="test-model",
                tensor_parallel_size=1,
                gpu_memory_utilization=0.8
            )
            mock_config.get_model_config.return_value = model_config
            
            # Make the load_from_yaml method return our mock instance
            mock_config_class.load_from_yaml.return_value = mock_config
            
            yield mock_config
    
    @pytest.fixture
    def mock_subprocess(self):
        """Mock subprocess to avoid actually running commands."""
        with mock.patch('src.model_engine.server_manager.subprocess') as mock_subprocess:
            # Mock the subprocess.Popen
            mock_process = mock.MagicMock()
            mock_process.pid = 12345
            mock_process.poll.return_value = None  # Process is running
            mock_subprocess.Popen.return_value = mock_process
            
            yield mock_subprocess
    
    @pytest.fixture
    def mock_psutil(self):
        """Mock psutil to avoid system calls."""
        with mock.patch('src.model_engine.server_manager.psutil') as mock_psutil:
            # Mock the psutil.Process
            mock_process = mock.MagicMock()
            mock_process.is_running.return_value = True
            mock_process.status.return_value = "running"
            mock_process.children.return_value = []
            
            # Mock the Process constructor to return our mock process
            mock_psutil.Process.return_value = mock_process
            
            yield mock_psutil
    
    @pytest.fixture
    def mock_requests(self):
        """Mock requests to avoid actual HTTP calls."""
        with mock.patch('src.model_engine.server_manager.requests') as mock_requests:
            # For server health check
            mock_response = mock.MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "data": [{"id": "test-model", "object": "model"}]
            }
            mock_requests.get.return_value = mock_response
            
            yield mock_requests
    
    @pytest.fixture
    def mock_asyncio(self):
        """Mock asyncio to avoid actually sleeping."""
        with mock.patch('src.model_engine.server_manager.asyncio') as mock_asyncio:
            # Make asyncio.sleep return immediately
            mock_asyncio.sleep = mock.AsyncMock()
            
            yield mock_asyncio
    
    @pytest.fixture
    def server_manager(self):
        """Create a ServerManager instance for testing."""
        return ServerManager(
            config_path="src/config/model_config.yaml",
            host="localhost",
            base_port=8000
        )
    
    def test_initialization(self, server_manager):
        """Test that ServerManager initializes correctly."""
        assert server_manager.config_path == "src/config/model_config.yaml"
        assert server_manager.host == "localhost"
        assert server_manager.base_port == 8000
        assert server_manager.servers == {}
    
    @pytest.mark.asyncio
    async def test_ensure_server_running_server_exists(
        self, 
        server_manager, 
        mock_config, 
        mock_requests
    ):
        """Test ensuring server is running when it's already tracked."""
        # Add a server to the tracked servers
        server_manager.servers["test_model_inference"] = {
            "process": mock.MagicMock(),
            "port": 8000,
            "url": "http://localhost:8000",
            "model_name": "test-model"
        }
        
        # Check that the server is marked as running
        result = await server_manager.ensure_server_running("test_model", "inference")
        assert result is True
        
        # Verify that we checked the server health
        mock_requests.get.assert_called_once_with(
            "http://localhost:8000/v1/models",
            timeout=5
        )
    
    @pytest.mark.asyncio
    async def test_ensure_server_running_start_new(
        self, 
        server_manager, 
        mock_config, 
        mock_subprocess, 
        mock_requests,
        mock_asyncio
    ):
        """Test starting a new server when none exists."""
        # Set up mock model config
        mock_config.get_model_config.return_value = mock.MagicMock(
            model_id="test-model",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.8
        )
        
        # Set up a successful server start
        result = await server_manager.ensure_server_running("test_model", "inference")
        
        # Verify result
        assert result is True
        
        # Verify that a subprocess was created
        mock_subprocess.Popen.assert_called_once()
        
        # Verify that we tracked the new server
        assert "test_model_inference" in server_manager.servers
        assert server_manager.servers["test_model_inference"]["url"] == "http://localhost:8000"
    
    @pytest.mark.asyncio
    async def test_ensure_server_running_health_check_retry(
        self, 
        server_manager, 
        mock_config, 
        mock_subprocess, 
        mock_requests,
        mock_asyncio
    ):
        """Test health check retries during server startup."""
        # Set up mock model config
        mock_config.get_model_config.return_value = mock.MagicMock(
            model_id="test-model",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.8
        )
        
        # First health check fails, second succeeds
        mock_requests.get.side_effect = [
            requests.exceptions.ConnectionError("Connection refused"),
            mock.MagicMock(status_code=200, json=lambda: {"data": [{"id": "test-model"}]})
        ]
        
        # Attempt to start the server
        result = await server_manager.ensure_server_running("test_model", "inference", timeout=10)
        
        # Verify result
        assert result is True
        
        # Verify that we retried the health check
        assert mock_requests.get.call_count == 2
        assert mock_asyncio.sleep.await_count >= 1
    
    @pytest.mark.asyncio
    async def test_ensure_server_running_timeout(
        self, 
        server_manager, 
        mock_config, 
        mock_subprocess, 
        mock_requests,
        mock_asyncio
    ):
        """Test server startup timeout."""
        # Set up mock model config
        mock_config.get_model_config.return_value = mock.MagicMock(
            model_id="test-model",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.8
        )
        
        # All health checks fail
        mock_requests.get.side_effect = requests.exceptions.ConnectionError("Connection refused")
        
        # Attempt to start the server with a short timeout
        result = await server_manager.ensure_server_running("test_model", "inference", timeout=1)
        
        # Verify result
        assert result is False
    
    def test_stop_all_servers(
        self, 
        server_manager
    ):
        """Test stopping all running servers."""
        # Add multiple servers to the tracked servers
        process1 = mock.MagicMock()
        process1.poll.return_value = None  # Process is running
        
        process2 = mock.MagicMock()
        process2.poll.return_value = None  # Process is running
        
        # Store the processes before calling stop_all_servers
        server_manager.servers["model1_inference"] = {
            "process": process1,
            "pid": 12345,
            "model": "model1",
            "url": "http://localhost:8000"
        }
        server_manager.servers["model2_embedding"] = {
            "process": process2,
            "pid": 12346,
            "model": "model2",
            "url": "http://localhost:8001"
        }
        
        # Stop all servers
        server_manager.stop_all_servers()
        
        # Verify all processes were terminated
        process1.terminate.assert_called_once()
        process2.terminate.assert_called_once()
        
        # Verify that all servers were removed from tracking
        assert len(server_manager.servers) == 0
    

    
    def test_stop_all_servers_with_exception(
        self, 
        server_manager
    ):
        """Test stopping all servers when one raises an exception."""
        # Add multiple servers to the tracked servers
        mock_process1 = mock.MagicMock()
        mock_process1.poll.return_value = None  # Process is running
        mock_process1.terminate.side_effect = Exception("Process won't terminate")
        mock_process1.wait.side_effect = Exception("Wait timeout")
        
        mock_process2 = mock.MagicMock()
        mock_process2.poll.return_value = None  # Process is running
        
        server_manager.servers["model1_inference"] = {
            "process": mock_process1,
            "pid": 12345,
            "model": "model1",
            "url": "http://localhost:8000"
        }
        server_manager.servers["model2_embedding"] = {
            "process": mock_process2,
            "pid": 12346,
            "model": "model2",
            "url": "http://localhost:8001"
        }
        
        # Mock os.kill to avoid actual kill calls
        with mock.patch('os.kill') as mock_kill:
            # Stop all servers
            server_manager.stop_all_servers()
            
            # Verify processes were terminated
            mock_process1.terminate.assert_called_once()
            # Since terminate failed, it should try to kill the process
            mock_kill.assert_called_once_with(12345, signal.SIGKILL)
            mock_process2.terminate.assert_called_once()
        
        # Verify that all servers were removed from tracking
        assert len(server_manager.servers) == 0
    

    

    

    

