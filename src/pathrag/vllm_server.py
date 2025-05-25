"""
vLLM server manager for HADES-PathRAG.

This module provides functionality to manage vLLM server instances,
including starting, stopping, and health checks.
"""

import subprocess
import time
import atexit
import os
import signal
import asyncio
import requests
from typing import Optional, Dict, Any, cast
import logging
from pathlib import Path
from src.config.vllm_config import VLLMConfig, VLLMModelConfig
from src.types.vllm_types import ServerStatusType, ModelMode

logger = logging.getLogger(__name__)

class VLLMServerManager:
    """Manages a vLLM server instance."""
    
    def __init__(self, config: Optional[VLLMConfig] = None):
        """
        Initialize the server manager.
        
        Args:
            config: vLLM configuration, uses default if not provided
        """
        self.config = config or VLLMConfig.load_from_yaml()
        self.process: Optional[subprocess.Popen] = None
        self.current_model_id: Optional[str] = None
        self.server_url = f"http://{self.config.server.host}:{self.config.server.port}"
        
        # Register cleanup on exit
        atexit.register(self.stop_server)
        
    async def ensure_server_running(self, model_alias: str, mode: ModelMode = "inference") -> bool:
        """Ensure a vLLM server is running for the specified model.
        
        Args:
            model_alias: The model alias to ensure is running
            mode: Either 'inference' or 'ingestion'
            
        Returns:
            True if server was started or is already running with the correct model,
            False otherwise
        """
        try:
            # Get model config
            model_config = self.config.get_model_config(model_alias, mode=mode)
            model_id = model_config.model_id
            
            # Check if server is running
            status = await self.check_server_status()
            current_model = status.get("model", "")
            
            # If server is running with the correct model, we're good
            if status.get("running") and current_model == model_id:
                logger.info(f"vLLM server already running with model {model_id}")
                return True
                
            # If server is running with a different model, stop it
            if status.get("running") and current_model != model_id:
                logger.info(f"Stopping vLLM server running with model {current_model}")
                await self.stop_server()
            
            # Start server with the requested model
            logger.info(f"Starting vLLM server with model {model_id}")
            success = await self.start_server(model_id)
            return success
        except Exception as e:
            logger.error(f"Error ensuring vLLM server is running: {str(e)}")
            return False
    
    async def start_server(self, model_id: str) -> bool:
        """
        Start a vLLM server with the specified model.
        
        Args:
            model_id: ID of the model to use
            
        Returns:
            True if the server was started successfully, False otherwise
        """
        # Check if server is already running with this model
        if self.is_server_running():
            if self.current_model_id == model_id:
                return True
            
            # Different model, stop current server
            logger.info(f"Stopping current server with model {self.current_model_id}")
            self.stop_server()
            
        logger.info(f"Starting vLLM server with model {model_id}")
        
        try:
            cmd = [
                "python", "-m", "vllm.entrypoints.openai.api_server",
                "--model", model_id,
                "--tensor-parallel-size", str(self.config.server.tensor_parallel_size),
                "--gpu-memory-utilization", str(self.config.server.gpu_memory_utilization),
                "--host", self.config.server.host,
                "--port", str(self.config.server.port),
                "--dtype", self.config.server.dtype,
            ]
            
            # Add max model length if specified
            if self.config.server.max_model_len:
                cmd.extend(["--max-model-len", str(self.config.server.max_model_len)])
            
            # Start the process in its own process group
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                preexec_fn=os.setsid  # Create a new process group
            )
            
            # Wait for server to start
            max_attempts = 30
            for i in range(max_attempts):
                try:
                    if self.is_server_running():
                        self.current_model_id = model_id
                        logger.info(f"vLLM server running with model {model_id}")
                        return True
                except Exception as e:
                    logger.debug(f"Server not yet running: {str(e)}")
                
                # Check if process has exited
                if self.process.poll() is not None:
                    stdout, stderr = self.process.communicate()
                    logger.error(f"vLLM server failed to start: {stderr}")
                    return False
                
                time.sleep(1)  # Wait before retrying
                
            # If we got here, server didn't start in time
            logger.error("vLLM server failed to start in time")
            self.stop_server()
            return False
            
        except Exception as e:
            logger.error(f"Error starting vLLM server: {str(e)}")
            return False
    
    async def stop_server(self) -> bool:
        """Stop the vLLM server if it's running.
        
        Returns:
            True if the server was stopped or wasn't running, False if stop failed
        """
        if self.process is None:
            return True
        
        try:
            logger.info("Stopping vLLM server")
            # Send SIGTERM to process group
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            except ProcessLookupError:
                logger.warning("Process not found when attempting to stop")
                self.process = None
                self.current_model_id = None
                return True
            
            # Wait for process to terminate
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                # Force kill if it doesn't terminate
                os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                
            self.process = None
            self.current_model_id = None
            return True
            
        except Exception as e:
            logger.error(f"Error stopping vLLM server: {str(e)}")
            return False
    
    def is_server_running(self) -> bool:
        """
        Check if the vLLM server is running.
        
        Returns:
            True if the server is running, False otherwise
        """
        try:
            response = requests.get(f"{self.server_url}/v1/models")
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"Error checking server status: {str(e)}")
            return False
    
    async def check_server_status(self) -> ServerStatusType:
        """Check if the vLLM server is running and which model is loaded.
        
        Returns:
            Dictionary with server status information
        """
        try:
            response = requests.get(f"{self.server_url}/v1/models")
            return {
                "running": response.status_code == 200,
                "model": response.json().get("model", ""),
            }
        except Exception as e:
            logger.debug(f"Error checking server status: {str(e)}")
            return {
                "running": False,
                "model": "",
            }
    
    def get_model_info(self, model_alias: str, mode: ModelMode = "inference") -> Dict[str, Any]:
        """Get model information for a given alias.
        
        Args:
            model_alias: Alias of the model to get information for
            mode: Either 'inference' or 'ingestion'
        
        Returns:
            Dictionary with model information
        """
        model_config = self.config.get_model_config(model_alias, mode=mode)
        return {
            "model_id": model_config.model_id,
            "temperature": model_config.temperature,
            "max_tokens": model_config.max_tokens,
            "top_p": model_config.top_p,
            "top_k": model_config.top_k,
            "context_window": model_config.context_window,
        }

# Global singleton instance
_server_manager = None

def get_server_manager(config: Optional[VLLMConfig] = None) -> VLLMServerManager:
    """
    Get or create the server manager singleton.
    
    Args:
        config: Optional configuration to use
        
    Returns:
        Server manager instance
    """
    global _server_manager
    if _server_manager is None:
        if config is None:
            config = VLLMConfig.load_from_yaml()
        _server_manager = VLLMServerManager(config)
    return _server_manager
