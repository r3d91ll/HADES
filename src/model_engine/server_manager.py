"""
Server manager for vLLM.

This module manages the lifecycle of vLLM servers, including starting,
stopping, and checking the status of model servers.
"""

import asyncio
import logging
import subprocess
import sys
import os
from pathlib import Path
from typing import Dict, Optional, List, Any
import json
import time
import signal
import psutil
import requests

from src.model_engine.adapters.vllm_adapter import start_vllm_server
from src.config.model_config import ModelConfig

# Set up logging
logger = logging.getLogger(__name__)


class ServerManager:
    """Manager for vLLM server instances."""
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        host: str = "localhost",
        base_port: int = 8000
    ):
        """
        Initialize the server manager.
        
        Args:
            config_path: Path to config file (None for default)
            host: Hostname for servers
            base_port: Base port number for servers
        """
        self.config_path = config_path
        self.host = host
        self.base_port = base_port
        self.servers = {}  # type: Dict[str, Dict[str, Any]]
    
    async def ensure_server_running(
        self,
        model_alias: str,
        mode: str = "inference",
        timeout: int = 180
    ) -> bool:
        """
        Ensure a server for the specified model is running.
        
        Args:
            model_alias: Alias of the model from config
            mode: Mode to use the model in ("inference" or "embedding")
            timeout: Timeout for server startup in seconds (default: 180 seconds for large models)
            
        Returns:
            True if server is running, False otherwise
        """
        # Load config
        config = ModelConfig.load_from_yaml(self.config_path)
        model_config = config.get_model_config(model_alias, mode=mode)
        
        # Check if server is already running
        server_url = f"http://{self.host}:{self.base_port}"
        server_key = f"{model_alias}_{mode}"
        
        if server_key in self.servers:
            # Check if still alive
            try:
                response = requests.get(
                    f"{server_url}/v1/models",
                    timeout=5
                )
                if response.status_code == 200:
                    return True
            except:
                # Server not responding, will restart
                pass
        
        # Start new server
        # If tensor_parallel_size is 1, specify a single GPU to use (GPU 0)
        # This helps ensure consistent loading
        cuda_visible_devices = None
        if config.server.tensor_parallel_size == 1:
            cuda_visible_devices = "0"  # Use first GPU by default
        
        command = start_vllm_server(
            model_name=model_config.model_id,
            port=self.base_port,
            # Pass model-specific context window as max_model_len
            max_model_len=model_config.context_window,
            tensor_parallel_size=config.server.tensor_parallel_size,
            gpu_memory_utilization=config.server.gpu_memory_utilization,
            # Add dtype and quantization from server config
            dtype=config.server.dtype if hasattr(config.server, 'dtype') else None,
            quantization=config.server.quantization if hasattr(config.server, 'quantization') else None,
            cuda_visible_devices=cuda_visible_devices,
            use_openai_api=True
        )
        
        # Log the command for debugging
        logging.info(f"Starting vLLM server with command: {command}")
        
        
        # Execute command in background
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Store process info
        self.servers[server_key] = {
            "process": process,
            "pid": process.pid,
            "model": model_config.model_id,
            "url": server_url,
            "started_at": time.time()
        }
        
        # Wait for server to be available
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(
                    f"{server_url}/v1/models",
                    timeout=5
                )
                if response.status_code == 200:
                    logger.info(f"vLLM server for {model_alias} in {mode} mode is now running")
                    return True
            except:
                pass
            
            # Check if process is still running
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                logger.error(f"vLLM server process exited: {stderr}")
                return False
            
            await asyncio.sleep(1)
        
        logger.error(f"Timeout waiting for vLLM server to start")
        return False
    
    def stop_all_servers(self):
        """Stop all running servers."""
        for server_key, server_info in self.servers.items():
            if "process" in server_info:
                try:
                    process = server_info["process"]
                    if process.poll() is None:  # Still running
                        process.terminate()
                        process.wait(timeout=5)
                except:
                    # Force kill if terminate doesn't work
                    try:
                        os.kill(server_info["pid"], signal.SIGKILL)
                    except:
                        pass
        
        self.servers = {}


# Global singleton
_manager = None

def get_server_manager(config_path=None):
    """Get the global server manager instance."""
    global _manager
    if _manager is None:
        _manager = ServerManager(config_path)
    return _manager
