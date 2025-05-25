"""
vLLM session manager for HADES-PathRAG.

This module provides a process manager for vLLM that can launch and manage 
vLLM instances as external processes, integrating with the ingestion pipeline.
"""

import asyncio
import logging
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, cast

from src.config.vllm_config import make_vllm_command, VLLMConfig, ModelMode
from src.types.vllm_types import VLLMProcessInfo

# Set up logging
logger = logging.getLogger(__name__)


async def get_vllm_base_url(model_alias: str, mode: ModelMode = ModelMode.INFERENCE) -> str:
    """
    Get the base URL for a vLLM model API endpoint.
    
    This function provides a convenient way to get the base URL for a vLLM model
    without having to manage the session context manually.
    
    Args:
        model_alias: Alias of the model to use
        mode: ModelMode (INFERENCE or INGESTION)
        
    Returns:
        Base URL for the model's API
        
    Raises:
        RuntimeError: If unable to get the base URL
    """
    try:
        # Import these here to avoid circular imports
        from src.config.vllm_config import VLLMConfig
        
        # Load the configuration
        config = VLLMConfig.load_from_yaml()
        
        # Get model details from config
        models = (
            config.inference_models if mode == ModelMode.INFERENCE
            else config.ingestion_models
        )
        model_config = models.get(model_alias)
        
        if not model_config:
            raise ValueError(f"Model {model_alias} not found in configuration")
        
        # Construct the base URL
        base_url = f"http://{config.server.host}:{model_config.port or config.server.port}"
        logger.debug(f"Base URL for model {model_alias}: {base_url}")
        return base_url
        
    except Exception as e:
        logger.error(f"Error getting base URL for model {model_alias}: {e}")
        raise RuntimeError(f"Failed to get base URL for model {model_alias}: {e}") from e


class VLLMProcessManager:
    """
    Manages vLLM processes for the HADES-PathRAG ingestion pipeline.
    
    This class provides methods to launch, monitor, and terminate vLLM instances
    using OS-level processes, allowing for flexible resource allocation and
    on-demand model loading.
    """
    
    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        vllm_executable: str = "/home/todd/bin/vllm"
    ) -> None:
        """
        Initialize the vLLM process manager.
        
        Args:
            config_path: Path to vLLM configuration file
            vllm_executable: Path to vLLM executable
        """
        self.config_path = config_path
        self.vllm_executable = vllm_executable
        
        # Track running processes
        self.processes: Dict[str, VLLMProcessInfo] = {}
        
        # Load configuration
        self.config = VLLMConfig.load_from_yaml(config_path)
        
        # Ensure executable exists
        if not os.path.exists(self.vllm_executable):
            logger.warning(f"vLLM executable not found at {self.vllm_executable}")
    
    def start_model(
        self,
        model_alias: str,
        mode: ModelMode = ModelMode.INFERENCE,
        wait_for_ready: bool = True,
        timeout: float = 60.0,
    ) -> Optional[VLLMProcessInfo]:
        """
        Start a vLLM model instance as an OS-level process.
        
        Args:
            model_alias: Alias of the model to start (from configuration)
            mode: ModelMode (INFERENCE or INGESTION)
            wait_for_ready: Whether to wait for the model to be ready
            timeout: Maximum wait time in seconds
            
        Returns:
            Process information if successful, None otherwise
        """
        process_key = f"{mode.value}_{model_alias}"
        
        # Check if model is already running
        if process_key in self.processes:
            if self._check_process_running(process_key):
                logger.info(f"Model {model_alias} ({mode.value}) already running")
                return self.processes[process_key]
            else:
                # Process died, clean up
                logger.warning(f"Found dead process for {model_alias}, cleaning up")
                self._cleanup_process(process_key)
        
        try:
            # Generate command to start model
            cmd = make_vllm_command(
                model_alias=model_alias,
                mode=mode,
                yaml_path=self.config_path,
                vllm_executable=self.vllm_executable
            )
            
            logger.info(f"Starting vLLM model {model_alias} with command: {' '.join(cmd)}")
            
            # Start the process
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                # Ensure process is killed when parent dies
                preexec_fn=os.setsid
            )
            
            # Get model details from config
            models = (
                self.config.inference_models if mode == ModelMode.INFERENCE
                else self.config.ingestion_models
            )
            model_config = models.get(model_alias)
            
            if not model_config:
                raise ValueError(f"Model {model_alias} not found in configuration")
            
            # Create process info
            process_info = VLLMProcessInfo(
                process=process,
                model_alias=model_alias,
                mode=mode,
                server_url=f"http://{self.config.server.host}:{model_config.port or self.config.server.port}",
                start_time=time.time()
            )
            
            # Track the process
            self.processes[process_key] = process_info
            
            # Wait for model to be ready if requested
            if wait_for_ready:
                if not self._wait_for_model_ready(process_info, timeout):
                    logger.error(f"Timeout waiting for model {model_alias} to be ready")
                    self.stop_model(model_alias, mode)
                    return None
            
            return process_info
            
        except Exception as e:
            logger.error(f"Error starting model {model_alias}: {e}")
            return None
    
    def stop_model(self, model_alias: str, mode: ModelMode = ModelMode.INFERENCE) -> bool:
        """
        Stop a running vLLM model instance.
        
        Args:
            model_alias: Alias of the model to stop
            mode: ModelMode (INFERENCE or INGESTION)
            
        Returns:
            True if successful, False otherwise
        """
        process_key = f"{mode.value}_{model_alias}"
        
        if process_key not in self.processes:
            logger.warning(f"Model {model_alias} ({mode.value}) not found in running processes")
            return False
        
        process_info = self.processes[process_key]
        
        try:
            # Try to terminate gracefully first
            process_info.process.terminate()
            
            # Wait for process to exit
            try:
                process_info.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Force kill if it doesn't exit gracefully
                logger.warning(f"Process for {model_alias} did not exit gracefully, killing")
                # Kill entire process group
                os.killpg(os.getpgid(process_info.process.pid), signal.SIGKILL)
            
            # Clean up
            self._cleanup_process(process_key)
            return True
            
        except Exception as e:
            logger.error(f"Error stopping model {model_alias}: {e}")
            return False
    
    def stop_all(self) -> None:
        """Stop all running vLLM model instances."""
        process_keys = list(self.processes.keys())
        for process_key in process_keys:
            parts = process_key.split('_', 1)
            if len(parts) == 2:
                mode_str, model_alias = parts
                mode = ModelMode(mode_str)
                self.stop_model(model_alias, mode)
    
    def get_running_models(self) -> Dict[str, VLLMProcessInfo]:
        """
        Get information about all running models.
        
        Returns:
            Dictionary mapping process keys to process information
        """
        # Update status of all processes
        running_processes = {}
        for process_key, process_info in self.processes.items():
            if self._check_process_running(process_key):
                running_processes[process_key] = process_info
        
        return running_processes
    
    def _check_process_running(self, process_key: str) -> bool:
        """Check if a process is still running."""
        if process_key not in self.processes:
            return False
        
        process_info = self.processes[process_key]
        return process_info.process.poll() is None
    
    def _cleanup_process(self, process_key: str) -> None:
        """Clean up resources for a process."""
        if process_key in self.processes:
            # Close file descriptors
            process_info = self.processes[process_key]
            if process_info.process.stdout:
                process_info.process.stdout.close()
            if process_info.process.stderr:
                process_info.process.stderr.close()
            
            # Remove from tracking dict
            del self.processes[process_key]
    
    def _wait_for_model_ready(
        self, 
        process_info: VLLMProcessInfo, 
        timeout: float = 60.0
    ) -> bool:
        """
        Wait for a model to be ready by checking its API endpoint.
        
        Args:
            process_info: Process information
            timeout: Maximum wait time in seconds
            
        Returns:
            True if model is ready, False otherwise
        """
        import requests
        from requests.exceptions import RequestException
        
        start_time = time.time()
        
        # Check if process is still running
        if process_info.process.poll() is not None:
            logger.error(f"Process for {process_info.model_alias} exited with code {process_info.process.returncode}")
            return False
        
        # URL to check
        health_url = f"{process_info.server_url}/v1/models"
        
        logger.info(f"Waiting for model {process_info.model_alias} to be ready at {health_url}")
        
        while time.time() - start_time < timeout:
            try:
                # Check if server is responding
                response = requests.get(health_url, timeout=2)
                if response.status_code == 200:
                    logger.info(f"Model {process_info.model_alias} is ready")
                    return True
            except RequestException:
                # Server not ready yet
                pass
            
            # Check if process is still running
            if process_info.process.poll() is not None:
                stdout = process_info.process.stdout.read() if process_info.process.stdout else ""
                stderr = process_info.process.stderr.read() if process_info.process.stderr else ""
                logger.error(
                    f"Process for {process_info.model_alias} exited with code {process_info.process.returncode}\n"
                    f"STDOUT: {stdout}\n"
                    f"STDERR: {stderr}"
                )
                return False
            
            # Wait before trying again
            time.sleep(1)
        
        return False


class VLLMSessionContext:
    """
    Context manager for vLLM sessions.
    
    This class provides a context manager interface for vLLM sessions,
    automatically starting and stopping models when entering and exiting
    the context.
    """
    
    def __init__(
        self,
        model_alias: str,
        mode: ModelMode = ModelMode.INFERENCE,
        manager: Optional[VLLMProcessManager] = None,
        config_path: Optional[Union[str, Path]] = None,
        vllm_executable: str = "/home/todd/bin/vllm"
    ) -> None:
        """
        Initialize the vLLM session context.
        
        Args:
            model_alias: Alias of the model to use
            mode: ModelMode (INFERENCE or INGESTION)
            manager: Optional existing vLLM process manager
            config_path: Path to vLLM configuration file
            vllm_executable: Path to vLLM executable
        """
        self.model_alias = model_alias
        self.mode = mode
        self.own_manager = manager is None
        self.manager = manager or VLLMProcessManager(config_path, vllm_executable)
        self.process_info: Optional[VLLMProcessInfo] = None
        self.server_url: Optional[str] = None
    
    def __enter__(self) -> 'VLLMSessionContext':
        """Enter the context, starting the model."""
        self.process_info = self.manager.start_model(
            model_alias=self.model_alias,
            mode=self.mode,
            wait_for_ready=True
        )
        
        if self.process_info is None:
            raise RuntimeError(f"Failed to start model {self.model_alias}")
        
        self.server_url = self.process_info.server_url
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the context, stopping the model."""
        if self.own_manager:
            # Only stop the specific model if we own the manager
            self.manager.stop_model(self.model_alias, self.mode)
            
    @property
    def base_url(self) -> str:
        """Get the base URL for the model's API."""
        if self.server_url is None:
            raise RuntimeError("Session not started")
        return self.server_url


class AsyncVLLMSessionContext:
    """
    Asynchronous context manager for vLLM sessions.
    
    This class provides an async context manager interface for vLLM sessions,
    similar to VLLMSessionContext but with async support.
    """
    
    def __init__(
        self,
        model_alias: str,
        mode: ModelMode = ModelMode.INFERENCE,
        manager: Optional[VLLMProcessManager] = None,
        config_path: Optional[Union[str, Path]] = None,
        vllm_executable: str = "/home/todd/bin/vllm"
    ) -> None:
        """
        Initialize the async vLLM session context.
        
        Args:
            model_alias: Alias of the model to use
            mode: ModelMode (INFERENCE or INGESTION)
            manager: Optional existing vLLM process manager
            config_path: Path to vLLM configuration file
            vllm_executable: Path to vLLM executable
        """
        self.model_alias = model_alias
        self.mode = mode
        self.own_manager = manager is None
        self.manager = manager or VLLMProcessManager(config_path, vllm_executable)
        self.process_info: Optional[VLLMProcessInfo] = None
        self.server_url: Optional[str] = None
    
    async def __aenter__(self) -> 'AsyncVLLMSessionContext':
        """Enter the context, starting the model asynchronously."""
        # Create a thread to start the model (blocking operation)
        loop = asyncio.get_event_loop()
        self.process_info = await loop.run_in_executor(
            None,
            lambda: self.manager.start_model(
                model_alias=self.model_alias,
                mode=self.mode,
                wait_for_ready=True
            )
        )
        
        if self.process_info is None:
            raise RuntimeError(f"Failed to start model {self.model_alias}")
        
        self.server_url = self.process_info.server_url
        return self
    
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the context, stopping the model asynchronously."""
        if self.own_manager:
            # Only stop the specific model if we own the manager
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.manager.stop_model(self.model_alias, self.mode)
            )
            
    @property
    def base_url(self) -> str:
        """Get the base URL for the model's API."""
        if self.server_url is None:
            raise RuntimeError("Session not started")
        return self.server_url


# Global singleton manager
_global_manager: Optional[VLLMProcessManager] = None


def get_vllm_manager(
    config_path: Optional[Union[str, Path]] = None,
    vllm_executable: str = "/home/todd/bin/vllm"
) -> VLLMProcessManager:
    """
    Get the global vLLM process manager instance.
    
    This function returns a singleton VLLMProcessManager instance,
    creating it if it doesn't exist yet.
    
    Args:
        config_path: Path to vLLM configuration file
        vllm_executable: Path to vLLM executable
        
    Returns:
        Global VLLMProcessManager instance
    """
    global _global_manager
    
    if _global_manager is None:
        _global_manager = VLLMProcessManager(config_path, vllm_executable)
        
    return _global_manager
