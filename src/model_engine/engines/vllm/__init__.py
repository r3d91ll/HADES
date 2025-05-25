"""vLLM-based model engine for HADES-PathRAG.

This engine uses vLLM for managing models in a separate process. It provides efficient
GPU memory management and high-performance inference for large language models.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from src.model_engine.base import ModelEngine
from src.model_engine.vllm_session import VLLMProcessManager, get_vllm_manager
from src.config.vllm_config import VLLMConfig, ModelMode


class VLLMModelEngine(ModelEngine):
    """vLLM-based model engine implementation.
    
    This engine uses the vLLM framework for managing models, with a process manager
    that communicates over HTTP to ensure efficient GPU memory usage.
    
    The engine can manage multiple models simultaneously, with different model types
    (embedding, completion, etc.) that can be loaded and used concurrently.
    
    Attributes:
        manager: The VLLMProcessManager instance for managing model processes
        config: The VLLMConfig instance with model and server settings
        running: Whether the model manager service is currently running
    """
    
    def __init__(self, config_path: Optional[str] = None) -> None:
        """Initialize the vLLM model engine.
        
        Args:
            config_path: Optional custom path to the vLLM configuration file
        """
        self.config_path = config_path
        self.manager = None
        self.config = VLLMConfig.load_from_yaml(config_path)
        self.running = False
        
    def start(self) -> bool:
        """Start the model engine service.
        
        This will initialize the vLLM process manager if it's not already running.
        
        Returns:
            True if the service was started successfully, False otherwise
        """
        if self.running and self.manager is not None:
            print("[VLLMModelEngine] Service is already running")
            return True
            
        try:
            self.manager = get_vllm_manager(self.config_path)
            self.running = True
            return True
        except Exception as e:
            print(f"[VLLMModelEngine] Failed to start service: {e}")
            self.running = False
            return False
            
    def stop(self) -> bool:
        """Stop the model engine service.
        
        This will shut down all running vLLM model processes and release resources.
        
        Returns:
            True if the service was stopped successfully, False otherwise
        """
        if not self.running or self.manager is None:
            print("[VLLMModelEngine] Service is not running")
            return True
            
        try:
            # Stop all running models
            self.manager.stop_all()
            self.running = False
            self.manager = None
            return True
        except Exception as e:
            print(f"[VLLMModelEngine] Failed to stop service: {e}")
            return False
            
    def restart(self) -> bool:
        """Restart the model engine service.
        
        This will stop and then start the vLLM process manager.
        
        Returns:
            True if the service was restarted successfully, False otherwise
        """
        self.stop()
        return self.start()
        
    def status(self) -> Dict[str, Any]:
        """Get the status of the model engine service.
        
        Returns:
            Dict containing status information including whether the service is running
            and information about loaded models
        """
        status_info = {"running": self.running}
        
        if self.running and self.manager is not None:
            try:
                # Get info about running models
                running_models = self.manager.get_running_models()
                status_info["running_models"] = {
                    key: {
                        "model_alias": info.model_alias,
                        "mode": info.mode.value,
                        "server_url": info.server_url,
                        "start_time": info.start_time
                    } for key, info in running_models.items()
                }
                status_info["model_count"] = len(running_models)
            except Exception as e:
                status_info["error"] = str(e)
                
        return status_info
        
    def __enter__(self):
        """Context manager entry - automatically start the service."""
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - automatically stop the service."""
        self.stop()
    
    def load_model(self, model_id: str, 
                  device: Optional[Union[str, List[str]]] = None,
                  mode: ModelMode = ModelMode.INFERENCE) -> str:
        """Load a model into memory using the vLLM process manager.
        
        Args:
            model_id: The alias of the model to load (from configuration)
            device: The device to load the model onto (e.g., "cuda:0")
                    If None, uses the default device.
            mode: Whether to load the model for inference or ingestion
                    
        Returns:
            Status string with server URL
            
        Raises:
            RuntimeError: If the service is not running or another error occurs
        """
        if not self.running or self.manager is None:
            self.start()
            if not self.running:
                raise RuntimeError("Model engine service is not running")
        
        # Start the model process
        process_info = self.manager.start_model(model_id, mode=mode)
        if process_info is None:
            raise RuntimeError(f"Failed to start model {model_id}")
            
        return process_info.server_url
    
    def unload_model(self, model_id: str, mode: ModelMode = ModelMode.INFERENCE) -> str:
        """Unload a model from memory.
        
        Args:
            model_id: The alias of the model to unload
            mode: Whether the model was loaded for inference or ingestion
            
        Returns:
            Status string ("stopped")
            
        Raises:
            RuntimeError: If the service is not running or another error occurs
        """
        if not self.running or self.manager is None:
            raise RuntimeError("Model engine service is not running")
            
        success = self.manager.stop_model(model_id, mode=mode)
        if not success:
            raise RuntimeError(f"Failed to stop model {model_id}")
            
        return "stopped"
    
    def get_loaded_models(self) -> Dict[str, Dict[str, Any]]:
        """Get information about currently loaded models.
        
        Returns:
            Dictionary mapping model IDs to information about each model
            
        Raises:
            RuntimeError: If the service is not running
        """
        if not self.running or self.manager is None:
            raise RuntimeError("Model engine service is not running")
            
        running_models = self.manager.get_running_models()
        return {
            key: {
                "model_alias": info.model_alias,
                "mode": info.mode.value,
                "server_url": info.server_url,
                "start_time": info.start_time
            } for key, info in running_models.items()
        }
    
    def get_model_url(self, model_id: str, mode: ModelMode = ModelMode.INFERENCE) -> str:
        """Get the URL for a loaded model.
        
        Args:
            model_id: The alias of the model
            mode: Whether the model was loaded for inference or ingestion
            
        Returns:
            Server URL for the model
            
        Raises:
            RuntimeError: If the model is not loaded or the service is not running
        """
        if not self.running or self.manager is None:
            raise RuntimeError("Model engine service is not running")
            
        process_key = f"{mode.value}_{model_id}"
        running_models = self.manager.get_running_models()
        
        if process_key not in running_models:
            # Try to load the model
            return self.load_model(model_id, mode=mode)
            
        return running_models[process_key].server_url
