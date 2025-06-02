"""vLLM-based model engine for HADES-PathRAG.

This engine uses vLLM for managing models in a separate process. It provides efficient
GPU memory management and high-performance inference for large language models.
"""
from __future__ import annotations

import asyncio
import logging
import requests  # type: ignore[import-untyped]
from typing import Any, Dict, List, Optional, Type, Union, cast

from src.config.vllm_config import VLLMConfig, ModelMode
from src.model_engine.base import ModelEngine
from src.model_engine.vllm_session import get_vllm_manager, VLLMProcessManager


# Provide a wrapper that implements the ModelEngine interface
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
        self.manager: Optional[VLLMProcessManager] = None
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
        
        # Service not running, need to start it
        try:
            # Fix type incompatibility by using type assertion
            manager = get_vllm_manager(self.config_path)
            self.manager = manager  # Assign after successful creation to avoid None references
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
        
        # Service is running, try to stop it
        assert self.manager is not None  # For type checker
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
        status_info: Dict[str, Any] = {"running": self.running}
        
        if self.running and self.manager is not None:
            assert self.manager is not None  # For type checker
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
        
    def __enter__(self) -> 'VLLMModelEngine':
        """Enter the context manager.
        
        This will start the model engine service if it's not already running.
        
        Returns:
            Self for use in a context manager
        """
        self.start()
        return self
        
    def __exit__(self, exc_type: Optional[Type[BaseException]], 
                 exc_val: Optional[BaseException], 
                 exc_tb: Optional[Any]) -> None:
        """Exit the context manager.
        
        This will stop the model engine service if it's running.
        
        Args:
            exc_type: Exception type if an exception was raised in the context
            exc_val: Exception value if an exception was raised in the context
            exc_tb: Exception traceback if an exception was raised in the context
        """
        self.stop()
    
    def load_model(self, model_id: str, device: Optional[Union[str, List[str]]] = None) -> str:
        """Load a model into memory.
        
        This will start a new vLLM process for the specified model if it's not already loaded.
        
        Args:
            model_id: The ID of the model to load
            device: Optional device or devices to load the model on (e.g., "cuda:0")
            
        Returns:
            URL to the model's API endpoint
            
        Raises:
            RuntimeError: If the model engine service is not running
        """
        if not self.running:
            raise RuntimeError("Model engine service is not running")
        
        if self.manager is None:
            raise RuntimeError("Process manager is not initialized")
        
        # Assert manager is not None for type checker
        assert self.manager is not None
        
        try:
            # Start the model if it's not already running
            # Note: device parameter is ignored since VLLMProcessManager.start_model doesn't accept it
            # Device configuration should be in the VLLMConfig
            model_info = self.manager.start_model(model_id)
            
            # Check if model_info is None before accessing server_url
            if model_info is None:
                return f"Error: Failed to start model {model_id}"
                
            return model_info.server_url
        except Exception as e:
            print(f"[VLLMModelEngine] Failed to load model {model_id}: {e}")
            # Return a string to match return type annotation
            return f"Error: {e}"
    
    def unload_model(self, model_id: str) -> str:
        """Unload a model from memory.
        
        This will stop the vLLM process for the specified model if it's running.
        
        Args:
            model_id: The ID of the model to unload
            
        Returns:
            Status string ("unloaded")
            
        Raises:
            RuntimeError: If the model engine service is not running
        """
        if not self.running:
            raise RuntimeError("Model engine service is not running")
        
        if self.manager is None:
            raise RuntimeError("Process manager is not initialized")
        
        # Assert manager is not None for type checker
        assert self.manager is not None
        
        try:
            # Stop the model process if it's running
            self.manager.stop_model(model_id)
            return "unloaded"
        except Exception as e:
            print(f"[VLLMModelEngine] Failed to unload model {model_id}: {e}")
            # Still return the expected string to match the return type
            return f"Error: {e}"
    
    def get_loaded_models(self) -> Dict[str, Dict[str, Any]]:
        """Get information about loaded models.
        
        Returns:
            Dictionary mapping model IDs to model information
            
        Raises:
            RuntimeError: If the model engine service is not running
        """
        if not self.running:
            raise RuntimeError("Model engine service is not running")
        
        if self.manager is None:
            return {}
        
        # Manager is not None, get running models
        running_models = self.manager.get_running_models()
        
        # Convert to dictionary format expected by the interface
        return {
            model_id: {
                "model_id": model_id,
                "alias": info.model_alias,
                "mode": info.mode.value,
                "server_url": info.server_url,
                "start_time": info.start_time
            }
            for model_id, info in running_models.items()
        }
    
    def get_model_url(self, model_id: str) -> str:
        """Get the URL for a loaded model's API endpoint.
        
        Args:
            model_id: The ID of the model
            
        Returns:
            URL to the model's API endpoint
            
        Raises:
            RuntimeError: If the model engine service is not running or the model is not loaded
        """
        if not self.running:
            raise RuntimeError("Model engine service is not running")
        
        if self.manager is None:
            raise RuntimeError("Process manager is not initialized")
        
        # Assert manager is not None for type checker
        assert self.manager is not None
        
        # Check if the model is loaded
        running_models = self.manager.get_running_models()
        
        if model_id not in running_models:
            # Try to load the model
            print(f"[VLLMModelEngine] Model {model_id} not loaded, attempting to load")
            return self.load_model(model_id)
        
        # Return the URL for the model's API endpoint
        return running_models[model_id].server_url
    
    def infer(self, model_id: str, inputs: Union[str, List[str]], 
              task: str, **kwargs: Any) -> Dict[str, Any]:
        """Run inference with a model.
        
        Args:
            model_id: The ID of the model to use
            inputs: Input text or list of texts
            task: The task to perform (e.g., "embedding", "generation", "chat")
            **kwargs: Additional task-specific parameters
            
        Returns:
            Task-specific outputs as a dictionary
            
        Raises:
            ValueError: If the task type is not supported
            RuntimeError: If the model service is not running
        """
        if not self.running:
            raise RuntimeError("Model engine service is not running")
        
        # Ensure model is loaded
        model_url = self.get_model_url(model_id)
        
        # Handle different task types
        task_lower = task.lower()
        
        if task_lower == "embedding":
            # Convert inputs to list if it's a single string
            text_inputs: List[str] = [inputs] if isinstance(inputs, str) else inputs
            
            # Direct API call to the vLLM server for embeddings
            try:
                response = requests.post(
                    f"{model_url}/v1/embeddings",
                    json={
                        "model": model_id,
                        "input": text_inputs,
                        **kwargs
                    }
                )
                response.raise_for_status()
                result = response.json()
                return {"embeddings": [item["embedding"] for item in result["data"]]}
            except Exception as e:
                return {"error": str(e)}
            
        elif task_lower == "generation" or task_lower == "completion":
            # For text generation tasks
            # Convert inputs to single string if it's a list
            text_input: str = inputs[0] if isinstance(inputs, list) else inputs
            
            # Direct API call to the vLLM server for completions
            try:
                response = requests.post(
                    f"{model_url}/v1/completions",
                    json={
                        "model": model_id,
                        "prompt": text_input,
                        "max_tokens": kwargs.get("max_tokens", 100),
                        "temperature": kwargs.get("temperature", 0.7),
                        **{k: v for k, v in kwargs.items() 
                           if k not in ["max_tokens", "temperature"]}
                    }
                )
                response.raise_for_status()
                result = response.json()
                return {"generated_text": result["choices"][0]["text"]}
            except Exception as e:
                return {"error": str(e)}
            
        elif task_lower == "chat":
            # For chat completion tasks, inputs should be a list of messages
            messages: Union[List[Dict[str, str]], Any]
            if isinstance(inputs, str):
                # Convert to simple user message format if string
                messages = [{"role": "user", "content": inputs}]
            elif isinstance(inputs, list) and all(isinstance(i, str) for i in inputs):
                # If list of strings, treat each as a separate message alternating roles
                messages_converted: List[Dict[str, str]] = []
                for i, msg in enumerate(inputs):
                    role = "user" if i % 2 == 0 else "assistant"
                    messages_converted.append({"role": role, "content": msg})
                messages = messages_converted
            else:
                # Assume it's already in the correct format
                messages = inputs
            
            # Direct API call to the vLLM server for chat completions
            try:
                response = requests.post(
                    f"{model_url}/v1/chat/completions",
                    json={
                        "model": model_id,
                        "messages": messages,
                        "max_tokens": kwargs.get("max_tokens", 100),
                        "temperature": kwargs.get("temperature", 0.7),
                        **{k: v for k, v in kwargs.items() 
                           if k not in ["max_tokens", "temperature"]}
                    }
                )
                response.raise_for_status()
                result = response.json()
                return {"chat_response": result["choices"][0]["message"]}
            except Exception as e:
                return {"error": str(e)}
            
        else:
            # Unknown task
            raise ValueError(f"Unknown task type: {task}")
    
    def health_check(self) -> Dict[str, Any]:
        """Check the health of the model engine service.
        
        Returns:
            Dictionary with health status information
        """
        status_info: Dict[str, Any] = {"status": "ok" if self.running else "not_running"}
        
        if not self.running:
            status_info["message"] = "Service is not running"
            return status_info
            
        if self.manager is None:
            status_info["status"] = "error"
            status_info["message"] = "Process manager is not initialized"
            return status_info
            
        try:
            # Basic health check - get loaded models
            loaded_models = self.get_loaded_models()
            status_info["loaded_models_count"] = len(loaded_models)
            status_info["models"] = list(loaded_models.keys())
        except Exception as e:
            status_info["status"] = "error"
            status_info["message"] = f"Health check failed: {str(e)}"
            
        return status_info
