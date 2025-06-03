"""vLLM-based model engine for HADES-PathRAG.

This engine uses vLLM for managing models in a separate process. It provides efficient
GPU memory management and high-performance inference for large language models.
"""
from __future__ import annotations

import asyncio
import logging
import requests
import time
from typing import Any, Dict, List, Optional, Type, Union, cast

from src.config.vllm_config import VLLMConfig, ModelMode
from src.model_engine.base import ModelEngine
from src.model_engine.vllm_session import get_vllm_manager, VLLMProcessManager


# Set up logging
logger = logging.getLogger(__name__)


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
    
    def __init__(self, config_path: Optional[str] = None, socket_path: Optional[str] = None) -> None:
        """Initialize the vLLM model engine.
        
        Args:
            config_path: Optional custom path to the vLLM configuration file
            socket_path: Optional socket path for IPC (not used, but kept for API consistency)
        """
        super().__init__(config_path=config_path)
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
        if self.running:
            logger.info("Model engine service is already running")
            return True
        
        try:
            # Initialize the vLLM process manager
            self.manager = get_vllm_manager()
            
            # Start the manager
            self.running = True
            logger.info("Model engine service started successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to start model engine service: {str(e)}")
            return False
    
    def stop(self) -> bool:
        """Stop the model engine service.
        
        This will shut down the vLLM process manager and all running model processes.
        
        Returns:
            True if the service was stopped successfully, False otherwise
        """
        if not self.running:
            logger.info("Model engine service is not running")
            return True
        
        try:
            # Check if the manager exists
            if self.manager is not None:
                # Stop all running model processes
                running_models = self.manager.get_running_models()
                for model_id in running_models:
                    logger.info(f"Stopping model: {model_id}")
                    try:
                        self.manager.stop_model(model_id)
                    except Exception as e:
                        logger.error(f"Error stopping model {model_id}: {str(e)}")
                
                # Reset the manager
                self.manager = None
            
            self.running = False
            logger.info("Model engine service stopped successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to stop model engine service: {str(e)}")
            return False
    
    def restart(self) -> bool:
        """Restart the model engine service.
        
        Returns:
            True if the service was restarted successfully, False otherwise
        """
        logger.info("Restarting model engine service")
        self.stop()
        return self.start()
    
    def load_model(self, model_id: str, 
                  device: Optional[Union[str, List[str]]] = None) -> str:
        """Load a model into memory.
        
        Args:
            model_id: The ID of the model to load (e.g., "meta-llama/Llama-2-7b-chat-hf")
            device: The device(s) to load the model onto (e.g., "cuda:0") - not directly used
                   in this implementation as the process manager handles device assignment.
                   
        Returns:
            URL for the model server or error message
            
        Raises:
            RuntimeError: If the model engine service is not running
        """
        if not self.running:
            raise RuntimeError("Model engine service is not running")
        
        if self.manager is None:
            raise RuntimeError("Process manager is not initialized")
        
        # Check if the model is already running
        running_models = self.manager.get_running_models()
        if model_id in running_models:
            logger.info(f"Model {model_id} is already loaded")
            return running_models[model_id].server_url
        
        # Get the appropriate model mode based on the model ID
        # This is a heuristic and might need to be adjusted
        if "embedding" in model_id.lower() or "bge" in model_id.lower():
            mode = ModelMode.INGESTION
        else:
            mode = ModelMode.INFERENCE
        
        # Start the model process
        try:
            result = self.manager.start_model(model_id, mode=mode)
            logger.info(f"Model {model_id} loaded successfully")
            if result is not None:
                return result.server_url
            raise RuntimeError(f"Failed to get server URL for model {model_id}")
        except Exception as e:
            error_msg = f"Failed to load model {model_id}: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def unload_model(self, model_id: str) -> str:
        """Unload a model from memory.
        
        Args:
            model_id: The ID of the model to unload
            
        Returns:
            Status string ("unloaded" or "not_loaded")
            
        Raises:
            RuntimeError: If the model engine service is not running
        """
        if not self.running:
            raise RuntimeError("Model engine service is not running")
        
        if self.manager is None:
            raise RuntimeError("Process manager is not initialized")
        
        # Check if the model is running
        running_models = self.manager.get_running_models()
        if model_id not in running_models:
            logger.info(f"Model {model_id} is not loaded")
            return "not_loaded"
        
        # Stop the model process
        try:
            self.manager.stop_model(model_id)
            logger.info(f"Model {model_id} unloaded successfully")
            return "unloaded"
        except Exception as e:
            error_msg = f"Failed to unload model {model_id}: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def get_loaded_models(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all loaded models.
        
        Returns:
            Dictionary mapping model IDs to model information
            
        Raises:
            RuntimeError: If the model engine service is not running
        """
        if not self.running:
            raise RuntimeError("Model engine service is not running")
        
        if self.manager is None:
            raise RuntimeError("Process manager is not initialized")
        
        # Get running models from the process manager
        running_models = self.manager.get_running_models()
        
        # Format the model information
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
        
        # Check if the model is loaded
        running_models = self.manager.get_running_models()
        
        if model_id not in running_models:
            # Try to load the model
            logger.info(f"Model {model_id} not loaded, attempting to load")
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
