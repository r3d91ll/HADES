"""Base interface for all model engines in HADES-PathRAG.

Each model engine implementation must conform to this interface to be
usable within the PathRAG framework. This allows for transparent swapping
of model backends (vLLM, Haystack, Ollama, etc) while maintaining 
consistent behavior in the rest of the system.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union


class ModelEngine(ABC):
    """Base class for all model engines in HADES-PathRAG.
    
    A ModelEngine is responsible for loading, unloading, and running models
    for inference. It abstracts away the details of how models are loaded
    and where they are run (GPU, CPU, remote service, etc).
    
    Implementations should handle:
    - Model loading and unloading
    - Efficient memory management (shared GPU memory, etc)
    - Batched inference
    - Error handling and recovery
    """
    
    @abstractmethod
    def load_model(self, model_id: str, 
                  device: Optional[Union[str, List[str]]] = None) -> str:
        """Load a model into memory.
        
        Args:
            model_id: The ID of the model to load (HF model ID or local path)
            device: The device(s) to load the model onto (e.g., "cuda:0")
                    If None, uses the default device for the engine.
                    
        Returns:
            Status string ("loaded" or "already_loaded")
            
        Raises:
            RuntimeError: If model loading fails
        """
        pass
    
    @abstractmethod
    def unload_model(self, model_id: str) -> str:
        """Unload a model from memory.
        
        Args:
            model_id: The ID of the model to unload
            
        Returns:
            Status string ("unloaded")
        """
        pass
    
    @abstractmethod
    def infer(self, model_id: str, inputs: Union[str, List[str]], 
             task: str, **kwargs) -> Any:
        """Run inference with a model.
        
        Args:
            model_id: The ID of the model to use
            inputs: Input text or list of texts
            task: The type of task to perform (e.g., "generate", "embed", "chunk")
            **kwargs: Task-specific parameters
            
        Returns:
            Model outputs in an appropriate format for the task
            
        Raises:
            ValueError: If the model doesn't support the task
            RuntimeError: If inference fails
        """
        pass
    
    @abstractmethod
    def get_loaded_models(self) -> Dict[str, Dict[str, Any]]:
        """Get information about currently loaded models.
        
        Returns:
            Dict mapping model_id to metadata about the model
        """
        pass
    
    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """Check the health of the engine.
        
        Returns:
            Dict containing health information (status, memory usage, etc)
        """
        pass
