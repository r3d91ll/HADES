"""Haystack-based model engine for HADES-PathRAG.

This engine uses Haystack and a Unix domain socket for managing models in a separate
process. It provides efficient GPU memory management through the ModelClient interface.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from src.model_engine.base import ModelEngine
from src.model_engine.engines.haystack.runtime import ModelClient


class HaystackModelEngine(ModelEngine):
    """Haystack-based model engine implementation.
    
    This engine uses the Haystack framework for managing models, with a custom
    runtime service that communicates over a Unix domain socket to ensure
    efficient GPU memory usage.
    
    The engine can manage multiple models simultaneously, with an LRU cache that
    will automatically unload the least recently used models if memory limits are reached.
    Different model types (embedding, classification, etc.) can be loaded and used
    concurrently.
    
    Attributes:
        client: The ModelClient instance for communicating with the runtime service
        running: Whether the model manager service is currently running
    """
    
    def __init__(self, socket_path: Optional[str] = None) -> None:
        """Initialize the Haystack model engine.
        
        Args:
            socket_path: Optional custom path to the Unix domain socket
        """
        self.socket_path = socket_path
        self.client: Optional[ModelClient] = None
        self.running = False
        
    def start(self) -> bool:
        """Start the model engine service.
        
        This will start the model manager server if it's not already running.
        
        Returns:
            True if the service was started successfully, False otherwise
        """
        # Early return path if already running
        if self.running:
            if self.client is not None:
                print("[HaystackModelEngine] Service is already running")
                return True
            else:
                # Running but no client is an inconsistent state - fix it
                self.running = False
        
        # Otherwise attempt to start    
        # Initialize client and test connection
        try:
            # Create new client
            client = ModelClient(socket_path=self.socket_path)
            self.client = client
                
            # Test the connection with a ping
            response = client.ping()
            self.running = response == "pong"
            return self.running
            
        except Exception as e:
            # Handle any errors during client creation or ping
            print(f"[HaystackModelEngine] Failed to start service: {e}")
            self.running = False
            self.client = None
            return False
            
    def stop(self) -> bool:
        """Stop the model engine service.
        
        This will shutdown the runtime service if it was started by this engine.
        Note that this will affect all clients that share the same socket.
        
        Returns:
            True if the service was stopped successfully, False otherwise
        """
        # Check running state first
        if not self.running:
            print("[HaystackModelEngine] Service is not marked as running")
            self.client = None  # Ensure client is cleared
            return True
            
        # Check client separately
        if self.client is None:
            print("[HaystackModelEngine] No client available to stop")
            self.running = False  # Fix inconsistent state
            return True
        
        # At this point we know service is running and client is not None
        try:
            # Attempt graceful shutdown
            self.client.shutdown()
            self.running = False
            self.client = None
            return True
        except Exception as e:
            print(f"[HaystackModelEngine] Error stopping service: {e}")
            # Keep the service marked as running on error
            return False
            
    def shutdown(self) -> bool:
        """Completely shut down the engine and service.
        
        This is a more decisive version of stop() that ensures the service
        is fully terminated. It may terminate the server process if necessary.
        
        Returns:
            True if shutdown was successful, False otherwise
        """
        # If we're not running, no action needed
        if not self.running and self.client is None:
            return True
            
        # Reset the state regardless of current status
        result = True
        
        # Try to stop gracefully if we have a client
        if self.client is not None:
            try:
                result = self.stop()  # This attempts a graceful shutdown
            except Exception as e:
                print(f"[HaystackModelEngine] Error during shutdown: {e}")
                result = False
        
        # Ensure our internal state is consistent
        self.running = False
        self.client = None
        
        return result
            
    def restart(self) -> bool:
        """Restart the model engine service.
        
        Returns:
            True if service was successfully restarted, False otherwise
        """
        # Try to stop first
        try:
            self.stop()
        except Exception as e:
            print(f"[HaystackModelEngine] Warning during stop in restart: {e}")
            # Continue to start attempt even if stop fails
            
        # Always attempt to start regardless of stop success
        start_result = self.start()
        return start_result
        
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the model engine.
        
        Returns:
            Dict with status information
        """
        status_info: Dict[str, Any] = {
            "running": self.running,
            "socket": self.socket_path,
        }
        
        # Add loaded models if service is running
        if self.running and self.client is not None:
            try:
                # Verify we have a client for mypy
                assert self.client is not None
                
                status_info["models"] = self.client.info()
            except Exception as e:
                status_info["error"] = str(e)
                
        return status_info
        
    def is_running(self) -> bool:
        """Check if the model engine service is running.
        
        Returns:
            True if the service is running, False otherwise
        """
        # Quick check based on internal state
        if not self.running:
            return False
            
        # If marked as running but no client, fix this inconsistent state
        if self.client is None:
            self.running = False
            return False
            
        # We have a client, try to ping the server
        try:
            ping_result = self.client.ping()
            is_ok = ping_result == "pong"
            
            # Update internal state if ping fails
            if not is_ok:
                self.running = False
                
            return is_ok
        except Exception:
            # Connection failed
            self.running = False
            return False
            
    def health_check(self) -> Dict[str, Any]:
        """Check the health of the engine.
        
        Returns:
            Dict with health status information
        """
        # Start with basic health info
        health_info: Dict[str, Any] = {
            "status": "healthy" if self.running else "not_running",
            "message": "Engine is running" if self.running else "Engine is not running",
            "socket_path": self.socket_path
        }
        
        # Only check more details if running
        if self.running:
            # First check if we have a client
            if self.client is None:
                health_info["status"] = "degraded"
                health_info["message"] = "Engine marked as running but client is not initialized"
                return health_info
                
            # Then try to use the client
            try:
                # Test server connection
                ping_result = self.client.ping()
                health_info["ping"] = ping_result == "pong"
                
                if ping_result != "pong":
                    health_info["status"] = "degraded"
                    health_info["message"] = f"Engine ping returned unexpected result: {ping_result}"
                    return health_info
                
                # Try to get memory usage info 
                try:
                    debug_info = self.client.debug()
                    if debug_info:
                        health_info["cache_size"] = debug_info.get("cache_size", "unknown")
                        health_info["max_cache_size"] = debug_info.get("max_cache_size", "unknown")
                except Exception as debug_error:
                    health_info["debug_error"] = str(debug_error)
            except Exception as e:
                health_info["status"] = "degraded"
                health_info["message"] = f"Engine is running but unresponsive: {e}"
        
        return health_info
        
    def __enter__(self) -> 'HaystackModelEngine':
        """Context manager entry - automatically start the service."""
        self.start()
        return self
        
    def __exit__(self, exc_type: Optional[type], exc_val: Optional[Exception], exc_tb: Optional[Any]) -> None:
        """Context manager exit - automatically stop the service."""
        self.stop()
    
    def load_model(self, model_id: str, 
                   device: Optional[Union[str, List[str]]] = None) -> str:
        """Load a model into memory using the runtime service.
        
        Args:
            model_id: The ID of the model to load (HF model ID or local path)
            device: The device to load the model onto (e.g., "cuda:0")
                    If None, uses the default device.
                    
        Returns:
            Status string ("loaded" or "already_loaded")
            
        Raises:
            RuntimeError: If the service is not running or another error occurs
        """
        if not self.running or self.client is None:
            started = self.start()
            if not started or self.client is None:
                raise RuntimeError("Model engine service is not running")
        
        # At this point we know self.client is not None
        assert self.client is not None
        return self.client.load(model_id, device=device if isinstance(device, str) else None)
    
    def unload_model(self, model_id: str) -> str:
        """Unload a model from memory.
        
        Args:
            model_id: The ID of the model to unload
            
        Returns:
            Status string ("unloaded")
            
        Raises:
            RuntimeError: If the service is not running or another error occurs
        """
        if not self.running or self.client is None:
            raise RuntimeError("Model engine service is not running")
        
        # At this point we know self.client is not None
        assert self.client is not None    
        return self.client.unload(model_id)
    
    def infer(self, model_id: str, inputs: Union[str, List[str]],
             task: str, **kwargs: Any) -> Any:
        """Run inference with a model.
        
        Args:
            model_id: The ID of the model to use
            inputs: Input text or list of texts
            task: The task to perform (e.g., "embedding", "generation")
            **kwargs: Additional task-specific parameters
            
        Returns:
            Task-specific outputs
            
        Raises:
            NotImplementedError: This method is not yet implemented
            RuntimeError: If the model service is not running
        """
        # Check client state first
        if not self.running:
            raise RuntimeError("Model engine service is not running")
        
        if self.client is None:
            raise RuntimeError("Model client is not initialized")
       
        # Handle different task types
        task_lower = task.lower()
        if task_lower == "embedding":
            # This feature is not yet implemented
            raise NotImplementedError("Embedding task not yet implemented in client")
        elif task_lower == "generation":
            # This feature is not yet implemented
            raise NotImplementedError("Generation task not yet implemented")
        else:
            # Unknown task
            raise ValueError(f"Unknown task type: {task}")
    
    def get_loaded_models(self) -> Dict[str, Dict[str, Any]]:
        """Get information about currently loaded models.
        
        Returns:
            Dictionary with model IDs as keys and info as values
            
        Raises:
            RuntimeError: If the service is not running or another error occurs
        """
        # Check running state first
        if not self.running:
            raise RuntimeError("Model engine service is not running")
        
        # Check if client is available
        if self.client is None:
            raise RuntimeError("Model client is not initialized")
            
        # Get model info from service
        try:
            result = self.client.info()
            return result
        except Exception as e:
            raise RuntimeError(f"Error getting model info: {e}") from e
    
    # The main health_check method is already implemented above
