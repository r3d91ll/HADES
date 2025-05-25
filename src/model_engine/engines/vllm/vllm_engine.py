"""vLLM-based model engine for HADES-PathRAG.

This engine implements the ModelEngine interface using vLLM's OpenAI-compatible API
for both embeddings and text generation tasks.
"""
from __future__ import annotations

import logging
import time
import os
import asyncio
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, AsyncIterator

import requests
import aiohttp
import numpy as np

from src.model_engine.base import ModelEngine
from src.config.vllm_config import VLLMConfig


# Set up logging
logger = logging.getLogger(__name__)


class VLLMModelEngine(ModelEngine):
    """vLLM-based model engine implementation.
    
    This engine uses vLLM's OpenAI-compatible API for model inference,
    supporting both embeddings and text generation. It manages the connection
    to the vLLM server and handles request retrying, batching, and error recovery.
    
    Attributes:
        server_url: URL of the vLLM server
        loaded_models: Dictionary of currently loaded models
        running: Whether the server is running
    """
    
    def __init__(
        self, 
        server_url: Optional[str] = None,
        device: str = "cuda",
        max_retries: int = 3,
        timeout: int = 60
    ) -> None:
        """Initialize the vLLM model engine.
        
        Args:
            server_url: URL of the vLLM server (if None, will use localhost:8000)
            device: Device to run the model on ('cuda' or 'cpu')
            max_retries: Maximum number of retries for API calls
            timeout: Timeout for API calls in seconds
        """
        self.server_url = server_url or "http://localhost:8000"
        self.device = device
        self.max_retries = max_retries
        self.timeout = timeout
        
        # Track loaded models
        self.loaded_models = {}
        self.running = False
        
        # Initialize session
        self._session = None
    
    def start(self) -> bool:
        """Start the vLLM server if it's not already running.
        
        Returns:
            True if the server was started successfully, False otherwise
        """
        if self.running:
            logger.info("vLLM server is already running")
            return True
        
        try:
            # Check if server is running
            response = requests.get(f"{self.server_url}/v1/models", timeout=5)
            if response.status_code == 200:
                logger.info("vLLM server is already running")
                self.running = True
                return True
        except (requests.RequestException, ConnectionError):
            logger.info("vLLM server not running, attempting to start")
        
        # TODO: Implement server starting logic
        # This would be similar to start_vllm_server in the adapter
        
        return self.running
    
    def stop(self) -> bool:
        """Stop the vLLM server.
        
        Returns:
            True if the server was stopped successfully, False otherwise
        """
        if not self.running:
            logger.info("vLLM server is not running")
            return True
        
        # TODO: Implement server stopping logic
        
        self.running = False
        return True
    
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
        if model_id in self.loaded_models:
            logger.info(f"Model {model_id} is already loaded")
            return "already_loaded"
        
        device = device or self.device
        
        # In vLLM, models are loaded by the server when needed
        # We'll just keep track of it in our loaded_models dict
        self.loaded_models[model_id] = {
            "device": device,
            "loaded_at": time.time()
        }
        
        return "loaded"
    
    def unload_model(self, model_id: str) -> str:
        """Unload a model from memory.
        
        Args:
            model_id: The ID of the model to unload
            
        Returns:
            Status string ("unloaded" or "not_loaded")
        """
        if model_id not in self.loaded_models:
            return "not_loaded"
        
        # Remove from our tracking dict
        del self.loaded_models[model_id]
        
        # Note: In vLLM, we don't have direct control over model unloading
        # The server manages this based on memory pressure
        
        return "unloaded"
    
    def is_model_loaded(self, model_id: str) -> bool:
        """Check if a model is loaded.
        
        Args:
            model_id: The ID of the model to check
            
        Returns:
            True if the model is loaded, False otherwise
        """
        return model_id in self.loaded_models
    
    def list_loaded_models(self) -> List[str]:
        """List all loaded models.
        
        Returns:
            List of loaded model IDs
        """
        return list(self.loaded_models.keys())
    
    async def generate_embeddings(
        self,
        texts: List[str],
        model_id: str = "BAAI/bge-large-en-v1.5",
        normalize: bool = True,
        batch_size: int = 32,
        **kwargs: Any
    ) -> List[List[float]]:
        """Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            model_id: Model ID to use for embedding
            normalize: Whether to normalize the embeddings
            batch_size: Batch size for processing
            
        Returns:
            List of embeddings (one per input text)
            
        Raises:
            RuntimeError: If embedding generation fails or model not loaded
        """
        if not self.running:
            raise RuntimeError("vLLM engine is not running. Call start() first.")
        if model_id not in self.loaded_models:
            raise RuntimeError(f"Model {model_id} not loaded. Call load_model() first.")
            
        all_embeddings = []
        # Use the session managed by the base class or overridden by tests
        if not self._session or self._session.closed:
            raise RuntimeError("AIOHTTP session is not available or closed.")
        session = self._session
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            # Retry logic
            retry_count = 0
            while retry_count < self.max_retries:
                try:
                    # Make async request to vLLM server
                    async with session.post(
                        f"{self.server_url}/v1/embeddings",
                        json={"input": batch, "model": model_id},
                        timeout=self.timeout
                    ) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            raise RuntimeError(f"Embedding request failed: {error_text}")
                        
                        data = await response.json()
                        batch_embeddings = [item["embedding"] for item in data["data"]]
                        
                        all_embeddings.extend(batch_embeddings)
                        break
                
                except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError) as e:
                    retry_count += 1
                    if retry_count >= self.max_retries:
                        raise RuntimeError(f"Embedding request failed after {self.max_retries} retries: {str(e)}") from e
                    wait_time = 2 ** retry_count # Exponential backoff
                    logger.warning(f"Embedding request failed, retrying in {wait_time}s: {str(e)}")
                    await asyncio.sleep(wait_time)
        
        # Combine results
        # Ensure all_embeddings contains lists of floats
        if not all_embeddings:
            return [] # Return empty list if no embeddings were generated
            
        embeddings_array = np.array(all_embeddings)
        
        # Normalize if requested
        if normalize:
            norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
            # Avoid division by zero for zero vectors
            norms[norms == 0] = 1e-12 
            embeddings_array = embeddings_array / norms
            
        return embeddings_array.tolist()

    async def generate_completion(
        self,
        prompt: str,
        model_id: str = "meta-llama/Llama-2-7b-chat-hf",
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        **kwargs: Any
    ) -> str:
        """Generate a completion for a prompt.
        
        Args:
            prompt: The prompt to complete
            model_id: Model ID to use for completion
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            
        Returns:
            Generated completion
            
        Raises:
            RuntimeError: If completion generation fails or model not loaded
        """
        if not self.running:
            raise RuntimeError("vLLM engine is not running. Call start() first.")
        if model_id not in self.loaded_models:
            raise RuntimeError(f"Model {model_id} not loaded. Call load_model() first.")

        # Use the session managed by the base class or overridden by tests
        if not self._session or self._session.closed:
            raise RuntimeError("AIOHTTP session is not available or closed.")
        session = self._session
        
        url = f"{self.server_url}/v1/completions"
        
        # Retry logic
        retry_count = 0
        while retry_count < self.max_retries:
            try:
                # Make async request to vLLM server
                async with session.post(
                    url,
                    json={
                        "model": model_id,
                        "prompt": prompt,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                        **kwargs
                    },
                    timeout=self.timeout
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise RuntimeError(f"Completion request failed: {error_text}")
                    
                    data = await response.json()
                    return data["choices"][0]["text"]
            
            except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError) as e:
                retry_count += 1
                if retry_count >= self.max_retries:
                    raise RuntimeError(f"Failed to generate completion after {self.max_retries} retries: {str(e)}")
                
                wait_time = 2 ** retry_count  # Exponential backoff
                logger.warning(f"Completion request failed, retrying in {wait_time}s: {str(e)}")
                await asyncio.sleep(wait_time)
    
    async def generate_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model_id: str = "meta-llama/Llama-2-7b-chat-hf",
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Generate a chat completion for a list of messages.
        
        Args:
            messages: List of messages in the conversation
            model_id: Model ID to use for chat completion
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            
        Returns:
            Chat completion response
            
        Raises:
            RuntimeError: If chat completion generation fails or model not loaded
        """
        if not self.running:
            raise RuntimeError("vLLM engine is not running. Call start() first.")
        if model_id not in self.loaded_models:
            raise RuntimeError(f"Model {model_id} not loaded. Call load_model() first.")
            
        # Use the session managed by the base class or overridden by tests
        if not self._session or self._session.closed:
            raise RuntimeError("AIOHTTP session is not available or closed.")
        session = self._session
        
        url = f"{self.server_url}/v1/chat/completions"
        
        # Retry logic
        retry_count = 0
        while retry_count < self.max_retries:
            try:
                # Make async request to vLLM server
                async with session.post(
                    url,
                    json={
                        "model": model_id,
                        "messages": messages,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                        **kwargs
                    },
                    timeout=self.timeout
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise RuntimeError(f"Chat completion request failed: {error_text}")
                    
                    return await response.json()
            
            except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError) as e:
                retry_count += 1
                if retry_count >= self.max_retries:
                    raise RuntimeError(f"Failed to generate chat completion after {self.max_retries} retries: {str(e)}")
                
                wait_time = 2 ** retry_count  # Exponential backoff
                logger.warning(f"Chat completion request failed, retrying in {wait_time}s: {str(e)}")
                await asyncio.sleep(wait_time)
    
    def _normalize_embedding(self, embedding: List[float]) -> List[float]:
        """Normalize an embedding vector to unit length.
        
        Args:
            embedding: The embedding vector to normalize
            
        Returns:
            Normalized embedding
        """
        norm = np.linalg.norm(embedding)
        if norm > 0:
            return (np.array(embedding) / norm).tolist()
        return embedding
