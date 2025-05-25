"""
vLLM adapter implementation for HADES-PathRAG.

This adapter provides a unified interface for both embedding generation
and text completion using vLLM's OpenAI-compatible API.
"""

import logging
import json
import requests
import aiohttp
import numpy as np
import time
import os
import asyncio
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, AsyncIterator

from src.model_engine.adapters.base import EmbeddingAdapter, CompletionAdapter, ChatAdapter
from src.config.vllm_config import VLLMConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VLLMAdapter(EmbeddingAdapter, CompletionAdapter, ChatAdapter):
    """Unified adapter for vLLM-powered models."""
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        server_url: Optional[str] = None,
        batch_size: int = 32,
        device: str = "cuda",
        normalize_embeddings: bool = True,
        max_retries: int = 3,
        timeout: int = 60,
        use_openai_api: bool = True,
        **kwargs
    ) -> None:
        """
        Initialize the vLLM adapter.
        
        Args:
            model_name: Name of the model to use
            server_url: URL of the vLLM server (if None, will use localhost:8000)
            batch_size: Batch size for processing
            device: Device to run the model on ('cuda' or 'cpu')
            normalize_embeddings: Whether to normalize embeddings
            max_retries: Maximum number of retries for failed requests
            timeout: Timeout for requests in seconds
            use_openai_api: Whether to use OpenAI-compatible API endpoints
            **kwargs: Additional parameters
        """
        self.model_name = model_name
        self.server_url = server_url or "http://localhost:8000"
        self.batch_size = batch_size
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self.max_retries = max_retries
        self.timeout = timeout
        self.use_openai_api = use_openai_api
        
        # If server URL ends with /, remove it
        if self.server_url.endswith("/"):
            self.server_url = self.server_url[:-1]
        
        # Configure API endpoints based on format
        if self.use_openai_api:
            self.embedding_endpoint = f"{self.server_url}/v1/embeddings"
            self.completion_endpoint = f"{self.server_url}/v1/completions"
            self.chat_endpoint = f"{self.server_url}/v1/chat/completions"
            self.health_endpoint = f"{self.server_url}/v1/models"
        else:
            self.embedding_endpoint = f"{self.server_url}/v1/embeddings"
            self.completion_endpoint = f"{self.server_url}/generate"
            self.chat_endpoint = f"{self.server_url}/generate"
            self.health_endpoint = f"{self.server_url}/health"
        
        # Check if server is available
        self._check_server()
    
    @property
    def is_available(self) -> bool:
        """Check if the vLLM server is available."""
        return self._check_server()
        
    def _check_server(self) -> bool:
        """
        Check if the vLLM server is available.
        
        Returns:
            True if server is available, False otherwise
        """
        try:
            response = requests.get(self.health_endpoint, timeout=self.timeout)
            if response.status_code == 200:
                logger.info(f"vLLM server at {self.server_url} is available")
                return True
            else:
                logger.warning(f"vLLM server at {self.server_url} returned status code {response.status_code}")
                return False
        except Exception as e:
            logger.warning(f"Could not connect to vLLM server at {self.server_url}: {e}")
            return False
    
    def _normalize(self, embeddings: List[List[float]]) -> List[List[float]]:
        """
        Normalize embeddings to unit length.
        
        Args:
            embeddings: List of embeddings to normalize
            
        Returns:
            Normalized embeddings
        """
        if not self.normalize_embeddings:
            return embeddings
        
        normalized = []
        for emb in embeddings:
            emb_np = np.array(emb)
            norm = np.linalg.norm(emb_np)
            if norm > 0:
                emb_np = emb_np / norm
            normalized.append(emb_np.tolist())
        
        return normalized
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors (as lists of floats)
            
        Raises:
            RuntimeError: If the embedding generation fails after retries
        """
        all_embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i+self.batch_size]
            
            # Prepare request
            payload = {
                "model": self.model_name,
                "input": batch_texts
            }
            
            # Send request to vLLM server
            for attempt in range(self.max_retries):
                try:
                    response = requests.post(
                        self.embedding_endpoint,
                        headers={"Content-Type": "application/json"},
                        data=json.dumps(payload),
                        timeout=self.timeout
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        # Extract embeddings from response
                        batch_embeddings = [item["embedding"] for item in data["data"]]
                        all_embeddings.extend(batch_embeddings)
                        break
                    else:
                        logger.warning(f"Error in embedding request (attempt {attempt+1}/{self.max_retries}): {response.text}")
                        time.sleep(1)
                except Exception as e:
                    logger.warning(f"Exception in embedding request (attempt {attempt+1}/{self.max_retries}): {e}")
                    time.sleep(1)
            else:
                raise RuntimeError(f"Failed to get embeddings after {self.max_retries} attempts")
        
        # Normalize if requested
        return self._normalize(all_embeddings)
    
    def complete(self, prompt: str, **kwargs) -> str:
        """
        Complete a text prompt.
        
        Args:
            prompt: Text prompt to complete
            **kwargs: Additional parameters for the model
            
        Returns:
            Generated text completion
        """
        # Synchronous wrapper around async completion
        return asyncio.run(self.complete_async(prompt, **kwargs))
    
    async def complete_async(self, prompt: str, **kwargs) -> str:
        """
        Complete a text prompt asynchronously.
        
        Args:
            prompt: Text prompt to complete
            **kwargs: Additional parameters for the model
            
        Returns:
            Generated text completion
            
        Raises:
            RuntimeError: If the completion request fails
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": kwargs.get("max_tokens", 1024),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "top_k": kwargs.get("top_k", 40),
            "stop": kwargs.get("stop", None)
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.completion_endpoint,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"Error in completion request: {error_text}")
                
                data = await response.json()
                return data["choices"][0]["text"]
    
    def chat_complete(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """
        Complete a chat conversation.
        
        Args:
            messages: List of chat messages (each with "role" and "content" keys)
            **kwargs: Additional parameters for the model
            
        Returns:
            Response from the API with completion
        """
        # Synchronous wrapper around async chat completion
        return asyncio.run(self.chat_complete_async(messages, **kwargs))
    
    async def chat_complete_async(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """
        Complete a chat conversation asynchronously.
        
        Args:
            messages: List of chat messages (each with "role" and "content" keys)
            **kwargs: Additional parameters for the model
            
        Returns:
            Response from the API with completion
            
        Raises:
            RuntimeError: If the chat completion request fails
        """
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 1024),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "top_k": kwargs.get("top_k", 40),
            "stop": kwargs.get("stop", None)
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.chat_endpoint,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"Error in chat completion request: {error_text}")
                
                return await response.json()


def start_vllm_server(
    model_name: str,
    port: int,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: Optional[int] = None,
    dtype: Optional[str] = None,
    quantization: Optional[str] = None,
    use_openai_api: bool = True,
    cuda_visible_devices: Optional[str] = None,
    seed: int = 42,
    **kwargs
) -> str:
    """
    Generate command to start a vLLM server.
    
    Args:
        model_name: Name of the model to use
        port: Port for the server
        tensor_parallel_size: Number of GPUs to use in parallel
        gpu_memory_utilization: Fraction of GPU memory to use
        use_openai_api: Whether to use OpenAI-compatible API
        max_model_len: Maximum sequence length for the model
        quantization: Quantization method ("fp8", "int8", "int4" or None)
        dtype: Data type for the model ("float16", "bfloat16", "float32")
        seed: Random seed for reproducibility
        **kwargs: Additional parameters to pass to the vLLM server
        
    Returns:
        Command string to run the server
    """
    # Base command with required parameters
    env_prefix = ""
    if cuda_visible_devices is not None:
        env_prefix = f"CUDA_VISIBLE_DEVICES={cuda_visible_devices} "
    # If tensor_parallel_size is greater than 1 and CUDA_VISIBLE_DEVICES is not set,
    # we need to make sure there are enough GPUs available
    elif tensor_parallel_size > 1:
        # Default to use all available GPUs
        env_prefix = "" # Let vLLM handle GPU selection
    
    cmd = f"{env_prefix}python -m vllm.entrypoints.openai.api_server "
    cmd += f"--model {model_name} "
    cmd += f"--port {port} "
    cmd += f"--tensor-parallel-size {tensor_parallel_size} "
    cmd += f"--gpu-memory-utilization {gpu_memory_utilization} "
    cmd += f"--seed {seed} "
    
    # Add optional parameters if provided
    if max_model_len is not None:
        cmd += f"--max-model-len {max_model_len} "
    
    if quantization is not None:
        cmd += f"--quantization {quantization} "

    if dtype is not None:
        cmd += f"--dtype {dtype} "

    # Add kwargs as additional CLI arguments
    for key, value in kwargs.items():
        key = key.replace("_", "-")
        if isinstance(value, bool) and value:
            cmd += f"--{key} "
        elif not isinstance(value, bool):
            cmd += f"--{key} {value} "

    # Return the complete command
    return cmd
