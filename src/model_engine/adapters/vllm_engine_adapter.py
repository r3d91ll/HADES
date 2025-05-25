"""
vLLM engine adapter implementation for HADES-PathRAG.

This adapter provides a unified interface for both embedding generation
and text completion using the vLLM engine.
"""

import logging
import json
import requests
import aiohttp
import numpy as np
import time
import asyncio
from typing import Dict, List, Any, Optional, Union, AsyncIterator

from src.model_engine.adapters.base import EmbeddingAdapter, CompletionAdapter, ChatAdapter
from src.model_engine.factory import get_vllm_engine
from src.config.vllm_config import ModelMode

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VLLMEngineAdapter(EmbeddingAdapter, CompletionAdapter, ChatAdapter):
    """Unified adapter for vLLM-powered models using the vLLM engine."""
    
    def __init__(
        self,
        model_name: str = "embedding",
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        max_retries: int = 3,
        timeout: int = 60,
        use_openai_api: bool = True,
        mode: ModelMode = ModelMode.INFERENCE,
        **kwargs
    ) -> None:
        """
        Initialize the vLLM engine adapter.
        
        Args:
            model_name: Alias of the model to use (from configuration)
            batch_size: Batch size for processing
            normalize_embeddings: Whether to normalize embeddings
            max_retries: Maximum number of retries for failed requests
            timeout: Timeout for requests in seconds
            use_openai_api: Whether to use OpenAI-compatible API endpoints
            mode: Whether to use the model for inference or ingestion
            **kwargs: Additional parameters
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.max_retries = max_retries
        self.timeout = timeout
        self.use_openai_api = use_openai_api
        self.mode = mode
        
        # Get the vLLM engine
        self.engine = get_vllm_engine()
        
        # Server URL will be set when the model is loaded
        self.server_url = None
        
        # Configure API endpoints (will be updated when model is loaded)
        self.embedding_endpoint = None
        self.completion_endpoint = None
        self.chat_endpoint = None
        self.health_endpoint = None
    
    def _load_model(self) -> None:
        """
        Load the model if it's not already loaded.
        
        Raises:
            RuntimeError: If the model fails to load
        """
        if self.server_url is not None:
            return
            
        try:
            # Start the engine if it's not running
            if not self.engine.running:
                self.engine.start()
                
            # Load the model
            self.server_url = self.engine.load_model(self.model_name, mode=self.mode)
            
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
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise RuntimeError(f"Failed to load model {self.model_name}: {e}")
    
    @property
    def is_available(self) -> bool:
        """Check if the vLLM server is available."""
        try:
            self._load_model()
            return self._check_server()
        except Exception:
            return False
        
    def _check_server(self) -> bool:
        """
        Check if the vLLM server is available.
        
        Returns:
            True if server is available, False otherwise
        """
        if self.server_url is None or self.health_endpoint is None:
            return False
            
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
        # Ensure the model is loaded
        self._load_model()
        
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
            RuntimeError: If the completion generation fails after retries
        """
        # Ensure the model is loaded
        self._load_model()
        
        # Extract parameters with defaults
        max_tokens = kwargs.get("max_tokens", 256)
        temperature = kwargs.get("temperature", 0.7)
        top_p = kwargs.get("top_p", 0.9)
        top_k = kwargs.get("top_k", 40)
        stop = kwargs.get("stop", None)
        
        # Prepare request
        if self.use_openai_api:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "n": 1,
                "stream": False
            }
            
            if stop is not None:
                payload["stop"] = stop if isinstance(stop, list) else [stop]
        else:
            payload = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "n": 1,
                "stream": False
            }
            
            if stop is not None:
                payload["stop"] = stop if isinstance(stop, list) else [stop]
        
        # Send request to vLLM server
        for attempt in range(self.max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self.completion_endpoint,
                        headers={"Content-Type": "application/json"},
                        json=payload,
                        timeout=self.timeout
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            # Extract completion from response
                            if self.use_openai_api:
                                return data["choices"][0]["text"]
                            else:
                                return data["text"][0]
                        else:
                            response_text = await response.text()
                            logger.warning(f"Error in completion request (attempt {attempt+1}/{self.max_retries}): {response_text}")
                            await asyncio.sleep(1)
            except Exception as e:
                logger.warning(f"Exception in completion request (attempt {attempt+1}/{self.max_retries}): {e}")
                await asyncio.sleep(1)
        
        raise RuntimeError(f"Failed to generate completion after {self.max_retries} attempts")
    
    async def complete_stream_async(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """
        Stream a text completion asynchronously.
        
        Args:
            prompt: Text prompt to complete
            **kwargs: Additional parameters for the model
            
        Yields:
            Chunks of generated text as they become available
            
        Raises:
            RuntimeError: If the completion generation fails after retries
        """
        # Ensure the model is loaded
        self._load_model()
        
        # Extract parameters with defaults
        max_tokens = kwargs.get("max_tokens", 256)
        temperature = kwargs.get("temperature", 0.7)
        top_p = kwargs.get("top_p", 0.9)
        top_k = kwargs.get("top_k", 40)
        stop = kwargs.get("stop", None)
        
        # Prepare request
        if self.use_openai_api:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "n": 1,
                "stream": True
            }
            
            if stop is not None:
                payload["stop"] = stop if isinstance(stop, list) else [stop]
        else:
            payload = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "n": 1,
                "stream": True
            }
            
            if stop is not None:
                payload["stop"] = stop if isinstance(stop, list) else [stop]
        
        # Send request to vLLM server
        for attempt in range(self.max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self.completion_endpoint,
                        headers={"Content-Type": "application/json"},
                        json=payload,
                        timeout=self.timeout
                    ) as response:
                        if response.status == 200:
                            # Process streaming response
                            async for line in response.content:
                                line = line.strip()
                                if not line or line == b"data: [DONE]":
                                    continue
                                    
                                if line.startswith(b"data: "):
                                    json_str = line[6:].decode("utf-8")
                                    try:
                                        data = json.loads(json_str)
                                        if self.use_openai_api:
                                            if "choices" in data and len(data["choices"]) > 0:
                                                chunk = data["choices"][0].get("text", "")
                                                if chunk:
                                                    yield chunk
                                        else:
                                            if "text" in data and len(data["text"]) > 0:
                                                chunk = data["text"][0]
                                                if chunk:
                                                    yield chunk
                                    except json.JSONDecodeError:
                                        logger.warning(f"Invalid JSON in streaming response: {json_str}")
                            return
                        else:
                            response_text = await response.text()
                            logger.warning(f"Error in streaming request (attempt {attempt+1}/{self.max_retries}): {response_text}")
                            await asyncio.sleep(1)
            except Exception as e:
                logger.warning(f"Exception in streaming request (attempt {attempt+1}/{self.max_retries}): {e}")
                await asyncio.sleep(1)
        
        raise RuntimeError(f"Failed to generate streaming completion after {self.max_retries} attempts")
    
    async def chat_async(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Generate a chat response asynchronously.
        
        Args:
            messages: List of message dictionaries with "role" and "content" keys
            **kwargs: Additional parameters for the model
            
        Returns:
            Generated chat response
            
        Raises:
            RuntimeError: If the chat generation fails after retries
        """
        # Ensure the model is loaded
        self._load_model()
        
        # Extract parameters with defaults
        max_tokens = kwargs.get("max_tokens", 256)
        temperature = kwargs.get("temperature", 0.7)
        top_p = kwargs.get("top_p", 0.9)
        top_k = kwargs.get("top_k", 40)
        stop = kwargs.get("stop", None)
        
        # Prepare request
        if self.use_openai_api:
            payload = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "n": 1,
                "stream": False
            }
            
            if stop is not None:
                payload["stop"] = stop if isinstance(stop, list) else [stop]
        else:
            # Convert messages to prompt format for non-OpenAI API
            prompt = ""
            for message in messages:
                role = message["role"]
                content = message["content"]
                if role == "system":
                    prompt += f"System: {content}\n\n"
                elif role == "user":
                    prompt += f"User: {content}\n\n"
                elif role == "assistant":
                    prompt += f"Assistant: {content}\n\n"
            prompt += "Assistant: "
            
            payload = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "n": 1,
                "stream": False
            }
            
            if stop is not None:
                payload["stop"] = stop if isinstance(stop, list) else [stop]
        
        # Send request to vLLM server
        for attempt in range(self.max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    endpoint = self.chat_endpoint if self.use_openai_api else self.completion_endpoint
                    async with session.post(
                        endpoint,
                        headers={"Content-Type": "application/json"},
                        json=payload,
                        timeout=self.timeout
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            # Extract completion from response
                            if self.use_openai_api:
                                return data["choices"][0]["message"]["content"]
                            else:
                                return data["text"][0]
                        else:
                            response_text = await response.text()
                            logger.warning(f"Error in chat request (attempt {attempt+1}/{self.max_retries}): {response_text}")
                            await asyncio.sleep(1)
            except Exception as e:
                logger.warning(f"Exception in chat request (attempt {attempt+1}/{self.max_retries}): {e}")
                await asyncio.sleep(1)
        
        raise RuntimeError(f"Failed to generate chat response after {self.max_retries} attempts")
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Generate a chat response.
        
        Args:
            messages: List of message dictionaries with "role" and "content" keys
            **kwargs: Additional parameters for the model
            
        Returns:
            Generated chat response
        """
        # Synchronous wrapper around async chat
        return asyncio.run(self.chat_async(messages, **kwargs))
