"""
vLLM adapter for HADES-PathRAG.

This module provides adapter functions for vLLM integration,
serving as the primary model engine implementation.
"""

import aiohttp
import numpy as np
from typing import List, Dict, Any, Optional, Union, AsyncIterator
import logging
import json
from datetime import datetime
from src.config.vllm_config import VLLMConfig
from src.pathrag.vllm_server import get_server_manager
from src.types.vllm_types import ModelMode

logger = logging.getLogger(__name__)

async def vllm_model_complete(
    prompt: str, 
    model_alias: Optional[str] = None, 
    **kwargs
) -> str:
    """
    Complete a prompt with the specified model using vLLM.
    
    Args:
        prompt: Text prompt to complete
        model_alias: Alias of the model to use (from config)
        **kwargs: Additional parameters to pass to the model
        
    Returns:
        Generated text completion
        
    Raises:
        RuntimeError: If the vLLM server fails to start or the API request fails
    """
    # Use default model if not specified
    config = VLLMConfig.load_from_yaml()
    if model_alias is None:
        model_alias = "default"
    
    # Make sure server is running with the right model
    server_manager = get_server_manager(config)
    server_running = await server_manager.ensure_server_running(model_alias, mode="inference")
    if not server_running:
        raise RuntimeError(f"Failed to start vLLM server with model {model_alias}")
    
    # Get model config
    model_config = config.get_model_config(model_alias, mode="inference")
    model_id = model_config.model_id
    
    # Format as OpenAI-style chat message
    messages = [{"role": "user", "content": prompt}]
    if "system_prompt" in kwargs and kwargs["system_prompt"]:
        messages.insert(0, {"role": "system", "content": kwargs["system_prompt"]})
    
    # Add history if provided
    if "history_messages" in kwargs and kwargs["history_messages"]:
        messages = kwargs["history_messages"] + messages
    
    # Clean kwargs of special parameters
    clean_kwargs = kwargs.copy()
    for k in ["system_prompt", "history_messages", "stream"]:
        if k in clean_kwargs:
            clean_kwargs.pop(k)
    
    # Call the chat API
    async with aiohttp.ClientSession() as session:
        payload = {
            "model": model_id,
            "messages": messages,
            "temperature": clean_kwargs.pop("temperature", model_config.temperature),
            "max_tokens": clean_kwargs.pop("max_tokens", model_config.max_tokens),
            "stream": False,  # We don't support streaming yet
            **clean_kwargs
        }
        
        url = f"http://{config.server.host}:{config.server.port}/v1/chat/completions"
        
        try:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    raise RuntimeError(f"API request failed with status {response.status}: {await response.text()}")
                    
                data = await response.json()
                return data["choices"][0]["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"Error calling vLLM API: {str(e)}")

async def vllm_model_if_cache(
    messages: List[Dict[str, str]], 
    model_alias: Optional[str] = None, 
    **kwargs
) -> Dict[str, Any]:
    """
    Chat completion with the specified model using vLLM.
    
    Args:
        messages: List of chat messages
        model_alias: Alias of the model to use (from config)
        **kwargs: Additional parameters to pass to the model
        
    Returns:
        JSON response from the vLLM API
        
    Raises:
        RuntimeError: If the vLLM server fails to start or the API request fails
    """
    # Use default model if not specified
    config = VLLMConfig.load_from_yaml()
    if model_alias is None:
        model_alias = "default"
    
    # Make sure server is running with the right model
    server_manager = get_server_manager(config)
    server_running = await server_manager.ensure_server_running(model_alias, mode="inference")
    if not server_running:
        raise RuntimeError(f"Failed to start vLLM server with model {model_alias}")
    
    # Get model config
    model_config = config.get_model_config(model_alias, mode="inference")
    model_id = model_config.model_id
    
    # Combine default model parameters with provided kwargs
    params = {
        "temperature": model_config.temperature,
        "top_p": model_config.top_p,
        "max_tokens": model_config.max_tokens,
    }
    params.update(kwargs)
    
    # Call the API
    async with aiohttp.ClientSession() as session:
        payload = {
            "model": model_id,
            "messages": messages,
            **params
        }
        
        url = f"http://{config.server.host}:{config.server.port}/v1/chat/completions"
        
        try:
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    logger.error(f"vLLM request failed: {error_text}")
                    raise RuntimeError(f"vLLM API request failed: {error_text}")
        except Exception as e:
            logger.error(f"Error calling vLLM API: {str(e)}")
            raise

async def vllm_embed(
    texts: List[str], 
    model_alias: Optional[str] = None, 
    **kwargs
) -> np.ndarray:
    """
    Generate embeddings for texts using vLLM.
    
    Args:
        texts: Texts to embed
        model_alias: Alias of the embedding model to use (from config)
        **kwargs: Additional parameters to pass to the model
        
    Returns:
        NumPy array of embeddings
        
    Raises:
        RuntimeError: If the vLLM server fails to start or the API request fails
    """
    # Use default embedding model if not specified
    config = VLLMConfig.load_from_yaml()
    if model_alias is None:
        model_alias = "embedding"
    
    # Make sure server is running with the right model
    server_manager = get_server_manager(config)
    server_running = await server_manager.ensure_server_running(model_alias, mode="ingestion")
    if not server_running:
        raise RuntimeError(f"Failed to start vLLM server with embedding model {model_alias}")
    
    # Get model config
    model_config = config.get_model_config(model_alias, mode="ingestion")
    model_id = model_config.model_id
    
    # Call the API
    async with aiohttp.ClientSession() as session:
        payload = {
            "model": model_id,
            "input": texts,
            **kwargs
        }
        
        url = f"http://{default_config.server.host}:{default_config.server.port}/v1/embeddings"
        
        try:
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    # Extract embeddings from response
                    embeddings = [item["embedding"] for item in data["data"]]
                    return np.array(embeddings)
                else:
                    error_text = await response.text()
                    logger.error(f"vLLM embedding request failed: {error_text}")
                    raise RuntimeError(f"vLLM API embedding request failed: {error_text}")
        except Exception as e:
            logger.error(f"Error calling vLLM embedding API: {str(e)}")
            raise
