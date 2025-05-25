"""
Model Engine adapter for HADES-PathRAG.

This module provides adapter functions to connect PathRAG queries to the model_engine,
abstracting away the specific underlying model implementations (vLLM and future backends).
"""

import aiohttp
import numpy as np
from typing import List, Dict, Any, Optional, Union, AsyncIterator
import logging
import json
from datetime import datetime

from src.config.model_config import ModelConfig
from src.model_engine.server_manager import get_server_manager
from src.model_engine.adapters.vllm_adapter import VLLMAdapter
from src.types.model_types import ModelMode

logger = logging.getLogger(__name__)

async def model_complete(
    prompt: str, 
    model_alias: Optional[str] = None, 
    **kwargs
) -> str:
    """
    Complete a prompt with the specified model.
    
    Args:
        prompt: Text prompt to complete
        model_alias: Alias of the model to use (from config)
        **kwargs: Additional parameters to pass to the model
        
    Returns:
        Generated text completion
        
    Raises:
        RuntimeError: If the model server fails to start or the API request fails
    """
    # Use default model if not specified
    config = ModelConfig.load_from_yaml()
    if model_alias is None:
        model_alias = "default"
    
    # Make sure server is running with the right model
    server_manager = get_server_manager(config)
    server_running = await server_manager.ensure_server_running(model_alias, mode="inference")
    if not server_running:
        raise RuntimeError(f"Failed to start model server with model {model_alias}")
    
    # Get model config
    model_config = config.get_model_config(model_alias, mode="inference")
    
    # Create adapter for the configured model backend
    # Currently only vLLM adapter is implemented
    adapter = VLLMAdapter(
        model_name=model_config.model_id,
        server_url=f"http://localhost:{config.server.port}",
        **kwargs
    )
    
    # Combine configuration and override parameters
    completion_params = {
        "max_tokens": model_config.max_tokens,
        "temperature": model_config.temperature,
        "top_p": model_config.top_p,
        "top_k": model_config.top_k
    }
    completion_params.update(kwargs)
    
    # Use the adapter to complete the prompt
    return await adapter.complete_async(prompt, **completion_params)


async def chat_complete(
    messages: List[Dict[str, str]], 
    model_alias: Optional[str] = None, 
    **kwargs
) -> Dict[str, Any]:
    """
    Chat completion with the specified model.
    
    Args:
        messages: List of chat messages
        model_alias: Alias of the model to use (from config)
        **kwargs: Additional parameters to pass to the model
        
    Returns:
        JSON response from the model API
        
    Raises:
        RuntimeError: If the model server fails to start or the API request fails
    """
    # Use default model if not specified
    config = ModelConfig.load_from_yaml()
    if model_alias is None:
        model_alias = "default"
    
    # Make sure server is running with the right model
    server_manager = get_server_manager(config)
    server_running = await server_manager.ensure_server_running(model_alias, mode="inference")
    if not server_running:
        raise RuntimeError(f"Failed to start model server with model {model_alias}")
    
    # Get model config
    model_config = config.get_model_config(model_alias, mode="inference")
    
    # Create adapter for the configured model backend
    adapter = VLLMAdapter(
        model_name=model_config.model_id,
        server_url=f"http://localhost:{config.server.port}",
        **kwargs
    )
    
    # Combine configuration and override parameters
    chat_params = {
        "max_tokens": model_config.max_tokens,
        "temperature": model_config.temperature,
        "top_p": model_config.top_p,
        "top_k": model_config.top_k
    }
    chat_params.update(kwargs)
    
    # Use the adapter to complete the chat
    return await adapter.chat_complete_async(messages, **chat_params)


async def embed(
    texts: List[str], 
    model_alias: Optional[str] = None, 
    **kwargs
) -> List[List[float]]:
    """
    Generate embeddings for texts.
    
    Args:
        texts: Texts to embed
        model_alias: Alias of the embedding model to use (from config)
        **kwargs: Additional parameters to pass to the model
        
    Returns:
        List of embedding vectors
        
    Raises:
        RuntimeError: If the model server fails to start or the API request fails
    """
    # Use default model if not specified
    config = ModelConfig.load_from_yaml()
    if model_alias is None:
        model_alias = "embedding"  # Default embedding model
    
    # Make sure server is running with the right model
    server_manager = get_server_manager(config)
    server_running = await server_manager.ensure_server_running(model_alias, mode="ingestion")
    if not server_running:
        raise RuntimeError(f"Failed to start model server with model {model_alias}")
    
    # Get model config
    model_config = config.get_model_config(model_alias, mode="ingestion")
    
    # Create adapter
    adapter = VLLMAdapter(
        model_name=model_config.model_id,
        server_url=f"http://localhost:{config.server.port}",
        normalize_embeddings=True,
        **kwargs
    )
    
    # Generate embeddings
    return adapter.get_embeddings(texts)
