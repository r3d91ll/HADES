"""vLLM adapter for embedding generation.

This module provides an implementation of the EmbeddingAdapter interface
that uses vLLM for generating embeddings.
"""

from __future__ import annotations

import aiohttp
import asyncio
import json
import logging
from typing import Dict, List, Optional, Union, Any, cast
import numpy as np

from src.model_engine.vllm_session import VLLMSessionContext, get_vllm_base_url
from src.types.common import EmbeddingVector
from src.embedding.base import EmbeddingAdapter, register_adapter

logger = logging.getLogger(__name__)


class VLLMEmbeddingAdapter:
    """Adapter for generating embeddings using vLLM."""
    
    def __init__(
        self,
        model_alias: str = "default",
        base_url: Optional[str] = None,
        timeout: float = 60.0,
        **kwargs
    ):
        """Initialize the vLLM embedding adapter.
        
        Args:
            model_alias: Alias of the embedding model to use
            base_url: Base URL of the vLLM server, will be auto-detected if None
            timeout: Request timeout in seconds
            **kwargs: Additional parameters for the embedding API
        """
        self.model_alias = model_alias
        self.base_url = base_url
        self.timeout = timeout
        self.additional_params = kwargs
    
    async def _get_base_url(self) -> str:
        """Get the base URL for the vLLM API.
        
        Returns:
            Base URL for the API
            
        Raises:
            RuntimeError: If the base URL cannot be determined
        """
        if self.base_url is not None:
            return self.base_url
        
        # Try to get base URL from the vLLM session manager
        try:
            return await get_vllm_base_url(self.model_alias)
        except Exception as e:
            raise RuntimeError(f"Failed to get vLLM base URL: {e}") from e
    
    async def embed(self, texts: List[str], **kwargs) -> List[EmbeddingVector]:
        """Generate embeddings for a list of texts using vLLM.
        
        Args:
            texts: List of text strings to embed
            **kwargs: Additional parameters to pass to the model API
            
        Returns:
            List of embedding vectors, one for each input text
            
        Raises:
            RuntimeError: If the API request fails
        """
        if not texts:
            return []
        
        base_url = await self._get_base_url()
        embed_url = f"{base_url}/v1/embeddings"
        
        # Prepare API parameters
        params = {
            "input": texts,
            "model": self.model_alias,
            **self.additional_params,
            **kwargs
        }
        
        # Make API request
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    embed_url,
                    json=params,
                    timeout=self.timeout
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise RuntimeError(
                            f"vLLM API request failed with status {response.status}: {error_text}"
                        )
                    
                    result = await response.json()
                    
                    # Extract embeddings from response
                    embeddings = [
                        data["embedding"] for data in result["data"]
                    ]
                    return cast(List[EmbeddingVector], embeddings)
            
            except asyncio.TimeoutError:
                raise RuntimeError(f"vLLM API request timed out after {self.timeout} seconds")
            except Exception as e:
                raise RuntimeError(f"vLLM API request failed: {e}") from e
    
    async def embed_single(self, text: str, **kwargs) -> EmbeddingVector:
        """Generate an embedding for a single text using vLLM.
        
        Args:
            text: Text string to embed
            **kwargs: Additional parameters to pass to the model API
            
        Returns:
            Embedding vector for the input text
            
        Raises:
            RuntimeError: If the API request fails
        """
        results = await self.embed([text], **kwargs)
        if not results:
            raise RuntimeError("vLLM API returned empty results")
        return results[0]


# Register the adapter
register_adapter("vllm", VLLMEmbeddingAdapter)
