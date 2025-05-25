"""Ollama adapter for embedding generation.

This module provides an implementation of the EmbeddingAdapter interface
that uses Ollama for generating embeddings.
"""

from __future__ import annotations

import aiohttp
import asyncio
import json
import logging
from typing import Dict, List, Optional, Union, Any, cast
import numpy as np

from src.embedding.base import EmbeddingAdapter, register_adapter
from src.types.common import EmbeddingVector

logger = logging.getLogger(__name__)


class OllamaEmbeddingAdapter:
    """Adapter for generating embeddings using Ollama."""
    
    def __init__(
        self,
        model_name: str = "llama3",
        base_url: str = "http://localhost:11434",
        timeout: float = 60.0,
        **kwargs
    ):
        """Initialize the Ollama embedding adapter.
        
        Args:
            model_name: Name of the Ollama model to use
            base_url: Base URL of the Ollama API server
            timeout: Request timeout in seconds
            **kwargs: Additional parameters for the embedding API
        """
        self.model_name = model_name
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.additional_params = kwargs
    
    async def embed(self, texts: List[str], **kwargs) -> List[EmbeddingVector]:
        """Generate embeddings for a list of texts using Ollama.
        
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
        
        embed_url = f"{self.base_url}/api/embeddings"
        
        # We'll process each text individually since Ollama's API
        # doesn't support batching embeddings in a single request
        async with aiohttp.ClientSession() as session:
            tasks = []
            for text in texts:
                params = {
                    "model": self.model_name,
                    "prompt": text,
                    **self.additional_params,
                    **kwargs
                }
                tasks.append(self._embed_single_request(session, embed_url, params))
            
            try:
                results = await asyncio.gather(*tasks)
                return cast(List[EmbeddingVector], results)
            except Exception as e:
                raise RuntimeError(f"Ollama embedding API request failed: {e}") from e
    
    async def _embed_single_request(
        self, 
        session: aiohttp.ClientSession, 
        url: str, 
        params: Dict[str, Any]
    ) -> EmbeddingVector:
        """Make a single embedding request to the Ollama API.
        
        Args:
            session: aiohttp client session
            url: URL to send the request to
            params: Parameters for the request
            
        Returns:
            Embedding vector from the response
            
        Raises:
            RuntimeError: If the API request fails
        """
        try:
            async with session.post(
                url,
                json=params,
                timeout=self.timeout
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(
                        f"Ollama API request failed with status {response.status}: {error_text}"
                    )
                
                result = await response.json()
                
                # Extract embedding from response
                if "embedding" not in result:
                    raise RuntimeError(f"Ollama API response missing 'embedding' field: {result}")
                
                return cast(EmbeddingVector, result["embedding"])
        
        except asyncio.TimeoutError:
            raise RuntimeError(f"Ollama API request timed out after {self.timeout} seconds")
        except Exception as e:
            raise RuntimeError(f"Ollama API request failed: {e}") from e
    
    async def embed_single(self, text: str, **kwargs) -> EmbeddingVector:
        """Generate an embedding for a single text using Ollama.
        
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
            raise RuntimeError("Ollama API returned empty results")
        return results[0]


# Register the adapter
register_adapter("ollama", OllamaEmbeddingAdapter)
