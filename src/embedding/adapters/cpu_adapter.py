"""CPU-optimized adapter for embedding generation.

This module provides an implementation of the EmbeddingAdapter protocol
that uses lightweight models optimized for CPU usage.
"""

from __future__ import annotations

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Union, Any, cast

import numpy as np

from src.types.common import EmbeddingVector
from src.embedding.base import EmbeddingAdapter, register_adapter

logger = logging.getLogger(__name__)

# Global thread pool executor for CPU-bound tasks
_CPU_EXECUTOR = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)


class CPUEmbeddingAdapter:
    """Adapter for generating embeddings using CPU-optimized models."""
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",  # Lightweight model for CPU
        max_length: int = 512,
        batch_size: int = 32,
        **kwargs
    ):
        """Initialize the CPU embedding adapter.
        
        Args:
            model_name: Name of the model to use for embeddings
            max_length: Maximum sequence length for tokenization
            batch_size: Batch size for processing
            **kwargs: Additional parameters for the embedding model
        """
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.additional_params = kwargs
        self._model = None
        
    @property
    def model(self):
        """Lazy-load the model when first needed."""
        if self._model is None:
            try:
                # Import sentence-transformers here to avoid dependency issues
                from sentence_transformers import SentenceTransformer
                
                logger.info(f"Loading CPU embedding model: {self.model_name}")
                self._model = SentenceTransformer(self.model_name, device="cpu")
                logger.info(f"Successfully loaded model {self.model_name}")
            except ImportError:
                logger.error("Failed to import sentence-transformers. Please install it with: pip install sentence-transformers")
                raise
            except Exception as e:
                logger.error(f"Failed to load model {self.model_name}: {e}")
                raise
        
        return self._model
    
    async def embed(self, texts: List[str], **kwargs) -> List[EmbeddingVector]:
        """Generate embeddings for a list of texts using CPU.
        
        Args:
            texts: List of text strings to embed
            **kwargs: Additional parameters to pass to the model
            
        Returns:
            List of embedding vectors, one for each input text
            
        Raises:
            RuntimeError: If the embedding operation fails
        """
        if not texts:
            return []
        
        try:
            # Run in thread pool to avoid blocking event loop
            loop = asyncio.get_event_loop()
            
            # Set additional parameters
            batch_size = kwargs.get("batch_size", self.batch_size)
            
            # Execute embedding generation in thread pool
            embeddings = await loop.run_in_executor(
                _CPU_EXECUTOR,
                lambda: self.model.encode(
                    texts,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    normalize_embeddings=True  # L2 normalize for cosine similarity
                )
            )
            
            # Convert to list of lists for JSON serialization if numpy arrays
            if isinstance(embeddings, np.ndarray):
                embeddings = embeddings.tolist()
            
            return cast(List[EmbeddingVector], embeddings)
        
        except Exception as e:
            logger.error(f"CPU embedding generation failed: {e}")
            raise RuntimeError(f"Failed to generate embeddings: {e}") from e
    
    async def embed_single(self, text: str, **kwargs) -> EmbeddingVector:
        """Generate an embedding for a single text.
        
        Args:
            text: Text string to embed
            **kwargs: Additional parameters to pass to the model
            
        Returns:
            Embedding vector for the input text
            
        Raises:
            RuntimeError: If the embedding operation fails
        """
        results = await self.embed([text], **kwargs)
        if not results:
            raise RuntimeError("Empty results from embedding model")
        return results[0]


# Register the adapter
register_adapter("cpu", CPUEmbeddingAdapter)
