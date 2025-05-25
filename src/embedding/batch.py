"""Utilities for batch embedding operations.

This module provides functions for efficiently processing batches of texts
for embedding generation with support for concurrency limits and progress tracking.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Dict, List, Optional, TypeVar, cast
import logging
from tqdm.asyncio import tqdm as tqdm_async

from .base import EmbeddingAdapter, EmbeddingVector

logger = logging.getLogger(__name__)

T = TypeVar('T')


async def batch_embed(
    texts: List[str],
    adapter: EmbeddingAdapter,
    batch_size: int = 32,
    max_concurrency: int = 8,
    show_progress: bool = False,
    **kwargs
) -> List[EmbeddingVector]:
    """Embed a large batch of texts efficiently with concurrency limits.
    
    Args:
        texts: List of texts to embed
        adapter: Embedding adapter to use
        batch_size: Maximum number of texts to embed in a single API call
        max_concurrency: Maximum number of concurrent API calls
        show_progress: Whether to show a progress bar
        **kwargs: Additional parameters to pass to the embedding model
        
    Returns:
        List of embedding vectors, one for each input text
        
    Raises:
        RuntimeError: If the embedding operation fails
    """
    if not texts:
        return []
    
    # Split texts into batches
    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    
    # Create a semaphore to limit concurrency
    semaphore = asyncio.Semaphore(max_concurrency)
    
    async def process_batch(batch: List[str]) -> List[EmbeddingVector]:
        async with semaphore:
            return await adapter.embed(batch, **kwargs)
    
    # Process batches with progress tracking if requested
    tasks = [process_batch(batch) for batch in batches]
    
    if show_progress:
        results = await tqdm_async.gather(*tasks, desc="Generating embeddings")
    else:
        results = await asyncio.gather(*tasks)
    
    # Flatten results
    all_embeddings: List[EmbeddingVector] = []
    for batch_result in results:
        all_embeddings.extend(batch_result)
    
    return all_embeddings
