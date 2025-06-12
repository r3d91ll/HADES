"""Embedding extension for the Haystack model server.

This module extends the Haystack model server with embedding generation capabilities
specifically optimized for ModernBERT and other transformer models.
"""

from __future__ import annotations

import torch
import numpy as np
from typing import List, Dict, Any, Union, Tuple, Optional

# Import local modules
from src.model_engine.engines.haystack.runtime.server import _CACHE

def calculate_embeddings(
    model_id: str,
    texts: List[str],
    pooling_strategy: str = "cls",
    normalize: bool = True,
    max_length: Optional[int] = None
) -> Dict[str, Any]:
    """Calculate embeddings for a list of texts using a loaded model.
    
    Args:
        model_id: ID of the loaded model to use
        texts: List of texts to embed
        pooling_strategy: Strategy for pooling embeddings (cls, mean, max)
        normalize: Whether to L2-normalize the embeddings
        max_length: Maximum sequence length for tokenization
        
    Returns:
        Dictionary with embedding results
        
    Raises:
        ValueError: If the model is not loaded or pooling strategy is invalid
    """
    # Get the model and tokenizer from cache
    cached = _CACHE.get(model_id)
    if not cached:
        raise ValueError(f"Model {model_id} is not loaded")
    
    model, tokenizer = cached
    
    # Ensure model is in eval mode
    model.eval()
    
    # Process texts in batches to avoid OOM
    all_embeddings = []
    
    with torch.no_grad():
        # Tokenize and get model outputs
        encoding = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length or tokenizer.model_max_length,
            return_tensors="pt"
        )
        
        # Move to the same device as model
        input_ids = encoding["input_ids"].to(model.device)
        attention_mask = encoding["attention_mask"].to(model.device)
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Get embeddings based on pooling strategy
        if pooling_strategy == "cls":
            # Use CLS token embedding (first token)
            embeddings = outputs.last_hidden_state[:, 0, :]
        elif pooling_strategy == "mean":
            # Mean pooling - average token embeddings
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        elif pooling_strategy == "max":
            # Max pooling - take maximum values
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            embeddings = torch.max(token_embeddings * input_mask_expanded, dim=1)[0]
        else:
            raise ValueError(f"Invalid pooling strategy: {pooling_strategy}")
        
        # Normalize if requested
        if normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        # Convert to numpy and then to list for JSON serialization
        embeddings_np = embeddings.cpu().numpy()
        all_embeddings = embeddings_np.tolist()
    
    # Return results
    return {
        "embeddings": all_embeddings,
        "dimensions": embeddings.shape[1],
        "model_id": model_id
    }
