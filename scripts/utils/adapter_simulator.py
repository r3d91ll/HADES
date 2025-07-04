"""
Adapter transformation simulator for testing and mocking Jina v4 LoRA adapters.

This module provides lightweight transformations that simulate the behavior of
different LoRA adapters without loading actual adapter weights. Useful for:
- Testing and debugging
- Quick experiments with task-specific transformations
- Understanding which embedding dimensions matter for different tasks
"""

import torch
import numpy as np
from typing import Union


def simulate_adapter_transform(
    embeddings: Union[torch.Tensor, np.ndarray], 
    adapter: str
) -> Union[torch.Tensor, np.ndarray]:
    """
    Apply simulated adapter-specific transformations to embeddings.
    
    This simulates the effect of LoRA adapters without loading actual weights.
    Useful for testing and experimentation.
    
    Args:
        embeddings: Input embeddings (torch.Tensor or numpy array)
        adapter: Adapter name ('text-matching', 'classification', 'retrieval')
        
    Returns:
        Transformed embeddings (same type as input)
    """
    is_numpy = isinstance(embeddings, np.ndarray)
    
    # Convert to torch if needed
    if is_numpy:
        embeddings = torch.from_numpy(embeddings)
        
    if adapter == 'text-matching':
        # Emphasize certain dimensions for text matching
        # Boost first 512 dimensions which often capture semantic similarity
        scale = torch.ones_like(embeddings[0] if len(embeddings.shape) > 1 else embeddings)
        scale[:512] *= 1.2  # Boost first 512 dimensions
        result = embeddings * scale
        
    elif adapter == 'classification':
        # Apply slight transformation for classification tasks
        # This dampening can help with classification boundaries
        result = embeddings * 0.95 + 0.05
        
    elif adapter == 'code':
        # For code embeddings, emphasize structural dimensions
        # Middle dimensions often capture syntax patterns
        scale = torch.ones_like(embeddings[0] if len(embeddings.shape) > 1 else embeddings)
        mid_start = len(scale) // 3
        mid_end = 2 * len(scale) // 3
        scale[mid_start:mid_end] *= 1.1
        result = embeddings * scale
        
    else:
        # Default retrieval adapter - no transformation
        result = embeddings
        
    # Convert back to numpy if needed
    if is_numpy:
        result = result.numpy()
        
    return result


def get_adapter_description(adapter: str) -> str:
    """Get a description of what the adapter transformation does."""
    descriptions = {
        'text-matching': 'Boosts first 512 dimensions by 1.2x for semantic similarity',
        'classification': 'Scales by 0.95 and adds 0.05 for better classification boundaries',
        'code': 'Boosts middle dimensions by 1.1x for structural patterns',
        'retrieval': 'No transformation - preserves original embeddings'
    }
    return descriptions.get(adapter, 'Unknown adapter')


if __name__ == '__main__':
    # Example usage
    import numpy as np
    
    # Create sample embeddings
    embeddings = np.random.randn(10, 2048).astype(np.float32)
    
    # Test different adapters
    for adapter in ['text-matching', 'classification', 'code', 'retrieval']:
        transformed = simulate_adapter_transform(embeddings, adapter)
        print(f"{adapter}: {get_adapter_description(adapter)}")
        print(f"  Shape: {transformed.shape}, Mean diff: {np.mean(np.abs(transformed - embeddings)):.4f}\n")