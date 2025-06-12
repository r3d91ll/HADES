"""
Graph Enhancement Components

This module provides graph enhancement components that implement the
GraphEnhancer protocol for various graph-based improvements to embeddings.

Available components:
- isne: Inductive Shallow Node Embedding (ISNE) graph enhancement
"""

from .isne import CoreISNE, ISNETrainingEnhancer, ISNEInferenceEnhancer

__all__ = [
    "CoreISNE",
    "ISNETrainingEnhancer", 
    "ISNEInferenceEnhancer"
]