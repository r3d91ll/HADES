"""
ISNE (Inductive Shallow Node Embedding) module.

This module implements the complete ISNE framework for graph-based embeddings
as described in "Unsupervised Graph Representation Learning with Inductive Shallow Node Embedding".
"""

from .models.isne_model import ISNEModel

__all__ = [
    'ISNEModel'
]