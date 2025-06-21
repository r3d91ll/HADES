"""
PathRAG Implementation

PathRAG (Path-based Retrieval Augmented Generation) implementation for HADES.
Based on the PathRAG paper: "PathRAG: Pruning Graph-based Retrieval Augmented 
Generation with Relational Paths".
"""

from .processor import PathRAGProcessor

__all__ = [
    "PathRAGProcessor"
]