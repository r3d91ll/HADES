"""
ISNE Trainer module for HADES-PathRAG.

This module provides components for training the ISNE (Inductive Shallow Node Embedding) model
as part of the HADES-PathRAG system, enhancing document embeddings through graph-based learning.
"""

from src.isne.trainer.training_orchestrator import ISNETrainingOrchestrator

__all__ = ["ISNETrainingOrchestrator"]
