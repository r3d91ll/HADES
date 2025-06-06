"""
Pipeline stages for HADES-PathRAG orchestration system.

This module provides the consolidated stage-based pipeline architecture
that combines the best features from both the original pipeline system
and the orchestration framework.
"""

from .base import (
    PipelineStage,
    PipelineStageResult,
    PipelineStageError,
    PipelineStageStatus
)

from .document_processor import DocumentProcessorStage
from .chunking import ChunkingStage
from .embedding import EmbeddingStage
from .isne import ISNEStage
from .storage import StorageStage

__all__ = [
    "PipelineStage",
    "PipelineStageResult",
    "PipelineStageError",
    "PipelineStageStatus",
    "DocumentProcessorStage",
    "ChunkingStage",
    "EmbeddingStage",
    "ISNEStage",
    "StorageStage"
]