"""
Data Ingestion Pipeline

This module contains stages and utilities for data ingestion pipelines.
These stages are reusable components that can be used across different
pipeline types (data ingestion, bootstrap, training, etc.).
"""

# Re-export stages for backward compatibility
from .stages.base import PipelineStage, PipelineStageResult, PipelineStageStatus, PipelineStageError
from .stages.document_processor import DocumentProcessorStage
from .stages.chunking import ChunkingStage
from .stages.embedding import EmbeddingStage
from .stages.isne import ISNEStage
from .stages.storage import StorageStage

__all__ = [
    'PipelineStage',
    'PipelineStageResult', 
    'PipelineStageStatus',
    'PipelineStageError',
    'DocumentProcessorStage',
    'ChunkingStage',
    'EmbeddingStage',
    'ISNEStage',
    'StorageStage'
]