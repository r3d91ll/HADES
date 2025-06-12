"""
HADES Pipeline System

This module provides the pipeline infrastructure for HADES, including
data ingestion, bootstrap, and training pipelines.

Currently includes:
- Data Ingestion Pipeline (reorganized from legacy stages)
- Bootstrap Pipeline (for ISNE model initialization)  
- Training Pipeline (for ongoing model training)
- Debug and test utilities
"""

# Import from reorganized pipeline structure
from .data_ingestion import (
    PipelineStage,
    PipelineStageResult,
    PipelineStageStatus,
    PipelineStageError,
    DocumentProcessorStage,
    ChunkingStage,
    EmbeddingStage,
    ISNEStage,
    StorageStage
)

# Import pipeline implementations
try:
    from .data_ingestion.pipeline import DataIngestionPipeline
except ImportError:
    # Handle case where pipeline hasn't been moved yet
    pass

try:
    from .bootstrap.pipeline import BootstrapPipeline, run_bootstrap_pipeline
except ImportError:
    # Handle case where bootstrap pipeline isn't fully set up
    pass

# Re-export for backward compatibility
__all__ = [
    # Stage classes
    'PipelineStage',
    'PipelineStageResult',
    'PipelineStageStatus', 
    'PipelineStageError',
    'DocumentProcessorStage',
    'ChunkingStage',
    'EmbeddingStage',
    'ISNEStage',
    'StorageStage',
    
    # Pipeline classes
    'BootstrapPipeline',
    'run_bootstrap_pipeline'
]

# Try to add DataIngestionPipeline if available
try:
    __all__.append('DataIngestionPipeline')
except NameError:
    pass
