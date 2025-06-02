"""
ISNE Pipeline package.

This package provides production-ready pipeline implementations for
processing documents with ISNE embeddings.
"""

from src.isne.pipeline.isne_pipeline import ISNEPipeline
from src.isne.pipeline.config import PipelineConfig

__all__ = ["ISNEPipeline", "PipelineConfig"]
