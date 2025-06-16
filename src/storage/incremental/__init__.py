"""
Incremental Storage Module for HADES

This module provides ArangoDB-based incremental storage capabilities,
enabling efficient document ingestion and ISNE model updates without
full reprocessing.
"""

from .manager import IncrementalManager
from .schema import SchemaManager, IncrementalSchema
from .graph_builder import GraphBuilder
from .model_updater import ModelUpdater
from .conflict_resolver import ConflictResolver
from .types import (
    IncrementalConfig,
    IngestionResult,
    ConflictStrategy,
    DocumentState,
    ModelUpdateResult
)

__all__ = [
    'IncrementalManager',
    'SchemaManager', 
    'IncrementalSchema',
    'GraphBuilder',
    'ModelUpdater',
    'ConflictResolver',
    'IncrementalConfig',
    'IngestionResult',
    'ConflictStrategy',
    'DocumentState',
    'ModelUpdateResult'
]