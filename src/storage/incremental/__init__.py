"""
Incremental Storage Module for HADES

This module provides ArangoDB-based incremental storage capabilities,
enabling efficient document ingestion and ISNE model updates without
full reprocessing.

Includes Sequential-ISNE modality-specific schema for cross-modal discovery.
"""

from .manager import IncrementalManager
from .schema import SchemaManager, IncrementalSchema
from .sequential_isne_schema import SequentialISNESchema, SequentialISNESchemaManager
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
from .sequential_isne_types import (
    SequentialISNEConfig,
    FileType,
    EdgeType,
    CodeFile,
    DocumentationFile,
    ConfigFile,
    Chunk,
    Embedding,
    IntraModalEdge,
    CrossModalEdge,
    ISNEModel,
    DirectoryStructure,
    ProcessingLog,
    classify_file_type,
    get_modality_collection,
    is_cross_modal_edge
)

__all__ = [
    # Original incremental storage
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
    'ModelUpdateResult',
    
    # Sequential-ISNE modality-specific components
    'SequentialISNESchema',
    'SequentialISNESchemaManager',
    'SequentialISNEConfig',
    
    # Enums and types
    'FileType',
    'EdgeType',
    
    # Data models
    'CodeFile',
    'DocumentationFile', 
    'ConfigFile',
    'Chunk',
    'Embedding',
    'IntraModalEdge',
    'CrossModalEdge',
    'ISNEModel',
    'DirectoryStructure',
    'ProcessingLog',
    
    # Utility functions
    'classify_file_type',
    'get_modality_collection',
    'is_cross_modal_edge'
]