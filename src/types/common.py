"""
Common type definitions for HADES-PathRAG.

This module provides common type annotations used across the codebase
to ensure consistency and improve type safety.
"""

from typing import Dict, List, Any, Optional, Union, TypedDict, NewType, ForwardRef
import numpy as np
from pathlib import Path
from datetime import datetime

# Basic type aliases
NodeID = NewType('NodeID', str)
EdgeID = NewType('EdgeID', str)
DocumentContent = NewType('DocumentContent', str)
EmbeddingVector = Union[List[float], np.ndarray]

# Code structure types
class Module(TypedDict, total=False):
    """Type definition for a Python module in the codebase."""
    path: str
    name: str
    content: str
    docstring: Optional[str]
    functions: List[Dict[str, Any]]
    classes: List[Dict[str, Any]]
    imports: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]


class DocumentationFile(TypedDict, total=False):
    """Type definition for a documentation file in the codebase."""
    path: str
    id: str
    type: str
    content: str
    title: Optional[str]
    headings: List[Dict[str, Any]]
    code_blocks: List[Dict[str, Any]]
    references: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]


# Structured data types

class StoreResults(TypedDict):
    node_count: int
    edge_count: int
    root_id: str
    failed_nodes: List[str]
    failed_edges: List[Dict[str, str]]

class IngestStats(TypedDict, total=False):
    """Type definition for ingestion statistics."""
    dataset_name: str
    directory: str
    start_time: str
    end_time: str
    duration_seconds: float
    file_stats: Dict[str, Any]
    document_count: int
    relationship_count: int
    storage_stats: Dict[str, Any]
    # Additional optional statistics populated during ingestion
    files_discovered: int
    files_processed: int
    entities_extracted: int
    relationships_extracted: int
    entities_stored: int
    relationships_stored: int
    repository_stats: Dict[str, Any]
    status: str


class MetadataExtractionConfig(TypedDict, total=False):
    """Configuration for metadata extraction."""
    extract_title: bool
    extract_authors: bool
    extract_date: bool
    use_filename_as_title: bool
    detect_language: bool


class EntityExtractionConfig(TypedDict, total=False):
    """Configuration for entity extraction."""
    extract_named_entities: bool
    extract_technical_terms: bool
    min_confidence: float


class ChunkingPreparationConfig(TypedDict, total=False):
    """Configuration for preparing content for chunking."""
    add_section_markers: bool
    preserve_metadata: bool
    mark_chunk_boundaries: bool


class PreProcessorConfig(TypedDict, total=False):
    """Complete configuration for document preprocessing."""
    input_dir: Path
    output_dir: Path
    exclude_patterns: List[str]
    recursive: bool
    max_workers: int
    file_type_map: Dict[str, List[str]]
    preprocessor_config: Dict[str, Dict[str, Any]]
    metadata_extraction: MetadataExtractionConfig
    entity_extraction: EntityExtractionConfig
    chunking_preparation: ChunkingPreparationConfig
    options: Dict[str, Any]

class NodeData(TypedDict, total=False):
    """Type definition for node data stored in graph databases."""
    id: str
    type: str
    content: str
    title: Optional[str]
    source: str
    embedding: Optional[EmbeddingVector]
    embedding_model: Optional[str]
    created_at: Optional[Union[str, datetime]]
    updated_at: Optional[Union[str, datetime]]
    metadata: Dict[str, Any]


class EdgeData(TypedDict, total=False):
    """Type definition for edge data stored in graph databases."""
    id: str
    source_id: str
    target_id: str
    type: str
    weight: float
    bidirectional: bool
    created_at: Optional[Union[str, datetime]]
    updated_at: Optional[Union[str, datetime]]
    metadata: Dict[str, Any]


class StorageConfig(TypedDict, total=False):
    """Configuration for storage systems."""
    storage_type: str
    host: str
    port: int
    username: str
    password: str
    database: str
    collection_prefix: str
    use_vector_index: bool
    vector_dimensions: int
    working_dir: str
    cache_dir: str
    # Embedding configuration nested inside storage config
    embedding: 'EmbeddingConfig'


class EmbeddingConfig(TypedDict, total=False):
    """Configuration for embedding models."""
    model_name: str
    model_provider: str
    model_dimension: int
    batch_size: int
    use_gpu: bool
    normalize_embeddings: bool
    cache_embeddings: bool
    pooling_strategy: str
    max_length: int
    api_key: Optional[str]


class GraphConfig(TypedDict, total=False):
    """Configuration for graph processing."""
    min_edge_weight: float
    max_distance: int
    include_self_loops: bool
    bidirectional_edges: bool
    graph_name: str
    node_collection: str
    edge_collection: str
    graph_algorithm: str


class PathRankingConfig(TypedDict, total=False):
    """Configuration for the PathRAG path ranking algorithm."""
    semantic_weight: float  # Weight for semantic relevance (default: 0.7)
    path_length_weight: float  # Weight for path length penalty (default: 0.1)
    edge_strength_weight: float  # Weight for edge strength (default: 0.2)
    max_path_length: int  # Maximum length of paths to consider (default: 5)
    max_paths: int  # Maximum number of paths to return (default: 20)
