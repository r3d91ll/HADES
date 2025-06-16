"""
Type definitions for incremental storage module.
"""

from typing import List, Dict, Any, Optional, Union
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from pydantic import BaseModel, Field


class ConflictStrategy(str, Enum):
    """Strategy for handling document conflicts."""
    SKIP = "skip"           # Skip conflicting documents
    UPDATE = "update"       # Update existing documents
    MERGE = "merge"         # Merge document content
    KEEP_BOTH = "keep_both" # Keep both versions


class DocumentState(str, Enum):
    """Document processing state."""
    NEW = "new"
    UPDATED = "updated"
    UNCHANGED = "unchanged"
    CONFLICT = "conflict"
    ERROR = "error"


class ModelExpansionStrategy(str, Enum):
    """Strategy for expanding ISNE model capacity."""
    GRADUAL = "gradual"     # Gradual capacity increase
    IMMEDIATE = "immediate" # Immediate full expansion
    ADAPTIVE = "adaptive"   # Adaptive based on data


class IncrementalConfig(BaseModel):
    """Configuration for incremental storage operations."""
    
    # Database settings
    db_name: str = Field(default="hades_incremental", description="ArangoDB database name")
    connection_pool_size: int = Field(default=10, description="Connection pool size")
    
    # Processing settings
    batch_size: int = Field(default=100, description="Batch size for processing")
    max_workers: int = Field(default=4, description="Maximum worker threads")
    
    # Conflict resolution
    conflict_strategy: ConflictStrategy = Field(default=ConflictStrategy.UPDATE, description="Conflict resolution strategy")
    similarity_threshold: float = Field(default=0.85, description="Similarity threshold for duplicates")
    
    # Graph construction
    edge_similarity_threshold: float = Field(default=0.4, description="Threshold for creating edges")
    max_edges_per_node: int = Field(default=50, description="Maximum edges per node")
    
    # Model settings
    model_expansion_strategy: ModelExpansionStrategy = Field(default=ModelExpansionStrategy.GRADUAL, description="Model expansion strategy")
    expansion_factor: float = Field(default=1.2, description="Factor for model expansion")
    
    # Caching
    enable_caching: bool = Field(default=True, description="Enable result caching")
    cache_ttl: int = Field(default=3600, description="Cache TTL in seconds")
    
    # Performance
    enable_parallel_processing: bool = Field(default=True, description="Enable parallel processing")
    chunk_size_limit: int = Field(default=10000, description="Maximum chunk size")


@dataclass
class DocumentInfo:
    """Information about a document."""
    file_path: str
    content_hash: str
    size: int
    modified_time: datetime
    state: DocumentState
    error_message: Optional[str] = None


@dataclass
class ChunkInfo:
    """Information about a chunk."""
    chunk_id: str
    document_id: str
    content: str
    content_hash: str
    start_pos: int
    end_pos: int
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None


@dataclass
class NodeInfo:
    """Information about a graph node."""
    node_id: str
    chunk_id: str
    embedding_id: str
    created_at: datetime
    metadata: Dict[str, Any]


@dataclass
class EdgeInfo:
    """Information about a graph edge."""
    edge_id: str
    from_node: str
    to_node: str
    weight: float
    edge_type: str
    created_at: datetime
    metadata: Dict[str, Any]


class IngestionResult(BaseModel):
    """Result of document ingestion operation."""
    
    # Processing statistics
    total_documents: int = Field(description="Total documents processed")
    new_documents: int = Field(description="Number of new documents")
    updated_documents: int = Field(description="Number of updated documents")
    unchanged_documents: int = Field(description="Number of unchanged documents")
    error_documents: int = Field(description="Number of documents with errors")
    
    # Content statistics
    total_chunks: int = Field(description="Total chunks created")
    new_chunks: int = Field(description="Number of new chunks")
    total_embeddings: int = Field(description="Total embeddings generated")
    
    # Graph statistics
    total_nodes: int = Field(description="Total graph nodes")
    new_nodes: int = Field(description="Number of new nodes")
    total_edges: int = Field(description="Total graph edges")
    new_edges: int = Field(description="Number of new edges")
    
    # Model statistics
    previous_model_size: int = Field(description="Previous model node count")
    new_model_size: int = Field(description="New model node count")
    model_expanded: bool = Field(description="Whether model was expanded")
    
    # Performance metrics
    processing_time_seconds: float = Field(description="Total processing time")
    documents_per_second: float = Field(description="Processing rate")
    
    # Detailed results
    document_results: List[DocumentInfo] = Field(default_factory=list, description="Per-document results")
    conflicts: List[Dict[str, Any]] = Field(default_factory=list, description="Conflict details")
    errors: List[Dict[str, Any]] = Field(default_factory=list, description="Error details")


class ModelUpdateResult(BaseModel):
    """Result of model update operation."""
    
    # Update statistics
    success: bool = Field(description="Whether update succeeded")
    previous_version: str = Field(description="Previous model version")
    new_version: str = Field(description="New model version")
    
    # Model metrics
    previous_node_count: int = Field(description="Previous node count")
    new_node_count: int = Field(description="New node count")
    nodes_added: int = Field(description="Number of nodes added")
    
    # Training metrics
    training_epochs: int = Field(description="Number of training epochs")
    final_loss: float = Field(description="Final training loss")
    convergence_achieved: bool = Field(description="Whether training converged")
    
    # Performance metrics
    training_time_seconds: float = Field(description="Training time")
    model_size_mb: float = Field(description="Model size in MB")
    
    # Paths and metadata
    model_path: str = Field(description="Path to saved model")
    backup_path: Optional[str] = Field(default=None, description="Path to backup model")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ConflictResolution(BaseModel):
    """Details of conflict resolution."""
    
    document_id: str = Field(description="Document ID")
    file_path: str = Field(description="File path")
    strategy_used: ConflictStrategy = Field(description="Resolution strategy used")
    original_hash: str = Field(description="Original content hash")
    new_hash: str = Field(description="New content hash")
    resolution_time: datetime = Field(description="When conflict was resolved")
    changes_summary: str = Field(description="Summary of changes")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class VersionInfo(BaseModel):
    """Version information for models and data."""
    
    version_id: str = Field(description="Unique version identifier")
    version_number: str = Field(description="Human-readable version number")
    created_at: datetime = Field(description="Creation timestamp")
    created_by: str = Field(description="Creator identifier")
    description: str = Field(description="Version description")
    
    # Data metrics
    document_count: int = Field(description="Number of documents")
    chunk_count: int = Field(description="Number of chunks")
    node_count: int = Field(description="Number of graph nodes")
    edge_count: int = Field(description="Number of graph edges")
    
    # Model metrics
    model_parameters: int = Field(description="Number of model parameters")
    embedding_dim: int = Field(description="Embedding dimension")
    
    # File paths
    model_path: Optional[str] = Field(default=None, description="Path to model file")
    data_snapshot_path: Optional[str] = Field(default=None, description="Path to data snapshot")
    
    # Parent version (for tracking lineage)
    parent_version: Optional[str] = Field(default=None, description="Parent version ID")
    
    # Tags and metadata
    tags: List[str] = Field(default_factory=list, description="Version tags")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class CacheConfig(BaseModel):
    """Configuration for caching layer."""
    
    enabled: bool = Field(default=True, description="Enable caching")
    backend: str = Field(default="redis", description="Cache backend")
    host: str = Field(default="localhost", description="Cache host")
    port: int = Field(default=6379, description="Cache port")
    db: int = Field(default=0, description="Cache database")
    password: Optional[str] = Field(default=None, description="Cache password")
    
    # TTL settings
    default_ttl: int = Field(default=3600, description="Default TTL in seconds")
    embedding_ttl: int = Field(default=86400, description="Embedding cache TTL")
    similarity_ttl: int = Field(default=7200, description="Similarity cache TTL")
    
    # Key prefixes
    embedding_prefix: str = Field(default="emb:", description="Embedding cache key prefix")
    similarity_prefix: str = Field(default="sim:", description="Similarity cache key prefix")
    graph_prefix: str = Field(default="graph:", description="Graph cache key prefix")


class DatabaseStatus(BaseModel):
    """Database status information."""
    
    # Connection status
    connected: bool = Field(description="Database connection status")
    db_name: str = Field(description="Database name")
    
    # Collection information
    collections: Dict[str, int] = Field(description="Collection document counts")
    
    # Index information
    indexes: Dict[str, List[str]] = Field(description="Index information per collection")
    
    # Schema version
    schema_version: str = Field(description="Current schema version")
    
    # Statistics
    total_documents: int = Field(description="Total documents across all collections")
    database_size_mb: float = Field(description="Database size in MB")
    
    # Health metrics
    last_updated: datetime = Field(description="Last update timestamp")
    uptime_seconds: float = Field(description="Database uptime")
    
    # Error information
    errors: List[str] = Field(default_factory=list, description="Recent errors")
    warnings: List[str] = Field(default_factory=list, description="Recent warnings")