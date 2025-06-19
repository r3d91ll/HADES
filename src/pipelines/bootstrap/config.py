"""
Bootstrap Pipeline Configuration

Configuration classes and utilities for the Sequential-ISNE Bootstrap Pipeline.
"""

from typing import Dict, List, Set, Any, Optional
from pathlib import Path
from pydantic import BaseModel, Field, field_validator
from enum import Enum


class FileTypeFilter(str, Enum):
    """File type filtering options."""
    ALL = "all"
    CODE_ONLY = "code_only"
    DOCS_ONLY = "docs_only"
    CONFIG_ONLY = "config_only"
    CODE_AND_DOCS = "code_and_docs"


class EdgeDiscoveryMethod(str, Enum):
    """Methods for discovering edges between files."""
    DIRECTORY_STRUCTURE = "directory_structure"
    IMPORT_ANALYSIS = "import_analysis"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    FILE_REFERENCES = "file_references"
    CO_LOCATION = "co_location"


class BootstrapConfig(BaseModel):
    """Configuration for Bootstrap Pipeline."""
    
    # Input/Output Configuration
    input_directory: Path = Field(..., description="Root directory to bootstrap from")
    output_database: str = Field(default="sequential_isne", description="Target database name")
    
    # File Processing Configuration
    file_type_filter: FileTypeFilter = Field(default=FileTypeFilter.ALL, description="Types of files to include")
    max_file_size: int = Field(default=10 * 1024 * 1024, description="Maximum file size in bytes")
    exclude_patterns: List[str] = Field(
        default_factory=lambda: [
            "*.pyc", "*.pyo", "*.pyd", "__pycache__", ".git", ".svn",
            "node_modules", ".venv", "venv", "*.log", "*.tmp"
        ],
        description="File patterns to exclude"
    )
    include_extensions: Set[str] = Field(
        default_factory=lambda: {
            ".py", ".js", ".ts", ".java", ".cpp", ".c", ".h", ".rs", ".go",
            ".md", ".txt", ".rst", ".pdf", ".html", ".tex",
            ".json", ".yaml", ".yml", ".toml", ".ini", ".xml", ".env", ".conf"
        },
        description="File extensions to include"
    )
    
    # Graph Construction Configuration
    enable_directory_bootstrap: bool = Field(default=True, description="Enable directory structure analysis")
    enable_import_analysis: bool = Field(default=True, description="Enable import statement analysis")
    enable_semantic_similarity: bool = Field(default=True, description="Enable semantic similarity analysis")
    enable_cross_modal_discovery: bool = Field(default=True, description="Enable cross-modal edge discovery")
    
    # Edge Discovery Configuration
    edge_discovery_methods: List[EdgeDiscoveryMethod] = Field(
        default_factory=lambda: [
            EdgeDiscoveryMethod.DIRECTORY_STRUCTURE,
            EdgeDiscoveryMethod.IMPORT_ANALYSIS,
            EdgeDiscoveryMethod.FILE_REFERENCES,
            EdgeDiscoveryMethod.CO_LOCATION
        ],
        description="Methods to use for edge discovery"
    )
    
    # Similarity Thresholds
    semantic_similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Threshold for semantic similarity edges")
    co_location_weight: float = Field(default=0.8, ge=0.0, le=1.0, description="Weight for co-location edges")
    import_weight: float = Field(default=0.9, ge=0.0, le=1.0, description="Weight for import edges")
    directory_hierarchy_weight: float = Field(default=0.6, ge=0.0, le=1.0, description="Weight for directory hierarchy edges")
    
    # Processing Configuration
    batch_size: int = Field(default=100, ge=1, description="Batch size for file processing")
    max_workers: int = Field(default=4, ge=1, description="Maximum worker threads")
    chunk_size: int = Field(default=512, ge=1, description="Text chunk size for analysis")
    chunk_overlap: int = Field(default=50, ge=0, description="Overlap between chunks")
    
    # Storage Configuration
    storage_component: str = Field(default="arangodb_v2", description="Storage component to use")
    storage_config: Dict[str, Any] = Field(default_factory=dict, description="Storage-specific configuration")
    
    # Performance Configuration
    enable_parallel_processing: bool = Field(default=True, description="Enable parallel file processing")
    enable_caching: bool = Field(default=True, description="Enable result caching")
    cache_ttl: int = Field(default=3600, description="Cache TTL in seconds")
    
    # Logging Configuration
    log_level: str = Field(default="INFO", description="Logging level")
    enable_progress_tracking: bool = Field(default=True, description="Enable progress tracking")
    
    @field_validator('input_directory')
    @classmethod
    def validate_input_directory(cls, v):
        """Validate input directory exists."""
        path = Path(v)
        if not path.exists():
            raise ValueError(f"Input directory does not exist: {path}")
        if not path.is_dir():
            raise ValueError(f"Input path is not a directory: {path}")
        return path
    
    @field_validator('max_file_size')
    @classmethod
    def validate_max_file_size(cls, v):
        """Validate max file size is reasonable."""
        if v < 1024:  # 1KB minimum
            raise ValueError("max_file_size must be at least 1024 bytes")
        if v > 100 * 1024 * 1024:  # 100MB maximum
            raise ValueError("max_file_size cannot exceed 100MB")
        return v
    
    class Config:
        use_enum_values = True


class BootstrapMetrics(BaseModel):
    """Metrics collected during bootstrap process."""
    
    # File Processing Metrics
    total_files_discovered: int = 0
    total_files_processed: int = 0
    files_by_type: Dict[str, int] = Field(default_factory=dict)
    processing_errors: int = 0
    
    # Graph Construction Metrics  
    total_nodes_created: int = 0
    total_edges_created: int = 0
    total_chunks_created: int = 0
    total_embeddings_created: int = 0
    edges_by_type: Dict[str, int] = Field(default_factory=dict)
    cross_modal_edges: int = 0
    
    # Performance Metrics
    total_processing_time: float = 0.0
    avg_file_processing_time: float = 0.0
    database_write_time: float = 0.0
    
    # Quality Metrics
    graph_density: float = 0.0
    connected_components: int = 0
    largest_component_size: int = 0
    modality_distribution: Dict[str, int] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True


class BootstrapResult(BaseModel):
    """Result of bootstrap pipeline execution."""
    
    success: bool = Field(description="Whether bootstrap completed successfully")
    config: BootstrapConfig = Field(description="Configuration used")
    metrics: BootstrapMetrics = Field(description="Metrics collected")
    database_name: str = Field(description="Database where graph was stored")
    node_count: int = Field(description="Total nodes in created graph")
    edge_count: int = Field(description="Total edges in created graph")
    errors: List[str] = Field(default_factory=list, description="Errors encountered")
    warnings: List[str] = Field(default_factory=list, description="Warnings generated")
    
    class Config:
        use_enum_values = True