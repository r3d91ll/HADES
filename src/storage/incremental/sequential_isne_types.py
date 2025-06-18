"""
Type definitions for Sequential-ISNE modality-specific storage.

This module provides Pydantic models and type definitions for the 
Sequential-ISNE modality-specific schema architecture.
"""

from typing import List, Dict, Any, Optional, Union, Set
from enum import Enum
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel, Field, field_validator
from dataclasses import dataclass


# ===== ENUMS =====

class FileType(str, Enum):
    """File type enumeration for modality classification."""
    CODE = "code"
    DOCUMENTATION = "documentation"
    CONFIG = "config"
    DATA = "data"
    UNKNOWN = "unknown"


class CodeFileType(str, Enum):
    """Specific code file types."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CPP = "cpp"
    C = "c"
    RUST = "rust"
    GO = "go"
    OTHER = "other"


class DocumentationType(str, Enum):
    """Specific documentation file types."""
    MARKDOWN = "markdown"
    TEXT = "text"
    RST = "rst"
    PDF = "pdf"
    HTML = "html"
    OTHER = "other"


class ConfigFileType(str, Enum):
    """Specific configuration file types."""
    JSON = "json"
    YAML = "yaml"
    TOML = "toml"
    INI = "ini"
    XML = "xml"
    ENV = "env"
    OTHER = "other"


class EdgeType(str, Enum):
    """Edge type enumeration for relationship classification."""
    # Intra-modal edges (within same modality)
    CODE_IMPORTS = "code_imports"
    CODE_CALLS = "code_calls"
    CODE_INHERITANCE = "code_inheritance"
    DOC_REFERENCES = "doc_references"
    DOC_HIERARCHY = "doc_hierarchy"
    CONFIG_INCLUDES = "config_includes"
    
    # Cross-modal edges (between different modalities)
    CODE_TO_DOC = "code_to_doc"           # Implementation → Documentation
    DOC_TO_CODE = "doc_to_code"           # Documentation → Implementation
    CODE_TO_CONFIG = "code_to_config"     # Code → Configuration
    CONFIG_TO_CODE = "config_to_code"     # Configuration → Code
    DOC_TO_CONFIG = "doc_to_config"       # Documentation → Configuration
    
    # Directory structure edges
    DIRECTORY_COLOCATION = "directory_colocation"
    DIRECTORY_HIERARCHY = "directory_hierarchy"
    
    # ISNE discovered edges
    ISNE_SIMILARITY = "isne_similarity"
    SEMANTIC_SIMILARITY = "semantic_similarity"


class ProcessingStatus(str, Enum):
    """Processing status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"
    SKIPPED = "skipped"


class OperationType(str, Enum):
    """Operation type for processing logs."""
    BOOTSTRAP = "bootstrap"
    TRAINING = "training"
    INGESTION = "ingestion"
    UPDATE = "update"


class EmbeddingType(str, Enum):
    """Embedding type classification."""
    ORIGINAL = "original"
    ISNE_ENHANCED = "isne_enhanced"


class SourceType(str, Enum):
    """Source type for embeddings."""
    FILE = "file"
    CHUNK = "chunk"


class ChunkType(str, Enum):
    """Chunk type classification."""
    TEXT = "text"
    CODE = "code"
    COMMENT = "comment"
    DOCSTRING = "docstring"
    CONFIG = "config"


# ===== BASE MODELS =====

class BaseFileModel(BaseModel):
    """Base model for all file types."""
    file_path: str = Field(..., description="Full path to the file")
    file_name: str = Field(..., description="Name of the file")
    directory: str = Field(..., description="Directory containing the file")
    extension: str = Field(..., description="File extension")
    content: str = Field(..., description="File content")
    content_hash: str = Field(..., description="SHA256 hash of content")
    size: int = Field(..., ge=0, description="File size in bytes")
    modified_time: datetime = Field(..., description="Last modification time")
    ingestion_time: datetime = Field(default_factory=datetime.now, description="Time when file was ingested")
    processing_status: ProcessingStatus = Field(default=ProcessingStatus.PENDING, description="Processing status")
    
    # Sequential-ISNE specific fields
    node_id: Optional[int] = Field(None, description="Node ID in the graph")
    embedding_id: Optional[str] = Field(None, description="Reference to embedding")
    chunk_count: int = Field(default=0, ge=0, description="Number of chunks extracted")
    directory_depth: int = Field(..., ge=0, description="Depth in directory hierarchy")
    
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        use_enum_values = True


# ===== MODALITY-SPECIFIC FILE MODELS =====

class CodeFile(BaseFileModel):
    """Model for code files."""
    file_type: CodeFileType = Field(..., description="Specific code file type")
    lines_of_code: int = Field(default=0, ge=0, description="Number of lines of code")
    
    # Code-specific metadata
    ast_metadata: Dict[str, Any] = Field(default_factory=dict, description="AST parsing metadata")
    imports: List[str] = Field(default_factory=list, description="Import statements")
    functions: List[Dict[str, Any]] = Field(default_factory=list, description="Function definitions")
    classes: List[Dict[str, Any]] = Field(default_factory=list, description="Class definitions")
    complexity_score: Optional[float] = Field(None, ge=0, description="Code complexity score")


class DocumentationFile(BaseFileModel):
    """Model for documentation files."""
    file_type: DocumentationType = Field(..., description="Specific documentation file type")
    word_count: int = Field(default=0, ge=0, description="Number of words")
    
    # Documentation-specific metadata
    document_structure: Dict[str, Any] = Field(default_factory=dict, description="Document structure")
    headings: List[Dict[str, Any]] = Field(default_factory=list, description="Document headings")
    links: List[str] = Field(default_factory=list, description="Links in the document")
    code_references: List[str] = Field(default_factory=list, description="References to code")
    readability_score: Optional[float] = Field(None, ge=0, description="Readability score")


class ConfigFile(BaseFileModel):
    """Model for configuration files."""
    file_type: ConfigFileType = Field(..., description="Specific config file type")
    
    # Config-specific metadata
    parsed_config: Dict[str, Any] = Field(default_factory=dict, description="Parsed configuration")
    config_schema: Dict[str, Any] = Field(default_factory=dict, description="Configuration schema")
    validation_status: Optional[str] = Field(None, description="Validation status")
    dependencies: List[str] = Field(default_factory=list, description="Configuration dependencies")


# ===== CHUNK MODEL =====

class Chunk(BaseModel):
    """Model for text chunks."""
    source_file_collection: str = Field(..., description="Source file collection name")
    source_file_id: str = Field(..., description="Source file ID")
    content: str = Field(..., description="Chunk content")
    content_hash: str = Field(..., description="SHA256 hash of chunk content")
    start_pos: int = Field(..., ge=0, description="Start position in source file")
    end_pos: int = Field(..., ge=0, description="End position in source file")
    chunk_index: int = Field(..., ge=0, description="Index of chunk in file")
    chunk_type: ChunkType = Field(..., description="Type of chunk")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation time")
    embedding_id: Optional[str] = Field(None, description="Reference to embedding")
    node_id: Optional[int] = Field(None, description="Node ID in the graph")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        use_enum_values = True


# ===== EMBEDDING MODEL =====

class Embedding(BaseModel):
    """Model for vector embeddings."""
    source_type: SourceType = Field(..., description="Type of source (file or chunk)")
    source_collection: str = Field(..., description="Source collection name")
    source_id: str = Field(..., description="Source ID")
    embedding_type: EmbeddingType = Field(..., description="Type of embedding")
    vector: List[float] = Field(..., description="Embedding vector")
    model_name: str = Field(..., description="Name of the embedding model")
    model_version: str = Field(..., description="Version of the embedding model")
    embedding_dim: int = Field(..., ge=1, description="Dimension of the embedding")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation time")
    isne_metadata: Dict[str, Any] = Field(default_factory=dict, description="ISNE-specific metadata")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @field_validator('embedding_dim')
    @classmethod
    def validate_embedding_dim(cls, v, info):
        """Validate embedding dimension matches vector length."""
        if info.data and 'vector' in info.data and len(info.data['vector']) != v:
            raise ValueError("Embedding dimension must match vector length")
        return v

    class Config:
        use_enum_values = True


# ===== EDGE MODELS =====

class IntraModalEdge(BaseModel):
    """Model for intra-modal edges (within same modality)."""
    from_node: str = Field(..., description="Source node ID")
    to_node: str = Field(..., description="Target node ID")
    edge_type: EdgeType = Field(..., description="Type of edge")
    weight: float = Field(..., ge=0, le=1, description="Edge weight")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")
    source: str = Field(..., description="Source of edge discovery")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation time")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        use_enum_values = True


class CrossModalEdge(BaseModel):
    """Model for cross-modal edges (theory-practice bridges)."""
    from_node: str = Field(..., description="Source node ID")
    to_node: str = Field(..., description="Target node ID")
    from_modality: FileType = Field(..., description="Source modality")
    to_modality: FileType = Field(..., description="Target modality")
    edge_type: EdgeType = Field(..., description="Type of edge")
    weight: float = Field(..., ge=0, le=1, description="Edge weight")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")
    similarity_score: float = Field(..., ge=0, le=1, description="Similarity score")
    source: str = Field(..., description="Source of edge discovery")
    discovery_method: str = Field(..., description="Method used for discovery")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation time")
    isne_metadata: Dict[str, Any] = Field(default_factory=dict, description="ISNE-specific metadata")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @field_validator('to_modality')
    @classmethod
    def validate_different_modalities(cls, v, info):
        """Ensure cross-modal edges connect different modalities."""
        if info.data and 'from_modality' in info.data:
            if v == info.data['from_modality']:
                raise ValueError("Cross-modal edges must connect different modalities")
        return v

    class Config:
        use_enum_values = True


# ===== ISNE MODEL =====

class ISNEModel(BaseModel):
    """Model for ISNE model metadata."""
    model_id: str = Field(..., description="Unique model identifier")
    version: str = Field(..., description="Model version")
    model_type: str = Field(..., description="Type of ISNE model")
    training_config: Dict[str, Any] = Field(..., description="Training configuration")
    architecture: Dict[str, Any] = Field(..., description="Model architecture")
    performance_metrics: Dict[str, Any] = Field(..., description="Performance metrics")
    model_path: str = Field(..., description="Path to saved model")
    node_count: int = Field(..., ge=0, description="Number of nodes in training graph")
    edge_count: int = Field(default=0, ge=0, description="Number of edges in training graph")
    embedding_dim: int = Field(..., ge=1, description="Embedding dimension")
    hidden_dim: int = Field(..., ge=1, description="Hidden dimension")
    num_layers: int = Field(..., ge=1, description="Number of layers")
    epochs_trained: int = Field(default=0, ge=0, description="Number of epochs trained")
    final_loss: Optional[float] = Field(None, description="Final training loss")
    convergence_achieved: bool = Field(default=False, description="Whether convergence was achieved")
    training_time_seconds: float = Field(default=0, ge=0, description="Training time in seconds")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation time")
    created_by: str = Field(..., description="Creator of the model")
    is_current: bool = Field(default=False, description="Whether this is the current model")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


# ===== DIRECTORY STRUCTURE MODEL =====

class DirectoryStructure(BaseModel):
    """Model for directory hierarchy."""
    directory_path: str = Field(..., description="Full directory path")
    parent_directory: Optional[str] = Field(None, description="Parent directory path")
    depth: int = Field(..., ge=0, description="Depth in directory hierarchy")
    file_count: int = Field(default=0, ge=0, description="Number of files in directory")
    subdirectory_count: int = Field(default=0, ge=0, description="Number of subdirectories")
    total_size: int = Field(default=0, ge=0, description="Total size of directory in bytes")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation time")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


# ===== PROCESSING LOG MODEL =====

class ProcessingLog(BaseModel):
    """Model for processing operation logs."""
    batch_id: str = Field(..., description="Unique batch identifier")
    operation_type: OperationType = Field(..., description="Type of operation")
    start_time: datetime = Field(..., description="Operation start time")
    end_time: Optional[datetime] = Field(None, description="Operation end time")
    status: ProcessingStatus = Field(..., description="Operation status")
    input_path: str = Field(..., description="Input path for operation")
    files_processed: int = Field(default=0, ge=0, description="Number of files processed")
    files_added: int = Field(default=0, ge=0, description="Number of files added")
    files_updated: int = Field(default=0, ge=0, description="Number of files updated")
    files_skipped: int = Field(default=0, ge=0, description="Number of files skipped")
    errors: List[str] = Field(default_factory=list, description="List of errors")
    config: Dict[str, Any] = Field(default_factory=dict, description="Operation configuration")
    results: Dict[str, Any] = Field(default_factory=dict, description="Operation results")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        use_enum_values = True


# ===== CONFIGURATION MODELS =====

class SequentialISNEConfig(BaseModel):
    """Configuration for Sequential-ISNE operations."""
    
    # Database settings
    db_name: str = Field(default="sequential_isne", description="Database name")
    connection_pool_size: int = Field(default=10, description="Connection pool size")
    
    # File processing settings
    batch_size: int = Field(default=100, description="Batch size for processing")
    max_workers: int = Field(default=4, description="Maximum worker threads")
    chunk_size: int = Field(default=512, description="Default chunk size")
    chunk_overlap: int = Field(default=50, description="Chunk overlap")
    
    # Graph construction settings
    enable_directory_bootstrap: bool = Field(default=True, description="Enable directory bootstrap")
    enable_cross_modal_discovery: bool = Field(default=True, description="Enable cross-modal discovery")
    similarity_threshold: float = Field(default=0.8, description="Similarity threshold for edges")
    max_edges_per_node: int = Field(default=50, description="Maximum edges per node")
    
    # ISNE training settings
    embedding_dim: int = Field(default=384, description="Embedding dimension")
    hidden_dim: int = Field(default=256, description="Hidden dimension")
    num_layers: int = Field(default=2, description="Number of layers")
    learning_rate: float = Field(default=0.001, description="Learning rate")
    epochs: int = Field(default=100, description="Training epochs")
    batch_size_training: int = Field(default=32, description="Training batch size")
    negative_samples: int = Field(default=5, description="Negative samples")
    device: str = Field(default="auto", description="Training device")
    
    # Performance settings
    enable_parallel_processing: bool = Field(default=True, description="Enable parallel processing")
    enable_caching: bool = Field(default=True, description="Enable caching")
    cache_ttl: int = Field(default=3600, description="Cache TTL in seconds")


# ===== UTILITY FUNCTIONS =====

def classify_file_type(file_path: Union[str, Path]) -> FileType:
    """Classify file type based on file path."""
    path = Path(file_path)
    extension = path.suffix.lower()
    
    # Code files
    code_extensions = {'.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.rs', '.go', '.php', '.rb', '.swift'}
    if extension in code_extensions:
        return FileType.CODE
    
    # Documentation files
    doc_extensions = {'.md', '.txt', '.rst', '.pdf', '.html', '.tex'}
    if extension in doc_extensions:
        return FileType.DOCUMENTATION
    
    # Configuration files
    config_extensions = {'.json', '.yaml', '.yml', '.toml', '.ini', '.xml', '.env', '.conf'}
    if extension in config_extensions:
        return FileType.CONFIG
    
    # Data files
    data_extensions = {'.csv', '.tsv', '.parquet', '.xlsx', '.db', '.sqlite'}
    if extension in data_extensions:
        return FileType.DATA
    
    return FileType.UNKNOWN


def get_specific_file_type(file_path: Union[str, Path], modality: FileType) -> str:
    """Get specific file type within a modality."""
    path = Path(file_path)
    extension = path.suffix.lower()
    
    if modality == FileType.CODE:
        mapping = {
            '.py': CodeFileType.PYTHON,
            '.js': CodeFileType.JAVASCRIPT,
            '.ts': CodeFileType.TYPESCRIPT,
            '.java': CodeFileType.JAVA,
            '.cpp': CodeFileType.CPP,
            '.c': CodeFileType.C,
            '.rs': CodeFileType.RUST,
            '.go': CodeFileType.GO
        }
        return mapping.get(extension, CodeFileType.OTHER).value
    
    elif modality == FileType.DOCUMENTATION:
        mapping = {
            '.md': DocumentationType.MARKDOWN,
            '.txt': DocumentationType.TEXT,
            '.rst': DocumentationType.RST,
            '.pdf': DocumentationType.PDF,
            '.html': DocumentationType.HTML
        }
        return mapping.get(extension, DocumentationType.OTHER).value
    
    elif modality == FileType.CONFIG:
        mapping = {
            '.json': ConfigFileType.JSON,
            '.yaml': ConfigFileType.YAML,
            '.yml': ConfigFileType.YAML,
            '.toml': ConfigFileType.TOML,
            '.ini': ConfigFileType.INI,
            '.xml': ConfigFileType.XML,
            '.env': ConfigFileType.ENV
        }
        return mapping.get(extension, ConfigFileType.OTHER).value
    
    return "other"


def get_modality_collection(file_type: FileType) -> str:
    """Get collection name for a file type modality."""
    mapping = {
        FileType.CODE: "code_files",
        FileType.DOCUMENTATION: "documentation_files",
        FileType.CONFIG: "config_files"
    }
    return mapping.get(file_type, "unknown_files")


def is_cross_modal_edge(edge_type: EdgeType) -> bool:
    """Check if an edge type is cross-modal."""
    cross_modal_edges = {
        EdgeType.CODE_TO_DOC,
        EdgeType.DOC_TO_CODE,
        EdgeType.CODE_TO_CONFIG,
        EdgeType.CONFIG_TO_CODE,
        EdgeType.DOC_TO_CONFIG,
        EdgeType.ISNE_SIMILARITY,
        EdgeType.SEMANTIC_SIMILARITY
    }
    return edge_type in cross_modal_edges