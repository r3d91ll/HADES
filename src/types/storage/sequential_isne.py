"""
Sequential-ISNE Types

This module defines types for the Sequential-ISNE storage system used by ArangoStorageV2.
These types support multi-modal knowledge graphs with code, documentation, and configuration files.
"""

from enum import Enum
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field


# Enums
class FileType(str, Enum):
    """Types of files in the knowledge graph."""
    CODE = "code"
    DOCUMENTATION = "documentation"
    CONFIG = "config"
    TEST = "test"
    UNKNOWN = "unknown"


class EdgeType(str, Enum):
    """Types of edges in the knowledge graph."""
    # Intra-modal edges (within same file type)
    IMPORTS = "imports"
    CALLS = "calls"
    INHERITS = "inherits"
    REFERENCES = "references"
    CONTAINS = "contains"
    
    # Cross-modal edges (between different file types)
    DOCUMENTS = "documents"
    CONFIGURES = "configures"
    TESTS = "tests"
    IMPLEMENTS = "implements"
    DESCRIBES = "describes"


class EmbeddingType(str, Enum):
    """Types of embeddings."""
    BASE = "base"
    ISNE_ENHANCED = "isne_enhanced"
    SEQUENTIAL_ISNE = "sequential_isne"


class SourceType(str, Enum):
    """Source types for entities."""
    FILE = "file"
    CHUNK = "chunk"
    SYMBOL = "symbol"
    SECTION = "section"


class ProcessingStatus(str, Enum):
    """Processing status for entities."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


# Base Models
class SequentialISNEConfig(BaseModel):
    """Configuration for Sequential-ISNE processing."""
    enable_cross_modal: bool = Field(default=True, description="Enable cross-modal edges")
    enable_sequential: bool = Field(default=True, description="Enable sequential processing")
    max_sequence_length: int = Field(default=10, description="Maximum sequence length")
    cross_modal_weight: float = Field(default=0.8, description="Weight for cross-modal edges")
    intra_modal_weight: float = Field(default=1.0, description="Weight for intra-modal edges")
    batch_size: int = Field(default=32, description="Batch size for processing")
    embedding_dim: int = Field(default=768, description="Embedding dimension")


# File Models
class BaseFile(BaseModel):
    """Base class for all file types."""
    id: str = Field(..., description="Unique identifier")
    path: str = Field(..., description="File path")
    content: str = Field(..., description="File content")
    file_type: FileType = Field(..., description="Type of file")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="File metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    status: ProcessingStatus = Field(default=ProcessingStatus.PENDING)


class CodeFile(BaseFile):
    """Code file representation."""
    file_type: FileType = Field(default=FileType.CODE)
    language: str = Field(..., description="Programming language")
    symbols: List[Dict[str, Any]] = Field(default_factory=list, description="Extracted symbols")
    imports: List[str] = Field(default_factory=list, description="Import statements")
    functions: List[str] = Field(default_factory=list, description="Function names")
    classes: List[str] = Field(default_factory=list, description="Class names")


class DocumentationFile(BaseFile):
    """Documentation file representation."""
    file_type: FileType = Field(default=FileType.DOCUMENTATION)
    format: str = Field(..., description="Documentation format (md, rst, txt)")
    sections: List[Dict[str, Any]] = Field(default_factory=list, description="Document sections")
    references: List[str] = Field(default_factory=list, description="Referenced files/symbols")


class ConfigFile(BaseFile):
    """Configuration file representation."""
    file_type: FileType = Field(default=FileType.CONFIG)
    format: str = Field(..., description="Config format (yaml, json, toml, ini)")
    config_data: Dict[str, Any] = Field(default_factory=dict, description="Parsed configuration")
    targets: List[str] = Field(default_factory=list, description="Target modules/components")


# Chunk and Embedding Models
class Chunk(BaseModel):
    """Text chunk from a file."""
    id: str = Field(..., description="Unique chunk identifier")
    file_id: str = Field(..., description="Parent file ID")
    content: str = Field(..., description="Chunk content")
    start_pos: int = Field(..., description="Start position in file")
    end_pos: int = Field(..., description="End position in file")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    source_type: SourceType = Field(default=SourceType.CHUNK)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Embedding(BaseModel):
    """Embedding representation."""
    id: str = Field(..., description="Unique embedding identifier")
    chunk_id: str = Field(..., description="Associated chunk ID")
    file_id: str = Field(..., description="Associated file ID")
    vector: List[float] = Field(..., description="Embedding vector")
    embedding_type: EmbeddingType = Field(default=EmbeddingType.BASE)
    model_name: str = Field(..., description="Model used for embedding")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


# Edge Models
class BaseEdge(BaseModel):
    """Base class for all edge types."""
    id: str = Field(..., description="Unique edge identifier")
    source_id: str = Field(..., description="Source node ID")
    target_id: str = Field(..., description="Target node ID")
    edge_type: EdgeType = Field(..., description="Type of edge")
    weight: float = Field(default=1.0, description="Edge weight")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class IntraModalEdge(BaseEdge):
    """Edge within the same modality (e.g., code-to-code)."""
    source_type: FileType = Field(..., description="Source file type")
    target_type: FileType = Field(..., description="Target file type (same as source)")
    
    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        # Ensure source and target types match for intra-modal edges
        if self.source_type != self.target_type:
            raise ValueError("IntraModalEdge must have matching source and target types")


class CrossModalEdge(BaseEdge):
    """Edge between different modalities (e.g., code-to-documentation)."""
    source_type: FileType = Field(..., description="Source file type")
    target_type: FileType = Field(..., description="Target file type (different from source)")
    cross_modal_score: float = Field(default=0.0, description="Cross-modal relevance score")
    
    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        # Ensure source and target types differ for cross-modal edges
        if self.source_type == self.target_type:
            raise ValueError("CrossModalEdge must have different source and target types")


# ISNE Model
class ISNEModel(BaseModel):
    """ISNE model configuration and state."""
    model_id: str = Field(..., description="Model identifier")
    version: str = Field(..., description="Model version")
    config: SequentialISNEConfig = Field(..., description="Model configuration")
    embedding_dim: int = Field(..., description="Embedding dimension")
    hidden_dim: int = Field(..., description="Hidden layer dimension")
    num_layers: int = Field(..., description="Number of layers")
    trained_at: Optional[datetime] = Field(default=None)
    metrics: Dict[str, float] = Field(default_factory=dict)
    checkpoint_path: Optional[str] = Field(default=None)


# Helper Functions
def classify_file_type(file_path: str) -> FileType:
    """
    Classify file type based on path and extension.
    
    Args:
        file_path: Path to the file
        
    Returns:
        FileType enum value
    """
    path_lower = file_path.lower()
    
    # Code files
    code_extensions = {'.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.hpp', 
                      '.go', '.rs', '.rb', '.php', '.swift', '.kt', '.scala'}
    if any(path_lower.endswith(ext) for ext in code_extensions):
        return FileType.CODE
    
    # Documentation files
    doc_extensions = {'.md', '.rst', '.txt', '.doc', '.docx', '.pdf'}
    doc_patterns = ['readme', 'changelog', 'contributing', 'license']
    if (any(path_lower.endswith(ext) for ext in doc_extensions) or
        any(pattern in path_lower for pattern in doc_patterns)):
        return FileType.DOCUMENTATION
    
    # Config files
    config_extensions = {'.yaml', '.yml', '.json', '.toml', '.ini', '.conf', '.cfg'}
    config_patterns = ['config', 'settings', '.env']
    if (any(path_lower.endswith(ext) for ext in config_extensions) or
        any(pattern in path_lower for pattern in config_patterns)):
        return FileType.CONFIG
    
    # Test files
    test_patterns = ['test_', '_test', '/test/', '/tests/', 'spec.']
    if any(pattern in path_lower for pattern in test_patterns):
        return FileType.TEST
    
    return FileType.UNKNOWN


def get_modality_collection(file_type: FileType) -> str:
    """
    Get the ArangoDB collection name for a file type.
    
    Args:
        file_type: FileType enum value
        
    Returns:
        Collection name string
    """
    collection_map = {
        FileType.CODE: "code_files",
        FileType.DOCUMENTATION: "doc_files",
        FileType.CONFIG: "config_files",
        FileType.TEST: "test_files",
        FileType.UNKNOWN: "unknown_files"
    }
    return collection_map.get(file_type, "unknown_files")


def is_cross_modal_edge(source_type: FileType, target_type: FileType) -> bool:
    """
    Check if an edge is cross-modal.
    
    Args:
        source_type: Source file type
        target_type: Target file type
        
    Returns:
        True if edge crosses modalities, False otherwise
    """
    return source_type != target_type


# Type Aliases for convenience
FileEntity = Union[CodeFile, DocumentationFile, ConfigFile]
EdgeEntity = Union[IntraModalEdge, CrossModalEdge]


__all__ = [
    # Enums
    "FileType",
    "EdgeType", 
    "EmbeddingType",
    "SourceType",
    "ProcessingStatus",
    
    # Config
    "SequentialISNEConfig",
    
    # File types
    "BaseFile",
    "CodeFile",
    "DocumentationFile",
    "ConfigFile",
    
    # Core types
    "Chunk",
    "Embedding",
    
    # Edge types
    "BaseEdge",
    "IntraModalEdge",
    "CrossModalEdge",
    
    # Model
    "ISNEModel",
    
    # Helper functions
    "classify_file_type",
    "get_modality_collection", 
    "is_cross_modal_edge",
    
    # Type aliases
    "FileEntity",
    "EdgeEntity",
    
    # Schema Manager
    "SequentialISNESchemaManager"
]


class SequentialISNESchemaManager:
    """Manager for Sequential ISNE schema operations."""
    
    def __init__(self, config: SequentialISNEConfig):
        """Initialize schema manager with configuration."""
        self.config = config
        self._initialized = False
        self._database_name = "hades_db"
        self._collections = {
            FileType.CODE: "code_files",
            FileType.DOCUMENTATION: "doc_files",
            FileType.CONFIG: "config_files",
            FileType.TEST: "test_files",
            FileType.UNKNOWN: "unknown_files"
        }
        
    def initialize_schema(self) -> bool:
        """Initialize the database schema."""
        # This would be implemented by the actual storage backend
        self._initialized = True
        return True
        
    def initialize_database(self) -> bool:
        """Initialize the database and collections."""
        return self.initialize_schema()
        
    def get_modality_collections(self) -> Dict[str, str]:
        """Get mapping of modality to collection names."""
        return {k.value: v for k, v in self._collections.items()}
        
    def get_database_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        return {
            "database": self._database_name,
            "collections": list(self._collections.values()),
            "initialized": self._initialized,
            "total_collections": len(self._collections),
            "schema_version": "2.0.0"
        }
        
    def validate_schema(self) -> bool:
        """Validate the current schema."""
        return self._initialized
        
    def get_metadata(self) -> Dict[str, Any]:
        """Get current schema metadata."""
        return {
            "version": "2.0.0",
            "initialized": self._initialized,
            "config": self.config.dict()
        }