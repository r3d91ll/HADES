"""
Component Contracts - Pydantic Models

This module defines the input/output contracts for all component types.
These contracts enable A/B testing and easy component swapping as long as
implementations conform to the same contract.

The key principle is that any implementation of a component type (e.g., docproc)
must accept the same Input contract and produce the same Output contract,
regardless of the internal implementation details.
"""

from typing import List, Dict, Any, Optional, Union, ClassVar
from pathlib import Path
from pydantic import BaseModel, Field, field_validator
from datetime import datetime, timezone
from enum import Enum

# ===== Enums =====

class ComponentType(str, Enum):
    """Component type enumeration."""
    DOCPROC = "docproc"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    GRAPH_ENHANCEMENT = "graph_enhancement"
    STORAGE = "storage"
    DATABASE = "database"
    SCHEMAS = "schemas"
    MODEL_ENGINE = "model_engine"
    GRAPH_ENGINE = "graph_engine"
    UNKNOWN = "unknown"

class ProcessingStatus(str, Enum):
    """Processing status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    SUCCESS = "success"
    ERROR = "error"
    PARTIAL = "partial"

class ContentCategory(str, Enum):
    """Content category enumeration."""
    TEXT = "text"
    CODE = "code"
    DATA = "data"
    DOCUMENT = "document"
    MARKDOWN = "markdown"
    UNKNOWN = "unknown"

# ===== Base Models =====

class ComponentMetadata(BaseModel):
    """Base metadata for all component operations."""
    component_type: ComponentType
    component_name: str
    component_version: str
    processing_time: Optional[float] = None
    processed_at: Optional[datetime] = Field(default_factory=datetime.utcnow)
    config: Dict[str, Any] = Field(default_factory=dict)
    status: ProcessingStatus = ProcessingStatus.PENDING

    class Config:
        """Pydantic configuration."""
        use_enum_values = True

# ===== Document Processing Contracts =====

class DocumentProcessingInput(BaseModel):
    """Input contract for document processing components."""
    file_path: Union[str, Path]
    content: Optional[str] = None  # Raw file content if already loaded
    file_type: Optional[str] = None  # MIME type or file extension
    processing_options: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('file_path')
    def validate_file_path(cls, v: Any) -> str:
        """Ensure file_path is converted to string."""
        return str(v)

    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True

class ProcessedDocument(BaseModel):
    """Processed document output structure."""
    id: str
    content: str
    content_type: str
    format: str
    content_category: ContentCategory
    entities: List[Dict[str, Any]] = Field(default_factory=list)
    sections: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    processing_time: Optional[float] = None

    class Config:
        """Pydantic configuration."""
        use_enum_values = True

class DocumentProcessingOutput(BaseModel):
    """Output contract for document processing components."""
    documents: List[ProcessedDocument]
    metadata: ComponentMetadata
    processing_stats: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    total_processed: int = 0
    total_errors: int = 0

    @field_validator('total_processed')
    @classmethod
    def set_total_processed(cls, v: Any, info: Any) -> int:
        """Auto-calculate total processed documents."""
        if hasattr(info, 'data') and info.data:
            documents = info.data.get('documents', [])
            return len([doc for doc in documents if getattr(doc, 'error', None) is None])
        return 0

    @field_validator('total_errors')
    @classmethod
    def set_total_errors(cls, v: Any, info: Any) -> int:
        """Auto-calculate total errors."""
        if hasattr(info, 'data') and info.data:
            documents = info.data.get('documents', [])
            base_errors = len([doc for doc in documents if getattr(doc, 'error', None) is not None])
            additional_errors = len(info.data.get('errors', []))
            return base_errors + additional_errors
        return 0

# ===== Chunking Contracts =====

class ChunkingInput(BaseModel):
    """Input contract for chunking components."""
    text: str
    document_id: str
    chunking_strategy: str = "default"
    chunk_size: int = Field(default=512, ge=1, le=8192)
    chunk_overlap: int = Field(default=50, ge=0)
    processing_options: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('chunk_overlap')
    @classmethod
    def validate_overlap(cls, v: int, info: Any) -> int:
        """Ensure overlap is less than chunk_size."""
        if hasattr(info, 'data') and info.data:
            chunk_size = info.data.get('chunk_size', 512)
            if v >= chunk_size:
                raise ValueError(f"chunk_overlap ({v}) must be less than chunk_size ({chunk_size})")
        return v

class DocumentsChunkingInput(BaseModel):
    """Input contract for document-based chunking components."""
    documents: List[ProcessedDocument]
    chunking_strategy: str = "default"
    chunk_size: int = Field(default=512, ge=1, le=8192)
    chunk_overlap: int = Field(default=50, ge=0)
    chunking_options: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('chunk_overlap')
    @classmethod
    def validate_overlap(cls, v: int, info: Any) -> int:
        """Ensure overlap is less than chunk_size."""
        if hasattr(info, 'data') and info.data:
            chunk_size = info.data.get('chunk_size', 512)
            if v >= chunk_size:
                raise ValueError(f"chunk_overlap ({v}) must be less than chunk_size ({chunk_size})")
        return v

class TextChunk(BaseModel):
    """Simple text chunk structure."""
    id: str
    text: str
    start_index: int = Field(default=0, ge=0)
    end_index: Optional[int] = Field(default=None, ge=0)
    chunk_index: int = Field(default=0, ge=0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    processing_time: Optional[float] = Field(default=None, ge=0.0)

    @field_validator('end_index')
    @classmethod
    def set_end_index(cls, v: Optional[int], info: Any) -> Optional[int]:
        """Auto-calculate end_index from text length if not provided."""
        if v is None and hasattr(info, 'data') and info.data:
            text = info.data.get('text', '')
            start_index = int(info.data.get('start_index', 0))
            return start_index + len(text)
        return v

    @field_validator('end_index')
    @classmethod
    def validate_positions(cls, v: Optional[int], info: Any) -> Optional[int]:
        """Ensure end_index > start_index."""
        if hasattr(info, 'data') and info.data:
            start_idx = info.data.get('start_index', 0)
            if v is not None and v <= start_idx:
                raise ValueError("end_index must be greater than start_index")
        return v

class DocumentChunk(BaseModel):
    """Individual chunk structure for document-based chunking."""
    id: str
    content: str
    document_id: str
    chunk_index: int = Field(ge=0)
    start_position: Optional[int] = Field(default=None, ge=0)
    end_position: Optional[int] = Field(default=None, ge=0)
    chunk_size: int = Field(ge=1)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('chunk_size')
    @classmethod
    def set_chunk_size(cls, v: int, info: Any) -> int:
        """Auto-calculate chunk size from content."""
        if hasattr(info, 'data') and info.data:
            content = info.data.get('content', '')
            return len(content)
        return v

    @field_validator('end_position')
    @classmethod
    def validate_positions(cls, v: Optional[int], info: Any) -> Optional[int]:
        """Ensure end_position > start_position."""
        if hasattr(info, 'data') and info.data:
            start_pos = info.data.get('start_position')
            if start_pos is not None and v is not None and v <= start_pos:
                raise ValueError("end_position must be greater than start_position")
        return v

class ChunkingOutput(BaseModel):
    """Output contract for chunking components."""
    chunks: List[TextChunk]
    metadata: ComponentMetadata
    processing_stats: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    total_chunks: int = 0
    total_characters: int = 0

    @field_validator('total_chunks')
    @classmethod
    def set_total_chunks(cls, v: int, info: Any) -> int:
        """Auto-calculate total chunks."""
        if hasattr(info, 'data') and info.data:
            return len(info.data.get('chunks', []))
        return 0

    @field_validator('total_characters')
    @classmethod
    def set_total_characters(cls, v: int, info: Any) -> int:
        """Auto-calculate total characters."""
        if hasattr(info, 'data') and info.data:
            chunks = info.data.get('chunks', [])
            return sum(len(getattr(chunk, 'text', '')) for chunk in chunks)
        return 0

class DocumentChunkingOutput(BaseModel):
    """Output contract for document-based chunking components."""
    chunks: List[DocumentChunk]
    metadata: ComponentMetadata
    chunking_stats: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    total_chunks: int = 0
    total_characters: int = 0

    @field_validator('total_chunks')
    @classmethod
    def set_total_chunks(cls, v: int, info: Any) -> int:
        """Auto-calculate total chunks."""
        if hasattr(info, 'data') and info.data:
            return len(info.data.get('chunks', []))
        return 0

    @field_validator('total_characters')
    @classmethod
    def set_total_characters(cls, v: int, info: Any) -> int:
        """Auto-calculate total characters."""
        if hasattr(info, 'data') and info.data:
            chunks = info.data.get('chunks', [])
            return sum(len(getattr(chunk, 'content', '')) for chunk in chunks)
        return 0

# ===== Embedding Contracts =====

class EmbeddingInput(BaseModel):
    """Input contract for embedding components."""
    chunks: List[DocumentChunk]
    model_name: str = "default"
    embedding_options: Dict[str, Any] = Field(default_factory=dict)
    batch_size: int = Field(default=32, ge=1, le=1000)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ChunkEmbedding(BaseModel):
    """Chunk with embedding vector."""
    chunk_id: str
    embedding: List[float]
    embedding_dimension: int = Field(ge=1)
    model_name: str
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('embedding_dimension')
    @classmethod
    def set_embedding_dimension(cls, v: int, info: Any) -> int:
        """Auto-calculate embedding dimension."""
        if hasattr(info, 'data') and info.data:
            embedding = info.data.get('embedding', [])
            return len(embedding)
        return v

    @field_validator('embedding')
    @classmethod
    def validate_embedding_values(cls, v: List[float]) -> List[float]:
        """Ensure all embedding values are finite."""
        import math
        if not all(math.isfinite(val) for val in v):
            raise ValueError("All embedding values must be finite numbers")
        return v

class EmbeddingOutput(BaseModel):
    """Output contract for embedding components."""
    embeddings: List[ChunkEmbedding]
    metadata: ComponentMetadata
    embedding_stats: Dict[str, Any] = Field(default_factory=dict)
    model_info: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    total_embeddings: int = 0
    avg_embedding_time: Optional[float] = None

    @field_validator('total_embeddings')
    @classmethod
    def set_total_embeddings(cls, v: int, info: Any) -> int:
        """Auto-calculate total embeddings."""
        if hasattr(info, 'data') and info.data:
            return len(info.data.get('embeddings', []))
        return 0

# ===== Graph Enhancement Contracts =====

class GraphEnhancementInput(BaseModel):
    """Input contract for graph enhancement components."""
    embeddings: List[ChunkEmbedding]
    enhancement_method: str = "default"
    enhancement_options: Dict[str, Any] = Field(default_factory=dict)
    graph_config: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class EnhancedEmbedding(BaseModel):
    """Enhanced embedding with graph information."""
    chunk_id: str
    original_embedding: List[float]
    enhanced_embedding: List[float]
    graph_features: Dict[str, Any] = Field(default_factory=dict)
    enhancement_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('enhanced_embedding')
    @classmethod
    def validate_enhanced_embedding(cls, v: List[float], info: Any) -> List[float]:
        """Ensure enhanced embedding has same dimension as original."""
        if hasattr(info, 'data') and info.data:
            original = info.data.get('original_embedding', [])
            if len(v) != len(original):
                raise ValueError("Enhanced embedding must have same dimension as original")
        return v

class GraphEnhancementOutput(BaseModel):
    """Output contract for graph enhancement components."""
    enhanced_embeddings: List[EnhancedEmbedding]
    metadata: ComponentMetadata
    graph_stats: Dict[str, Any] = Field(default_factory=dict)
    model_info: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    total_enhanced: int = 0
    enhancement_quality_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @field_validator('total_enhanced')
    @classmethod
    def set_total_enhanced(cls, v: int, info: Any) -> int:
        """Auto-calculate total enhanced embeddings."""
        if hasattr(info, 'data') and info.data:
            return len(info.data.get('enhanced_embeddings', []))
        return 0

# ===== Storage Contracts =====

class StorageInput(BaseModel):
    """Input contract for storage components."""
    enhanced_embeddings: List[EnhancedEmbedding]
    storage_method: str = "default"
    storage_options: Dict[str, Any] = Field(default_factory=dict)
    index_config: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class StoredItem(BaseModel):
    """Stored item reference."""
    item_id: str
    storage_location: str
    storage_timestamp: datetime = Field(default_factory=datetime.utcnow)
    retrieval_metadata: Dict[str, Any] = Field(default_factory=dict)
    index_status: ProcessingStatus = ProcessingStatus.SUCCESS

    class Config:
        """Pydantic configuration."""
        use_enum_values = True

class StorageOutput(BaseModel):
    """Output contract for storage components."""
    stored_items: List[StoredItem]
    metadata: ComponentMetadata
    storage_stats: Dict[str, Any] = Field(default_factory=dict)
    index_info: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    total_stored: int = 0
    storage_size_bytes: Optional[int] = Field(default=None, ge=0)

    @field_validator('total_stored')
    @classmethod
    def set_total_stored(cls, v: int, info: Any) -> int:
        """Auto-calculate total stored items."""
        if hasattr(info, 'data') and info.data:
            return len(info.data.get('stored_items', []))
        return 0

# ===== Query/Retrieval Contracts =====

class QueryInput(BaseModel):
    """Input contract for query/retrieval operations."""
    query: str
    query_embedding: Optional[List[float]] = None
    top_k: int = Field(default=10, ge=1, le=1000)
    filters: Dict[str, Any] = Field(default_factory=dict)
    search_options: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('query')
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Ensure query is not empty."""
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()

class RetrievalResult(BaseModel):
    """Individual retrieval result."""
    item_id: str
    content: str
    score: float = Field(ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    chunk_metadata: Optional[Dict[str, Any]] = None
    document_metadata: Optional[Dict[str, Any]] = None

class QueryOutput(BaseModel):
    """Output contract for query operations."""
    results: List[RetrievalResult]
    metadata: ComponentMetadata
    search_stats: Dict[str, Any] = Field(default_factory=dict)
    query_info: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    total_results: int = 0
    search_time: Optional[float] = Field(default=None, ge=0.0)

    @field_validator('total_results')
    @classmethod
    def set_total_results(cls, v: int, info: Any) -> int:
        """Auto-calculate total results."""
        if hasattr(info, 'data') and info.data:
            return len(info.data.get('results', []))
        return 0

    @field_validator('results')
    @classmethod
    def validate_results_sorted(cls, v: List[RetrievalResult]) -> List[RetrievalResult]:
        """Ensure results are sorted by score (descending)."""
        if len(v) > 1:
            scores = [result.score for result in v]
            if scores != sorted(scores, reverse=True):
                raise ValueError("Results must be sorted by score in descending order")
        return v


# ===== Model Engine Contracts =====

class ModelEngineInput(BaseModel):
    """Input contract for model engine operations."""
    requests: List[Dict[str, Any]]  # List of inference requests
    engine_config: Dict[str, Any] = Field(default_factory=dict)
    batch_config: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('requests')
    @classmethod
    def validate_requests_not_empty(cls, v: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Ensure requests list is not empty."""
        if not v:
            raise ValueError("Requests list cannot be empty")
        return v

class ModelInferenceResult(BaseModel):
    """Individual model inference result."""
    request_id: str
    response_data: Dict[str, Any]
    processing_time: float = Field(ge=0.0)
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class ModelEngineOutput(BaseModel):
    """Output contract for model engine operations."""
    results: List[ModelInferenceResult]
    metadata: ComponentMetadata
    engine_stats: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)

    @field_validator('results')
    @classmethod
    def validate_results_count(cls, v: List[ModelInferenceResult]) -> List[ModelInferenceResult]:
        """Validate results."""
        return v