"""
Document type definitions for ISNE.

This module provides document and relationship types for the ISNE system,
including both TypedDict and dataclass versions for different use cases.
"""

from typing import Any, Dict, List, Optional, TypedDict, Union
from dataclasses import dataclass, field
from datetime import datetime
from pydantic import BaseModel, Field

from src.types.isne.base import DocumentType, RelationType, EmbeddingVector


# TypedDict versions for high-performance operations

class IngestDocumentDict(TypedDict, total=False):
    """Document representation for ISNE graph construction (TypedDict version)."""
    
    id: str  # Unique identifier for the document
    content: Optional[str]  # Document content (may be omitted to save space)
    source: str  # Source path or identifier
    document_type: str  # Type of document (see DocumentType enum)
    metadata: Dict[str, Any]  # Document metadata
    chunks: List[Dict[str, Any]]  # List of document chunks
    embeddings: Dict[str, List[float]]  # Chunk embeddings (key = chunk_id)
    embedding_model: str  # Model used to generate embeddings
    embedding: Optional[EmbeddingVector]  # Original document embedding
    enhanced_embedding: Optional[EmbeddingVector]  # ISNE-enhanced embedding
    created_at: Optional[str]  # Creation timestamp
    processed_at: Optional[str]  # Processing timestamp


class DocumentRelationDict(TypedDict, total=False):
    """Relationship between document entities (TypedDict version)."""
    
    source_id: str  # ID of the source entity
    target_id: str  # ID of the target entity
    relation_type: str  # Type of relationship (see RelationType enum)
    weight: float  # Relationship weight/strength
    metadata: Dict[str, Any]  # Additional metadata about the relationship
    created_at: Optional[str]  # Creation timestamp
    confidence: Optional[float]  # Confidence score for the relationship


class LoaderResultDict(TypedDict):
    """Result of loading documents through a loader (TypedDict version)."""
    
    documents: List[IngestDocumentDict]  # Loaded documents
    relations: List[DocumentRelationDict]  # Document relationships
    metadata: Dict[str, Any]  # Additional metadata
    total_documents: int  # Total number of documents loaded
    total_relations: int  # Total number of relations loaded
    load_time: float  # Time taken to load (seconds)
    loader_type: str  # Type of loader used


# Dataclass versions for rich object functionality

@dataclass
class IngestDocument:
    """Document representation for ISNE graph construction (dataclass version).
    
    This class holds the document content, metadata, and computed embeddings.
    It supports different modalities through the document_type field.
    """
    
    id: str  # Unique identifier
    content: str  # Document content
    source: str  # Source path or identifier
    document_type: Union[str, DocumentType] = "text"  # Document type (see DocumentType)
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata
    embedding: Optional[EmbeddingVector] = None  # Original embedding
    enhanced_embedding: Optional[EmbeddingVector] = None  # ISNE-enhanced embedding
    chunks: List[Dict[str, Any]] = field(default_factory=list)  # Document chunks
    embeddings: Dict[str, List[float]] = field(default_factory=dict)  # Chunk embeddings
    embedding_model: str = "modernbert"  # Model used to generate embeddings
    created_at: Optional[datetime] = None  # Creation timestamp
    processed_at: Optional[datetime] = None  # Processing timestamp
    
    def __post_init__(self) -> None:
        """Validate document fields after initialization."""
        # Validate document type
        try:
            # Try to convert to DocumentType enum
            if isinstance(self.document_type, str):
                self.document_type = DocumentType(self.document_type)
        except ValueError:
            # If invalid, set to UNKNOWN
            self.document_type = DocumentType.UNKNOWN
            
        # Metadata is initialized by default_factory, no need to check
        
        # Set timestamps if not provided
        if self.created_at is None:
            self.created_at = datetime.now()
        
        # Ensure model info exists in metadata
        if "model_used" not in self.metadata:
            self.metadata["model_used"] = self.embedding_model
    
    def to_dict(self) -> IngestDocumentDict:
        """Convert to TypedDict format."""
        return {
            "id": self.id,
            "content": self.content,
            "source": self.source,
            "document_type": str(self.document_type),
            "metadata": self.metadata,
            "chunks": self.chunks,
            "embeddings": self.embeddings,
            "embedding_model": self.embedding_model,
            "embedding": self.embedding,
            "enhanced_embedding": self.enhanced_embedding,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
        }
    
    @classmethod
    def from_dict(cls, data: IngestDocumentDict) -> "IngestDocument":
        """Create from TypedDict format."""
        created_at = None
        created_at_str = data.get("created_at")
        if created_at_str is not None:
            created_at = datetime.fromisoformat(created_at_str)
        
        processed_at = None
        processed_at_str = data.get("processed_at")
        if processed_at_str is not None:
            processed_at = datetime.fromisoformat(processed_at_str)
        
        return cls(
            id=data["id"],
            content=data.get("content") or "",
            source=data.get("source", ""),
            document_type=data.get("document_type", "text"),
            metadata=data.get("metadata", {}),
            embedding=data.get("embedding"),
            enhanced_embedding=data.get("enhanced_embedding"),
            chunks=data.get("chunks", []),
            embeddings=data.get("embeddings", {}),
            embedding_model=data.get("embedding_model", "modernbert"),
            created_at=created_at,
            processed_at=processed_at,
        )


@dataclass
class DocumentRelation:
    """Relationship between two documents (dataclass version).
    
    This class defines edges in the document graph, with types that
    determine edge weights and traversal behavior.
    """
    
    source_id: str  # Source document ID
    target_id: str  # Target document ID
    relation_type: RelationType = RelationType.GENERIC  # Relationship type
    weight: float = 1.0  # Edge weight/strength
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata
    confidence: Optional[float] = None  # Confidence score for the relationship
    created_at: Optional[datetime] = None  # Creation timestamp
    
    def __post_init__(self) -> None:
        """Set default weights and timestamps after initialization."""
        # Set creation timestamp if not provided
        if self.created_at is None:
            self.created_at = datetime.now()
        
        # Set default weights based on relation type if not explicitly set
        if self.weight == 1.0:  # Only if not explicitly set
            from src.types.isne.base import RELATIONSHIP_WEIGHTS
            self.weight = RELATIONSHIP_WEIGHTS.get(self.relation_type, 0.5)
    
    def to_dict(self) -> DocumentRelationDict:
        """Convert to TypedDict format."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": str(self.relation_type),
            "weight": self.weight,
            "metadata": self.metadata,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
    
    @classmethod
    def from_dict(cls, data: DocumentRelationDict) -> "DocumentRelation":
        """Create from TypedDict format."""
        created_at = None
        created_at_str = data.get("created_at")
        if created_at_str is not None:
            created_at = datetime.fromisoformat(created_at_str)
        
        relation_type = RelationType(data.get("relation_type", "generic"))
        
        return cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            relation_type=relation_type,
            weight=data.get("weight", 1.0),
            metadata=data.get("metadata", {}),
            confidence=data.get("confidence"),
            created_at=created_at,
        )


@dataclass
class LoaderResult:
    """Result of loading documents through a loader (dataclass version).
    
    This class contains the loaded documents and their relationships,
    forming the input to the ISNE pipeline.
    """
    
    documents: List[IngestDocument] = field(default_factory=list)  # Loaded documents
    relations: List[DocumentRelation] = field(default_factory=list)  # Document relationships
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata
    loader_type: str = "base"  # Type of loader used
    load_time: float = 0.0  # Time taken to load (seconds)
    
    @property
    def document_count(self) -> int:
        """Get the number of documents."""
        return len(self.documents)
    
    @property
    def relation_count(self) -> int:
        """Get the number of relations."""
        return len(self.relations)
    
    def get_document_by_id(self, doc_id: str) -> Optional[IngestDocument]:
        """Get a document by its ID."""
        for doc in self.documents:
            if doc.id == doc_id:
                return doc
        return None
    
    def get_relations_for_document(self, doc_id: str) -> List[DocumentRelation]:
        """Get all relations involving a document."""
        return [rel for rel in self.relations 
                if rel.source_id == doc_id or rel.target_id == doc_id]
    
    def to_dict(self) -> LoaderResultDict:
        """Convert to TypedDict format."""
        return {
            "documents": [doc.to_dict() for doc in self.documents],
            "relations": [rel.to_dict() for rel in self.relations],
            "metadata": self.metadata,
            "total_documents": self.document_count,
            "total_relations": self.relation_count,
            "load_time": self.load_time,
            "loader_type": self.loader_type,
        }
    
    @classmethod
    def from_dict(cls, data: LoaderResultDict) -> "LoaderResult":
        """Create from TypedDict format."""
        documents = [IngestDocument.from_dict(doc) for doc in data["documents"]]
        relations = [DocumentRelation.from_dict(rel) for rel in data["relations"]]
        
        return cls(
            documents=documents,
            relations=relations,
            metadata=data.get("metadata", {}),
            loader_type=data.get("loader_type", "base"),
            load_time=data.get("load_time", 0.0),
        )


# Pydantic models for validation and API use

class PydanticIngestDocument(BaseModel):
    """Pydantic model for document validation."""
    
    id: str = Field(..., description="Unique document identifier")
    content: str = Field(..., description="Document content")
    source: str = Field(..., description="Source path or identifier")
    document_type: DocumentType = Field(default=DocumentType.TEXT, description="Document type")
    metadata: Dict[str, Any] = Field(default_factory=lambda: {}, description="Additional metadata")
    embedding: Optional[EmbeddingVector] = Field(None, description="Original embedding")
    enhanced_embedding: Optional[EmbeddingVector] = Field(None, description="ISNE-enhanced embedding")
    chunks: List[Dict[str, Any]] = Field(default_factory=list, description="Document chunks")
    embeddings: Dict[str, List[float]] = Field(default_factory=lambda: {}, description="Chunk embeddings")
    embedding_model: str = Field(default="modernbert", description="Embedding model used")
    created_at: Optional[datetime] = Field(default_factory=datetime.now, description="Creation timestamp")
    processed_at: Optional[datetime] = Field(None, description="Processing timestamp")
    
    class Config:
        extra = "allow"
        arbitrary_types_allowed = True


class PydanticDocumentRelation(BaseModel):
    """Pydantic model for document relationship validation."""
    
    source_id: str = Field(..., description="Source document ID")
    target_id: str = Field(..., description="Target document ID")
    relation_type: RelationType = Field(default=RelationType.GENERIC, description="Relationship type")
    weight: float = Field(default=1.0, description="Relationship weight", ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=lambda: {}, description="Additional metadata")
    confidence: Optional[float] = Field(None, description="Confidence score", ge=0.0, le=1.0)
    created_at: Optional[datetime] = Field(default_factory=datetime.now, description="Creation timestamp")
    
    class Config:
        extra = "allow"


# Type aliases for convenience
AnyIngestDocument = Union[IngestDocument, IngestDocumentDict, PydanticIngestDocument]
AnyDocumentRelation = Union[DocumentRelation, DocumentRelationDict, PydanticDocumentRelation]
AnyLoaderResult = Union[LoaderResult, LoaderResultDict]