"""
Document schema standardization for HADES-PathRAG.

This module defines Pydantic models for document schema validation,
ensuring consistent structure and validation throughout the pipeline.
"""
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Literal, ClassVar, Type
import uuid

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator


# Type alias for embedding vectors
EmbeddingVector = Union[List[float], np.ndarray]


class SchemaVersion(str, Enum):
    """Schema version enumeration for backward compatibility tracking."""
    V1 = "1.0.0"
    V2 = "2.0.0"  # Current version


class RelationType(str, Enum):
    """Enum representing types of relationships between documents."""
    CONTAINS = "contains"
    REFERENCES = "references"
    CONNECTS_TO = "connects_to"
    SIMILAR_TO = "similar_to"
    DEPENDS_ON = "depends_on"
    PART_OF = "part_of"
    RELATED_TO = "related_to"
    DERIVED_FROM = "derived_from"
    IMPLEMENTS = "implements"
    CUSTOM = "custom"
    
    # New relation types can be added here as the system evolves


class DocumentType(str, Enum):
    """Enum representing types of documents."""
    CODE = "code"
    MARKDOWN = "markdown"
    TEXT = "text"
    PDF = "pdf"
    HTML = "html"
    JSON = "json"
    XML = "xml"
    YAML = "yaml"
    CSV = "csv"
    UNKNOWN = "unknown"


class ChunkMetadata(BaseModel):
    """Metadata for document chunks."""
    start_offset: int = Field(..., description="Start position in the original document")
    end_offset: int = Field(..., description="End position in the original document")
    chunk_type: str = Field(default="text", description="Type of the chunk (text, code, etc.)")
    chunk_index: int = Field(..., description="Sequential index of the chunk")
    parent_id: str = Field(..., description="ID of the parent document")
    
    # Additional metadata fields
    context_before: Optional[str] = Field(default=None, description="Text context before the chunk")
    context_after: Optional[str] = Field(default=None, description="Text context after the chunk")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional chunk-specific metadata")

    model_config = {
        "arbitrary_types_allowed": True,
        "validate_assignment": True,
        "extra": "allow",
        "json_schema_extra": {
            "examples": [
                {
                    "start_offset": 0,
                    "end_offset": 100,
                    "chunk_type": "text",
                    "chunk_index": 0,
                    "parent_id": "doc123",
                    "metadata": {
                        "importance": "high"
                    }
                }
            ]
        }
    }


class DocumentRelationSchema(BaseModel):
    """Schema for document relations."""
    source_id: str = Field(..., description="ID of the source document")
    target_id: str = Field(..., description="ID of the target document")
    relation_type: RelationType = Field(..., description="Type of relationship")
    weight: float = Field(default=1.0, description="Weight or strength of the relationship")
    bidirectional: bool = Field(default=False, description="Whether the relationship is bidirectional")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional relation metadata")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique ID for this relation")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")

    model_config = {
        "arbitrary_types_allowed": True,
        "validate_assignment": True,
        "extra": "allow",
        "json_schema_extra": {
            "examples": [
                {
                    "source_id": "doc123",
                    "target_id": "doc456",
                    "relation_type": "contains",
                    "weight": 0.8,
                    "bidirectional": False
                }
            ]
        }
    }

    @field_validator('relation_type', mode='before')
    @classmethod
    def validate_relation_type(cls, v: Any) -> RelationType:
        """Validate relation type."""
        if isinstance(v, str) and v not in [item.value for item in RelationType]:
            return RelationType.CUSTOM
        if isinstance(v, RelationType):
            return v
        # If it's not a string or RelationType, convert to CUSTOM
        return RelationType.CUSTOM


class DocumentSchema(BaseModel):
    """
    Pydantic schema for document validation and standardization.
    
    This model enforces structure and type safety for documents in the HADES-PathRAG system.
    """
    # Core document properties with validation
    id: str = Field(..., description="Unique identifier for the document")
    content: str = Field(..., description="Document content text")
    source: str = Field(..., description="Origin of the document (filename, URL, etc.)")
    document_type: DocumentType = Field(..., description="Type of document")
    
    # Schema versioning
    schema_version: SchemaVersion = Field(default=SchemaVersion.V2, description="Schema version for compatibility")
    
    # Document metadata
    title: Optional[str] = Field(default=None, description="Document title")
    author: Optional[str] = Field(default=None, description="Document author")
    created_at: Optional[datetime] = Field(default=None, description="Document creation timestamp")
    updated_at: Optional[datetime] = Field(default=None, description="Document last update timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional document metadata")
    
    # Embedding data
    embedding: Optional[EmbeddingVector] = Field(default=None, description="Document embedding vector")
    embedding_model: Optional[str] = Field(default=None, description="Model used to generate the embedding")
    
    # Processing metadata
    chunks: List[ChunkMetadata] = Field(default_factory=list, description="Document chunks metadata")
    tags: List[str] = Field(default_factory=list, description="Document tags for categorization")

    model_config = {
        "arbitrary_types_allowed": True,
        "validate_assignment": True,
        "extra": "allow",
        "json_schema_extra": {
            "examples": [
                {
                    "id": "doc123",
                    "content": "Example document content",
                    "source": "example.txt",
                    "document_type": "text",
                    "title": "Example Document"
                }
            ]
        }
    }

    @field_validator('document_type', mode='before')
    @classmethod
    def validate_document_type(cls, v: Any) -> DocumentType:
        """Validate document type."""
        if isinstance(v, str):
            try:
                return DocumentType(v)
            except ValueError:
                return DocumentType.UNKNOWN
        if isinstance(v, DocumentType):
            return v
        # If it's not a string or DocumentType, return UNKNOWN
        return DocumentType.UNKNOWN

    @field_validator('id')
    @classmethod
    def validate_id(cls, v: Optional[str]) -> str:
        """Validate ID is not empty."""
        if not v:
            return str(uuid.uuid4())
        return v

    @model_validator(mode='after')
    def ensure_timestamps_and_title(self) -> 'DocumentSchema':
        """Ensure timestamps are present and derive title from source if not provided."""
        # Ensure timestamps
        if not self.created_at:
            self.created_at = datetime.now()
        if not self.updated_at:
            self.updated_at = datetime.now()
            
        # Derive title from source if not provided
        if not self.title and self.source:
            source = self.source
            # Extract filename from path if source looks like a path
            if '/' in source:
                source = source.split('/')[-1]
            if '.' in source:
                source = source.split('.')[0]
            self.title = source
            
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert document to dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the document
        """
        data = self.model_dump()
        
        # Convert datetime objects to ISO format strings
        if data.get('created_at'):
            data['created_at'] = data['created_at'].isoformat()
        if data.get('updated_at'):
            data['updated_at'] = data['updated_at'].isoformat()
            
        # Convert embedding to list if it's a numpy array
        if isinstance(data.get('embedding'), np.ndarray):
            data['embedding'] = data['embedding'].tolist()
            
        return data
    
    @classmethod
    def from_ingest_document(cls, doc: Any) -> 'DocumentSchema':
        """
        Convert an existing IngestDocument to a DocumentSchema.
        
        Args:
            doc: An IngestDocument instance
            
        Returns:
            DocumentSchema: A validated document schema instance
        """
        # Extract base fields
        document_data = {
            "id": getattr(doc, "id", str(uuid.uuid4())),
            "content": getattr(doc, "content", ""),
            "source": getattr(doc, "source", "unknown"),
            "document_type": getattr(doc, "document_type", DocumentType.UNKNOWN)
        }
        
        # Extract metadata if available
        if hasattr(doc, "metadata") and isinstance(doc.metadata, dict):
            document_data["metadata"] = doc.metadata
            
            # Extract common metadata fields
            for field in ["title", "author", "created_at", "updated_at"]:
                if field in doc.metadata:
                    document_data[field] = doc.metadata[field]
        
        # Extract embedding if available
        if hasattr(doc, "embedding") and doc.embedding is not None:
            document_data["embedding"] = doc.embedding
            document_data["embedding_model"] = getattr(doc, "embedding_model", None)
            
        # Extract chunks if available
        if hasattr(doc, "chunks") and isinstance(doc.chunks, list):
            document_data["chunks"] = [
                {
                    "chunk_index": i,
                    "start_offset": getattr(chunk, "start_offset", 0),
                    "end_offset": getattr(chunk, "end_offset", 0),
                    "parent_id": document_data["id"],
                    "chunk_type": getattr(chunk, "chunk_type", "text"),
                    "content": getattr(chunk, "content", "")
                }
                for i, chunk in enumerate(doc.chunks)
            ]
            
        return cls.model_validate(document_data)


class DatasetSchema(BaseModel):
    """Schema for dataset containing multiple documents and their relations."""
    id: str = Field(..., description="Unique identifier for the dataset")
    name: str = Field(..., description="Name of the dataset")
    description: Optional[str] = Field(default=None, description="Dataset description")
    schema_version: SchemaVersion = Field(default=SchemaVersion.V2, description="Schema version for compatibility")
    documents: Dict[str, DocumentSchema] = Field(default_factory=dict, description="Documents in the dataset")
    relations: List[DocumentRelationSchema] = Field(
            default_factory=list, description="Relations between documents"
        )
    created_at: datetime = Field(default_factory=datetime.now, description="Dataset creation timestamp")
    updated_at: Optional[datetime] = Field(default=None, description="Dataset last update timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional dataset metadata")

    model_config = {
        "arbitrary_types_allowed": True,
        "validate_assignment": True,
        "extra": "allow",
        "json_schema_extra": {
            "examples": [
                {
                    "id": "dataset123",
                    "name": "Example Dataset",
                    "description": "A collection of example documents"
                }
            ]
        }
    }

    @field_validator('id')
    @classmethod
    def validate_id(cls, v: Optional[str]) -> str:
        """Validate ID is not empty."""
        if not v:
            return str(uuid.uuid4())
        return v

    @model_validator(mode='after')
    def ensure_timestamps(self) -> 'DatasetSchema':
        """Ensure timestamps are present and valid."""
        if not self.created_at:
            self.created_at = datetime.now()
        if not self.updated_at:
            self.updated_at = datetime.now()
        return self

    def add_document(self, document: DocumentSchema) -> None:
        """Add a document to the dataset."""
        self.documents[document.id] = document

    def add_relation(self, relation: DocumentRelationSchema) -> None:
        """Add a relationship between documents to the dataset."""
        self.relations.append(relation)
    
    @classmethod
    def from_ingest_dataset(cls, dataset: Any) -> 'DatasetSchema':
        """
        Convert an existing IngestDataset to a DatasetSchema.
        
        Args:
            dataset: An IngestDataset instance
            
        Returns:
            DatasetSchema: A validated dataset schema instance
        """
        # Extract base fields
        dataset_data = {
            "id": getattr(dataset, "id", str(uuid.uuid4())),
            "name": getattr(dataset, "name", "Unnamed Dataset"),
            "description": getattr(dataset, "description", None)
        }
        
        # Extract metadata if available
        if hasattr(dataset, "metadata") and isinstance(dataset.metadata, dict):
            dataset_data["metadata"] = dataset.metadata
        
        # Create dataset instance
        dataset_schema = cls.model_validate(dataset_data)
        
        # Add documents if available
        if hasattr(dataset, "documents") and isinstance(dataset.documents, dict):
            for doc_id, doc in dataset.documents.items():
                document_schema = DocumentSchema.from_ingest_document(doc)
                dataset_schema.add_document(document_schema)
                
        # Add relations if available
        if hasattr(dataset, "relations") and isinstance(dataset.relations, list):
            for relation in dataset.relations:
                relation_data = {
                    "source_id": getattr(relation, "source_id", ""),
                    "target_id": getattr(relation, "target_id", ""),
                    "relation_type": getattr(relation, "relation_type", RelationType.RELATED_TO),
                    "weight": getattr(relation, "weight", 1.0),
                    "bidirectional": getattr(relation, "bidirectional", False)
                }
                relation_schema = DocumentRelationSchema.model_validate(relation_data)
                dataset_schema.add_relation(relation_schema)
                
        return dataset_schema
