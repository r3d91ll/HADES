"""Base document type definitions.

This module provides foundational type definitions for documents,
entities, and metadata that can be extended for specific document types.
"""

from typing import Any, Dict, List, Optional, Union
from datetime import datetime


class BaseEntity:
    """Base class for entities extracted from documents."""
    
    def __init__(
        self, 
        entity_type: str,
        text: str,
        start_pos: Optional[int] = None,
        end_pos: Optional[int] = None,
        confidence: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize a base entity.
        
        Args:
            entity_type: Type of the entity (e.g., "person", "organization")
            text: The text content of the entity
            start_pos: Starting position in the document text
            end_pos: Ending position in the document text
            confidence: Confidence score for the entity extraction
            metadata: Additional metadata for the entity
        """
        self.entity_type = entity_type
        self.text = text
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.confidence = confidence
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary representation."""
        return {
            "entity_type": self.entity_type,
            "text": self.text,
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
            "confidence": self.confidence,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseEntity":
        """Create entity from dictionary representation."""
        return cls(
            entity_type=data["entity_type"],
            text=data["text"],
            start_pos=data.get("start_pos"),
            end_pos=data.get("end_pos"),
            confidence=data.get("confidence"),
            metadata=data.get("metadata", {})
        )


class BaseMetadata:
    """Base class for document metadata."""
    
    def __init__(
        self,
        source_path: Optional[str] = None,
        title: Optional[str] = None,
        author: Optional[Union[str, List[str]]] = None,
        created_date: Optional[Union[str, datetime]] = None,
        modified_date: Optional[Union[str, datetime]] = None,
        language: Optional[str] = None,
        content_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        custom_metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize document metadata.
        
        Args:
            source_path: Path to the source document
            title: Document title
            author: Document author(s)
            created_date: Document creation date
            modified_date: Document last modification date
            language: Document language
            content_type: Document content type
            tags: List of tags for the document
            custom_metadata: Additional custom metadata
        """
        self.source_path = source_path
        self.title = title
        self.author = author
        self.created_date = created_date
        self.modified_date = modified_date
        self.language = language
        self.content_type = content_type
        self.tags = tags or []
        self.custom_metadata = custom_metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary representation."""
        return {
            "source_path": self.source_path,
            "title": self.title,
            "author": self.author,
            "created_date": self.created_date.isoformat() if isinstance(self.created_date, datetime) else self.created_date,
            "modified_date": self.modified_date.isoformat() if isinstance(self.modified_date, datetime) else self.modified_date,
            "language": self.language,
            "content_type": self.content_type,
            "tags": self.tags,
            "custom_metadata": self.custom_metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseMetadata":
        """Create metadata from dictionary representation."""
        return cls(
            source_path=data.get("source_path"),
            title=data.get("title"),
            author=data.get("author"),
            created_date=data.get("created_date"),
            modified_date=data.get("modified_date"),
            language=data.get("language"),
            content_type=data.get("content_type"),
            tags=data.get("tags"),
            custom_metadata=data.get("custom_metadata", {})
        )


class BaseDocument:
    """Base class for documents."""
    
    def __init__(
        self,
        document_id: str,
        content: str,
        metadata: Optional[Union[BaseMetadata, Dict[str, Any]]] = None,
        entities: Optional[List[Union[BaseEntity, Dict[str, Any]]]] = None,
        chunks: Optional[List[Dict[str, Any]]] = None,
        errors: Optional[List[Dict[str, Any]]] = None
    ):
        """Initialize a base document.
        
        Args:
            document_id: Unique identifier for the document
            content: Document content
            metadata: Document metadata
            entities: Entities extracted from the document
            chunks: Document chunks after chunking
            errors: Processing errors
        """
        self.document_id = document_id
        self.content = content
        
        # Convert metadata dictionary to BaseMetadata if needed
        if isinstance(metadata, dict):
            self.metadata = BaseMetadata.from_dict(metadata)
        else:
            self.metadata = metadata or BaseMetadata()
        
        # Process entities
        self.entities = []
        if entities:
            for entity in entities:
                if isinstance(entity, dict):
                    self.entities.append(BaseEntity.from_dict(entity))
                else:
                    self.entities.append(entity)
        
        self.chunks = chunks or []
        self.errors = errors or []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary representation."""
        return {
            "document_id": self.document_id,
            "content": self.content,
            "metadata": self.metadata.to_dict(),
            "entities": [entity.to_dict() for entity in self.entities],
            "chunks": self.chunks,
            "errors": self.errors
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseDocument":
        """Create document from dictionary representation."""
        return cls(
            document_id=data["document_id"],
            content=data["content"],
            metadata=data.get("metadata", {}),
            entities=data.get("entities", []),
            chunks=data.get("chunks", []),
            errors=data.get("errors", [])
        )
