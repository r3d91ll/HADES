"""
Data models for the ISNE pipeline.

This module defines the core data models used throughout the ISNE pipeline,
including document representations, relationship types, and embedding vectors.
These models are designed to support multiple modalities (text, code, etc.)
for both ingestion and query processing.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Union, Any, Set, TypeVar, Generic
import numpy as np
from pathlib import Path

# Type alias for embedding vectors
EmbeddingVector = Union[List[float], np.ndarray]


class DocumentType(str, Enum):
    """
    Enum representing the type of document being processed.
    
    This determines which processing pipeline and ISNE model will be used.
    """
    TEXT = "text"                   # General text documents 
    MARKDOWN = "markdown"           # Markdown documentation
    PDF = "pdf"                     # PDF documents
    CODE_PYTHON = "code_python"     # Python source code
    CODE_JAVASCRIPT = "code_javascript"  # JavaScript source code
    CODE_OTHER = "code_other"       # Other code types
    HTML = "html"                   # HTML documents
    JSON = "json"                   # JSON data
    XML = "xml"                     # XML data
    YAML = "yaml"                   # YAML data
    UNKNOWN = "unknown"             # Default/unknown type


class RelationType(str, Enum):
    """
    Enum representing the types of relationships between documents.
    
    The relationship type determines edge weights and traversal behavior
    during both ISNE processing and PathRAG queries.
    """
    # Structural relationships (primarily for code)
    CALLS = "calls"                 # Function calls another function
    IMPORTS = "imports"             # Module imports another module
    CONTAINS = "contains"           # Module/class contains function/method
    IMPLEMENTS = "implements"       # Class implements interface
    EXTENDS = "extends"             # Class extends another class
    REFERENCES = "references"       # Code references a symbol
    
    # Documentation relationships
    DOCUMENTS = "documents"         # Documentation describes code
    REFERS_TO = "refers_to"         # Document references another document
    FOLLOWS = "follows"             # Document follows another in sequence
    
    # Semantic relationships
    SIMILAR_TO = "similar_to"       # Documents are semantically similar
    RELATED_TO = "related_to"       # Documents are contextually related
    
    # Default relationship
    GENERIC = "generic"             # Generic relationship


@dataclass
class IngestDocument:
    """
    Represents a document to be processed by the ISNE pipeline.
    
    This class holds the document content, metadata, and computed embeddings.
    It supports different modalities through the document_type field, which
    determines how the document is processed during both ingestion and queries.
    """
    id: str                                # Unique identifier
    content: str                           # Document content
    source: str                            # Source path or identifier
    document_type: str = "text"            # Document type (see DocumentType)
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata
    embedding: Optional[EmbeddingVector] = None  # Original embedding
    enhanced_embedding: Optional[EmbeddingVector] = None  # ISNE-enhanced embedding
    chunks: List[Dict[str, Any]] = field(default_factory=list)  # Document chunks
    
    def __post_init__(self) -> None:
        """Validate document fields after initialization."""
        # Ensure document_type is a valid type
        if not hasattr(DocumentType, self.document_type.upper()):
            # Default to UNKNOWN if not valid
            self.document_type = DocumentType.UNKNOWN
        
        # Ensure metadata is a dictionary
        if not isinstance(self.metadata, dict):
            self.metadata = {}
            
        # Add model_used metadata if not present
        if "model_used" not in self.metadata:
            self.metadata["model_used"] = "modernbert"


@dataclass
class DocumentRelation:
    """
    Represents a relationship between two documents.
    
    This class defines edges in the document graph, with types that
    determine edge weights and traversal behavior.
    """
    source_id: str                         # Source document ID
    target_id: str                         # Target document ID
    relation_type: RelationType = RelationType.GENERIC  # Relationship type
    weight: float = 1.0                    # Edge weight/strength
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata
    
    def __post_init__(self) -> None:
        """Set default weights based on relation type."""
        if self.weight == 1.0:  # Only if not explicitly set
            # Set default weights based on PathRAG architecture recommendations
            if self.relation_type in (RelationType.CALLS, RelationType.CONTAINS, 
                                      RelationType.IMPLEMENTS):
                # Primary relationships
                self.weight = 0.9
            elif self.relation_type in (RelationType.IMPORTS, RelationType.REFERENCES, 
                                       RelationType.EXTENDS):
                # Secondary relationships
                self.weight = 0.6
            elif self.relation_type in (RelationType.SIMILAR_TO,):
                # Tertiary relationships
                self.weight = 0.3
            else:
                # Default relationship weight
                self.weight = 0.5


@dataclass
class LoaderResult:
    """
    Result of loading documents through a loader.
    
    This class contains the loaded documents and their relationships,
    forming the input to the ISNE pipeline.
    """
    documents: List[IngestDocument] = field(default_factory=list)  # Loaded documents
    relations: List[DocumentRelation] = field(default_factory=list)  # Document relationships
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata
    
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
