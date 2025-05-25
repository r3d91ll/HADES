"""
Dataset schemas for HADES-PathRAG.

This module defines schemas for collections of documents and their relationships,
representing complete datasets in the system.
"""
from __future__ import annotations

from datetime import datetime
import uuid
from typing import Any, Dict, List, Optional

from pydantic import Field, field_validator, model_validator

from ..common.base import BaseSchema
from ..common.enums import SchemaVersion
from ..common.types import MetadataDict
from .base import DocumentSchema
from .relations import DocumentRelationSchema


class DatasetSchema(BaseSchema):
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
    metadata: MetadataDict = Field(default_factory=dict, description="Additional dataset metadata")
    
    model_config = {
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
    
    @field_validator("id")
    @classmethod
    def validate_id(cls, v: Optional[str]) -> str:
        """Validate ID is not empty."""
        if not v:
            return str(uuid.uuid4())
        return v
        
    @model_validator(mode="after")
    def ensure_timestamps(self) -> DatasetSchema:
        """Ensure timestamps are present and valid."""
        # Set update time to creation time if not provided
        if self.updated_at is None:
            self.updated_at = self.created_at
        return self
        
    def add_document(self, document: DocumentSchema) -> None:
        """Add a document to the dataset.
        
        Args:
            document: The document to add
        """
        self.documents[document.id] = document
        self.updated_at = datetime.now()
        
    def add_relation(self, relation: DocumentRelationSchema) -> None:
        """Add a relationship between documents to the dataset.
        
        Args:
            relation: The relation to add
        """
        # Verify that both documents exist in the dataset
        if relation.source_id not in self.documents:
            raise ValueError(f"Source document {relation.source_id} not in dataset")
        if relation.target_id not in self.documents:
            raise ValueError(f"Target document {relation.target_id} not in dataset")
            
        self.relations.append(relation)
        self.updated_at = datetime.now()
        
    @classmethod
    def from_ingest_dataset(cls, dataset: Any) -> DatasetSchema:
        """Convert an existing IngestDataset to a DatasetSchema.
        
        Args:
            dataset: An IngestDataset instance
            
        Returns:
            DatasetSchema: A validated dataset schema instance
        """
        # Handle conversion from old types
        data = dataset.to_dict() if hasattr(dataset, 'to_dict') else dataset
        
        # Handle documents conversion
        documents = {}
        if 'documents' in data:
            for doc_id, doc_data in data['documents'].items():
                documents[doc_id] = DocumentSchema.from_ingest_document(doc_data)
                
        # Update data with converted documents
        data['documents'] = documents
        
        # Handle relations conversion
        relations = []
        if 'relations' in data:
            for rel_data in data['relations']:
                relations.append(DocumentRelationSchema(**rel_data))
                
        # Update data with converted relations  
        data['relations'] = relations
        
        # Create and validate new schema instance
        return cls(**data)
