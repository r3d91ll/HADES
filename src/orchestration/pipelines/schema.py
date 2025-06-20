"""
Schema definitions for pipeline data structures.

This module provides Pydantic models for validating and documenting the
data structures used throughout the orchestration pipeline system.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator


class ValidationSeverity(str, Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationIssue(BaseModel):
    """Model for representing a validation issue."""
    severity: ValidationSeverity
    message: str
    location: Optional[str] = None
    context: Dict[str, Any] = Field(default_factory=lambda: {})


class ValidationResult(BaseModel):
    """Model for validation results."""
    is_valid: bool
    issues: List[ValidationIssue] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)
    stage_name: Optional[str] = None
    
    def get_issues_by_severity(self, severity: ValidationSeverity) -> List[ValidationIssue]:
        """Get issues filtered by severity level.
        
        Args:
            severity: Severity level to filter by
            
        Returns:
            List of issues with the specified severity
        """
        return [issue for issue in self.issues if issue.severity == severity]
    
    @property
    def has_errors(self) -> bool:
        """Check if there are any error or critical issues.
        
        Returns:
            True if there are any ERROR or CRITICAL issues
        """
        return any(issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] 
                 for issue in self.issues)


class Relationship(BaseModel):
    """Model for chunk relationships."""
    source: str
    target: str
    type: str
    weight: float = 1.0
    metadata: Dict[str, Any] = Field(default_factory=lambda: {})


class ChunkSchema(BaseModel):
    """Schema for document chunks."""
    id: str
    text: str
    embedding: Optional[List[float]] = None
    isne_embedding: Optional[List[float]] = None
    relationships: List[Relationship] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=lambda: {})
    validation: Optional[ValidationResult] = None
    
    @validator('embedding', 'isne_embedding')
    def check_embedding_dimensions(cls, v: Optional[List[float]], values: Dict[str, Any]) -> Optional[List[float]]:
        """Validate embedding dimensions."""
        if v is not None and len(v) == 0:
            raise ValueError("Embedding must not be empty")
        return v


class DocumentSchema(BaseModel):
    """Schema for processed documents."""
    file_id: str
    file_path: str
    file_name: str
    file_type: str
    content_type: str = "text"
    chunks: List[ChunkSchema] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=lambda: {})
    processed_at: datetime = Field(default_factory=datetime.now)
    validation: Optional[ValidationResult] = None
    
    @property
    def total_chunks(self) -> int:
        """Get the total number of chunks in the document.
        
        Returns:
            Number of chunks
        """
        return len(self.chunks)
    
    @property
    def has_embeddings(self) -> bool:
        """Check if document chunks have embeddings.
        
        Returns:
            True if all chunks have embeddings
        """
        return all(chunk.embedding is not None for chunk in self.chunks)
    
    @property
    def has_isne_embeddings(self) -> bool:
        """Check if document chunks have ISNE embeddings.
        
        Returns:
            True if all chunks have ISNE embeddings
        """
        return all(chunk.isne_embedding is not None for chunk in self.chunks)
    
    @property
    def has_relationships(self) -> bool:
        """Check if document chunks have relationships.
        
        Returns:
            True if any chunks have relationships
        """
        return any(len(chunk.relationships) > 0 for chunk in self.chunks)
    
    @property
    def total_relationships(self) -> int:
        """Get the total number of relationships in the document.
        
        Returns:
            Number of relationships
        """
        return sum(len(chunk.relationships) for chunk in self.chunks)


class BatchProcessingStats(BaseModel):
    """Statistics for batch document processing."""
    total_documents: int = 0
    documents_processed: int = 0
    total_chunks: int = 0
    chunks_with_embeddings: int = 0
    chunks_with_isne: int = 0
    chunks_with_relationships: int = 0
    total_relationships: int = 0
    processing_times: Dict[str, float] = Field(default_factory=lambda: {})
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    
    def add_document_stats(self, doc: DocumentSchema) -> None:
        """Add stats from a processed document.
        
        Args:
            doc: Processed document
        """
        self.documents_processed += 1
        self.total_chunks += doc.total_chunks
        self.chunks_with_embeddings += sum(1 for chunk in doc.chunks if chunk.embedding is not None)
        self.chunks_with_isne += sum(1 for chunk in doc.chunks if chunk.isne_embedding is not None)
        self.chunks_with_relationships += sum(1 for chunk in doc.chunks if len(chunk.relationships) > 0)
        self.total_relationships += doc.total_relationships


class StorageResult(BaseModel):
    """Result of storage operations."""
    stored_documents: int = 0
    stored_chunks: int = 0
    stored_relationships: int = 0
    operation_mode: str
    database_name: str
    collections: Dict[str, int] = Field(default_factory=lambda: {})
    execution_time: float
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)


class PipelineExecutionContext(BaseModel):
    """Context for pipeline execution with orchestration capabilities."""
    pipeline_name: str
    execution_mode: str = "sequential"  # sequential, batch, parallel
    batch_size: int = 10
    enable_monitoring: bool = True
    enable_alerts: bool = True
    output_dir: Optional[str] = None
    worker_config: Dict[str, Any] = Field(default_factory=lambda: {})
    queue_config: Dict[str, Any] = Field(default_factory=lambda: {})
    stage_configs: Dict[str, Dict[str, Any]] = Field(default_factory=lambda: {})
    timestamp: datetime = Field(default_factory=datetime.now)


class PipelineExecutionResult(BaseModel):
    """Result of complete pipeline execution."""
    pipeline_name: str
    execution_context: PipelineExecutionContext
    total_execution_time: float
    stage_results: List[Dict[str, Any]] = Field(default_factory=list)
    batch_stats: BatchProcessingStats = Field(default_factory=BatchProcessingStats)
    storage_result: Optional[StorageResult] = None
    success: bool
    error_summary: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)