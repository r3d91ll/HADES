"""
Text pipeline schemas for HADES-PathRAG.

This module defines schemas specific to text processing pipelines,
including configuration, document processing, and result types.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import Field, field_validator, model_validator

from ..common.base import BaseSchema
from ..common.enums import DocumentType, SchemaVersion, ProcessingStage
from ..common.types import MetadataDict
from .base import PipelineConfigSchema, PipelineStage


class ChunkingStrategy(str, Enum):
    """Strategies for document chunking."""
    FIXED_SIZE = "fixed_size"
    SEMANTIC = "semantic"
    RECURSIVE = "recursive"
    OVERLAP = "overlap"
    CODE_AWARE = "code_aware"
    CHONKY = "chonky"


class TextPipelineConfigSchema(PipelineConfigSchema):
    """Configuration for text processing pipeline."""
    
    input_dir: str = Field(..., description="Input directory path")
    output_dir: Optional[str] = Field(default=None, description="Output directory path")
    file_types: List[str] = Field(default_factory=lambda: ["*.txt", "*.md", "*.pdf"], 
                                 description="File types to process")
    recursive: bool = Field(default=True, description="Whether to recursively search for files")
    exclude_patterns: List[str] = Field(default_factory=list, 
                                       description="Patterns to exclude from processing")
    
    # Document loading config
    max_file_size_mb: float = Field(default=10.0, 
                                   description="Maximum file size in MB to process")
    encoding: str = Field(default="utf-8", 
                         description="Character encoding for text files")
    
    # Preprocessing config
    remove_stopwords: bool = Field(default=False, 
                                  description="Whether to remove stopwords")
    clean_html: bool = Field(default=True, 
                            description="Whether to clean HTML tags")
    language: Optional[str] = Field(default=None, 
                                   description="Document language (e.g., 'en')")
    
    # Chunking config
    chunk_strategy: ChunkingStrategy = Field(default=ChunkingStrategy.SEMANTIC, 
                                            description="Chunking strategy")
    chunk_size: int = Field(default=1000, 
                           description="Target chunk size in characters")
    chunk_overlap: int = Field(default=200, 
                              description="Overlap between chunks in characters")
    
    # Embedding config
    embedding_model: str = Field(default="modernbert", 
                                description="Name of embedding model to use")
    normalize_embeddings: bool = Field(default=True, 
                                      description="Whether to normalize embeddings")
    batch_size: int = Field(default=32, 
                           description="Batch size for embedding generation")
    
    # Storage config
    store_embeddings: bool = Field(default=True, 
                                  description="Whether to store embeddings")
    store_raw_documents: bool = Field(default=True, 
                                     description="Whether to store raw documents")
    store_chunks: bool = Field(default=True, 
                              description="Whether to store document chunks")
    
    # Database config
    db_connection_string: Optional[str] = Field(default=None, 
                                               description="Database connection string")
    db_name: Optional[str] = Field(default=None, 
                                  description="Database name")
    create_new_collections: bool = Field(default=False, 
                                        description="Whether to create new collections/graph")
    
    @field_validator("chunk_size", "chunk_overlap", "batch_size")
    @classmethod
    def validate_positive_int(cls, v: int) -> int:
        """Validate integer values are positive."""
        if v <= 0:
            raise ValueError(f"Value must be positive, got {v}")
        return v


class TextPipelineResultSchema(BaseSchema):
    """Result of text pipeline processing for a single document."""
    
    document_id: str = Field(..., description="ID of the processed document")
    source: str = Field(..., description="Source path of the document")
    document_type: DocumentType = Field(..., description="Type of the document")
    processing_stage: ProcessingStage = Field(default=ProcessingStage.RAW, 
                                             description="Current processing stage")
    success: bool = Field(default=True, description="Whether processing was successful")
    error: Optional[str] = Field(default=None, description="Error message if processing failed")
    chunks: List[Dict[str, Any]] = Field(default_factory=list, description="Processed chunks")
    metadata: MetadataDict = Field(default_factory=dict, description="Processing metadata")
    
    @model_validator(mode='after')
    def validate_error_consistency(self) -> TextPipelineResultSchema:
        """Ensure error state is consistent."""
        if self.error and self.success:
            self.success = False
        return self
