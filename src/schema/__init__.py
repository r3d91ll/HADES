"""
Schema standardization module for HADES-PathRAG.

This package provides Pydantic-based schema definitions and validation utilities
to ensure consistent data structures and validation throughout the pipeline.

IMPORTANT: This module is deprecated. Please use 'src.schemas' instead.
"""

# Import from compatibility layer
from src.schema.compatibility import (
    SchemaVersion,
    RelationType,
    DocumentType,
    ChunkMetadata,
    DocumentRelationSchema,
    DocumentSchema,
    DatasetSchema,
    ValidationStage,
    ValidationResult,
    validate_document,
    validate_dataset,
    validate_or_raise,
    ValidationCheckpoint,
    upgrade_schema_version
)

__all__ = [
    # Schema models
    "SchemaVersion",
    "RelationType",
    "DocumentType",
    "ChunkMetadata",
    "DocumentRelationSchema",
    "DocumentSchema",
    "DatasetSchema",
    
    # Validation utilities
    "ValidationStage",
    "ValidationResult",
    "validate_document",
    "validate_dataset",
    "validate_or_raise",
    "ValidationCheckpoint",
    "upgrade_schema_version"
]
