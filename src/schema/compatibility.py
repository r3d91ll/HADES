"""
Compatibility layer for schema migration.

This module provides backward compatibility while migrating from 
src/schema to src/schemas. It should be removed once the migration is complete.
"""
import warnings
from typing import Any, Dict, List, Optional, Type, Union

# Import from new location but re-export with old names
from src.schemas.documents.base import DocumentSchema, ChunkMetadata
from src.schemas.documents.dataset import DatasetSchema
from src.schemas.documents.relations import DocumentRelationSchema
from src.schemas.common.enums import DocumentType, RelationType, SchemaVersion
from src.schemas.common.validation import (
    ValidationStage, 
    ValidationResult,
    validate_document,
    validate_dataset,
    validate_or_raise,
    ValidationCheckpoint,
    upgrade_schema_version
)

# Show deprecation warning
warnings.warn(
    "The 'src.schema' module is deprecated and will be removed in a future version. "
    "Please use 'src.schemas' instead.",
    DeprecationWarning,
    stacklevel=2
)
