"""
Python document schemas for document processing validation.

This module provides Pydantic models that define the structure for Python
code documents processed in the pipeline. These models ensure proper
validation of Python-specific fields and relationships.

NOTE: This module uses types that have been migrated to src/types/docproc.
New code should import types directly from there instead of this module.
"""

from __future__ import annotations

# Re-export from centralized types
from src.types.docproc.python import (
    PythonMetadata,
    PythonEntity,
    CodeRelationship,
    CodeElement,
    FunctionElement,
    MethodElement,
    ClassElement,
    ImportElement,
    SymbolTable,
    PythonDocument,
    # Also re-export utility functions
    typed_field_validator,
    typed_model_validator,
)

# Import for backward compatibility
from src.schemas.common.base import BaseSchema
from src.types.common import MetadataDict
from src.docproc.schemas.base import BaseDocument, BaseEntity, BaseMetadata
from src.types.docproc.enums import RelationshipType, AccessLevel


# All classes are now imported from src/types/docproc/python.py
# This file only exists for backward compatibility
