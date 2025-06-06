"""
Validation type definitions for HADES-PathRAG.

This package contains type definitions for validation functionality,
including validation results, document structures, and enums.
"""

from .base import (
    ChunkIdentifier,
    DocumentIdentifier,
    ValidationStage,
    ValidationSeverity,
    ValidationStatus,
)
from .results import (
    PreValidationResult,
    PostValidationResult,
    ValidationDiscrepancies,
    ValidationSummary,
)
from .documents import (
    ValidatedDocumentList,
    DocumentListWithValidation,
)

__all__ = [
    # Base types
    "ChunkIdentifier",
    "DocumentIdentifier", 
    "ValidationStage",
    "ValidationSeverity",
    "ValidationStatus",
    
    # Result types
    "PreValidationResult",
    "PostValidationResult",
    "ValidationDiscrepancies",
    "ValidationSummary",
    
    # Document types
    "ValidatedDocumentList",
    "DocumentListWithValidation",
]