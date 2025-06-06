"""
Common enumeration types for HADES-PathRAG.

This module re-exports enumerations from the centralized type system
for backward compatibility and convenience.
"""
from __future__ import annotations

# Import all enums from the centralized type system
from src.types.common import (
    DocumentType,
    RelationType, 
    ProcessingStage,
    ProcessingStatus,
    SchemaVersion
)

# Re-export for backward compatibility
__all__ = [
    "DocumentType",
    "RelationType", 
    "ProcessingStage",
    "ProcessingStatus",
    "SchemaVersion"
]
