"""
Data models for Python code analysis.

This module defines the type structures used to represent Python code elements
in a hierarchical JSON format. These models provide type safety and validation
for the output of the Python code processing pipeline.

NOTE: This module uses types that have been migrated to src/types/docproc.
New code should import types directly from there instead of this module.
"""

from __future__ import annotations

# Re-export centralized types
from src.types.docproc.enums import RelationshipType, ImportSourceType, AccessLevel
from src.types.docproc.code_elements import (
    CodeRelationship,
    ElementRelationship,
    LineRange,
    Annotation,
    ImportElement,
    FunctionElement,
    MethodElement,
    ClassElement,
    ModuleElement,
    PySymbolTable,
    PythonDocument
)

# This file is kept for backward compatibility and utility functions
# All type definitions have been migrated to src/types/docproc/


def get_default_relationship_weight(rel_type: RelationshipType) -> float:
    """Get the default weight for a relationship type based on PathRAG architecture."""
    
    # Primary relationships (0.8-1.0)
    if rel_type in (RelationshipType.CALLS, RelationshipType.CONTAINS, RelationshipType.IMPLEMENTS):
        return 0.9
        
    # Secondary relationships (0.5-0.7)
    if rel_type in (RelationshipType.IMPORTS, RelationshipType.REFERENCES, RelationshipType.EXTENDS):
        return 0.7
        
    # Tertiary relationships (0.2-0.4)
    if rel_type in (RelationshipType.SIMILAR_TO, RelationshipType.RELATED_TO):
        return 0.3
        
    return 0.5  # Default weight
