"""
Type definitions for JSON document processing.

This module provides type definitions specific to JSON document processing,
including node representation, path expressions, and validation schemas.
"""

from typing import Dict, Any, List, Optional, Union, TypedDict, Set


class JSONNodeInfo(TypedDict, total=False):
    """Information about a node in a JSON structure."""
    
    path: str  # JSON path to this node (e.g., $.store.book[0].title)
    type: str  # Type of node (object, array, string, number, boolean, null)
    value: Optional[Any]  # Value of the node (if scalar)
    value_preview: Optional[str]  # Preview/summary of the value
    parent_path: Optional[str]  # Path to parent node
    key: Optional[str]  # Key in parent object (if applicable)
    index: Optional[int]  # Index in parent array (if applicable)
    children: List[str]  # List of child paths
    size: int  # Size in bytes of this node
    depth: int  # Depth in the JSON structure
    is_leaf: bool  # Whether this is a leaf node
    schema_type: Optional[str]  # JSON Schema type, if available
    # Additional fields for adapter compatibility
    name: Optional[str]  # Name/identifier for the node
    start_char: Optional[int]  # Start character position in source text
    end_char: Optional[int]  # End character position in source text
    line_start: Optional[int]  # Start line number
    line_end: Optional[int]  # End line number
    value_type: Optional[str]  # Type name as string
    parent: Optional[str]  # Parent node ID


class JSONQueryResult(TypedDict, total=False):
    """Result of a JSON path query."""
    
    path: str  # Path that was queried
    matches: List[str]  # List of paths that matched
    values: List[Any]  # Values at those paths
    count: int  # Number of matches


class JSONRelationship(TypedDict, total=False):
    """Relationship between JSON nodes."""
    
    source_path: str  # Path to source node
    target_path: str  # Path to target node
    type: str  # Type of relationship (reference, containment, etc.)
    metadata: Dict[str, Any]  # Additional relationship metadata


class JSONPathSegment(TypedDict):
    """Segment of a JSON path expression."""
    
    type: str  # Type of segment (property, index, wildcard, etc.)
    value: Optional[str]  # Value of the segment
    is_wildcard: bool  # Whether this is a wildcard segment
    position: int  # Position in the path


class JSONSchemaValidationResult(TypedDict):
    """Result of validating JSON against a schema."""
    
    valid: bool  # Whether validation passed
    errors: List[str]  # List of validation errors
    schema_id: Optional[str]  # ID of schema used
    validated_paths: List[str]  # Paths that were validated
