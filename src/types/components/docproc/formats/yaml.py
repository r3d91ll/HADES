"""
Type definitions for YAML document processing.

This module provides type definitions specific to YAML document processing,
including node information, path expressions, and validation schemas.
"""

from typing import Dict, Any, List, Optional, Union, TypedDict, Set


class YAMLNodeInfo(TypedDict, total=False):
    """Information about a node in a YAML structure."""
    
    path: str  # YAML path to this node
    type: str  # Type of node (mapping, sequence, scalar)
    value: Optional[Any]  # Value of the node (if scalar)
    tag: Optional[str]  # YAML tag for the node
    parent_path: Optional[str]  # Path to parent node
    key: Optional[str]  # Key in parent mapping (if applicable)
    index: Optional[int]  # Index in parent sequence (if applicable)
    children: List[str]  # List of child paths
    line: int  # Line number in source
    column: int  # Column number in source
    is_leaf: bool  # Whether this is a leaf node
    anchor: Optional[str]  # Anchor name if node has anchor
    alias_for: Optional[str]  # Path this node is an alias for


class YAMLValidationResult(TypedDict):
    """Result of validating YAML against a schema."""
    
    valid: bool  # Whether validation passed
    errors: List[str]  # List of validation errors
    schema_id: Optional[str]  # ID of schema used
    validated_paths: List[str]  # Paths that were validated
    

class YAMLRelationship(TypedDict, total=False):
    """Relationship between YAML nodes."""
    
    source_path: str  # Path to source node
    target_path: str  # Path to target node
    type: str  # Type of relationship (reference, anchor, etc.)
    metadata: Dict[str, Any]  # Additional relationship metadata


class YAMLDocumentInfo(TypedDict):
    """Information about a YAML document."""
    
    node_count: int  # Total number of nodes
    document_count: int  # Number of documents in stream
    max_depth: int  # Maximum nesting depth
    has_anchors: bool  # Whether document uses anchors
    has_tags: bool  # Whether document uses tags
    has_directives: bool  # Whether document has directives
    structure_summary: Dict[str, int]  # Count of each node type
