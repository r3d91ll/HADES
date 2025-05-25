"""
Common type definitions for HADES-PathRAG.

This module defines type annotations and aliases used across multiple
components of the system for consistent type checking.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union, TypeAlias
import numpy as np
from pydantic import Field, GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema

# Type alias for embedding vectors with special JSON schema handling
EmbeddingVector: TypeAlias = Union[List[float], np.ndarray]


# Custom type for UUID strings with validation
class UUIDStr(str):
    """String type that must conform to UUID format."""
    
    # This is required for string validation to work in Pydantic v2
    def __new__(cls, content):
        # Validate UUID format first
        from uuid import UUID
        try:
            UUID(content)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid UUID format: {content}")
        # Then create the string instance
        return super().__new__(cls, content)
    
    @classmethod
    def __get_validators__(cls):
        # This is for Pydantic v1 compatibility
        from uuid import UUID
        
        def validate(v):
            if not isinstance(v, str):
                raise TypeError("UUID string required")
            # Validate UUID format
            try:
                UUID(v)
            except (ValueError, TypeError):
                raise ValueError(f"Invalid UUID format: {v}")
            return v
        
        yield validate
    
    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: Any, _handler: GetJsonSchemaHandler
    ) -> core_schema.CoreSchema:
        """Get Pydantic core schema for v2 compatibility."""
        # Use a simple schema that ensures we have a string and validates via __new__
        return core_schema.is_instance_schema(str)
    
    @classmethod
    def __get_json_schema__(
        cls, _handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        """Get JSON schema for UUIDStr."""
        return {
            "type": "string",
            "format": "uuid",
            "pattern": r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
        }


# PathSpec type for defining traversal paths in graphs
PathSpec: TypeAlias = List[str]

# ArangoDocument type for Arango document collections
ArangoDocument: TypeAlias = Dict[str, Any]

# GraphNode type for node representation in graph operations
GraphNode: TypeAlias = Dict[str, Any]

# MetadataDict type for consistent metadata handling
MetadataDict: TypeAlias = Dict[str, Any]
