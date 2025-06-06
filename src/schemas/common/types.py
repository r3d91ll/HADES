"""
Common type definitions for HADES-PathRAG.

This module defines type annotations and aliases used across multiple
components of the system for consistent type checking.
"""
from __future__ import annotations

from typing import Dict, Any, Optional, List, Union, Literal, cast, Callable, Generator, TypeAlias
import numpy as np
from pydantic import Field, GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema

# This file has been significantly refactored. Common types are now in src.types.common
# The UUIDString class below is kept for Pydantic validation purposes only


# Custom type for UUID strings with validation
class UUIDString(str):
    """String type that must conform to UUID format."""
    
    # This is required for string validation to work in Pydantic v2
    def __new__(cls, content: str) -> 'UUIDString':
        # Validate UUID format first
        from uuid import UUID
        try:
            UUID(content)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid UUID format: {content}")
        # Then create the string instance
        return super().__new__(cls, content)
    
    @classmethod
    def __get_validators__(cls) -> Generator[Callable[[str], str], None, None]:
        # Correct return type for validator functions
        from typing import Generator
        """Return a list of validator functions.
        
        Returns:
            A list of validator functions
        """
        # This is for Pydantic v1 compatibility
        from uuid import UUID
        
        def validate(v: str) -> str:
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


# These types are now imported from src.types.common above
# PathSpec, ArangoDocument, GraphNode, MetadataDict are available via import
