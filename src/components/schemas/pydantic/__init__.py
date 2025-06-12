"""
Pydantic Schema Validation Component

This module provides Pydantic-based schema validation implementing the
SchemaValidator protocol.
"""

from .validator import PydanticValidator

__all__ = [
    "PydanticValidator"
]