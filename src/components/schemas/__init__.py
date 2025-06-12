"""
Schema Validation Components

This module contains schema validation components that implement the
SchemaValidator protocol for different validation approaches.

Available components:
- pydantic: Pydantic-based schema validation
"""

from .pydantic import PydanticValidator

__all__ = [
    "PydanticValidator"
]