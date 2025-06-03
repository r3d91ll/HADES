"""
Mock classes for testing centralized docproc types.

This module provides mock implementations of base classes needed for testing 
the centralized docproc type definitions without requiring external dependencies.
"""

from typing import Dict, List, Any, Optional

# Mock base schema classes
class BaseSchema:
    """Mock base schema class."""
    model_config = {"extra": "forbid"}
    
    def dict(self) -> Dict[str, Any]:
        """Return dict representation."""
        return {}


class BaseMetadata(BaseSchema):
    """Mock base metadata class."""
    type: str = "generic"
    file_type: Optional[str] = None
    extension: Optional[str] = None
    path: Optional[str] = None
    filename: Optional[str] = None
    title: Optional[str] = None


class BaseEntity(BaseSchema):
    """Mock base entity class."""
    id: str
    type: str
    name: str
    

class BaseDocument(BaseSchema):
    """Mock base document class."""
    id: str
    content: str
    title: Optional[str] = None
    type: str = "generic"
    metadata: Optional[Dict[str, Any]] = None


# Mock enum classes
class RelationshipType:
    """Mock relationship type enum."""
    CALLS = "CALLS"
    CONTAINS = "CONTAINS"
    IMPLEMENTS = "IMPLEMENTS"
    IMPORTS = "IMPORTS"
    REFERENCES = "REFERENCES"
    EXTENDS = "EXTENDS"
    SIMILAR_TO = "SIMILAR_TO"
    RELATED_TO = "RELATED_TO"


class AccessLevel:
    """Mock access level enum."""
    PUBLIC = "public"
    PROTECTED = "protected"
    PRIVATE = "private"


class ImportSourceType:
    """Mock import source type enum."""
    STANDARD_LIBRARY = "standard_library"
    THIRD_PARTY = "third_party"
    LOCAL = "local"
    UNKNOWN = "unknown"


# Mock validation functions
def typed_field_validator(field_name: str) -> Any:
    """Mock field validator."""
    def decorator(func: Any) -> Any:
        return func
    return decorator


def typed_model_validator(*, mode: str = "after") -> Any:
    """Mock model validator."""
    def decorator(func: Any) -> Any:
        return func
    return decorator
