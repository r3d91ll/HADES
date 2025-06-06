"""
Base types and enums for validation functionality.

This module contains foundational types used throughout the validation system.
"""

from enum import Enum
from typing import TypeAlias

# Type aliases for clarity
ChunkIdentifier: TypeAlias = str
DocumentIdentifier: TypeAlias = str


class ValidationStage(str, Enum):
    """Stages of validation in the processing pipeline."""
    PRE_ISNE = "pre_isne"
    POST_ISNE = "post_isne"
    PRE_STORAGE = "pre_storage"
    POST_STORAGE = "post_storage"


class ValidationSeverity(str, Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationStatus(str, Enum):
    """Overall validation status."""
    PASSED = "passed"
    FAILED = "failed"
    WARNINGS = "warnings"
    SKIPPED = "skipped"