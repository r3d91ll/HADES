"""
Common types and utilities for component implementations.

This module provides shared types and utilities used across
component implementations in the HADES system.
"""

from typing import Dict, Any, Optional, Union
from dataclasses import dataclass

# Import BaseSchema from the main types module
from ..common import BaseSchema

# Base class for all HADES component models
BaseHADESModel = BaseSchema


@dataclass
class ComponentResult:
    """Generic result wrapper for component operations."""
    success: bool
    data: Optional[Any] = None
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ComponentError(Exception):
    """Base exception for component-related errors."""
    pass


class ComponentConfigError(ComponentError):
    """Raised when component configuration is invalid."""
    pass