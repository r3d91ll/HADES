"""
Common types and utilities for model engine component implementations.

This module provides shared types and utilities used across
model engine component implementations in the HADES system.
"""

from typing import Dict, Any, Optional, Union
from dataclasses import dataclass

# Import BaseSchema from the main types module
from ...common import BaseSchema

# Base class for all HADES model engine models
BaseHADESModel = BaseSchema


@dataclass
class ModelEngineResult:
    """Generic result wrapper for model engine operations."""
    success: bool
    data: Optional[Any] = None
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ModelEngineError(Exception):
    """Base exception for model engine-related errors."""
    pass


class ModelEngineConfigError(ModelEngineError):
    """Raised when model engine configuration is invalid."""
    pass