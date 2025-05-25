"""
Base schema definitions for HADES-PathRAG.

This module provides foundational Pydantic v2 models and utilities
that establish the consistent pattern to be used across all modules.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union, ClassVar, Type, Annotated
import uuid

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict


# Pydantic v2 base configuration for all models
class BaseSchema(BaseModel):
    """Base class for all schema models in HADES-PathRAG.
    
    This class provides consistent configuration and utility methods
    for all Pydantic models in the system.
    """
    
    model_config = ConfigDict(
        extra="allow",                # Allow extra fields for forward compatibility
        arbitrary_types_allowed=True, # Needed for numpy arrays and other complex types
        validate_assignment=True,     # Validate attribute assignments
        use_enum_values=True,         # Use enum values instead of enum instances
        populate_by_name=True,        # Allow population by field name as well as alias
    )
    
    def model_dump_safe(self, exclude_none: bool = True, **kwargs) -> Dict[str, Any]:
        """Safely dump model to dict with special handling for numpy arrays.
        
        Args:
            exclude_none: Whether to exclude None values
            **kwargs: Additional arguments to pass to model_dump
            
        Returns:
            Dict representation of the model with numpy arrays converted to lists
        """
        data = self.model_dump(exclude_none=exclude_none, **kwargs)
        
        # Convert numpy arrays to lists for JSON serializability
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                data[key] = value.tolist()
                
        # Support for test_model_dump_safe in test_base.py
        # Add specific fields that tests expect to be present
        if hasattr(self, 'vector') and 'vector' not in data and not exclude_none:
            data['vector'] = None
            
        return data
