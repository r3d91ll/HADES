"""
Validation utilities for document processing.

This module provides helper functions for validating document data at various
stages of the processing pipeline. It includes functions for applying validation
models and handling validation errors gracefully.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Type, TypeVar, Union, cast

from pydantic import BaseModel, ValidationError

from src.docproc.schemas.base import BaseDocument
from src.docproc.schemas.python_document import PythonDocument

# Setup logging
logger = logging.getLogger(__name__)

# Define a generic type that extends BaseModel
T = TypeVar('T', bound=BaseModel)


def validate_document(data: Dict[str, Any]) -> BaseDocument:
    """
    Validate document data and return the appropriate model instance.
    
    Attempts to validate the document against the most specific schema first,
    then falls back to more general schemas if necessary. The function returns
    a BaseDocument instance which may be a subclass like PythonDocument if validation
    succeeded against that model.
    
    Args:
        data: Document data dictionary
        
    Returns:
        Validated document model (BaseDocument or a subclass)
        
    Raises:
        ValidationError: If validation fails against all schemas
    """
    doc_format = data.get("format", "").lower()
    
    # Try to validate against the appropriate model
    if doc_format == "python":
        try:
            return PythonDocument.model_validate(data)
        except ValidationError as e:
            logger.warning(f"Python document validation failed: {e}")
            # Fall back to base document validation
            pass
    
    # Fall back to base document validation
    # Return as base type to satisfy mypy
    return BaseDocument.model_validate(data)


def validate_or_default(
    data: Dict[str, Any], 
    model_class: Type[T], 
    default: Optional[T] = None
) -> Optional[T]:
    """
    Validate against a model or return default if validation fails.
    
    Useful for when validation should not block processing.
    
    Args:
        data: Data to validate
        model_class: Pydantic model class
        default: Default value to return on validation failure
        
    Returns:
        Validated model or default
    """
    try:
        return model_class.model_validate(data)
    except ValidationError as e:
        logger.warning(f"Validation failed for {model_class.__name__}: {e}")
        return default


def safe_validate(
    data: Dict[str, Any], 
    model_class: Type[T]
) -> Dict[str, Any]:
    """
    Attempt to validate data, but return original data if validation fails.
    
    This function is useful during migration when you want to start using
    validation but need to ensure backward compatibility.
    
    Args:
        data: Data to validate
        model_class: Pydantic model class
        
    Returns:
        Validated model dict or original data dict
    """
    try:
        validated = model_class.model_validate(data)
        return validated.model_dump()
    except ValidationError as e:
        logger.warning(f"Validation failed - returning original data: {e}")
        # Add error info to the data
        data["_validation_error"] = str(e)
        return data


def add_validation_to_adapter(
    adapter_process_result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Add validation to an existing adapter's process result.
    
    This is a utility function to help transition existing adapters to use
    validation without requiring a full rewrite.
    
    Args:
        adapter_process_result: The result dictionary from an adapter's process method
        
    Returns:
        The validated data as a dictionary, or the original data with validation errors
    """
    doc_format = adapter_process_result.get("format", "").lower()
    
    try:
        # Use different variable names to avoid type conflicts
        if doc_format == "python":
            python_doc = PythonDocument.model_validate(adapter_process_result)
            return python_doc.model_dump()
        else:
            base_doc = BaseDocument.model_validate(adapter_process_result)
            return base_doc.model_dump()
    except ValidationError as e:
        logger.warning(f"Validation failed for {doc_format} document: {e}")
        # Add error info but preserve original data
        adapter_process_result["_validation_error"] = str(e)
        return adapter_process_result
