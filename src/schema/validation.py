"""
Validation utilities for HADES-PathRAG document schemas.

This module provides functions to validate documents against schemas at different
pipeline stages and handle validation errors gracefully.
"""
import json
import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Type, Union, Callable, TypeVar, cast

from pydantic import BaseModel, ValidationError

from src.schema.document_schema import (
    DocumentSchema, 
    DatasetSchema,
    DocumentRelationSchema,
    SchemaVersion
)

# Set up logger
logger = logging.getLogger(__name__)

# Type variables for generic functions
T = TypeVar('T', bound=BaseModel)
F = TypeVar('F', bound=Callable)


class ValidationStage(str, Enum):
    """Enumeration of pipeline stages where validation occurs."""
    INGESTION = "ingestion"
    PREPROCESSING = "preprocessing"
    CHUNKING = "chunking"
    ENRICHMENT = "enrichment"
    EMBEDDING = "embedding"
    STORAGE = "storage"
    RETRIEVAL = "retrieval"


class ValidationResult:
    """Result of a validation operation."""
    
    def __init__(
        self, 
        is_valid: bool, 
        stage: ValidationStage,
        errors: Optional[List[Dict[str, Any]]] = None,
        document_id: Optional[str] = None
    ):
        """
        Initialize validation result.
        
        Args:
            is_valid: Whether validation passed
            stage: Pipeline stage where validation occurred
            errors: List of validation errors if any
            document_id: ID of the document being validated
        """
        self.is_valid = is_valid
        self.stage = stage
        self.errors = errors or []
        self.document_id = document_id
    
    def __bool__(self) -> bool:
        """Allow using the result in boolean context."""
        return self.is_valid
    
    def format_errors(self) -> str:
        """Format errors for logging or display."""
        if not self.errors:
            return "No validation errors"
            
        error_msgs = []
        doc_info = f" for document {self.document_id}" if self.document_id else ""
        error_msgs.append(f"Validation errors at {self.stage.value} stage{doc_info}:")
        
        for i, error in enumerate(self.errors, 1):
            loc = ".".join(str(l) for l in error.get("loc", []))
            msg = error.get("msg", "Unknown error")
            error_msgs.append(f"  {i}. {loc}: {msg}")
            
        return "\n".join(error_msgs)


def validate_document(
    document_data: Dict[str, Any], 
    stage: ValidationStage,
    schema_version: SchemaVersion = SchemaVersion.V2
) -> ValidationResult:
    """
    Validate document data against the schema.
    
    Args:
        document_data: Document data to validate
        stage: Current pipeline stage
        schema_version: Schema version to validate against
        
    Returns:
        ValidationResult: Result of validation
    """
    doc_id = document_data.get("id", "unknown")
    
    try:
        DocumentSchema.model_validate(document_data)
        return ValidationResult(
            is_valid=True, 
            stage=stage,
            document_id=doc_id
        )
    except ValidationError as e:
        errors = [dict(err) for err in e.errors()]
        logger.error(f"Document validation failed at {stage.value} stage for {doc_id}: {e}")
        return ValidationResult(
            is_valid=False, 
            stage=stage,
            errors=errors,
            document_id=doc_id
        )


def validate_dataset(
    dataset_data: Dict[str, Any],
    stage: ValidationStage,
    schema_version: SchemaVersion = SchemaVersion.V2
) -> ValidationResult:
    """
    Validate dataset data against the schema.
    
    Args:
        dataset_data: Dataset data to validate
        stage: Current pipeline stage
        schema_version: Schema version to validate against
        
    Returns:
        ValidationResult: Result of validation
    """
    dataset_id = dataset_data.get("id", "unknown")
    
    try:
        DatasetSchema.model_validate(dataset_data)
        return ValidationResult(
            is_valid=True, 
            stage=stage,
            document_id=dataset_id
        )
    except ValidationError as e:
        errors = [dict(err) for err in e.errors()]
        logger.error(f"Dataset validation failed at {stage.value} stage for {dataset_id}: {e}")
        return ValidationResult(
            is_valid=False, 
            stage=stage,
            errors=errors,
            document_id=dataset_id
        )


def validate_or_raise(
    data: Dict[str, Any],
    schema_class: Type[T],
    stage: ValidationStage,
    error_message: Optional[str] = None
) -> T:
    """
    Validate data against a schema and raise exception if invalid.
    
    Args:
        data: Data to validate
        schema_class: Pydantic model class to validate against
        stage: Current pipeline stage
        error_message: Custom error message prefix
        
    Returns:
        BaseModel: Validated model instance
        
    Raises:
        ValueError: If validation fails
    """
    try:
        return schema_class.model_validate(data)
    except ValidationError as e:
        msg = error_message or f"Validation failed at {stage.value} stage"
        errors = e.errors()
        error_details = "\n".join([
            f"  - {'.'.join(str(loc) for loc in error['loc'])}: {error['msg']}"
            for error in errors
        ])
        raise ValueError(f"{msg}:\n{error_details}")


class ValidationCheckpoint:
    """
    Pipeline validation checkpoint to enforce schema validation.
    
    This class can be used as a decorator on pipeline components to
    validate input and output data against schemas.
    """
    
    def __init__(
        self, 
        stage: ValidationStage,
        input_schema: Optional[Type[BaseModel]] = None,
        output_schema: Optional[Type[BaseModel]] = None,
        strict: bool = False
    ):
        """
        Initialize validation checkpoint.
        
        Args:
            stage: Pipeline stage where validation occurs
            input_schema: Schema to validate input data against
            output_schema: Schema to validate output data against
            strict: Whether to raise exceptions on validation failure
        """
        self.stage = stage
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.strict = strict
    
    def __call__(self, func: F) -> F:
        """
        Decorate a function with validation.
        
        Args:
            func: Function to decorate
            
        Returns:
            Callable: Wrapped function with validation
        """
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Extract the document data from arguments
            # This assumes the first positional argument after self is the data
            # or a keyword argument 'document' or 'data' exists
            data = None
            if len(args) > 1:
                data = args[1]  # Skip self
            elif 'document' in kwargs:
                data = kwargs['document']
            elif 'data' in kwargs:
                data = kwargs['data']
            
            # Validate input if schema specified and data found
            if self.input_schema and data:
                if isinstance(data, dict):
                    try:
                        if self.strict:
                            validate_or_raise(
                                data=data,
                                schema_class=self.input_schema,
                                stage=self.stage
                            )
                        else:
                            result = validate_document(
                                document_data=data,
                                stage=self.stage
                            )
                            if not result:
                                logger.warning(result.format_errors())
                    except Exception as e:
                        if self.strict:
                            raise
                        logger.error(f"Input validation failed at {self.stage.value} stage: {e}")
            
            # Call the original function
            result = func(*args, **kwargs)
            
            # Validate output if schema specified and result is a dict
            if self.output_schema and result and isinstance(result, dict):
                try:
                    if self.strict:
                        validate_or_raise(
                            data=result,
                            schema_class=self.output_schema,
                            stage=self.stage
                        )
                    else:
                        validation_result = validate_document(
                            document_data=result,
                            stage=self.stage
                        )
                        if not validation_result:
                            logger.warning(validation_result.format_errors())
                except Exception as e:
                    if self.strict:
                        raise
                    logger.error(f"Output validation failed at {self.stage.value} stage: {e}")
            
            return result
        
        # Preserve function metadata
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        
        return cast(F, wrapper)


def upgrade_schema_version(
    document_data: Dict[str, Any],
    target_version: SchemaVersion = SchemaVersion.V2
) -> Dict[str, Any]:
    """
    Upgrade a document to a newer schema version.
    
    Args:
        document_data: Document data to upgrade
        target_version: Target schema version
        
    Returns:
        Dict[str, Any]: Upgraded document data
    """
    # Make a copy to avoid modifying the original
    data = document_data.copy()
    
    # Get current version or assume V1 if not specified
    current_version = data.get("schema_version", SchemaVersion.V1.value)
    
    # If already at target version, return as is
    if current_version == target_version.value:
        return data
    
    # Set the target version
    data["schema_version"] = target_version.value
    
    # V1 to V2 upgrade
    if current_version == SchemaVersion.V1.value and target_version == SchemaVersion.V2:
        # Add any missing fields with default values
        if "metadata" not in data:
            data["metadata"] = {}
        
        if "tags" not in data:
            data["tags"] = []
            
        # Ensure chunks have proper metadata
        if "chunks" in data and isinstance(data["chunks"], list):
            for i, chunk in enumerate(data["chunks"]):
                # Ensure chunk has metadata
                if not isinstance(chunk, dict):
                    continue
                    
                # Add required fields if missing
                if "chunk_index" not in chunk:
                    chunk["chunk_index"] = i
                    
                if "parent_id" not in chunk and "id" in data:
                    chunk["parent_id"] = data["id"]
                    
                if "chunk_type" not in chunk:
                    chunk["chunk_type"] = "text"
                    
                # Add start_offset and end_offset if missing
                if "start_offset" not in chunk:
                    chunk["start_offset"] = 0
                    
                if "end_offset" not in chunk:
                    # If content is available, use its length as end_offset
                    if "content" in chunk and isinstance(chunk["content"], str):
                        chunk["end_offset"] = len(chunk["content"])
                    else:
                        chunk["end_offset"] = 0
    
    # Add more version upgrade paths here as needed
    
    return data
