"""
Validation utilities for HADES-PathRAG document schemas.

This module provides functions to validate documents against schemas at different
pipeline stages and handle validation errors gracefully.
"""
import json
import logging
from enum import Enum
from functools import wraps
from typing import Any, Dict, List, Optional, Type, Union, Callable, TypeVar, cast, overload, ParamSpec, Protocol, Generic

from pydantic import BaseModel, ValidationError, field_validator, model_validator

from src.schemas.documents.base import DocumentSchema
from src.schemas.documents.dataset import DatasetSchema
from src.schemas.documents.relations import DocumentRelationSchema
from src.schemas.common.enums import SchemaVersion

# Set up logger
logger = logging.getLogger(__name__)

# Type variables for generic functions
T = TypeVar('T', bound=BaseModel)
F = TypeVar('F', bound=Callable)
P = ParamSpec('P')
R = TypeVar('R')

# Check if running with Pydantic v1 or v2
_HAS_PYDANTIC_V2 = hasattr(BaseModel, 'model_config')

# Import validator from pydantic if available (for backwards compatibility)
try:
    from pydantic import validator
except ImportError:
    validator = None

# Typed wrappers for Pydantic validators to preserve type information
def typed_field_validator(*fields: str) -> Callable[[Callable[..., R]], Callable[..., R]]:
    """Type-preserving wrapper for Pydantic's field_validator.
    
    Works with both Pydantic v1 and v2 validation decorators.
    
    Args:
        fields: Field names to validate
        
    Returns:
        Decorated validator function with preserved type information
    """
    def decorator(f: Callable[..., R]) -> Callable[..., R]:
        # Use the appropriate validator based on what's available
        if _HAS_PYDANTIC_V2:
            validator_func = field_validator(*fields)(f)
        elif validator is not None:
            validator_func = validator(*fields)(f)
        else:
            # If neither is available, just return the function unchanged
            validator_func = f
            
        return cast(Callable[..., R], validator_func)
    return decorator

def typed_model_validator(mode: str = 'after') -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Type-preserving wrapper for Pydantic's model_validator.
    
    Works with both Pydantic v1 and v2 validation decorators.
    
    Args:
        mode: Validator mode ('before' or 'after')
        
    Returns:
        Decorated validator function with preserved type information
    """
    def decorator(f: Callable[..., T]) -> Callable[..., T]:
        # Use the appropriate validator based on what's available
        if _HAS_PYDANTIC_V2:
            validator_func = model_validator(mode=mode)(f)
        elif validator is not None:
            # In v1, root validators were used for model validation
            from pydantic import root_validator
            validator_func = root_validator(pre=(mode == 'before'))(f)
        else:
            # If neither is available, just return the function unchanged
            validator_func = f
            
        return cast(Callable[..., T], validator_func)
    return decorator


class ValidationStage(str, Enum):
    """Enumeration of pipeline stages where validation occurs."""
    INGESTION = "ingestion"
    PROCESSING = "processing"
    STORAGE = "storage"
    RETRIEVAL = "retrieval"
    EMBEDDING = "embedding"
    OUTPUT = "output"
    CUSTOM = "custom"


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
        
        formatted = [f"Validation errors at stage {self.stage}"]
        if self.document_id:
            formatted.append(f"Document ID: {self.document_id}")
            
        for i, error in enumerate(self.errors, 1):
            error_str = json.dumps(error, indent=2)
            formatted.append(f"Error {i}: {error_str}")
            
        return "\n".join(formatted)


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
    document_id = document_data.get('id', 'unknown')
    
    try:
        # Ensure schema version in data matches requested version
        if 'schema_version' not in document_data or document_data['schema_version'] != schema_version:
            document_data['schema_version'] = schema_version
            
        DocumentSchema(**document_data)
        return ValidationResult(True, stage, document_id=document_id)
    except ValidationError as e:
        logger.warning(f"Document validation failed at {stage} stage: {str(e)}")
        return ValidationResult(
            False, 
            stage,
            errors=[error.dict() for error in e.errors()],
            document_id=document_id
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
    dataset_id = dataset_data.get('id', 'unknown')
    
    try:
        # Ensure schema version in data matches requested version
        if 'schema_version' not in dataset_data or dataset_data['schema_version'] != schema_version:
            dataset_data['schema_version'] = schema_version
            
        DatasetSchema(**dataset_data)
        return ValidationResult(True, stage, document_id=dataset_id)
    except ValidationError as e:
        logger.warning(f"Dataset validation failed at {stage} stage: {str(e)}")
        return ValidationResult(
            False, 
            stage,
            errors=[error.dict() for error in e.errors()],
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
        # Cast the return value to type T for mypy
        result = schema_class(**data)
        return cast(T, result)
    except ValidationError as e:
        msg = error_message or f"Validation failed at {stage} stage"
        error_details = "\n".join(str(err) for err in e.errors())
        raise ValueError(f"{msg}: {error_details}")


class ValidationCheckpoint:
    """Pipeline validation checkpoint to enforce schema validation.
    
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
        # Type annotation to preserve the original function's type
        # Explicitly state return type to be the same as the input type
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Validate input if schema is provided
            if self.input_schema and len(args) > 0:
                input_data = args[0]
                if isinstance(input_data, dict):
                    if self.strict:
                        validate_or_raise(
                            input_data, 
                            self.input_schema, 
                            self.stage,
                            f"Input validation failed for {func.__name__}"
                        )
                    else:
                        result = ValidationResult(True, self.stage)
                        try:
                            self.input_schema(**input_data)
                        except ValidationError as e:
                            errors = [error.dict() for error in e.errors()]
                            result = ValidationResult(
                                False, 
                                self.stage,
                                errors=errors
                            )
                            logger.warning(
                                f"Input validation warning in {func.__name__}: "
                                f"{result.format_errors()}"
                            )
            
            # Call the original function
            output = func(*args, **kwargs)
            
            # Validate output if schema is provided
            if self.output_schema and output is not None:
                if isinstance(output, dict):
                    if self.strict:
                        validate_or_raise(
                            output, 
                            self.output_schema, 
                            self.stage,
                            f"Output validation failed for {func.__name__}"
                        )
                    else:
                        result = ValidationResult(True, self.stage)
                        try:
                            self.output_schema(**output)
                        except ValidationError as e:
                            errors = [error.dict() for error in e.errors()]
                            result = ValidationResult(
                                False, 
                                self.stage,
                                errors=errors
                            )
                            logger.warning(
                                f"Output validation warning in {func.__name__}: "
                                f"{result.format_errors()}"
                            )
            
            return output
        
        # Copy metadata from the original function
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        wrapper.__module__ = func.__module__
        
        # Use cast to explicitly tell mypy that the wrapper has the same type as func
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
    document = document_data.copy()
    
    # Get current version
    current_version = document.get('schema_version', SchemaVersion.V1)
    
    # If already at target version, return as-is
    if current_version == target_version:
        return document
    
    # Handle specific version upgrades
    if current_version == SchemaVersion.V1 and target_version == SchemaVersion.V2:
        # V1 to V2 upgrade logic
        
        # Ensure all required fields for V2 exist
        if 'embedding' not in document:
            document['embedding'] = None
            
        if 'embedding_model' not in document:
            document['embedding_model'] = None
            
        if 'chunks' not in document:
            document['chunks'] = []
            
        if 'tags' not in document:
            document['tags'] = []
            
        # Update metadata structure if needed
        if 'metadata' in document and not isinstance(document['metadata'], dict):
            document['metadata'] = {'legacy_metadata': document['metadata']}
            
        # Set the new schema version
        document['schema_version'] = SchemaVersion.V2
        
        logger.info(f"Upgraded document {document.get('id', 'unknown')} from V1 to V2")
        
    else:
        logger.warning(
            f"Unsupported schema version upgrade from {current_version} to {target_version}. "
            f"Document {document.get('id', 'unknown')} returned as-is."
        )
    
    return document
