"""Validation type definitions."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from ..common import BaseSchema


class ValidationResult(BaseSchema):
    """Result of a validation check."""
    is_valid: bool = Field(..., description="Whether validation passed")
    errors: List[str] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class EmbeddingValidationResult(ValidationResult):
    """Result of embedding validation."""
    embedding_dim: Optional[int] = Field(None, description="Detected embedding dimension")
    has_nan: bool = Field(False, description="Whether embedding contains NaN values")
    has_inf: bool = Field(False, description="Whether embedding contains infinity values")
    norm: Optional[float] = Field(None, description="Embedding norm")
    stats: Dict[str, float] = Field(default_factory=dict, description="Embedding statistics")


class BatchValidationResult(BaseSchema):
    """Result of batch validation."""
    total_items: int = Field(..., description="Total items validated")
    valid_items: int = Field(..., description="Number of valid items")
    invalid_items: int = Field(..., description="Number of invalid items")
    item_results: List[ValidationResult] = Field(default_factory=list, description="Individual results")
    summary: Dict[str, Any] = Field(default_factory=dict, description="Validation summary")