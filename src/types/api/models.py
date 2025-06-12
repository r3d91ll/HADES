"""
API data models for HADES.

This module contains Pydantic models for API data structures.
"""

from pydantic import BaseModel, Field
from typing import Dict, Any


class QueryResult(BaseModel):
    """
    A single result from a PathRAG query.
    
    Attributes:
        content: The content snippet for this result
        path: Source file path or identifier
        confidence: Confidence score for this result (0-1)
        metadata: Additional information about this result
    """
    content: str
    path: str
    confidence: float
    metadata: Dict[str, Any] = Field(default_factory=lambda: {})