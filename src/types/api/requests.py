"""
API request models for HADES-PathRAG.

This module contains Pydantic models for API request structures.
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional


class WriteRequest(BaseModel):
    """
    Generic write request that can handle documents, code, or relationships.
    
    Attributes:
        content: Document/code content or serialized relationship data
        path: File path or identifier (optional)
        metadata: Any additional information about the content
    """
    content: str = Field(..., description="Document/code content or serialized relationship data")
    path: Optional[str] = Field(None, description="File path or identifier")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class QueryRequest(BaseModel):
    """
    Request to query the PathRAG system.
    
    Attributes:
        query: The natural language query to process
        max_results: Maximum number of results to return (default: 5)
    """
    query: str = Field(..., description="Natural language query")
    max_results: int = Field(5, description="Maximum number of results to return")