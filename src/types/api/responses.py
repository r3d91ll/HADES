"""
API response models for HADES.

This module contains Pydantic models for API response structures.
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List

from .models import QueryResult


class QueryResponse(BaseModel):
    """
    Response to a PathRAG query.
    
    Attributes:
        results: List of query results
        execution_time_ms: Time taken to execute the query in milliseconds
    """
    results: List[QueryResult]
    execution_time_ms: float


class WriteResponse(BaseModel):
    """
    Response to a write operation.
    
    Attributes:
        status: Operation status (success/error)
        id: Identifier of the written/updated entity
        message: Additional information about the operation
    """
    status: str
    id: str
    message: Optional[str] = None


class StatusResponse(BaseModel):
    """
    Response to a status check.
    
    Attributes:
        status: Current system status (online/offline/degraded)
        document_count: Number of documents in the database
        version: API version number
    """
    status: str
    document_count: int
    version: str