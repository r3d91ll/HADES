"""
Pydantic models for the HADES-PathRAG API.
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List


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
    metadata: Dict[str, Any] = Field(default_factory=dict)


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
