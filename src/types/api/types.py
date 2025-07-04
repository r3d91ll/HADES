"""API type definitions."""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from datetime import datetime

from ..common import BaseSchema


class APIRequest(BaseSchema):
    """Base API request type."""
    request_id: Optional[str] = Field(None, description="Unique request identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Request timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Request metadata")


class APIResponse(BaseSchema):
    """Standard API response format."""
    success: bool = Field(..., description="Whether the request was successful")
    data: Optional[Dict[str, Any]] = Field(None, description="Response data")
    error: Optional[str] = Field(None, description="Error message if failed")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    request_id: Optional[str] = Field(None, description="Original request ID")
    

class APIError(Exception):
    """Base exception for API errors."""
    pass


class ProcessDirectoryRequest(APIRequest):
    """Request to process a directory."""
    directory_path: str = Field(..., description="Path to directory to process")
    recursive: bool = Field(True, description="Process subdirectories recursively")
    file_extensions: Optional[List[str]] = Field(None, description="File extensions to process")
    options: Dict[str, Any] = Field(default_factory=dict, description="Processing options")


class ProcessFileRequest(APIRequest):
    """Request to process a single file."""
    file_path: str = Field(..., description="Path to file to process")
    options: Dict[str, Any] = Field(default_factory=dict, description="Processing options")


class ProcessTextRequest(APIRequest):
    """Request to process raw text."""
    text: str = Field(..., description="Text to process")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Text metadata")
    options: Dict[str, Any] = Field(default_factory=dict, description="Processing options")


class QueryRequest(APIRequest):
    """Request to query the knowledge graph."""
    query: str = Field(..., description="Query text")
    mode: Optional[str] = Field("hybrid", description="Query mode")
    limit: int = Field(10, description="Maximum results", ge=1, le=100)
    filters: Dict[str, Any] = Field(default_factory=dict, description="Query filters")
    options: Dict[str, Any] = Field(default_factory=dict, description="Query options")