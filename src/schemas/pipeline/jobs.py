"""
Job management schemas for pipeline processing in HADES-PathRAG.

This module defines schema models for job execution, tracking, and batch processing
within the pipeline system.
"""

from typing import Dict, List, Optional, Any, Union
from enum import Enum
from datetime import datetime
import uuid

from pydantic import Field, field_validator, model_validator

from src.schemas.common.base import BaseSchema


class JobStatus(str, Enum):
    """Status values for pipeline jobs."""
    PENDING = "pending"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    RETRY = "retry"


class JobSchema(BaseSchema):
    """Schema for pipeline job definitions."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    pipeline: str
    status: JobStatus = JobStatus.PENDING
    params: Dict[str, Any] = Field(default_factory=dict)
    priority: int = 0
    created_at: Union[str, datetime] = Field(default_factory=datetime.now)
    started_at: Optional[Union[str, datetime]] = None
    completed_at: Optional[Union[str, datetime]] = None
    scheduled_at: Optional[Union[str, datetime]] = None
    max_retries: int = 0
    retry_count: int = 0
    timeout: Optional[int] = None
    owner: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator('priority', 'max_retries', 'retry_count', 'timeout')
    def validate_positive_int(cls, v, info):
        """Validate positive integer values."""
        if v is not None and info.field_name != 'priority' and v < 0:
            raise ValueError(f"{info.field_name} must be a non-negative integer")
        return v


class JobResultSchema(BaseSchema):
    """Schema for pipeline job execution results."""
    
    job_id: str
    status: JobStatus
    output: Optional[Dict[str, Any]] = Field(default_factory=dict)  # Default to empty dict to avoid None
    error: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None
    completed_at: Union[str, datetime] = Field(default_factory=datetime.now)
    duration: Optional[float] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    
    @model_validator(mode='before')
    @classmethod
    def validate_result(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate job result based on status. Using 'before' mode to avoid recursion."""
        # Get status, which could be a string or enum
        status = data.get('status')
        status_value = status.value if hasattr(status, 'value') else status
        
        # Verify failed jobs have an error message
        if status_value == "failed" and not data.get('error'):
            raise ValueError("Error message is required for failed jobs")
        
        # For completed jobs, ensure output is at least an empty dict
        if status_value == "completed" and data.get('output') is None:
            data['output'] = {}
            
        return data


class BatchJobSchema(BaseSchema):
    """Schema for batch job processing."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    jobs: List[JobSchema]
    strategy: str = "parallel"  # parallel, sequential, dependency
    max_concurrent: int = 4
    priority: int = 0
    owner: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: Union[str, datetime] = Field(default_factory=datetime.now)
    timeout: Optional[int] = None
    retries_enabled: bool = False
    
    @field_validator('strategy')
    def validate_strategy(cls, v):
        """Validate batch execution strategy."""
        valid_strategies = ["parallel", "sequential", "dependency"]
        if v not in valid_strategies:
            raise ValueError(f"Invalid strategy: {v}. Must be one of {valid_strategies}")
        return v
    
    @field_validator('max_concurrent')
    def validate_max_concurrent(cls, v):
        """Validate max_concurrent value."""
        if v <= 0:
            raise ValueError("max_concurrent must be a positive integer")
        return v
    
    @model_validator(mode='after')
    def validate_jobs(self):
        """Validate batch job configuration."""
        if not self.jobs:
            raise ValueError("Batch job must contain at least one job")
            
        if self.strategy == "dependency":
            # TODO: Implement dependency validation
            pass
            
        return self
