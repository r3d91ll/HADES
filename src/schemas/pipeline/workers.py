"""
Worker schemas for HADES-PathRAG pipeline orchestration.

This module defines schemas for configuring and tracking workers
that execute pipeline tasks in parallel.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import Field, field_validator

from ..common.base import BaseSchema
from ..common.types import MetadataDict


class WorkerStatus(str, Enum):
    """Status of a worker."""
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    TERMINATED = "terminated"


class WorkerType(str, Enum):
    """Types of workers in the system."""
    PROCESSOR = "processor"
    LOADER = "loader"
    CHUNKER = "chunker"
    EMBEDDER = "embedder"
    INDEXER = "indexer"
    STORAGE = "storage"
    GENERIC = "generic"


class WorkerConfigSchema(BaseSchema):
    """Configuration for a worker instance."""
    
    worker_type: WorkerType = Field(..., description="Type of worker")
    worker_id: Optional[str] = Field(default=None, description="Unique worker identifier")
    max_batch_size: int = Field(default=10, description="Maximum batch size for this worker")
    max_retries: int = Field(default=3, description="Maximum number of retries")
    timeout_seconds: int = Field(default=300, description="Timeout in seconds")
    priority: int = Field(default=0, description="Worker priority (higher is more important)")
    resource_limits: Dict[str, Any] = Field(default_factory=dict, description="Resource limits")
    metadata: MetadataDict = Field(default_factory=dict, description="Additional worker metadata")
    
    @field_validator("max_batch_size", "max_retries", "timeout_seconds")
    @classmethod
    def validate_positive(cls, v: int) -> int:
        """Validate value is positive."""
        if v <= 0:
            raise ValueError(f"Value must be positive, got {v}")
        return v


class WorkerStatsSchema(BaseSchema):
    """Statistics for a worker instance."""
    
    worker_id: str = Field(..., description="Unique worker identifier")
    worker_type: WorkerType = Field(..., description="Type of worker")
    status: WorkerStatus = Field(default=WorkerStatus.IDLE, description="Current worker status")
    start_time: datetime = Field(default_factory=datetime.now, description="Worker start time")
    last_active: datetime = Field(default_factory=datetime.now, description="Last active timestamp")
    tasks_processed: int = Field(default=0, description="Number of tasks processed")
    tasks_failed: int = Field(default=0, description="Number of tasks failed")
    avg_processing_time_ms: float = Field(default=0.0, description="Average processing time in milliseconds")
    current_task: Optional[str] = Field(default=None, description="ID of current task")
    errors: List[Dict[str, Any]] = Field(default_factory=list, description="List of errors encountered")
    
    def update_status(self, status: WorkerStatus) -> None:
        """Update worker status and last active timestamp.
        
        Args:
            status: New worker status
        """
        self.status = status
        self.last_active = datetime.now()
    
    def record_task_processed(self, processing_time_ms: float) -> None:
        """Record a successfully processed task.
        
        Args:
            processing_time_ms: Processing time in milliseconds
        """
        self.tasks_processed += 1
        self.last_active = datetime.now()
        
        # Update average processing time
        if self.tasks_processed == 1:
            self.avg_processing_time_ms = processing_time_ms
        else:
            self.avg_processing_time_ms = (
                (self.avg_processing_time_ms * (self.tasks_processed - 1) + processing_time_ms)
                / self.tasks_processed
            )
    
    def record_error(self, error: str, task_id: Optional[str] = None) -> None:
        """Record an error encountered by the worker.
        
        Args:
            error: Error message
            task_id: Optional ID of task that caused the error
        """
        error_entry = {
            "error": error,
            "timestamp": datetime.now().isoformat()
        }
        if task_id:
            error_entry["task_id"] = task_id
            
        self.errors.append(error_entry)
        self.tasks_failed += 1
        self.update_status(WorkerStatus.ERROR)


class WorkerPoolConfigSchema(BaseSchema):
    """Configuration for a worker pool."""
    
    min_workers: int = Field(default=1, description="Minimum number of workers")
    max_workers: int = Field(default=4, description="Maximum number of workers")
    worker_configs: Dict[WorkerType, WorkerConfigSchema] = Field(
        default_factory=dict, description="Worker configurations by type"
    )
    scaling_enabled: bool = Field(default=True, description="Whether to enable auto-scaling")
    scaling_cooldown_seconds: int = Field(default=60, description="Cooldown period between scaling events")
    task_queue_size: int = Field(default=100, description="Maximum size of the task queue")
    
    @field_validator("min_workers", "max_workers")
    @classmethod
    def validate_worker_count(cls, v: int) -> int:
        """Validate worker count is positive."""
        if v <= 0:
            raise ValueError(f"Worker count must be positive, got {v}")
        return v
    
    @field_validator("scaling_cooldown_seconds")
    @classmethod
    def validate_cooldown(cls, v: int) -> int:
        """Validate cooldown period is non-negative."""
        if v < 0:
            raise ValueError(f"Cooldown period must be non-negative, got {v}")
        return v
