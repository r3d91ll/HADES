"""
Helper functions and classes for pipeline schema tests.

This module provides mock implementations and helper utilities for testing 
pipeline schemas without depending on actual implementation details.
"""

from datetime import datetime
from enum import Enum, IntEnum
from typing import Dict, Any, Optional

class MockTaskPriority(IntEnum):
    """Mock priority levels for tasks in the queue."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class MockTaskStatus(str, Enum):
    """Mock status of a task in the queue."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELED = "canceled"


class MockPipelineStage(str, Enum):
    """Mock pipeline stages."""
    LOAD = "load"
    PREPROCESS = "preprocess"
    CHUNK = "chunk"
    EMBED = "embed"
    INDEX = "index"
    STORE = "store"
    COMPLETE = "complete"
    FAILED = "failed"


class MockJobStatus(str, Enum):
    """Mock status values for pipeline jobs."""
    PENDING = "pending"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    RETRY = "retry"


class MockTaskSchema:
    """Mock schema for a task in the processing queue."""
    
    def __init__(
        self,
        task_id: str,
        task_type: str,
        priority: MockTaskPriority = MockTaskPriority.NORMAL,
        status: MockTaskStatus = MockTaskStatus.PENDING,
        payload: Dict[str, Any] = None,
        created_at: datetime = None,
        started_at: Optional[datetime] = None,
        completed_at: Optional[datetime] = None,
        worker_id: Optional[str] = None,
        retry_count: int = 0,
        max_retries: int = 3,
        error: Optional[str] = None,
        parent_task_id: Optional[str] = None,
        depends_on: list = None,
        metadata: Dict[str, Any] = None
    ):
        self.task_id = task_id
        self.task_type = task_type
        self.priority = priority
        self.status = status
        self.payload = payload or {}
        self.created_at = created_at or datetime.now()
        self.started_at = started_at
        self.completed_at = completed_at
        self.worker_id = worker_id
        self.retry_count = retry_count
        self.max_retries = max_retries
        self.error = error
        self.parent_task_id = parent_task_id
        self.depends_on = depends_on or []
        self.metadata = metadata or {}
