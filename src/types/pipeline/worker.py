"""Worker pool type definitions for orchestration pipeline.

This module defines TypedDict and other types related to worker management,
task execution, and worker metrics in the pipeline system.
"""

from typing import Any, Dict, List, Literal, Optional, TypedDict, Union
from enum import Enum


class WorkerStatus(str, Enum):
    """Status of a worker in the pipeline system."""
    
    IDLE = "idle"
    """Worker is idle and available for tasks."""
    
    BUSY = "busy"
    """Worker is currently processing a task."""
    
    STOPPED = "stopped"
    """Worker has been stopped and is no longer accepting tasks."""
    
    ERROR = "error"
    """Worker is in an error state."""


class WorkerConfig(TypedDict, total=False):
    """Configuration for worker pools in the pipeline system."""
    
    count: int
    """Number of workers in the pool."""
    
    timeout_seconds: int
    """Maximum time a task can run before timing out."""
    
    retry_count: int
    """Number of times to retry a failed task."""
    
    retry_delay_seconds: int
    """Delay between retry attempts."""


class WorkerTaskResult(TypedDict, total=False):
    """Result of a task executed by a worker."""
    
    task_id: str
    """Unique identifier for the task."""
    
    status: Literal["success", "error", "timeout", "cancelled"]
    """Status of the task execution."""
    
    result: Any
    """Result data from the task (if successful)."""
    
    error: Optional[str]
    """Error message (if status is 'error')."""
    
    execution_time: float
    """Time taken to execute the task in seconds."""
    
    retries: int
    """Number of retry attempts made for this task."""


class WorkerPoolMetrics(TypedDict, total=False):
    """Metrics for a worker pool in the pipeline system."""
    
    name: str
    """Worker pool identifier name."""
    
    max_workers: int
    """Maximum number of workers in the pool."""
    
    active_workers: int
    """Number of currently active workers."""
    
    active_tasks: int
    """Number of tasks currently being processed."""
    
    completed_tasks: int
    """Total number of successfully completed tasks."""
    
    failed_tasks: int
    """Total number of failed tasks."""
    
    avg_execution_time: float
    """Average execution time in seconds."""
    
    uptime_seconds: float
    """Time since the worker pool was created."""
