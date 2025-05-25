"""
Queue management schemas for HADES-PathRAG pipeline orchestration.

This module defines schemas for task queues and message handling in the
distributed pipeline system.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional, Union

from pydantic import Field, field_validator, model_validator

from ..common.base import BaseSchema
from ..common.types import MetadataDict


class TaskPriority(IntEnum):
    """Priority levels for tasks in the queue."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class TaskStatus(str, Enum):
    """Status of a task in the queue."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELED = "canceled"


class TaskSchema(BaseSchema):
    """Schema for a task in the processing queue."""
    
    task_id: str = Field(..., description="Unique task identifier")
    task_type: str = Field(..., description="Type of task")
    priority: TaskPriority = Field(default=TaskPriority.NORMAL, description="Task priority")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Current task status")
    payload: Dict[str, Any] = Field(..., description="Task payload data")
    created_at: datetime = Field(default_factory=datetime.now, description="Task creation timestamp")
    started_at: Optional[datetime] = Field(default=None, description="Processing start timestamp")
    completed_at: Optional[datetime] = Field(default=None, description="Task completion timestamp")
    worker_id: Optional[str] = Field(default=None, description="ID of assigned worker")
    retry_count: int = Field(default=0, description="Number of retry attempts")
    max_retries: int = Field(default=3, description="Maximum number of retries")
    error: Optional[str] = Field(default=None, description="Error message if task failed")
    parent_task_id: Optional[str] = Field(default=None, description="ID of parent task if subtask")
    depends_on: List[str] = Field(default_factory=list, description="IDs of tasks this task depends on")
    metadata: MetadataDict = Field(default_factory=dict, description="Additional task metadata")
    
    @model_validator(mode='after')
    def validate_status_consistency(self) -> TaskSchema:
        """Ensure task status is consistent with timestamps."""
        if self.completed_at is not None and self.status not in (TaskStatus.COMPLETED, TaskStatus.FAILED):
            self.status = TaskStatus.COMPLETED
            
        if self.started_at is not None and self.status == TaskStatus.PENDING:
            self.status = TaskStatus.PROCESSING
            
        if self.error is not None and self.status != TaskStatus.FAILED:
            self.status = TaskStatus.FAILED
            
        return self


class QueueConfigSchema(BaseSchema):
    """Configuration for a task queue."""
    
    queue_name: str = Field(..., description="Name of the queue")
    max_size: int = Field(default=1000, description="Maximum queue size")
    dequeue_batch_size: int = Field(default=10, description="Batch size for dequeue operations")
    enable_priority: bool = Field(default=True, description="Whether to respect task priorities")
    enable_persistence: bool = Field(default=True, description="Whether to persist queue to disk")
    persistence_path: Optional[str] = Field(default=None, description="Path for queue persistence")
    enable_dead_letter: bool = Field(default=True, description="Whether to use dead letter queue")
    max_retry_delay_seconds: int = Field(default=300, description="Maximum delay for retries")
    metadata: MetadataDict = Field(default_factory=dict, description="Additional queue metadata")
    
    @field_validator("max_size", "dequeue_batch_size")
    @classmethod
    def validate_positive(cls, v: int) -> int:
        """Validate value is positive."""
        if v <= 0:
            raise ValueError(f"Value must be positive, got {v}")
        return v


class QueueStatsSchema(BaseSchema):
    """Statistics for a task queue."""
    
    queue_name: str = Field(..., description="Name of the queue")
    current_size: int = Field(default=0, description="Current number of tasks in queue")
    peak_size: int = Field(default=0, description="Peak queue size")
    total_enqueued: int = Field(default=0, description="Total tasks enqueued")
    total_dequeued: int = Field(default=0, description="Total tasks dequeued")
    total_completed: int = Field(default=0, description="Total tasks completed")
    total_failed: int = Field(default=0, description="Total tasks failed")
    total_retries: int = Field(default=0, description="Total retry attempts")
    average_wait_time_ms: float = Field(default=0.0, description="Average wait time in milliseconds")
    average_processing_time_ms: float = Field(default=0.0, description="Average processing time in milliseconds")
    tasks_by_priority: Dict[str, int] = Field(default_factory=dict, description="Task counts by priority")
    tasks_by_status: Dict[str, int] = Field(default_factory=dict, description="Task counts by status")
    
    def update_stats(self, task: TaskSchema, event_type: str) -> None:
        """Update queue statistics based on a task event.
        
        Args:
            task: The task involved in the event
            event_type: Type of event (e.g., 'enqueue', 'dequeue', 'complete', 'fail')
        """
        now = datetime.now()
        
        # Update counts
        if event_type == 'enqueue':
            self.total_enqueued += 1
            self.current_size += 1
            if self.current_size > self.peak_size:
                self.peak_size = self.current_size
                
            # Update tasks by priority - handle both enum objects and primitive values
            if hasattr(task.priority, 'name'):
                # If it's an enum object, use the name
                priority_key = str(task.priority.name)
            else:
                # If it's a primitive value (integer or string), use it directly
                priority_key = str(task.priority)
            self.tasks_by_priority[priority_key] = self.tasks_by_priority.get(priority_key, 0) + 1
            
            # Update tasks by status - handle both enum objects and string values
            if hasattr(task.status, 'value'):
                # If it's an enum object, use the value
                status_key = str(task.status.value)
            else:
                # If it's a string, use it directly
                status_key = str(task.status)
            self.tasks_by_status[status_key] = self.tasks_by_status.get(status_key, 0) + 1
            
        elif event_type == 'dequeue':
            self.total_dequeued += 1
            self.current_size = max(0, self.current_size - 1)
            
            # Calculate wait time
            if task.created_at and task.started_at:
                wait_time_ms = (task.started_at - task.created_at).total_seconds() * 1000
                
                # Update average wait time
                if self.total_dequeued == 1:
                    self.average_wait_time_ms = wait_time_ms
                else:
                    self.average_wait_time_ms = (
                        (self.average_wait_time_ms * (self.total_dequeued - 1) + wait_time_ms)
                        / self.total_dequeued
                    )
                    
            # Update tasks by status counts
            old_status = 'pending'
            new_status = 'processing'
            self.tasks_by_status[old_status] = max(0, self.tasks_by_status.get(old_status, 0) - 1)
            self.tasks_by_status[new_status] = self.tasks_by_status.get(new_status, 0) + 1
            
        elif event_type == 'complete':
            self.total_completed += 1
            
            # Calculate processing time
            if task.started_at and task.completed_at:
                processing_time_ms = (task.completed_at - task.started_at).total_seconds() * 1000
                
                # Update average processing time
                if self.total_completed == 1:
                    self.average_processing_time_ms = processing_time_ms
                else:
                    self.average_processing_time_ms = (
                        (self.average_processing_time_ms * (self.total_completed - 1) + processing_time_ms)
                        / self.total_completed
                    )
                    
            # Update tasks by status counts
            old_status = 'processing'
            new_status = 'completed'
            self.tasks_by_status[old_status] = max(0, self.tasks_by_status.get(old_status, 0) - 1)
            self.tasks_by_status[new_status] = self.tasks_by_status.get(new_status, 0) + 1
            
        elif event_type == 'fail':
            self.total_failed += 1
            
            # Update tasks by status counts
            old_status = 'processing'
            new_status = 'failed'
            self.tasks_by_status[old_status] = max(0, self.tasks_by_status.get(old_status, 0) - 1)
            self.tasks_by_status[new_status] = self.tasks_by_status.get(new_status, 0) + 1
            
        elif event_type == 'retry':
            self.total_retries += 1
            
            # Update tasks by status counts
            old_status = 'failed'
            new_status = 'retrying'
            self.tasks_by_status[old_status] = max(0, self.tasks_by_status.get(old_status, 0) - 1)
            self.tasks_by_status[new_status] = self.tasks_by_status.get(new_status, 0) + 1
