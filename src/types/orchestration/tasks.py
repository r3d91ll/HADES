"""
Task type definitions for orchestration components.

This module provides task-related types for queuing, scheduling,
and execution within the orchestration system.
"""

from typing import Any, Callable, Dict, List, Optional, TypedDict, Union
from datetime import datetime
from enum import Enum


class TaskStatus(str, Enum):
    """Status of tasks in the system."""
    
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"
    PAUSED = "paused"


class TaskPriority(str, Enum):
    """Priority levels for tasks."""
    
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class TaskType(str, Enum):
    """Types of tasks in the system."""
    
    DOCUMENT_PROCESSING = "document_processing"
    EMBEDDING_GENERATION = "embedding_generation"
    MODEL_TRAINING = "model_training"
    DATA_INGESTION = "data_ingestion"
    ANALYSIS = "analysis"
    CLEANUP = "cleanup"
    HEALTH_CHECK = "health_check"
    CUSTOM = "custom"


# Base task types

class TaskMetadata(TypedDict, total=False):
    """Metadata for tasks."""
    created_by: str
    created_at: str  # ISO timestamp
    tags: List[str]
    category: str
    description: str
    estimated_duration: float
    resource_requirements: Dict[str, Any]
    dependencies: List[str]


class TaskResult(TypedDict, total=False):
    """Result of task execution."""
    task_id: str
    status: str  # TaskStatus enum value
    start_time: str  # ISO timestamp
    end_time: str  # ISO timestamp
    duration: float
    output: Any
    error: Optional[str]
    metrics: Dict[str, float]
    logs: List[str]


class BaseTask(TypedDict, total=False):
    """Base task definition."""
    task_id: str
    task_type: str  # TaskType enum value
    priority: str  # TaskPriority enum value
    status: str  # TaskStatus enum value
    
    # Task content
    function_name: str
    args: List[Any]
    kwargs: Dict[str, Any]
    
    # Scheduling
    scheduled_time: Optional[str]  # ISO timestamp
    deadline: Optional[str]  # ISO timestamp
    timeout: Optional[float]
    
    # Execution control
    max_retries: int
    retry_count: int
    retry_delay: float
    
    # Metadata
    metadata: TaskMetadata
    result: Optional[TaskResult]


# Specialized task types

class DocumentProcessingTask(BaseTask, total=False):
    """Task for document processing operations."""
    document_path: str
    output_path: str
    processing_config: Dict[str, Any]
    chunk_size: int
    overlap_size: int
    format_type: str


class EmbeddingTask(BaseTask, total=False):
    """Task for embedding generation."""
    input_texts: List[str]
    model_name: str
    embedding_config: Dict[str, Any]
    batch_size: int
    normalize: bool
    cache_key: Optional[str]


class TrainingTask(BaseTask, total=False):
    """Task for model training operations."""
    model_type: str
    training_data_path: str
    validation_data_path: str
    model_config: Dict[str, Any]
    training_config: Dict[str, Any]
    checkpoint_path: str
    epochs: int


class AnalysisTask(BaseTask, total=False):
    """Task for analysis operations."""
    analysis_type: str
    input_data: Union[str, Dict[str, Any]]
    analysis_config: Dict[str, Any]
    output_format: str


# Task collections and workflows

class TaskBatch(TypedDict, total=False):
    """Collection of related tasks."""
    batch_id: str
    batch_name: str
    tasks: List[BaseTask]
    batch_priority: str  # TaskPriority enum value
    parallel_execution: bool
    max_concurrent: int
    batch_metadata: TaskMetadata
    progress: Dict[str, int]  # status counts


class TaskWorkflow(TypedDict, total=False):
    """Workflow definition with task dependencies."""
    workflow_id: str
    workflow_name: str
    workflow_version: str
    
    # Workflow structure
    stages: List[Dict[str, Any]]
    dependencies: Dict[str, List[str]]  # task_id -> [dependency_task_ids]
    
    # Execution settings
    parallel_stages: bool
    fail_fast: bool
    retry_strategy: str
    
    # Workflow metadata
    metadata: TaskMetadata
    execution_history: List[Dict[str, Any]]


# Queue and scheduling types

class TaskQueue(TypedDict, total=False):
    """Task queue definition."""
    queue_id: str
    queue_name: str
    queue_type: str  # "fifo", "priority", "delay"
    max_size: int
    current_size: int
    
    # Queue behavior
    priority_levels: List[str]
    allow_duplicates: bool
    persistence: bool
    
    # Statistics
    total_enqueued: int
    total_dequeued: int
    average_wait_time: float
    peak_size: int


class TaskSchedule(TypedDict, total=False):
    """Task scheduling configuration."""
    schedule_id: str
    schedule_name: str
    schedule_type: str  # "one_time", "recurring", "cron"
    
    # Schedule definition
    cron_expression: Optional[str]
    interval_seconds: Optional[float]
    start_time: Optional[str]  # ISO timestamp
    end_time: Optional[str]  # ISO timestamp
    
    # Task template
    task_template: BaseTask
    
    # Execution control
    max_instances: int
    overlap_policy: str  # "allow", "skip", "queue"
    timezone: str
    
    # Status
    enabled: bool
    next_run: Optional[str]  # ISO timestamp
    last_run: Optional[str]  # ISO timestamp
    run_count: int


# Execution context types

class ExecutionContext(TypedDict, total=False):
    """Context for task execution."""
    execution_id: str
    worker_id: str
    node_id: str
    
    # Environment
    environment: Dict[str, str]
    working_directory: str
    temp_directory: str
    
    # Resources
    allocated_cpu: float
    allocated_memory: int  # bytes
    allocated_gpu: Optional[int]
    
    # Monitoring
    start_time: str  # ISO timestamp
    heartbeat_interval: float
    last_heartbeat: str  # ISO timestamp
    
    # Logging
    log_level: str
    log_file: Optional[str]
    capture_stdout: bool
    capture_stderr: bool


class WorkerAssignment(TypedDict, total=False):
    """Assignment of task to worker."""
    assignment_id: str
    task_id: str
    worker_id: str
    queue_id: str
    
    # Assignment timing
    assigned_at: str  # ISO timestamp
    started_at: Optional[str]  # ISO timestamp
    
    # Worker info
    worker_type: str
    worker_capabilities: List[str]
    worker_load: float
    
    # Assignment metadata
    assignment_reason: str
    estimated_completion: Optional[str]  # ISO timestamp
    actual_completion: Optional[str]  # ISO timestamp


# Performance and monitoring types

class TaskMetrics(TypedDict, total=False):
    """Performance metrics for tasks."""
    task_id: str
    
    # Timing metrics
    queue_time: float
    execution_time: float
    total_time: float
    
    # Resource metrics
    cpu_usage: float
    memory_usage: int  # bytes
    disk_io: Dict[str, float]
    network_io: Dict[str, float]
    
    # Task-specific metrics
    input_size: int
    output_size: int
    items_processed: int
    error_count: int
    
    # Custom metrics
    custom_metrics: Dict[str, float]


class TaskAlert(TypedDict, total=False):
    """Alert related to task execution."""
    alert_id: str
    task_id: str
    alert_type: str  # "timeout", "error", "resource", "custom"
    severity: str  # "low", "medium", "high", "critical"
    
    # Alert content
    title: str
    message: str
    details: Dict[str, Any]
    
    # Alert timing
    triggered_at: str  # ISO timestamp
    acknowledged_at: Optional[str]  # ISO timestamp
    resolved_at: Optional[str]  # ISO timestamp
    
    # Actions
    auto_resolve: bool
    actions_taken: List[str]
    escalation_level: int