"""
Orchestration type definitions for HADES.

This module provides centralized type definitions for the orchestration system,
including pipelines, workers, queues, monitors, tasks, and configurations.
"""

# Base types and protocols
from .base import (
    # Type aliases
    PathType,
    ConfigDict,
    MetricsDict,
    ResultDict,
    
    # Enums
    PipelineState,
    ComponentState,
    PipelineType,
    Priority,
    WorkerType,
    
    # Protocols
    MonitorProtocol,
    WorkerProtocol,
    QueueProtocol,
    PipelineProtocol,
    
    # Abstract base classes
    OrchestrationComponent,
    BasePipeline,
)

# Configuration types
from .config import (
    # Base configurations
    BaseConfig,
    ResourceConfig,
    
    # Component configurations
    WorkerConfig,
    WorkerPoolConfig,
    QueueConfig,
    MonitoringConfig,
    
    # Pipeline configurations
    PipelineConfig,
    TextPipelineConfig,
    ISNETrainingConfig,
    EmbeddingPipelineConfig,
    
    # Stage and system configurations
    StageConfig,
    OrchestrationConfig,
    ResultConfig,
    ScheduleConfig,
)

# Result types
from .results import (
    # Enums
    ResultStatus,
    ErrorSeverity,
    
    # Base result types
    BaseResult,
    ErrorInfo,
    MetricsInfo,
    
    # Component result types
    WorkerResult,
    QueueResult,
    MonitoringResult,
    
    # Pipeline result types
    StageResult,
    PipelineResult,
    TextPipelineResult,
    ISNETrainingResult,
    EmbeddingPipelineResult,
    
    # Collection and batch results
    BatchResult,
    CollectionResult,
    
    # Health and status results
    HealthCheckResult,
    StatusResult,
    
    # Performance results
    PerformanceResult,
    BenchmarkResult,
)

# Task types
from .tasks import (
    # Enums
    TaskStatus,
    TaskPriority,
    TaskType,
    
    # Basic task types
    TaskMetadata,
    TaskResult,
    BaseTask,
    
    # Specialized task types
    DocumentProcessingTask,
    EmbeddingTask,
    TrainingTask,
    AnalysisTask,
    
    # Task collections
    TaskBatch,
    TaskWorkflow,
    
    # Queue and scheduling
    TaskQueue,
    TaskSchedule,
    
    # Execution context
    ExecutionContext,
    WorkerAssignment,
    
    # Monitoring
    TaskMetrics,
    TaskAlert,
)

__all__ = [
    # Base types
    "PathType",
    "ConfigDict", 
    "MetricsDict",
    "ResultDict",
    
    # Base enums
    "PipelineState",
    "ComponentState",
    "PipelineType",
    "Priority",
    "WorkerType",
    
    # Protocols
    "MonitorProtocol",
    "WorkerProtocol",
    "QueueProtocol", 
    "PipelineProtocol",
    
    # Base classes
    "OrchestrationComponent",
    "BasePipeline",
    
    # Configuration types
    "BaseConfig",
    "ResourceConfig",
    "WorkerConfig",
    "WorkerPoolConfig",
    "QueueConfig",
    "MonitoringConfig",
    "PipelineConfig",
    "TextPipelineConfig",
    "ISNETrainingConfig",
    "EmbeddingPipelineConfig",
    "StageConfig",
    "OrchestrationConfig",
    "ResultConfig",
    "ScheduleConfig",
    
    # Result enums
    "ResultStatus",
    "ErrorSeverity",
    
    # Result types
    "BaseResult",
    "ErrorInfo",
    "MetricsInfo",
    "WorkerResult",
    "QueueResult",
    "MonitoringResult",
    "StageResult",
    "PipelineResult",
    "TextPipelineResult",
    "ISNETrainingResult",
    "EmbeddingPipelineResult",
    "BatchResult",
    "CollectionResult",
    "HealthCheckResult",
    "StatusResult",
    "PerformanceResult",
    "BenchmarkResult",
    
    # Task enums
    "TaskStatus",
    "TaskPriority",
    "TaskType",
    
    # Task types
    "TaskMetadata",
    "TaskResult",
    "BaseTask",
    "DocumentProcessingTask",
    "EmbeddingTask",
    "TrainingTask",
    "AnalysisTask",
    "TaskBatch",
    "TaskWorkflow",
    "TaskQueue",
    "TaskSchedule",
    "ExecutionContext",
    "WorkerAssignment",
    "TaskMetrics",
    "TaskAlert",
]