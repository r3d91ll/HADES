"""
Configuration type definitions for orchestration components.

This module provides TypedDict definitions for various orchestration
component configurations.
"""

from typing import Any, Dict, List, Optional, TypedDict, Union
from pathlib import Path


# Base configuration types

class BaseConfig(TypedDict, total=False):
    """Base configuration for all orchestration components."""
    name: str
    enabled: bool
    timeout: float
    retry_count: int
    log_level: str


class ResourceConfig(TypedDict, total=False):
    """Resource allocation configuration."""
    cpu_count: int
    memory_limit: str  # e.g., "2GB", "512MB"
    gpu_count: int
    gpu_memory_limit: str
    disk_space: str
    network_bandwidth: str


# Worker configuration types

class WorkerConfig(BaseConfig, total=False):
    """Configuration for worker components."""
    worker_type: str  # WorkerType enum value
    max_tasks: int
    batch_size: int
    resource_limits: ResourceConfig
    warm_up_time: float
    cool_down_time: float


class WorkerPoolConfig(BaseConfig, total=False):
    """Configuration for worker pools."""
    min_workers: int
    max_workers: int
    scaling_factor: float
    idle_timeout: float
    worker_config: WorkerConfig


# Queue configuration types

class QueueConfig(BaseConfig, total=False):
    """Configuration for queue components."""
    max_size: int
    priority_levels: List[str]
    persistence: bool
    persistence_path: str
    batch_timeout: float


# Monitoring configuration types

class MonitoringConfig(BaseConfig, total=False):
    """Configuration for monitoring components."""
    metrics_interval: float
    history_size: int
    alert_thresholds: Dict[str, float]
    export_metrics: bool
    export_path: str
    dashboard_enabled: bool
    dashboard_port: int


# Pipeline configuration types

class PipelineConfig(BaseConfig, total=False):
    """Base configuration for pipelines."""
    pipeline_type: str  # PipelineType enum value
    input_path: Union[str, List[str]]
    output_path: str
    parallel: bool
    batch_size: int
    max_retries: int
    checkpoint_interval: int
    checkpoint_path: str


class TextPipelineConfig(PipelineConfig, total=False):
    """Configuration for text processing pipelines."""
    document_formats: List[str]
    chunk_size: int
    overlap_size: int
    embedding_model: str
    embedding_batch_size: int
    preprocessing: Dict[str, Any]
    postprocessing: Dict[str, Any]


class ISNETrainingConfig(PipelineConfig, total=False):
    """Configuration for ISNE training pipelines."""
    epochs: int
    learning_rate: float
    training_batch_size: int  # Renamed to avoid conflict with PipelineConfig.batch_size
    validation_split: float
    model_config: Dict[str, Any]
    training_data_path: str
    validation_data_path: str
    model_save_path: str
    early_stopping: bool
    patience: int


class EmbeddingPipelineConfig(PipelineConfig, total=False):
    """Configuration for embedding generation pipelines."""
    embedding_model: str
    embedding_dim: int
    normalize_embeddings: bool
    cache_embeddings: bool
    cache_path: str
    similarity_threshold: float


# Stage configuration types

class StageConfig(BaseConfig, total=False):
    """Configuration for individual pipeline stages."""
    stage_type: str
    dependencies: List[str]
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    resources: ResourceConfig
    error_handling: str  # "ignore", "retry", "fail"
    max_stage_retries: int


# Orchestration system configuration

class OrchestrationConfig(BaseConfig, total=False):
    """Main orchestration system configuration."""
    system_name: str
    data_path: str
    temp_path: str
    log_path: str
    
    # Component configurations
    monitoring: MonitoringConfig
    default_worker_pool: WorkerPoolConfig
    default_queue: QueueConfig
    
    # Pipeline configurations
    pipelines: Dict[str, PipelineConfig]
    stages: Dict[str, StageConfig]
    
    # Global settings
    max_concurrent_pipelines: int
    pipeline_timeout: float
    cleanup_interval: float
    health_check_interval: float


# Result type configurations

class ResultConfig(TypedDict, total=False):
    """Configuration for result handling."""
    store_intermediate: bool
    compress_results: bool
    result_format: str  # "json", "pickle", "parquet"
    result_path: str
    retention_days: int
    cleanup_on_success: bool


# Scheduling configuration

class ScheduleConfig(TypedDict, total=False):
    """Configuration for pipeline scheduling."""
    schedule_type: str  # "immediate", "delayed", "cron", "interval"
    schedule_expression: str  # cron expression or interval
    max_instances: int
    overlap_policy: str  # "allow", "skip", "queue"
    priority: str  # Priority enum value
    dependencies: List[str]
    conditions: Dict[str, Any]