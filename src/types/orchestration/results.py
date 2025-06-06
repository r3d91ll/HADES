"""
Result type definitions for orchestration components.

This module provides structured result types for pipeline stages,
workers, and orchestration operations.
"""

from typing import Any, Dict, List, Optional, TypedDict, Union
from datetime import datetime
from enum import Enum


class ResultStatus(str, Enum):
    """Status of operation results."""
    
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    RETRYING = "retrying"


class ErrorSeverity(str, Enum):
    """Severity levels for errors."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# Base result types

class BaseResult(TypedDict, total=False):
    """Base result structure for all operations."""
    status: str  # ResultStatus enum value
    start_time: str  # ISO timestamp
    end_time: str  # ISO timestamp
    duration: float  # seconds
    message: str
    metadata: Dict[str, Any]


class ErrorInfo(TypedDict, total=False):
    """Error information structure."""
    error_type: str
    error_message: str
    error_code: Optional[str]
    severity: str  # ErrorSeverity enum value
    traceback: Optional[str]
    context: Dict[str, Any]
    recoverable: bool


class MetricsInfo(TypedDict, total=False):
    """Metrics information structure."""
    cpu_usage: float
    memory_usage: float
    gpu_usage: float
    disk_io: Dict[str, float]
    network_io: Dict[str, float]
    custom_metrics: Dict[str, float]


# Component result types

class WorkerResult(BaseResult, total=False):
    """Result from worker operations."""
    worker_id: str
    worker_type: str
    task_count: int
    tasks_completed: int
    tasks_failed: int
    average_task_time: float
    throughput: float  # tasks per second
    metrics: MetricsInfo
    errors: List[ErrorInfo]


class QueueResult(BaseResult, total=False):
    """Result from queue operations."""
    queue_id: str
    items_processed: int
    items_remaining: int
    average_wait_time: float
    peak_size: int
    throughput: float  # items per second
    metrics: MetricsInfo


class MonitoringResult(BaseResult, total=False):
    """Result from monitoring operations."""
    component_count: int
    alerts_generated: int
    metrics_collected: int
    health_checks: int
    system_metrics: MetricsInfo
    component_metrics: Dict[str, MetricsInfo]


# Pipeline result types

class StageResult(BaseResult, total=False):
    """Result from individual pipeline stages."""
    stage_name: str
    stage_type: str
    input_size: int
    output_size: int
    items_processed: int
    items_failed: int
    retry_count: int
    checkpoint_created: bool
    checkpoint_path: Optional[str]
    stage_metrics: MetricsInfo
    stage_errors: List[ErrorInfo]
    outputs: Dict[str, Any]


class PipelineResult(BaseResult, total=False):
    """Result from complete pipeline execution."""
    pipeline_id: str
    pipeline_type: str
    pipeline_name: str
    total_stages: int
    completed_stages: int
    failed_stages: int
    skipped_stages: int
    total_items: int
    processed_items: int
    failed_items: int
    stage_results: Dict[str, StageResult]
    final_output: Dict[str, Any]
    checkpoints: List[str]
    resource_usage: MetricsInfo
    errors: List[ErrorInfo]


# Specific pipeline result types

class TextPipelineResult(PipelineResult, total=False):
    """Result from text processing pipeline."""
    documents_processed: int
    chunks_generated: int
    embeddings_created: int
    total_tokens: int
    average_document_size: int
    processing_rate: float  # documents per second
    embedding_stats: Dict[str, float]


class ISNETrainingResult(PipelineResult, total=False):
    """Result from ISNE training pipeline."""
    model_path: str
    training_epochs: int
    validation_score: float
    training_loss: float
    validation_loss: float
    training_time: float
    model_size: int  # bytes
    convergence_epoch: Optional[int]
    best_metrics: Dict[str, float]
    learning_curves: Dict[str, List[float]]


class EmbeddingPipelineResult(PipelineResult, total=False):
    """Result from embedding generation pipeline."""
    embeddings_generated: int
    embedding_dimension: int
    model_used: str
    cache_hits: int
    cache_misses: int
    similarity_comparisons: int
    average_embedding_time: float
    embedding_quality_metrics: Dict[str, float]


# Batch and collection results

class BatchResult(BaseResult, total=False):
    """Result from batch operations."""
    batch_id: str
    batch_size: int
    items_processed: int
    items_failed: int
    batch_metrics: MetricsInfo
    item_results: List[Dict[str, Any]]
    partial_results: bool


class CollectionResult(BaseResult, total=False):
    """Result from collection of operations."""
    collection_id: str
    total_operations: int
    successful_operations: int
    failed_operations: int
    operation_results: List[BaseResult]
    aggregate_metrics: MetricsInfo
    summary_stats: Dict[str, float]


# Health and status results

class HealthCheckResult(BaseResult, total=False):
    """Result from health check operations."""
    component_name: str
    component_type: str
    is_healthy: bool
    response_time: float
    last_error: Optional[ErrorInfo]
    uptime: float
    version: str
    dependencies: Dict[str, bool]


class StatusResult(BaseResult, total=False):
    """Result from status check operations."""
    system_status: str
    component_statuses: Dict[str, str]
    active_pipelines: int
    queued_tasks: int
    system_load: float
    available_resources: Dict[str, float]
    alerts: List[Dict[str, Any]]


# Performance and analysis results

class PerformanceResult(BaseResult, total=False):
    """Result from performance analysis."""
    operation_type: str
    sample_size: int
    average_time: float
    median_time: float
    min_time: float
    max_time: float
    percentiles: Dict[str, float]  # e.g., {"p95": 1.2, "p99": 2.1}
    throughput: float
    error_rate: float
    bottlenecks: List[str]
    recommendations: List[str]


class BenchmarkResult(BaseResult, total=False):
    """Result from benchmark operations."""
    benchmark_name: str
    benchmark_version: str
    test_cases: int
    passed_tests: int
    failed_tests: int
    performance_scores: Dict[str, float]
    comparison_baseline: Optional[str]
    improvement_percentage: Optional[float]
    detailed_results: List[PerformanceResult]