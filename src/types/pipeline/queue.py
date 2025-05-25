"""Queue type definitions for orchestration pipeline.

This module defines TypedDict and other types related to queue management,
backpressure control, and queue metrics in the pipeline system.
"""

from typing import Any, Dict, List, Literal, Optional, TypedDict, Union


class QueueBackpressureConfig(TypedDict, total=False):
    """Configuration for queue backpressure mechanisms."""
    
    backpressure_threshold: float
    """Threshold (0.0-1.0) at which to start applying backpressure."""
    
    backoff_strategy: Literal["linear", "exponential"]
    """Strategy for backoff when queue is full or backpressure is active."""
    
    backoff_base_seconds: float
    """Base backoff time in seconds."""
    
    backoff_max_seconds: float
    """Maximum backoff time in seconds."""
    
    memory_check_interval_sec: int
    """How often to check memory usage (seconds)."""


class QueueConfig(TypedDict, total=False):
    """Configuration for a queue in the pipeline system."""
    
    max_size: int
    """Maximum number of items in the queue."""
    
    max_memory_mb: int
    """Maximum estimated memory usage in MB."""
    
    backpressure: QueueBackpressureConfig
    """Backpressure configuration for the queue."""


class QueueMetrics(TypedDict, total=False):
    """Metrics for a queue in the pipeline system."""
    
    name: str
    """Queue identifier name."""
    
    current_size: int
    """Current number of items in the queue."""
    
    max_size: int
    """Maximum queue size."""
    
    enqueued: int
    """Total number of items added to the queue since creation."""
    
    dequeued: int
    """Total number of items removed from the queue since creation."""
    
    backpressure_events: int
    """Number of times backpressure has been applied."""
    
    estimated_memory_mb: float
    """Estimated memory usage in MB."""
    
    is_paused: bool
    """Whether the queue is currently paused due to backpressure."""
    
    last_put_time: float
    """Timestamp of last successful put operation."""
    
    last_get_time: float
    """Timestamp of last successful get operation."""
