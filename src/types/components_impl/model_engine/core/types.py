"""
Core Model Engine Types

Type definitions specifically for the core model engine component.
These types are used by the CoreModelEngine implementation.
"""

from typing import Dict, Any, List, Optional, Union, Literal
from pydantic import BaseModel, Field, validator
from datetime import datetime, timezone

from ...common import BaseHADESModel


class CoreModelEngineConfig(BaseHADESModel):
    """Configuration for core model engine component."""
    
    # Engine selection
    engine_type: Literal["vllm", "haystack", "custom"] = Field(
        default="vllm",
        description="Type of underlying engine to use"
    )
    
    # Model configuration
    model_name: str = Field(
        default="BAAI/bge-large-en-v1.5",
        description="Model name or path"
    )
    model_revision: Optional[str] = Field(
        default=None,
        description="Model revision/version"
    )
    
    # Server configuration
    server_host: str = Field(
        default="localhost",
        description="Model server host"
    )
    server_port: int = Field(
        default=8000,
        ge=1024,
        le=65535,
        description="Model server port"
    )
    
    # Performance configuration
    max_retries: int = Field(
        default=3,
        ge=1,
        description="Maximum retry attempts"
    )
    timeout_seconds: int = Field(
        default=60,
        ge=1,
        description="Request timeout in seconds"
    )
    batch_size: int = Field(
        default=32,
        ge=1,
        description="Default batch size for inference"
    )
    
    # Resource limits
    max_concurrent_requests: int = Field(
        default=100,
        ge=1,
        description="Maximum concurrent requests"
    )
    gpu_memory_utilization: float = Field(
        default=0.8,
        gt=0,
        le=1,
        description="GPU memory utilization ratio"
    )
    
    # Engine-specific configuration
    engine_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Engine-specific configuration parameters"
    )
    
    @validator('engine_config')
    def validate_engine_config(cls, v: Any) -> Dict[str, Any]:
        """Validate engine-specific configuration."""
        if not isinstance(v, dict):
            raise ValueError("engine_config must be a dictionary")
        return v


class CoreModelEngineState(BaseHADESModel):
    """State information for core model engine component."""
    
    # Engine state
    engine_initialized: bool = Field(
        default=False,
        description="Whether the underlying engine is initialized"
    )
    server_running: bool = Field(
        default=False,
        description="Whether the model server is running"
    )
    last_health_check: Optional[datetime] = Field(
        default=None,
        description="Timestamp of last health check"
    )
    
    # Performance metrics
    total_requests: int = Field(
        default=0,
        ge=0,
        description="Total number of requests processed"
    )
    successful_requests: int = Field(
        default=0,
        ge=0,
        description="Number of successful requests"
    )
    failed_requests: int = Field(
        default=0,
        ge=0,
        description="Number of failed requests"
    )
    
    # Timing metrics
    average_processing_time: float = Field(
        default=0.0,
        ge=0,
        description="Average processing time in seconds"
    )
    last_request_time: Optional[datetime] = Field(
        default=None,
        description="Timestamp of last request"
    )
    
    # Resource usage
    memory_usage_mb: float = Field(
        default=0.0,
        ge=0,
        description="Current memory usage in MB"
    )
    gpu_memory_usage_mb: float = Field(
        default=0.0,
        ge=0,
        description="Current GPU memory usage in MB"
    )
    
    @validator('successful_requests', 'failed_requests')
    def validate_request_counts(cls, v: int, values: Dict[str, Any]) -> int:
        """Ensure request counts are consistent."""
        total = values.get('total_requests', 0)
        if v > total:
            raise ValueError("Request counts cannot exceed total requests")
        return v


class CoreModelEngineMetrics(BaseHADESModel):
    """Performance metrics for core model engine component."""
    
    # Processing metrics
    requests_per_second: float = Field(
        default=0.0,
        ge=0,
        description="Current requests per second"
    )
    latency_p50: float = Field(
        default=0.0,
        ge=0,
        description="50th percentile latency in milliseconds"
    )
    latency_p95: float = Field(
        default=0.0,
        ge=0,
        description="95th percentile latency in milliseconds"
    )
    latency_p99: float = Field(
        default=0.0,
        ge=0,
        description="99th percentile latency in milliseconds"
    )
    
    # Error metrics
    error_rate: float = Field(
        default=0.0,
        ge=0,
        le=1,
        description="Current error rate (0.0 to 1.0)"
    )
    timeout_rate: float = Field(
        default=0.0,
        ge=0,
        le=1,
        description="Current timeout rate (0.0 to 1.0)"
    )
    
    # Resource metrics
    cpu_utilization: float = Field(
        default=0.0,
        ge=0,
        le=1,
        description="CPU utilization (0.0 to 1.0)"
    )
    memory_utilization: float = Field(
        default=0.0,
        ge=0,
        le=1,
        description="Memory utilization (0.0 to 1.0)"
    )
    gpu_utilization: float = Field(
        default=0.0,
        ge=0,
        le=1,
        description="GPU utilization (0.0 to 1.0)"
    )
    
    # Queue metrics
    queue_size: int = Field(
        default=0,
        ge=0,
        description="Current queue size"
    )
    queue_wait_time: float = Field(
        default=0.0,
        ge=0,
        description="Average queue wait time in seconds"
    )


class CoreModelEngineRequest(BaseHADESModel):
    """Request format for core model engine."""
    
    # Request identification
    request_id: str = Field(
        description="Unique request identifier"
    )
    request_type: Literal["generate", "embed", "chat", "complete"] = Field(
        description="Type of request"
    )
    
    # Input data
    input_text: str = Field(
        description="Input text for processing"
    )
    input_tokens: Optional[List[int]] = Field(
        default=None,
        description="Pre-tokenized input (optional)"
    )
    
    # Parameters
    max_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum tokens to generate"
    )
    temperature: Optional[float] = Field(
        default=None,
        ge=0,
        le=2,
        description="Sampling temperature"
    )
    top_p: Optional[float] = Field(
        default=None,
        gt=0,
        le=1,
        description="Top-p sampling parameter"
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        description="Top-k sampling parameter"
    )
    
    # Control parameters
    stop_sequences: List[str] = Field(
        default_factory=list,
        description="Stop sequences for generation"
    )
    stream: bool = Field(
        default=False,
        description="Whether to stream the response"
    )
    
    # Metadata
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional request metadata"
    )


class CoreModelEngineResponse(BaseHADESModel):
    """Response format for core model engine."""
    
    # Request identification
    request_id: str = Field(
        description="Original request identifier"
    )
    
    # Response data
    output_text: Optional[str] = Field(
        default=None,
        description="Generated text output"
    )
    output_tokens: Optional[List[int]] = Field(
        default=None,
        description="Generated tokens"
    )
    embeddings: Optional[List[float]] = Field(
        default=None,
        description="Generated embeddings"
    )
    
    # Generation metadata
    finish_reason: Optional[str] = Field(
        default=None,
        description="Reason generation finished"
    )
    token_count: int = Field(
        default=0,
        ge=0,
        description="Number of tokens generated"
    )
    
    # Performance data
    processing_time: float = Field(
        ge=0,
        description="Processing time in seconds"
    )
    queue_time: float = Field(
        default=0.0,
        ge=0,
        description="Time spent in queue in seconds"
    )
    
    # Error handling
    error: Optional[str] = Field(
        default=None,
        description="Error message if request failed"
    )
    warning: Optional[str] = Field(
        default=None,
        description="Warning message if applicable"
    )
    
    # Additional metadata
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional response metadata"
    )


# Type aliases for convenience
CoreEngineConfig = CoreModelEngineConfig
CoreEngineState = CoreModelEngineState
CoreEngineMetrics = CoreModelEngineMetrics
CoreEngineRequest = CoreModelEngineRequest
CoreEngineResponse = CoreModelEngineResponse