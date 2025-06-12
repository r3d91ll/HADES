"""
vLLM Model Engine Types

Type definitions specifically for the vLLM model engine component.
These types are used by the vLLMModelEngine implementation.
"""

from typing import Dict, Any, List, Optional, Union, Literal
from pydantic import BaseModel, Field, validator
from datetime import datetime

from ...common import BaseHADESModel


class vLLMServerConfig(BaseHADESModel):
    """Configuration for vLLM server."""
    
    # Model configuration
    model_name: str = Field(description="Model name or path")
    model_revision: Optional[str] = Field(default=None, description="Model revision/version")
    tokenizer_name: Optional[str] = Field(default=None, description="Tokenizer name if different from model")
    
    # Server configuration
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, ge=1024, le=65535, description="Server port")
    uvicorn_log_level: str = Field(default="info", description="Uvicorn log level")
    
    # Hardware configuration
    tensor_parallel_size: int = Field(default=1, ge=1, description="Tensor parallel size")
    pipeline_parallel_size: int = Field(default=1, ge=1, description="Pipeline parallel size")
    max_parallel_loading_workers: Optional[int] = Field(default=None, description="Max parallel loading workers")
    
    # Memory configuration
    gpu_memory_utilization: float = Field(default=0.9, gt=0, le=1, description="GPU memory utilization")
    max_model_len: Optional[int] = Field(default=None, ge=1, description="Maximum model length")
    block_size: int = Field(default=16, ge=1, description="Block size for attention")
    seed: int = Field(default=0, description="Random seed")
    
    # Performance configuration
    swap_space: int = Field(default=4, ge=0, description="Swap space in GB")
    disable_log_stats: bool = Field(default=False, description="Disable logging stats")
    max_num_seqs: int = Field(default=256, ge=1, description="Maximum number of sequences")
    max_num_batched_tokens: Optional[int] = Field(default=None, description="Max batched tokens")
    
    # API configuration
    api_key: Optional[str] = Field(default=None, description="API key for authentication")
    served_model_name: Optional[str] = Field(default=None, description="Served model name")
    response_role: str = Field(default="assistant", description="Response role")
    
    # Advanced configuration
    trust_remote_code: bool = Field(default=False, description="Trust remote code")
    download_dir: Optional[str] = Field(default=None, description="Download directory")
    load_format: str = Field(default="auto", description="Load format")
    dtype: str = Field(default="auto", description="Data type")
    kv_cache_dtype: str = Field(default="auto", description="KV cache data type")
    quantization_param_path: Optional[str] = Field(default=None, description="Quantization param path")
    
    @validator('gpu_memory_utilization')
    def validate_gpu_memory(cls, v: float) -> float:
        if v <= 0 or v > 1:
            raise ValueError("GPU memory utilization must be between 0 and 1")
        return v


class vLLMRequestConfig(BaseHADESModel):
    """Configuration for vLLM inference requests."""
    
    # Sampling parameters
    temperature: float = Field(default=1.0, ge=0, description="Sampling temperature")
    top_p: float = Field(default=1.0, ge=0, le=1, description="Top-p sampling")
    top_k: int = Field(default=-1, description="Top-k sampling")
    frequency_penalty: float = Field(default=0.0, description="Frequency penalty")
    presence_penalty: float = Field(default=0.0, description="Presence penalty")
    repetition_penalty: float = Field(default=1.0, description="Repetition penalty")
    
    # Generation parameters
    max_tokens: Optional[int] = Field(default=None, ge=1, description="Maximum tokens to generate")
    min_tokens: int = Field(default=0, ge=0, description="Minimum tokens to generate")
    stop: Optional[Union[str, List[str]]] = Field(default=None, description="Stop sequences")
    stop_token_ids: Optional[List[int]] = Field(default=None, description="Stop token IDs")
    
    # Advanced parameters
    logprobs: Optional[int] = Field(default=None, ge=0, description="Number of log probabilities")
    prompt_logprobs: Optional[int] = Field(default=None, ge=0, description="Prompt log probabilities")
    skip_special_tokens: bool = Field(default=True, description="Skip special tokens in output")
    spaces_between_special_tokens: bool = Field(default=True, description="Spaces between special tokens")
    
    # Beam search
    use_beam_search: bool = Field(default=False, description="Use beam search")
    best_of: Optional[int] = Field(default=None, ge=1, description="Best of N sequences")
    
    # Other parameters
    include_stop_str_in_output: bool = Field(default=False, description="Include stop string in output")
    length_penalty: float = Field(default=1.0, description="Length penalty")
    early_stopping: bool = Field(default=False, description="Early stopping")


class vLLMServerStatus(BaseHADESModel):
    """Status information for vLLM server."""
    
    # Server status
    is_running: bool = Field(default=False, description="Whether server is running")
    pid: Optional[int] = Field(default=None, description="Process ID")
    start_time: Optional[datetime] = Field(default=None, description="Server start time")
    uptime_seconds: Optional[float] = Field(default=None, description="Server uptime in seconds")
    
    # Model status
    model_loaded: bool = Field(default=False, description="Whether model is loaded")
    model_name: Optional[str] = Field(default=None, description="Loaded model name")
    model_load_time: Optional[datetime] = Field(default=None, description="Model load time")
    
    # Resource status
    gpu_count: int = Field(default=0, ge=0, description="Number of GPUs")
    gpu_memory_total: List[float] = Field(default_factory=list, description="Total GPU memory per device")
    gpu_memory_used: List[float] = Field(default_factory=list, description="Used GPU memory per device")
    
    # Request status
    active_requests: int = Field(default=0, ge=0, description="Number of active requests")
    queued_requests: int = Field(default=0, ge=0, description="Number of queued requests")
    total_requests: int = Field(default=0, ge=0, description="Total requests processed")
    
    # Health status
    health_status: Literal["healthy", "unhealthy", "unknown"] = Field(default="unknown", description="Health status")
    last_health_check: Optional[datetime] = Field(default=None, description="Last health check time")


class vLLMInferenceStats(BaseHADESModel):
    """Statistics for vLLM inference operations."""
    
    # Request statistics
    total_requests: int = Field(default=0, ge=0, description="Total requests")
    successful_requests: int = Field(default=0, ge=0, description="Successful requests")
    failed_requests: int = Field(default=0, ge=0, description="Failed requests")
    
    # Timing statistics
    avg_time_to_first_token: float = Field(default=0.0, ge=0, description="Average time to first token")
    avg_inter_token_latency: float = Field(default=0.0, ge=0, description="Average inter-token latency")
    avg_total_time: float = Field(default=0.0, ge=0, description="Average total processing time")
    
    # Throughput statistics
    tokens_per_second: float = Field(default=0.0, ge=0, description="Tokens per second")
    requests_per_second: float = Field(default=0.0, ge=0, description="Requests per second")
    
    # Token statistics
    total_input_tokens: int = Field(default=0, ge=0, description="Total input tokens")
    total_output_tokens: int = Field(default=0, ge=0, description="Total output tokens")
    avg_input_length: float = Field(default=0.0, ge=0, description="Average input length")
    avg_output_length: float = Field(default=0.0, ge=0, description="Average output length")
    
    # Resource statistics
    avg_gpu_utilization: float = Field(default=0.0, ge=0, le=100, description="Average GPU utilization")
    peak_memory_usage: float = Field(default=0.0, ge=0, description="Peak memory usage in GB")
    
    # Collection metadata
    stats_period_start: datetime = Field(default_factory=datetime.utcnow, description="Stats period start")
    stats_period_end: datetime = Field(default_factory=datetime.utcnow, description="Stats period end")


class vLLMBatchResult(BaseHADESModel):
    """Result of a vLLM batch processing operation."""
    
    # Batch metadata
    batch_id: str = Field(description="Unique batch identifier")
    batch_size: int = Field(ge=1, description="Number of requests in batch")
    
    # Individual results
    results: List[Dict[str, Any]] = Field(description="Individual request results")
    
    # Batch statistics
    total_processing_time: float = Field(ge=0, description="Total batch processing time")
    avg_processing_time: float = Field(ge=0, description="Average per-request processing time")
    successful_count: int = Field(ge=0, description="Number of successful requests")
    failed_count: int = Field(ge=0, description="Number of failed requests")
    
    # Token statistics
    total_input_tokens: int = Field(default=0, ge=0, description="Total input tokens")
    total_output_tokens: int = Field(default=0, ge=0, description="Total output tokens")
    
    # Errors
    errors: List[str] = Field(default_factory=list, description="Batch-level errors")
    
    # Timestamps
    batch_start_time: datetime = Field(default_factory=datetime.utcnow, description="Batch start time")
    batch_end_time: datetime = Field(default_factory=datetime.utcnow, description="Batch end time")


# Export all types
__all__ = [
    "vLLMServerConfig",
    "vLLMRequestConfig", 
    "vLLMServerStatus",
    "vLLMInferenceStats",
    "vLLMBatchResult"
]