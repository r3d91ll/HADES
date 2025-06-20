"""
Chonky Chunking Types

Type definitions for GPU-accelerated "chonky" chunking components.
"""

from typing import Dict, Any, List, Optional, Literal
from pydantic import Field, validator
from datetime import datetime, timezone

from ...common import BaseHADESModel


class ChonkyChunkingConfig(BaseHADESModel):
    """Configuration for chonky (GPU-accelerated) chunking component."""
    
    # GPU configuration
    device: str = Field(default="cuda", description="Device to use (cuda, cpu)")
    gpu_id: int = Field(default=0, ge=0, description="GPU ID to use")
    batch_size: int = Field(default=32, ge=1, description="Batch size for GPU processing")
    max_memory_gb: float = Field(default=8.0, gt=0, description="Maximum GPU memory to use")
    
    # Parallel processing
    num_workers: int = Field(default=4, ge=1, description="Number of worker processes")
    parallel_chunk_processing: bool = Field(default=True, description="Enable parallel chunk processing")
    
    # Advanced chunking
    use_transformer_tokenizer: bool = Field(default=True, description="Use transformer tokenizer")
    tokenizer_model: str = Field(default="bert-base-uncased", description="Tokenizer model to use")
    max_token_length: int = Field(default=512, ge=1, description="Maximum token length")
    
    # Embedding-aware chunking
    use_embedding_similarity: bool = Field(default=False, description="Use embedding similarity for chunking")
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", description="Embedding model")
    similarity_threshold: float = Field(default=0.8, ge=0, le=1, description="Similarity threshold")
    
    # Performance optimization
    use_mixed_precision: bool = Field(default=True, description="Use mixed precision")
    prefetch_factor: int = Field(default=2, ge=1, description="Prefetch factor for data loading")
    pin_memory: bool = Field(default=True, description="Pin memory for faster GPU transfer")
    
    @validator('device')
    def validate_device(cls, v: str) -> str:
        if v not in ['cuda', 'cpu', 'auto']:
            raise ValueError("Device must be 'cuda', 'cpu', or 'auto'")
        return v


class ChonkyChunkingPerformance(BaseHADESModel):
    """Performance metrics for chonky chunking operations."""
    
    # GPU metrics
    gpu_utilization_percent: float = Field(default=0.0, ge=0, le=100, description="GPU utilization")
    gpu_memory_used_gb: float = Field(default=0.0, ge=0, description="GPU memory used")
    gpu_memory_total_gb: float = Field(default=0.0, ge=0, description="Total GPU memory")
    gpu_temperature_celsius: Optional[float] = Field(default=None, description="GPU temperature")
    
    # Processing metrics
    chunks_per_second: float = Field(default=0.0, ge=0, description="Chunks processed per second")
    tokens_per_second: float = Field(default=0.0, ge=0, description="Tokens processed per second")
    batch_processing_time: float = Field(default=0.0, ge=0, description="Average batch processing time")
    
    # Parallelization metrics
    parallel_efficiency: float = Field(default=0.0, ge=0, le=1, description="Parallel processing efficiency")
    worker_utilization: Dict[int, float] = Field(default_factory=dict, description="Per-worker utilization")
    
    # Memory metrics
    cpu_memory_used_mb: float = Field(default=0.0, ge=0, description="CPU memory used")
    shared_memory_used_mb: float = Field(default=0.0, ge=0, description="Shared memory used")
    
    # Quality metrics
    tokenization_accuracy: float = Field(default=0.0, ge=0, le=1, description="Tokenization accuracy")
    chunk_quality_score: float = Field(default=0.0, ge=0, le=1, description="Overall chunk quality score")
    
    # Collection metadata
    measurement_timestamp: datetime = Field(default_factory=datetime.utcnow, description="Measurement timestamp")
    measurement_duration: float = Field(default=0.0, ge=0, description="Measurement duration")


class ChonkyChunkingBatch(BaseHADESModel):
    """Batch processing result for chonky chunking."""
    
    # Batch metadata
    batch_id: str = Field(description="Unique batch identifier")
    batch_size: int = Field(ge=1, description="Number of documents in batch")
    
    # Processing results
    successful_chunks: int = Field(default=0, ge=0, description="Successfully processed chunks")
    failed_chunks: int = Field(default=0, ge=0, description="Failed chunks")
    total_tokens_processed: int = Field(default=0, ge=0, description="Total tokens processed")
    
    # Performance metrics
    batch_processing_time: float = Field(ge=0, description="Total batch processing time")
    gpu_processing_time: float = Field(ge=0, description="GPU processing time")
    cpu_processing_time: float = Field(ge=0, description="CPU processing time")
    
    # Resource utilization
    peak_gpu_memory_gb: float = Field(default=0.0, ge=0, description="Peak GPU memory usage")
    avg_gpu_utilization: float = Field(default=0.0, ge=0, le=100, description="Average GPU utilization")
    
    # Error information
    errors: List[str] = Field(default_factory=list, description="Processing errors")
    warnings: List[str] = Field(default_factory=list, description="Processing warnings")
    
    # Timestamps
    batch_start_time: datetime = Field(default_factory=datetime.utcnow, description="Batch start time")
    batch_end_time: datetime = Field(default_factory=datetime.utcnow, description="Batch end time")


class ChonkyChunkingState(BaseHADESModel):
    """State information for chonky chunking component."""
    
    # GPU state
    gpu_available: bool = Field(default=False, description="Whether GPU is available")
    gpu_device_name: Optional[str] = Field(default=None, description="GPU device name")
    cuda_version: Optional[str] = Field(default=None, description="CUDA version")
    
    # Model state
    tokenizer_loaded: bool = Field(default=False, description="Whether tokenizer is loaded")
    embedding_model_loaded: bool = Field(default=False, description="Whether embedding model is loaded")
    
    # Processing state
    active_batches: int = Field(default=0, ge=0, description="Number of active batches")
    queued_batches: int = Field(default=0, ge=0, description="Number of queued batches")
    
    # Performance state
    current_throughput: float = Field(default=0.0, ge=0, description="Current processing throughput")
    avg_gpu_utilization: float = Field(default=0.0, ge=0, le=100, description="Average GPU utilization")
    
    # Health state
    component_health: Literal["optimal", "degraded", "error"] = Field(
        default="optimal",
        description="Component health status"
    )
    last_health_check: Optional[datetime] = Field(default=None, description="Last health check")


# Export all types
__all__ = [
    "ChonkyChunkingConfig",
    "ChonkyChunkingPerformance",
    "ChonkyChunkingBatch", 
    "ChonkyChunkingState"
]