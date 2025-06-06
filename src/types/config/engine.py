"""
Engine configuration types for HADES-PathRAG.

This module contains Pydantic models for engine configuration structures.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union, Literal, Any, Set
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict


class StageConfig(BaseModel):
    """Configuration for a specific pipeline stage."""
    
    timeout_seconds: int = 60
    
    model_config = ConfigDict(extra="allow")  # Allow extra fields for stage-specific config


class DocProcConfig(StageConfig):
    """Configuration for document processing stage."""
    pass


class ChunkConfig(StageConfig):
    """Configuration for chunking stage."""
    
    max_tokens: int = 2048
    use_overlap: bool = True


class EmbedConfig(StageConfig):
    """Configuration for embedding stage."""
    
    model: str = "modern-bert"
    batch_size: int = 32


class ISNEConfig(StageConfig):
    """Configuration for ISNE stage."""
    
    gnn_layers: int = 2
    embedding_dim: int = 768


class PipelineLayoutConfig(BaseModel):
    """Configuration for pipeline stage-to-GPU layout."""
    
    gpu0: List[str] = Field(default_factory=lambda: ["DocProc", "Chunk"])
    gpu1: List[str] = Field(default_factory=lambda: ["Embed", "ISNE"])
    
    @field_validator("gpu0", "gpu1")
    @classmethod
    def validate_stages(cls, v: List[str]) -> List[str]:
        """Validate that stages are recognized."""
        valid_stages = {"DocProc", "Chunk", "Embed", "ISNE"}
        for stage in v:
            if stage not in valid_stages:
                raise ValueError(f"Invalid stage: {stage}. Valid stages: {valid_stages}")
        return v
    
    @model_validator(mode='after')
    def validate_all_stages_assigned(self) -> 'PipelineLayoutConfig':
        """Validate that all stages are assigned to exactly one GPU."""
        gpu0 = self.gpu0
        gpu1 = self.gpu1
        
        # Check for duplicate assignments
        intersection = set(gpu0).intersection(set(gpu1))
        if intersection:
            raise ValueError(f"Stages assigned to multiple GPUs: {intersection}")
        
        # Check that all stages are assigned
        all_stages = {"DocProc", "Chunk", "Embed", "ISNE"}
        assigned = set(gpu0).union(set(gpu1))
        missing = all_stages - assigned
        if missing:
            raise ValueError(f"Stages not assigned to any GPU: {missing}")
        
        return self


class StagesConfig(BaseModel):
    """Configuration for all pipeline stages."""
    
    DocProc: DocProcConfig = Field(default_factory=DocProcConfig)
    Chunk: ChunkConfig = Field(default_factory=ChunkConfig)
    Embed: EmbedConfig = Field(default_factory=EmbedConfig)
    ISNE: ISNEConfig = Field(default_factory=ISNEConfig)


class PipelineConfig(BaseModel):
    """Configuration for the pipeline."""
    
    layout: PipelineLayoutConfig = Field(default_factory=PipelineLayoutConfig)
    batch_size: int = 128
    queue_depth: int = 4
    nvlink_peer_copy: bool = True
    stages: StagesConfig = Field(default_factory=StagesConfig)


class MonitoringConfig(BaseModel):
    """Configuration for performance monitoring."""
    
    enabled: bool = True
    prometheus_port: int = 9090
    log_metrics: bool = True
    metrics: List[str] = Field(
        default_factory=lambda: [
            "gpu_utilization",
            "queue_length",
            "batch_latency",
            "memory_usage",
        ]
    )


class ErrorHandlingConfig(BaseModel):
    """Configuration for error handling."""
    
    max_retries: int = 3
    backoff_factor: int = 2
    dead_letter_queue: bool = True


class EngineConfig(BaseModel):
    """Configuration for the GPU-orchestrated batch engine."""
    
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    error_handling: ErrorHandlingConfig = Field(default_factory=ErrorHandlingConfig)