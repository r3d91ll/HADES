"""
Jina v4 processor types for HADES.

This module defines types specific to the Jina v4 document processor.
"""

from typing import Dict, Any, Optional, List, Union, Literal
from datetime import datetime
from pydantic import Field
from pathlib import Path

from ..common import BaseSchema, EmbeddingVector, DocumentType


class JinaV4Config(BaseSchema):
    """Configuration for Jina v4 processor."""
    model_name: str = "jinaai/jina-embeddings-v4"
    device: str = "cuda"
    output_mode: Literal["single-vector", "multi-vector"] = "multi-vector"
    late_chunking: bool = True
    extract_keywords: bool = True
    
    # Model settings
    max_length: int = 8192
    batch_size: int = 32
    
    # Feature flags
    features: Dict[str, Any] = Field(default_factory=lambda: {
        "ast_analysis": {"enabled": True},
        "keyword_extraction": {"enabled": True, "max_keywords": 10},
        "semantic_chunking": {"enabled": True},
        "multimodal": {"enabled": True}
    })
    
    # vLLM settings
    vllm: Dict[str, Any] = Field(default_factory=lambda: {
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.9,
        "max_model_len": 8192,
        "dtype": "float16"
    })
    
    # LoRA adapter
    lora_adapter: str = "retrieval"  # "retrieval", "text-matching", "classification"


class DocumentInput(BaseSchema):
    """Input for document processing."""
    file_path: Optional[Path] = None
    content: Optional[str] = None
    document_type: Optional[DocumentType] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    hierarchy: Optional[Dict[str, Any]] = None  # Filesystem hierarchy info
    options: Dict[str, Any] = Field(default_factory=dict)


class ChunkData(BaseSchema):
    """Data for a document chunk."""
    chunk_id: str
    text: str
    embeddings: Union[EmbeddingVector, List[EmbeddingVector]]  # Single or multi-vector
    keywords: List[str] = Field(default_factory=list)
    chunk_index: int
    start_pos: int
    end_pos: int
    metadata: Dict[str, Any] = Field(default_factory=dict)
    relationships: List[Dict[str, Any]] = Field(default_factory=list)


class ASTAnalysis(BaseSchema):
    """AST analysis results for code files."""
    imports: List[Dict[str, Any]]
    functions: List[Dict[str, Any]]
    classes: List[Dict[str, Any]]
    global_vars: List[str]
    keywords: List[str]
    complexity_metrics: Dict[str, float] = Field(default_factory=dict)
    dependencies: List[str] = Field(default_factory=list)
    stats: Dict[str, int] = Field(default_factory=dict)


class ProcessingResult(BaseSchema):
    """Result from document processing."""
    chunks: List[ChunkData]
    document_metadata: Dict[str, Any]
    document_keywords: List[str] = Field(default_factory=list)
    ast_analysis: Optional[ASTAnalysis] = None
    images: List[Dict[str, Any]] = Field(default_factory=list)
    processing_time: float
    token_count: int
    model_used: str


class EmbeddingExtractionConfig(BaseSchema):
    """Configuration for embedding extraction."""
    mode: Literal["api", "local"] = "api"
    adapter: str = "retrieval"
    batch_size: int = 32
    normalize: bool = True
    pooling_strategy: str = "mean"  # For multi-vector to single-vector
    

class VLLMConfig(BaseSchema):
    """Configuration for vLLM integration."""
    api_url: Optional[str] = None
    api_key: Optional[str] = None
    model_path: Optional[str] = None
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: int = 8192
    dtype: Literal["float16", "bfloat16", "float32"] = "float16"
    

class KeywordExtractionResult(BaseSchema):
    """Result from keyword extraction."""
    keywords: List[str]
    keyword_scores: Dict[str, float]
    extraction_method: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    

class SemanticBoundary(BaseSchema):
    """Semantic boundary detection result."""
    position: int
    confidence: float
    boundary_type: str  # "paragraph", "section", "topic_shift"
    context_before: str
    context_after: str
    

class LateCunkingConfig(BaseSchema):
    """Configuration for late chunking."""
    strategy: Literal["semantic_similarity", "fixed_size", "sentence_boundary"] = "semantic_similarity"
    min_chunk_tokens: int = 128
    max_chunk_tokens: int = 512
    overlap_tokens: int = 50
    similarity_threshold: float = 0.85
    

class MultimodalContent(BaseSchema):
    """Multimodal content representation."""
    text: Optional[str] = None
    images: List[Dict[str, Any]] = Field(default_factory=list)
    tables: List[Dict[str, Any]] = Field(default_factory=list)
    code_blocks: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)