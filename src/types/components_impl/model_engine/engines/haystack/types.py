"""
Haystack Model Engine Types

Type definitions specifically for the Haystack model engine component.
These types are used by the HaystackModelEngine implementation.
"""

from typing import Dict, Any, List, Optional, Union, Literal
from pydantic import BaseModel, Field, validator
from datetime import datetime, timezone

from ...common import BaseHADESModel


class HaystackPipelineConfig(BaseHADESModel):
    """Configuration for Haystack pipeline."""
    
    # Pipeline type
    pipeline_type: Literal["embedding", "qa", "summarization", "generation"] = Field(
        default="embedding", 
        description="Type of Haystack pipeline"
    )
    
    # Model configuration
    model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Name of the model to use"
    )
    
    # Document store configuration
    document_store_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            "type": "memory",
            "embedding_dim": 384
        },
        description="Configuration for document store"
    )
    
    # Retriever configuration
    retriever_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            "top_k": 10,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
        },
        description="Configuration for retriever"
    )
    
    # Reader configuration (for QA pipelines)
    reader_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration for reader component"
    )
    
    # Generator configuration (for generation pipelines)
    generator_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration for generator component"
    )


class HaystackDocumentStoreConfig(BaseHADESModel):
    """Configuration for Haystack document store."""
    
    store_type: Literal["memory", "elasticsearch", "faiss", "weaviate"] = Field(
        default="memory",
        description="Type of document store"
    )
    
    # Memory store config
    embedding_dim: int = Field(default=384, ge=1, description="Embedding dimension")
    similarity: Literal["cosine", "dot_product", "l2"] = Field(
        default="cosine",
        description="Similarity metric"
    )
    
    # Elasticsearch config
    host: Optional[str] = Field(default=None, description="Elasticsearch host")
    port: Optional[int] = Field(default=None, ge=1, le=65535, description="Elasticsearch port")
    username: Optional[str] = Field(default=None, description="Elasticsearch username")
    password: Optional[str] = Field(default=None, description="Elasticsearch password")
    index: Optional[str] = Field(default=None, description="Elasticsearch index")
    
    # FAISS config
    faiss_index_factory_str: Optional[str] = Field(default=None, description="FAISS index factory string")
    faiss_config_path: Optional[str] = Field(default=None, description="FAISS config file path")
    
    # General config
    duplicate_documents: Literal["skip", "overwrite", "fail"] = Field(
        default="skip",
        description="How to handle duplicate documents"
    )
    return_embedding: bool = Field(default=False, description="Whether to return embeddings")


class HaystackRetrieverConfig(BaseHADESModel):
    """Configuration for Haystack retriever."""
    
    retriever_type: Literal["embedding", "bm25", "dpr"] = Field(
        default="embedding",
        description="Type of retriever"
    )
    
    # Common config
    top_k: int = Field(default=10, ge=1, description="Number of documents to retrieve")
    
    # Embedding retriever config
    embedding_model: Optional[str] = Field(
        default=None,
        description="Embedding model for retriever"
    )
    model_format: Optional[str] = Field(default=None, description="Model format")
    pooling_strategy: Optional[Literal["cls_token", "mean", "max", "mean_max"]] = Field(
        default=None,
        description="Pooling strategy for embeddings"
    )
    
    # BM25 config
    language: Optional[str] = Field(default="english", description="Language for BM25")
    
    # DPR config
    query_embedding_model: Optional[str] = Field(default=None, description="Query embedding model for DPR")
    passage_embedding_model: Optional[str] = Field(default=None, description="Passage embedding model for DPR")


class HaystackReaderConfig(BaseHADESModel):
    """Configuration for Haystack reader (for QA)."""
    
    reader_type: Literal["farm", "transformers"] = Field(
        default="transformers",
        description="Type of reader"
    )
    
    # Model config
    model_name_or_path: str = Field(
        default="deepset/roberta-base-squad2",
        description="Model name or path"
    )
    
    # Processing config
    context_window_size: int = Field(default=150, ge=1, description="Context window size")
    return_no_answer: bool = Field(default=False, description="Whether to return no answer")
    max_seq_len: int = Field(default=384, ge=1, description="Maximum sequence length")
    doc_stride: int = Field(default=128, ge=1, description="Document stride")
    
    # Answer extraction config
    top_k: int = Field(default=10, ge=1, description="Top K answers to return")
    no_ans_boost: float = Field(default=0.0, description="No answer boost")
    confidence_threshold: float = Field(default=0.0, ge=0, le=1, description="Confidence threshold")


class HaystackPipelineStatus(BaseHADESModel):
    """Status information for Haystack pipeline."""
    
    # Pipeline status
    pipeline_initialized: bool = Field(default=False, description="Whether pipeline is initialized")
    pipeline_type: Optional[str] = Field(default=None, description="Pipeline type")
    
    # Component status
    document_store_initialized: bool = Field(default=False, description="Document store status")
    retriever_initialized: bool = Field(default=False, description="Retriever status")
    reader_initialized: bool = Field(default=False, description="Reader status")
    generator_initialized: bool = Field(default=False, description="Generator status")
    
    # Model status
    models_loaded: List[str] = Field(default_factory=list, description="Loaded models")
    model_load_times: Dict[str, datetime] = Field(default_factory=dict, description="Model load times")
    
    # Processing status
    documents_indexed: int = Field(default=0, ge=0, description="Number of indexed documents")
    last_query_time: Optional[datetime] = Field(default=None, description="Last query time")
    
    # Health status
    health_status: Literal["healthy", "unhealthy", "unknown"] = Field(
        default="unknown",
        description="Overall health status"
    )
    last_health_check: Optional[datetime] = Field(default=None, description="Last health check")


class HaystackQueryResult(BaseHADESModel):
    """Result from Haystack query operation."""
    
    # Query metadata
    query: str = Field(description="Original query")
    query_id: Optional[str] = Field(default=None, description="Query identifier")
    pipeline_type: str = Field(description="Pipeline type used")
    
    # Results
    answers: List[Dict[str, Any]] = Field(default_factory=list, description="Answer results")
    documents: List[Dict[str, Any]] = Field(default_factory=list, description="Retrieved documents")
    
    # Metadata
    retrieval_time: float = Field(ge=0, description="Time spent on retrieval")
    processing_time: float = Field(ge=0, description="Total processing time")
    
    # Query statistics
    num_documents_retrieved: int = Field(ge=0, description="Number of documents retrieved")
    num_answers_returned: int = Field(ge=0, description="Number of answers returned")
    
    # Confidence scores
    max_answer_confidence: Optional[float] = Field(default=None, ge=0, le=1, description="Max answer confidence")
    avg_document_score: Optional[float] = Field(default=None, description="Average document score")
    
    # Timestamps
    query_timestamp: datetime = Field(default_factory=datetime.utcnow, description="Query timestamp")


class HaystackPerformanceMetrics(BaseHADESModel):
    """Performance metrics for Haystack operations."""
    
    # Query metrics
    total_queries: int = Field(default=0, ge=0, description="Total queries processed")
    successful_queries: int = Field(default=0, ge=0, description="Successful queries")
    failed_queries: int = Field(default=0, ge=0, description="Failed queries")
    
    # Timing metrics
    avg_query_time: float = Field(default=0.0, ge=0, description="Average query time")
    avg_retrieval_time: float = Field(default=0.0, ge=0, description="Average retrieval time")
    avg_reading_time: float = Field(default=0.0, ge=0, description="Average reading time")
    
    # Document metrics
    total_documents_indexed: int = Field(default=0, ge=0, description="Total documents indexed")
    avg_documents_per_query: float = Field(default=0.0, ge=0, description="Average documents per query")
    
    # Answer metrics
    avg_answers_per_query: float = Field(default=0.0, ge=0, description="Average answers per query")
    avg_answer_confidence: float = Field(default=0.0, ge=0, le=1, description="Average answer confidence")
    
    # Resource metrics
    memory_usage_mb: float = Field(default=0.0, ge=0, description="Memory usage in MB")
    
    # Collection metadata
    metrics_start_time: datetime = Field(default_factory=datetime.utcnow, description="Metrics start time")
    metrics_end_time: datetime = Field(default_factory=datetime.utcnow, description="Metrics end time")


# Export all types
__all__ = [
    "HaystackPipelineConfig",
    "HaystackDocumentStoreConfig",
    "HaystackRetrieverConfig",
    "HaystackReaderConfig",
    "HaystackPipelineStatus",
    "HaystackQueryResult",
    "HaystackPerformanceMetrics"
]