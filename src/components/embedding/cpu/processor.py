"""
CPU Embedding Component

This module provides CPU-based embedding component that implements the
Embedder protocol specifically for CPU inference using sentence transformers
and other CPU-optimized models.
"""

import logging
import numpy as np
import psutil
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

# Import component contracts and protocols
from src.types.components.contracts import (
    ComponentType,
    ComponentMetadata,
    EmbeddingInput,
    EmbeddingOutput,
    ChunkEmbedding,
    ProcessingStatus
)
from src.types.components.protocols import Embedder

# Import model engine factory
from src.components.model_engine.factory import create_model_engine


class CPUEmbedder(Embedder):
    """
    CPU embedding component implementing Embedder protocol.
    
    This component specializes in CPU-based embedding generation using
    sentence transformers and other CPU-optimized models.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize CPU embedder.
        
        Args:
            config: Optional configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self._config = config or {}
        
        # Component metadata
        self._metadata = ComponentMetadata(
            component_type=ComponentType.EMBEDDING,
            component_name="cpu",
            component_version="1.0.0",
            config=self._config
        )
        
        # Configuration settings
        self._model_name = self._config.get('model_name', 'all-MiniLM-L6-v2')
        self._batch_size = self._config.get('batch_size', 32)
        self._normalize_embeddings = self._config.get('normalize_embeddings', True)
        self._max_length = self._config.get('max_length', 512)
        self._pooling_strategy = self._config.get('pooling_strategy', 'mean')
        
        # Model components
        self._model = None
        self._tokenizer = None
        self._model_loaded = False
        
        # Performance tracking
        self._total_embeddings_created = 0
        self._total_processing_time = 0.0
        
        # Model engine will be initialized on demand
        self._model_engine_type = self._config.get('model_engine_type', 'haystack')
        
        # Monitoring integration - standardized stats tracking
        self._startup_time = datetime.now(timezone.utc)
        self._stats = {
            "total_embeddings_created": 0,
            "successful_embeddings": 0,
            "failed_embeddings": 0,
            "total_processing_time": 0.0,
            "total_characters_processed": 0,
            "total_tokens_processed": 0,
            "last_processing_time": None,
            "initialization_count": 1,
            "errors": [],
            "model_loads": 0,
            "embedding_method_counts": {}  # Track usage of different embedding methods
        }
        
        self.logger.info(f"Initialized CPU embedder with model: {self._model_name}")
    
    @property
    def name(self) -> str:
        """Component name for identification."""
        return "cpu"
    
    @property
    def version(self) -> str:
        """Component version string."""
        return "1.0.0"
    
    @property
    def component_type(self) -> ComponentType:
        """Type of component."""
        return ComponentType.EMBEDDING
    
    def configure(self, config: Dict[str, Any]) -> None:
        """
        Configure component with parameters.
        
        Args:
            config: Configuration dictionary containing component parameters
            
        Raises:
            ValueError: If configuration is invalid
        """
        if not self.validate_config(config):
            raise ValueError("Invalid configuration provided")
        
        self._config.update(config)
        self._metadata.config = self._config
        self._metadata.processed_at = datetime.utcnow()
        
        # Update configuration parameters
        if 'model_name' in config:
            self._model_name = config['model_name']
            # Reset model to force reload
            self._model = None
            self._tokenizer = None
            self._model_loaded = False
        
        if 'batch_size' in config:
            self._batch_size = config['batch_size']
        
        if 'max_length' in config:
            self._max_length = config['max_length']
        
        self.logger.info("Updated CPU embedder configuration")
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate configuration parameters.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            True if configuration is valid, False otherwise
        """
        if not isinstance(config, dict):
            return False
        
        # Validate batch size
        if 'batch_size' in config:
            batch_size = config['batch_size']
            if not isinstance(batch_size, int) or batch_size < 1:
                return False
        
        # Validate max length
        if 'max_length' in config:
            max_length = config['max_length']
            if not isinstance(max_length, int) or max_length < 1:
                return False
        
        # Validate pooling strategy
        if 'pooling_strategy' in config:
            pooling = config['pooling_strategy']
            if pooling not in ['mean', 'max', 'cls']:
                return False
        
        return True
    
    def get_config_schema(self) -> Dict[str, Any]:
        """
        Get the JSON schema for component configuration.
        
        Returns:
            JSON schema dictionary describing valid configuration
        """
        return {
            "type": "object",
            "properties": {
                "model_name": {
                    "type": "string",
                    "description": "Name of the sentence transformer model",
                    "default": "all-MiniLM-L6-v2"
                },
                "batch_size": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 256,
                    "default": 32,
                    "description": "Batch size for CPU processing"
                },
                "max_length": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 8192,
                    "default": 512,
                    "description": "Maximum token length per text"
                },
                "normalize_embeddings": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to normalize embeddings"
                },
                "pooling_strategy": {
                    "type": "string",
                    "enum": ["mean", "max", "cls"],
                    "default": "mean",
                    "description": "Pooling strategy for embeddings"
                },
                "device": {
                    "type": "string",
                    "enum": ["cpu", "auto"],
                    "default": "cpu",
                    "description": "Device to use for processing"
                }
            }
        }
    
    def health_check(self) -> bool:
        """
        Check if component is healthy and ready to process data.
        
        Returns:
            True if component is healthy, False otherwise
        """
        try:
            # Try to initialize model if not already loaded
            if not self._model_loaded:
                self._initialize_model()
            
            if not self._model_loaded:
                return False
            
            # Test with simple text
            test_text = "This is a test text for embedding."
            embedding = self._embed_single_text(test_text)
            
            return len(embedding) > 0
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get component performance metrics.
        
        Returns:
            Dictionary containing performance metrics
        """
        avg_processing_time = (
            self._total_processing_time / max(self._total_embeddings_created, 1)
        )
        
        return {
            "component_name": self.name,
            "component_version": self.version,
            "model_name": self._model_name,
            "model_loaded": self._model_loaded,
            "batch_size": self._batch_size,
            "max_length": self._max_length,
            "total_embeddings_created": self._total_embeddings_created,
            "total_processing_time": self._total_processing_time,
            "avg_processing_time": avg_processing_time,
            "last_health_check": datetime.utcnow().isoformat()
        }
    
    def embed(self, input_data: EmbeddingInput) -> EmbeddingOutput:
        """
        Generate embeddings according to the contract.
        
        Args:
            input_data: Input data conforming to EmbeddingInput contract
            
        Returns:
            Output data conforming to EmbeddingOutput contract
        """
        errors = []
        
        try:
            start_time = datetime.utcnow()
            
            # Initialize model if needed
            if not self._model_loaded:
                self._initialize_model()
            
            if not self._model_loaded:
                raise ValueError("Model not initialized")
            
            # Extract texts from chunks
            texts = [chunk.content for chunk in input_data.chunks]
            chunk_ids = [chunk.id for chunk in input_data.chunks]
            
            # Generate embeddings
            embedding_vectors = self._embed_texts(texts)
            
            # Convert to contract format
            embeddings = []
            for i, (chunk_id, vector) in enumerate(zip(chunk_ids, embedding_vectors)):
                embedding = ChunkEmbedding(
                    chunk_id=chunk_id,
                    embedding=vector.tolist() if hasattr(vector, 'tolist') else vector,
                    embedding_dimension=len(vector),
                    model_name=input_data.model_name or self._model_name,
                    confidence=1.0,  # CPU embeddings have consistent confidence
                    metadata=input_data.metadata
                )
                embeddings.append(embedding)
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Update statistics - both old and new tracking
            self._total_embeddings_created += len(embeddings)
            self._total_processing_time += processing_time
            
            # Update standardized stats
            total_chars = sum(len(chunk.content) for chunk in input_data.chunks)
            total_tokens = self.estimate_tokens(input_data)
            
            self._stats["total_embeddings_created"] += len(embeddings)
            self._stats["successful_embeddings"] += 1
            self._stats["total_processing_time"] += processing_time
            self._stats["total_characters_processed"] += total_chars
            self._stats["total_tokens_processed"] += total_tokens
            self._stats["last_processing_time"] = datetime.now(timezone.utc).isoformat()
            
            # Track embedding method used
            method_used = "model_engine" if hasattr(self._model, 'embed_texts') else "fallback"
            self._stats["embedding_method_counts"][method_used] = self._stats["embedding_method_counts"].get(method_used, 0) + 1
            
            # Update metadata
            metadata = ComponentMetadata(
                component_type=self.component_type,
                component_name=self.name,
                component_version=self.version,
                processing_time=processing_time,
                processed_at=datetime.utcnow(),
                config=self._config,
                status=ProcessingStatus.SUCCESS
            )
            
            return EmbeddingOutput(
                embeddings=embeddings,
                metadata=metadata,
                embedding_stats={
                    "processing_time": processing_time,
                    "embedding_count": len(embeddings),
                    "total_texts": len(texts),
                    "batch_size": self._batch_size,
                    "model_name": self._model_name,
                    "pooling_strategy": self._pooling_strategy,
                    "throughput_embeddings_per_second": len(embeddings) / max(processing_time, 0.001)
                },
                model_info={
                    "model_name": self._model_name,
                    "embedding_dimension": len(embedding_vectors[0]) if embedding_vectors else 0,
                    "max_length": self._max_length,
                    "device": "cpu"
                },
                errors=errors
            )
            
        except Exception as e:
            error_msg = f"CPU embedding failed: {str(e)}"
            errors.append(error_msg)
            self.logger.error(error_msg)
            
            # Update error stats
            self._stats["failed_embeddings"] += 1
            error_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": error_msg
            }
            self._stats["errors"].append(error_entry)
            
            # Keep only last 50 errors to prevent memory bloat
            if len(self._stats["errors"]) > 50:
                self._stats["errors"] = self._stats["errors"][-50:]
            
            metadata = ComponentMetadata(
                component_type=self.component_type,
                component_name=self.name,
                component_version=self.version,
                processed_at=datetime.utcnow(),
                config=self._config,
                status=ProcessingStatus.ERROR
            )
            
            return EmbeddingOutput(
                embeddings=[],
                metadata=metadata,
                embedding_stats={},
                model_info={},
                errors=errors
            )
    
    def estimate_tokens(self, input_data: EmbeddingInput) -> int:
        """
        Estimate number of tokens that will be processed.
        
        Args:
            input_data: Input data to estimate for
            
        Returns:
            Estimated number of tokens
        """
        try:
            total_chars = sum(len(chunk.content) for chunk in input_data.chunks)
            # Simple estimation: ~4 characters per token
            estimated_tokens = total_chars // 4
            return max(1, estimated_tokens)
            
        except Exception:
            return len(input_data.chunks)  # Fallback estimate
    
    def supports_model(self, model_name: str) -> bool:
        """
        Check if embedder supports the given model.
        
        Args:
            model_name: Model name to check
            
        Returns:
            True if model is supported, False otherwise
        """
        # Common sentence transformer models
        supported_models = [
            'all-MiniLM-L6-v2',
            'all-mpnet-base-v2',
            'all-distilroberta-v1',
            'paraphrase-MiniLM-L6-v2',
            'sentence-transformers'
        ]
        
        return any(model in model_name.lower() for model in supported_models)
    
    def get_embedding_dimension(self, model_name: str) -> int:
        """
        Get the embedding dimension for a given model.
        
        Args:
            model_name: Model name
            
        Returns:
            Embedding dimension
        """
        # Common model dimensions
        model_dims = {
            'all-MiniLM-L6-v2': 384,
            'all-mpnet-base-v2': 768,
            'all-distilroberta-v1': 768,
            'paraphrase-MiniLM-L6-v2': 384
        }
        
        for model, dim in model_dims.items():
            if model in model_name.lower():
                return dim
        
        return 384  # Default dimension
    
    @property
    def embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this embedder."""
        return self.get_embedding_dimension(self._model_name)
    
    @property
    def max_sequence_length(self) -> int:
        """Get the maximum sequence length this embedder can handle."""
        return self._max_length
    
    def supports_batch_processing(self) -> bool:
        """Check if embedder supports batch processing."""
        return True
    
    def get_optimal_batch_size(self) -> int:
        """Get the optimal batch size for this embedder."""
        return self._batch_size
    
    def estimate_embedding_time(self, input_data: EmbeddingInput) -> float:
        """
        Estimate time to generate embeddings.
        
        Args:
            input_data: Input data to estimate for
            
        Returns:
            Estimated processing time in seconds
        """
        try:
            num_chunks = len(input_data.chunks)
            total_tokens = self.estimate_tokens(input_data)
            
            # CPU processing is generally slower than GPU
            # Estimate based on tokens and batch size
            batches_needed = (num_chunks + self._batch_size - 1) // self._batch_size
            
            # Base time per batch on CPU (conservative estimate)
            time_per_batch = 0.5  # 500ms per batch
            
            # Scale by token complexity
            token_complexity_factor = min(total_tokens / 10000, 2.0)  # Cap at 2x
            
            estimated_time = batches_needed * time_per_batch * token_complexity_factor
            
            return max(0.1, estimated_time)
            
        except Exception:
            # Fallback estimate
            return len(input_data.chunks) * 0.1
    
    def get_infrastructure_metrics(self) -> Dict[str, Any]:
        """
        Get infrastructure metrics for monitoring.
        
        Returns:
            Dictionary containing infrastructure metrics
        """
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "component_name": self.name,
            "component_version": self.version,
            "model_name": self._model_name,
            "model_loaded": self._model_loaded,
            "model_engine_type": self._model_engine_type,
            "batch_size": self._batch_size,
            "max_length": self._max_length,
            "pooling_strategy": self._pooling_strategy,
            "normalize_embeddings": self._normalize_embeddings,
            "memory_usage": {
                "rss_mb": memory_info.rss / 1024 / 1024,
                "vms_mb": memory_info.vms / 1024 / 1024
            },
            "startup_time": self._startup_time.isoformat(),
            "uptime_seconds": (datetime.now(timezone.utc) - self._startup_time).total_seconds()
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for monitoring.
        
        Returns:
            Dictionary containing performance metrics
        """
        total_operations = self._stats["successful_embeddings"] + self._stats["failed_embeddings"]
        success_rate = (self._stats["successful_embeddings"] / max(total_operations, 1)) * 100
        
        avg_processing_time = (
            self._stats["total_processing_time"] / max(self._stats["successful_embeddings"], 1)
        )
        
        uptime_seconds = (datetime.now(timezone.utc) - self._startup_time).total_seconds()
        embeddings_per_second = self._stats["total_embeddings_created"] / max(uptime_seconds, 1)
        
        return {
            "component_name": self.name,
            "total_embeddings_created": self._stats["total_embeddings_created"],
            "successful_operations": self._stats["successful_embeddings"],
            "failed_operations": self._stats["failed_embeddings"],
            "success_rate_percent": success_rate,
            "embeddings_per_second": embeddings_per_second,
            "average_processing_time": avg_processing_time,
            "total_processing_time": self._stats["total_processing_time"],
            "total_characters_processed": self._stats["total_characters_processed"],
            "total_tokens_processed": self._stats["total_tokens_processed"],
            "avg_chars_per_embedding": self._stats["total_characters_processed"] / max(self._stats["total_embeddings_created"], 1),
            "avg_tokens_per_embedding": self._stats["total_tokens_processed"] / max(self._stats["total_embeddings_created"], 1),
            "last_processing_time": self._stats["last_processing_time"],
            "initialization_count": self._stats["initialization_count"],
            "model_loads": self._stats["model_loads"],
            "embedding_method_distribution": self._stats["embedding_method_counts"],
            "recent_errors": self._stats["errors"][-10:] if self._stats["errors"] else [],
            "error_count": len(self._stats["errors"]),
            "uptime_seconds": uptime_seconds
        }
    
    def export_metrics_prometheus(self) -> str:
        """
        Export metrics in Prometheus format.
        
        Returns:
            String containing Prometheus metrics
        """
        infra_metrics = self.get_infrastructure_metrics()
        perf_metrics = self.get_performance_metrics()
        
        lines = [
            "# HELP hades_embedding_cpu_info Component information",
            "# TYPE hades_embedding_cpu_info gauge",
            f'hades_embedding_cpu_info{{component="{infra_metrics["component_name"]}",version="{infra_metrics["component_version"]}",model="{infra_metrics["model_name"]}",engine="{infra_metrics["model_engine_type"]}"}} 1',
            "",
            "# HELP hades_embedding_cpu_uptime_seconds Component uptime in seconds",
            "# TYPE hades_embedding_cpu_uptime_seconds gauge",
            f'hades_embedding_cpu_uptime_seconds{{component="{infra_metrics["component_name"]}"}} {infra_metrics["uptime_seconds"]:.2f}',
            "",
            "# HELP hades_embedding_cpu_memory_rss_bytes Resident Set Size memory usage",
            "# TYPE hades_embedding_cpu_memory_rss_bytes gauge",
            f'hades_embedding_cpu_memory_rss_bytes{{component="{infra_metrics["component_name"]}"}} {infra_metrics["memory_usage"]["rss_mb"] * 1024 * 1024:.0f}',
            "",
            "# HELP hades_embedding_cpu_memory_vms_bytes Virtual Memory Size",
            "# TYPE hades_embedding_cpu_memory_vms_bytes gauge",
            f'hades_embedding_cpu_memory_vms_bytes{{component="{infra_metrics["component_name"]}"}} {infra_metrics["memory_usage"]["vms_mb"] * 1024 * 1024:.0f}',
            "",
            "# HELP hades_embedding_cpu_embeddings_created_total Total number of embeddings created",
            "# TYPE hades_embedding_cpu_embeddings_created_total counter",
            f'hades_embedding_cpu_embeddings_created_total{{component="{perf_metrics["component_name"]}"}} {perf_metrics["total_embeddings_created"]}',
            "",
            "# HELP hades_embedding_cpu_operations_total Total embedding operations",
            "# TYPE hades_embedding_cpu_operations_total counter",
            f'hades_embedding_cpu_operations_total{{component="{perf_metrics["component_name"]}",status="success"}} {perf_metrics["successful_operations"]}',
            f'hades_embedding_cpu_operations_total{{component="{perf_metrics["component_name"]}",status="failed"}} {perf_metrics["failed_operations"]}',
            "",
            "# HELP hades_embedding_cpu_success_rate_percent Success rate percentage",
            "# TYPE hades_embedding_cpu_success_rate_percent gauge",
            f'hades_embedding_cpu_success_rate_percent{{component="{perf_metrics["component_name"]}"}} {perf_metrics["success_rate_percent"]:.2f}',
            "",
            "# HELP hades_embedding_cpu_processing_time_seconds Total processing time",
            "# TYPE hades_embedding_cpu_processing_time_seconds counter",
            f'hades_embedding_cpu_processing_time_seconds{{component="{perf_metrics["component_name"]}"}} {perf_metrics["total_processing_time"]:.6f}',
            "",
            "# HELP hades_embedding_cpu_throughput_embeddings_per_second Current throughput",
            "# TYPE hades_embedding_cpu_throughput_embeddings_per_second gauge",
            f'hades_embedding_cpu_throughput_embeddings_per_second{{component="{perf_metrics["component_name"]}"}} {perf_metrics["embeddings_per_second"]:.2f}',
            "",
            "# HELP hades_embedding_cpu_characters_processed_total Total characters processed",
            "# TYPE hades_embedding_cpu_characters_processed_total counter",
            f'hades_embedding_cpu_characters_processed_total{{component="{perf_metrics["component_name"]}"}} {perf_metrics["total_characters_processed"]}',
            "",
            "# HELP hades_embedding_cpu_tokens_processed_total Total tokens processed",
            "# TYPE hades_embedding_cpu_tokens_processed_total counter",
            f'hades_embedding_cpu_tokens_processed_total{{component="{perf_metrics["component_name"]}"}} {perf_metrics["total_tokens_processed"]}',
            "",
            "# HELP hades_embedding_cpu_errors_total Total number of errors",
            "# TYPE hades_embedding_cpu_errors_total counter",
            f'hades_embedding_cpu_errors_total{{component="{perf_metrics["component_name"]}"}} {perf_metrics["error_count"]}',
            "",
            "# HELP hades_embedding_cpu_model_loads_total Total number of model loads",
            "# TYPE hades_embedding_cpu_model_loads_total counter",
            f'hades_embedding_cpu_model_loads_total{{component="{perf_metrics["component_name"]}"}} {perf_metrics["model_loads"]}',
            ""
        ]
        
        # Add method distribution metrics
        for method, count in perf_metrics["embedding_method_distribution"].items():
            lines.extend([
                "# HELP hades_embedding_cpu_method_usage_total Usage count by embedding method",
                "# TYPE hades_embedding_cpu_method_usage_total counter",
                f'hades_embedding_cpu_method_usage_total{{component="{perf_metrics["component_name"]}",method="{method}"}} {count}',
                ""
            ])
        
        return "\n".join(lines)
    
    def _initialize_model(self) -> None:
        """Initialize the CPU model engine."""
        try:
            self.logger.info(f"Initializing CPU model engine for model: {self._model_name}")
            
            # Get CPU model engine from factory
            engine_config = {
                'model_name': self._model_name,
                'device': 'cpu',
                **self._config
            }
            self._model = create_model_engine(
                engine_type=self._model_engine_type,
                config=engine_config
            )
            
            if self._model:
                self._model_loaded = True
                self._stats["model_loads"] += 1
                self.logger.info("CPU model engine initialized successfully")
            else:
                self.logger.warning("Failed to create CPU model engine, using fallback")
                self._model = "basic_fallback"
                self._model_loaded = True
                self._stats["model_loads"] += 1
            
        except Exception as e:
            self.logger.error(f"Failed to initialize CPU model engine: {e}")
            self._model = "basic_fallback"
            self._model_loaded = True
            self._stats["model_loads"] += 1
            
            # Track initialization error
            error_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": f"Model initialization failed: {str(e)}"
            }
            self._stats["errors"].append(error_entry)
            if len(self._stats["errors"]) > 50:
                self._stats["errors"] = self._stats["errors"][-50:]
            
            self.logger.info("Using basic fallback embedding implementation")
    
    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        if not self._model_loaded:
            raise ValueError("Model not initialized")
        
        try:
            if hasattr(self._model, 'embed_texts'):
                # Use model engine
                embeddings = self._model.embed_texts(
                    texts=texts,
                    batch_size=self._batch_size,
                    max_length=self._max_length,
                    normalize=self._normalize_embeddings
                )
                return embeddings
            
            else:
                # Use fallback implementation
                return self._embed_texts_fallback(texts)
            
        except Exception as e:
            self.logger.error(f"Embedding generation failed: {e}")
            # Return fallback embeddings
            return self._embed_texts_fallback(texts)
    
    def _embed_single_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        embeddings = self._embed_texts([text])
        return embeddings[0] if embeddings else []
    
    def _embed_texts_fallback(self, texts: List[str]) -> List[List[float]]:
        """Fallback embedding method using basic text features."""
        embeddings = []
        
        for text in texts:
            # Simple feature-based embedding
            features = self._extract_text_features(text)
            embeddings.append(features)
        
        return embeddings
    
    def _extract_text_features(self, text: str) -> List[float]:
        """Extract basic text features for fallback embedding."""
        import re
        import math
        
        # Basic text statistics (normalized to create embedding-like vector)
        features = []
        
        # Length features
        features.append(min(len(text) / 1000.0, 1.0))  # Normalized length
        features.append(min(len(text.split()) / 100.0, 1.0))  # Normalized word count
        
        # Character distribution features
        char_counts = {}
        for char in text.lower():
            char_counts[char] = char_counts.get(char, 0) + 1
        
        # Add features for common characters
        common_chars = 'abcdefghijklmnopqrstuvwxyz0123456789 '
        for char in common_chars:
            freq = char_counts.get(char, 0) / max(len(text), 1)
            features.append(freq)
        
        # Pad or truncate to fixed dimension
        target_dim = 384
        while len(features) < target_dim:
            features.append(0.0)
        
        features = features[:target_dim]
        
        # Normalize if requested
        if self._normalize_embeddings:
            norm = math.sqrt(sum(f * f for f in features))
            if norm > 0:
                features = [f / norm for f in features]
        
        return features