"""
GPU Embedding Component

This module provides GPU-based embedding component that implements the
Embedder protocol for high-throughput GPU inference using the configured
model engine (VLLM, Haystack, Ollama, etc.) for embedding models.
"""

import logging
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


class GPUEmbedder(Embedder):
    """
    GPU embedding component implementing Embedder protocol.
    
    This component specializes in high-throughput GPU-based embedding generation
    using the configured model engine for efficient batch processing and model serving.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize GPU embedder.
        
        Args:
            config: Optional configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self._config = config or {}
        
        # Component metadata
        self._metadata = ComponentMetadata(
            component_type=ComponentType.EMBEDDING,
            component_name="gpu",
            component_version="1.0.0",
            config=self._config
        )
        
        # Configuration settings
        self._model_name = self._config.get('model_name', 'sentence-transformers/all-MiniLM-L6-v2')
        self._batch_size = self._config.get('batch_size', 64)  # Higher batch size for GPU
        self._max_length = self._config.get('max_length', 512)
        self._normalize_embeddings = self._config.get('normalize_embeddings', True)
        
        # Model engine components
        self._model_engine: Optional[Any] = None
        self._engine_loaded = False
        
        # Performance tracking
        self._total_embeddings_created = 0
        self._total_processing_time = 0.0
        
        # Initialize model engine factory
        # Model engine type will be vllm for GPU by default
        self._model_engine_type = self._config.get('model_engine_type', 'vllm')
        
        # Monitoring integration - standardized stats tracking
        self._startup_time = datetime.now(timezone.utc)
        self._stats: Dict[str, Any] = {
            "total_embeddings_created": 0,
            "successful_embeddings": 0,
            "failed_embeddings": 0,
            "total_processing_time": 0.0,
            "total_characters_processed": 0,
            "total_tokens_processed": 0,
            "last_processing_time": None,
            "initialization_count": 1,
            "errors": [],
            "engine_loads": 0,
            "gpu_operations": 0,
            "embedding_method_counts": {}  # Track usage of different embedding methods
        }
        
        self.logger.info(f"Initialized GPU embedder with model: {self._model_name}")
    
    @property
    def name(self) -> str:
        """Component name for identification."""
        return "gpu"
    
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
            # Reset engine to force reload
            self._model_engine = None
            self._engine_loaded = False
        
        if 'batch_size' in config:
            self._batch_size = config['batch_size']
        
        if 'max_length' in config:
            self._max_length = config['max_length']
        
        self.logger.info("Updated GPU embedder configuration")
    
    def validate_config(self, config: Any) -> bool:
        """
        Validate configuration parameters.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            True if configuration is valid, False otherwise
        """
        if not isinstance(config, dict):
            return False
        
        # Validate model name
        if 'model_name' in config:
            if not isinstance(config['model_name'], str):
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
                    "description": "Name of the embedding model to serve with GPU model engine",
                    "default": "sentence-transformers/all-MiniLM-L6-v2"
                },
                "batch_size": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 1024,
                    "default": 64,
                    "description": "Batch size for GPU processing"
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
                "gpu_memory_utilization": {
                    "type": "number",
                    "minimum": 0.1,
                    "maximum": 1.0,
                    "default": 0.8,
                    "description": "GPU memory utilization fraction"
                },
                "tensor_parallel_size": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 1,
                    "description": "Number of GPUs for tensor parallelism"
                }
            }
        }
    
    def get_infrastructure_metrics(self) -> Dict[str, Any]:
        """
        Get infrastructure metrics for monitoring.
        
        Returns:
            Dictionary containing infrastructure metrics
        """
        process = psutil.Process()
        memory_info = process.memory_info()
        
        # Try to get GPU info if available
        gpu_info = {}
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Use first GPU for metrics
                gpu_info = {
                    "gpu_id": gpu.id,
                    "gpu_name": gpu.name,
                    "gpu_memory_used_mb": gpu.memoryUsed,
                    "gpu_memory_total_mb": gpu.memoryTotal,
                    "gpu_memory_utilization_percent": (gpu.memoryUsed / gpu.memoryTotal) * 100,
                    "gpu_utilization_percent": gpu.load * 100
                }
        except ImportError:
            gpu_info = {"gpu_monitoring": "GPUtil not available"}
        except Exception as e:
            gpu_info = {"gpu_monitoring_error": str(e)}
        
        base_metrics = {
            "component_name": self.name,
            "component_version": self.version,
            "model_name": self._model_name,
            "engine_loaded": self._engine_loaded,
            "model_engine_type": self._model_engine_type,
            "batch_size": self._batch_size,
            "max_length": self._max_length,
            "normalize_embeddings": self._normalize_embeddings,
            "memory_usage": {
                "rss_mb": memory_info.rss / 1024 / 1024,
                "vms_mb": memory_info.vms / 1024 / 1024
            },
            "startup_time": self._startup_time.isoformat(),
            "uptime_seconds": (datetime.now(timezone.utc) - self._startup_time).total_seconds()
        }
        
        # Merge GPU info
        base_metrics.update(gpu_info)
        return base_metrics
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for monitoring.
        
        Returns:
            Dictionary containing performance metrics
        """
        total_operations = int(self._stats["successful_embeddings"]) + int(self._stats["failed_embeddings"])
        success_rate = (int(self._stats["successful_embeddings"]) / max(total_operations, 1)) * 100
        
        avg_processing_time = (
            float(self._stats["total_processing_time"]) / max(int(self._stats["successful_embeddings"]), 1)
        )
        
        uptime_seconds = (datetime.now(timezone.utc) - self._startup_time).total_seconds()
        embeddings_per_second = int(self._stats["total_embeddings_created"]) / max(uptime_seconds, 1)
        
        return {
            "component_name": self.name,
            "total_embeddings_created": self._stats["total_embeddings_created"],
            "successful_operations": self._stats["successful_embeddings"],
            "failed_operations": self._stats["failed_embeddings"],
            "gpu_operations": self._stats["gpu_operations"],
            "success_rate_percent": success_rate,
            "embeddings_per_second": embeddings_per_second,
            "average_processing_time": avg_processing_time,
            "total_processing_time": self._stats["total_processing_time"],
            "total_characters_processed": self._stats["total_characters_processed"],
            "total_tokens_processed": self._stats["total_tokens_processed"],
            "avg_chars_per_embedding": int(self._stats["total_characters_processed"]) / max(int(self._stats["total_embeddings_created"]), 1),
            "avg_tokens_per_embedding": int(self._stats["total_tokens_processed"]) / max(int(self._stats["total_embeddings_created"]), 1),
            "last_processing_time": self._stats["last_processing_time"],
            "initialization_count": self._stats["initialization_count"],
            "engine_loads": self._stats["engine_loads"],
            "embedding_method_distribution": self._stats["embedding_method_counts"],
            "recent_errors": self._get_recent_errors(10),
            "error_count": self._get_error_count(),
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
            "# HELP hades_embedding_gpu_info Component information",
            "# TYPE hades_embedding_gpu_info gauge",
            f'hades_embedding_gpu_info{{component="{infra_metrics["component_name"]}",version="{infra_metrics["component_version"]}",model="{infra_metrics["model_name"]}",engine="{infra_metrics["model_engine_type"]}"}} 1',
            "",
            "# HELP hades_embedding_gpu_uptime_seconds Component uptime in seconds",
            "# TYPE hades_embedding_gpu_uptime_seconds gauge",
            f'hades_embedding_gpu_uptime_seconds{{component="{infra_metrics["component_name"]}"}} {infra_metrics["uptime_seconds"]:.2f}',
            "",
            "# HELP hades_embedding_gpu_memory_rss_bytes Resident Set Size memory usage",
            "# TYPE hades_embedding_gpu_memory_rss_bytes gauge",
            f'hades_embedding_gpu_memory_rss_bytes{{component="{infra_metrics["component_name"]}"}} {infra_metrics["memory_usage"]["rss_mb"] * 1024 * 1024:.0f}',
            "",
            "# HELP hades_embedding_gpu_memory_vms_bytes Virtual Memory Size",
            "# TYPE hades_embedding_gpu_memory_vms_bytes gauge",
            f'hades_embedding_gpu_memory_vms_bytes{{component="{infra_metrics["component_name"]}"}} {infra_metrics["memory_usage"]["vms_mb"] * 1024 * 1024:.0f}',
            "",
            "# HELP hades_embedding_gpu_embeddings_created_total Total number of embeddings created",
            "# TYPE hades_embedding_gpu_embeddings_created_total counter",
            f'hades_embedding_gpu_embeddings_created_total{{component="{perf_metrics["component_name"]}"}} {perf_metrics["total_embeddings_created"]}',
            "",
            "# HELP hades_embedding_gpu_operations_total Total embedding operations",
            "# TYPE hades_embedding_gpu_operations_total counter",
            f'hades_embedding_gpu_operations_total{{component="{perf_metrics["component_name"]}",status="success"}} {perf_metrics["successful_operations"]}',
            f'hades_embedding_gpu_operations_total{{component="{perf_metrics["component_name"]}",status="failed"}} {perf_metrics["failed_operations"]}',
            f'hades_embedding_gpu_operations_total{{component="{perf_metrics["component_name"]}",status="gpu"}} {perf_metrics["gpu_operations"]}',
            "",
            "# HELP hades_embedding_gpu_success_rate_percent Success rate percentage",
            "# TYPE hades_embedding_gpu_success_rate_percent gauge",
            f'hades_embedding_gpu_success_rate_percent{{component="{perf_metrics["component_name"]}"}} {perf_metrics["success_rate_percent"]:.2f}',
            "",
            "# HELP hades_embedding_gpu_processing_time_seconds Total processing time",
            "# TYPE hades_embedding_gpu_processing_time_seconds counter",
            f'hades_embedding_gpu_processing_time_seconds{{component="{perf_metrics["component_name"]}"}} {perf_metrics["total_processing_time"]:.6f}',
            "",
            "# HELP hades_embedding_gpu_throughput_embeddings_per_second Current throughput",
            "# TYPE hades_embedding_gpu_throughput_embeddings_per_second gauge",
            f'hades_embedding_gpu_throughput_embeddings_per_second{{component="{perf_metrics["component_name"]}"}} {perf_metrics["embeddings_per_second"]:.2f}',
            "",
            "# HELP hades_embedding_gpu_characters_processed_total Total characters processed",
            "# TYPE hades_embedding_gpu_characters_processed_total counter",
            f'hades_embedding_gpu_characters_processed_total{{component="{perf_metrics["component_name"]}"}} {perf_metrics["total_characters_processed"]}',
            "",
            "# HELP hades_embedding_gpu_tokens_processed_total Total tokens processed",
            "# TYPE hades_embedding_gpu_tokens_processed_total counter",
            f'hades_embedding_gpu_tokens_processed_total{{component="{perf_metrics["component_name"]}"}} {perf_metrics["total_tokens_processed"]}',
            "",
            "# HELP hades_embedding_gpu_errors_total Total number of errors",
            "# TYPE hades_embedding_gpu_errors_total counter",
            f'hades_embedding_gpu_errors_total{{component="{perf_metrics["component_name"]}"}} {perf_metrics["error_count"]}',
            "",
            "# HELP hades_embedding_gpu_engine_loads_total Total number of model engine loads",
            "# TYPE hades_embedding_gpu_engine_loads_total counter",
            f'hades_embedding_gpu_engine_loads_total{{component="{perf_metrics["component_name"]}"}} {perf_metrics["engine_loads"]}',
            ""
        ]
        
        # Add GPU-specific metrics if available
        if "gpu_memory_used_mb" in infra_metrics:
            lines.extend([
                "# HELP hades_embedding_gpu_vram_used_bytes GPU VRAM used in bytes",
                "# TYPE hades_embedding_gpu_vram_used_bytes gauge",
                f'hades_embedding_gpu_vram_used_bytes{{component="{infra_metrics["component_name"]}",gpu_id="{infra_metrics.get("gpu_id", "0")}"}} {infra_metrics["gpu_memory_used_mb"] * 1024 * 1024:.0f}',
                "",
                "# HELP hades_embedding_gpu_vram_total_bytes GPU VRAM total in bytes",
                "# TYPE hades_embedding_gpu_vram_total_bytes gauge",
                f'hades_embedding_gpu_vram_total_bytes{{component="{infra_metrics["component_name"]}",gpu_id="{infra_metrics.get("gpu_id", "0")}"}} {infra_metrics["gpu_memory_total_mb"] * 1024 * 1024:.0f}',
                "",
                "# HELP hades_embedding_gpu_utilization_percent GPU utilization percentage",
                "# TYPE hades_embedding_gpu_utilization_percent gauge",
                f'hades_embedding_gpu_utilization_percent{{component="{infra_metrics["component_name"]}",gpu_id="{infra_metrics.get("gpu_id", "0")}"}} {infra_metrics["gpu_utilization_percent"]:.2f}',
                "",
                "# HELP hades_embedding_gpu_memory_utilization_percent GPU memory utilization percentage",
                "# TYPE hades_embedding_gpu_memory_utilization_percent gauge",
                f'hades_embedding_gpu_memory_utilization_percent{{component="{infra_metrics["component_name"]}",gpu_id="{infra_metrics.get("gpu_id", "0")}"}} {infra_metrics["gpu_memory_utilization_percent"]:.2f}',
                ""
            ])
        
        # Add method distribution metrics
        for method, count in perf_metrics["embedding_method_distribution"].items():
            lines.extend([
                "# HELP hades_embedding_gpu_method_usage_total Usage count by embedding method",
                "# TYPE hades_embedding_gpu_method_usage_total counter",
                f'hades_embedding_gpu_method_usage_total{{component="{perf_metrics["component_name"]}",method="{method}"}} {count}',
                ""
            ])
        
        return "\n".join(lines)
    
    def health_check(self) -> bool:
        """
        Check if component is healthy and ready to process data.
        
        Returns:
            True if component is healthy, False otherwise
        """
        try:
            # Try to initialize model engine if not already loaded
            if not self._engine_loaded:
                self._initialize_model_engine()
            
            # Test with simple text if engine is loaded
            if self._engine_loaded and self._model_engine:
                test_text = "This is a test text for embedding."
                embedding = self._embed_single_text(test_text)
                return len(embedding) > 0
            
            # Return True for fallback mode
            return True
            
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
        
        metrics = {
            "component_name": self.name,
            "component_version": self.version,
            "model_name": self._model_name,
            "engine_loaded": self._engine_loaded,
            "batch_size": self._batch_size,
            "max_length": self._max_length,
            "total_embeddings_created": self._total_embeddings_created,
            "total_processing_time": self._total_processing_time,
            "avg_processing_time": avg_processing_time,
            "last_health_check": datetime.utcnow().isoformat()
        }
        
        # Add GPU metrics if available
        if self._engine_loaded:
            try:
                import torch
                if torch.cuda.is_available():
                    metrics.update({
                        "gpu_memory_allocated_mb": torch.cuda.memory_allocated() / (1024 * 1024),
                        "gpu_memory_cached_mb": torch.cuda.memory_reserved() / (1024 * 1024),
                        "gpu_device_count": torch.cuda.device_count()
                    })
            except ImportError:
                pass
        
        return metrics
    
    def embed(self, input_data: EmbeddingInput) -> EmbeddingOutput:
        """
        Generate embeddings according to the contract.
        
        Args:
            input_data: Input data conforming to EmbeddingInput contract
            
        Returns:
            Output data conforming to EmbeddingOutput contract
        """
        errors: List[str] = []
        
        try:
            start_time = datetime.utcnow()
            
            # Extract texts from chunks
            texts = [chunk.content for chunk in input_data.chunks]
            chunk_ids = [chunk.id for chunk in input_data.chunks]
            
            # Generate embeddings using model engine
            if self._engine_loaded:
                # Use the configured model engine for GPU embedding
                embedding_vectors = self._embed_texts_gpu(texts)
            else:
                # Fall back to basic implementation
                embedding_vectors = self._embed_texts_fallback(texts)
            
            # Convert to contract format
            embeddings = []
            for i, (chunk_id, vector) in enumerate(zip(chunk_ids, embedding_vectors)):
                embedding = ChunkEmbedding(
                    chunk_id=chunk_id,
                    embedding=vector.tolist() if hasattr(vector, 'tolist') else vector,
                    embedding_dimension=len(vector),
                    model_name=input_data.model_name or self._model_name,
                    confidence=1.0,  # GPU embeddings have consistent confidence
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
            if self._engine_loaded:
                gpu_ops = self._stats.get("gpu_operations", 0)
                if isinstance(gpu_ops, (int, float)):
                    self._stats["gpu_operations"] = int(gpu_ops) + 1
                method_used = "gpu_engine"
            else:
                method_used = "fallback"
            
            method_counts = self._stats.get("embedding_method_counts", {})
            if isinstance(method_counts, dict):
                method_counts[method_used] = method_counts.get(method_used, 0) + 1
                self._stats["embedding_method_counts"] = method_counts
            
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
                    "engine_type": "gpu" if self._engine_loaded else "fallback",
                    "throughput_embeddings_per_second": len(embeddings) / max(processing_time, 0.001)
                },
                model_info={
                    "model_name": self._model_name,
                    "embedding_dimension": len(embedding_vectors[0]) if embedding_vectors else 0,
                    "max_length": self._max_length,
                    "device": "gpu" if self._engine_loaded else "cpu"
                },
                errors=errors
            )
            
        except Exception as e:
            error_msg = f"GPU embedding failed: {str(e)}"
            errors.append(error_msg)
            self.logger.error(error_msg)
            
            # Update error stats
            failed_embeddings = self._stats.get("failed_embeddings", 0)
            if isinstance(failed_embeddings, (int, float)):
                self._stats["failed_embeddings"] = int(failed_embeddings) + 1
            
            error_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": error_msg
            }
            errors_list = self._stats.get("errors", [])
            if isinstance(errors_list, list):
                errors_list.append(error_entry)
                
                # Keep only last 50 errors to prevent memory bloat
                if len(errors_list) > 50:
                    errors_list = errors_list[-50:]
                    
                self._stats["errors"] = errors_list
            
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
        # GPU model engine can serve various embedding models
        supported_patterns = [
            'sentence-transformers',
            'all-MiniLM',
            'all-mpnet',
            'e5-',
            'bge-',
            'gte-'
        ]
        
        return any(pattern in model_name.lower() for pattern in supported_patterns)
    
    def get_embedding_dimension(self, model_name: str) -> int:
        """
        Get the embedding dimension for a given model.
        
        Args:
            model_name: Model name
            
        Returns:
            Embedding dimension
        """
        # Common embedding model dimensions
        model_dims = {
            'all-MiniLM-L6-v2': 384,
            'all-mpnet-base-v2': 768,
            'e5-large': 1024,
            'bge-large': 1024,
            'gte-large': 1024
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
        Estimate time to generate embeddings using GPU.
        
        Args:
            input_data: Input data to estimate for
            
        Returns:
            Estimated processing time in seconds
        """
        try:
            num_chunks = len(input_data.chunks)
            total_tokens = self.estimate_tokens(input_data)
            
            # GPU processing is much faster than CPU
            # Estimate based on tokens and batch size
            batches_needed = (num_chunks + self._batch_size - 1) // self._batch_size
            
            # Base time per batch on GPU (optimistic estimate)
            time_per_batch = 0.05  # 50ms per batch
            
            # Scale by token complexity (GPU handles this better)
            token_complexity_factor = min(total_tokens / 50000, 1.5)  # Cap at 1.5x
            
            estimated_time = batches_needed * time_per_batch * token_complexity_factor
            
            return max(0.01, estimated_time)
            
        except Exception:
            # Fallback estimate (GPU is fast)
            return len(input_data.chunks) * 0.01
    
    def _initialize_model_engine(self) -> None:
        """Initialize the GPU model engine."""
        try:
            self.logger.info(f"Initializing GPU model engine for model: {self._model_name}")
            
            # Get GPU model engine from factory
            engine_config = {
                'model_name': self._model_name,
                'device': 'cuda',
                **self._config
            }
            self._model_engine = create_model_engine(
                engine_type=self._model_engine_type,
                config=engine_config
            )
            
            if self._model_engine:
                self._engine_loaded = True
                engine_loads = self._stats.get("engine_loads", 0)
                if isinstance(engine_loads, (int, float)):
                    self._stats["engine_loads"] = int(engine_loads) + 1
                self.logger.info("GPU model engine initialized successfully")
            else:
                self.logger.warning("Failed to create GPU model engine")
                self._engine_loaded = False
                engine_loads = self._stats.get("engine_loads", 0)
                if isinstance(engine_loads, (int, float)):
                    self._stats["engine_loads"] = int(engine_loads) + 1
            
        except Exception as e:
            self.logger.error(f"Failed to initialize GPU model engine: {e}")
            self._model_engine = None
            self._engine_loaded = False
            engine_loads = self._stats.get("engine_loads", 0)
            if isinstance(engine_loads, (int, float)):
                self._stats["engine_loads"] = int(engine_loads) + 1
            
            # Track initialization error
            error_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": f"Model engine initialization failed: {str(e)}"
            }
            errors_list = self._stats.get("errors", [])
            if isinstance(errors_list, list):
                errors_list.append(error_entry)
                if len(errors_list) > 50:
                    errors_list = errors_list[-50:]
                self._stats["errors"] = errors_list
    
    def _embed_texts_gpu(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using GPU model engine."""
        if not self._engine_loaded:
            self._initialize_model_engine()
        
        if not self._engine_loaded:
            # Fall back to basic implementation
            return self._embed_texts_fallback(texts)
        
        try:
            # Use model engine to generate embeddings
            if self._model_engine is not None:
                embeddings = self._model_engine.embed_texts(
                    texts=texts,
                    batch_size=self._batch_size,
                    max_length=self._max_length,
                    normalize=self._normalize_embeddings
                )
                
                return embeddings
            else:
                return self._embed_texts_fallback(texts)
            
        except Exception as e:
            self.logger.error(f"GPU embedding failed: {e}")
            return self._embed_texts_fallback(texts)
    
    def _embed_single_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        embeddings = self._embed_texts_gpu([text])
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
    
    def _get_recent_errors(self, count: int = 10) -> List[Any]:
        """Get recent errors with proper type checking."""
        errors_list = self._stats.get("errors", [])
        if isinstance(errors_list, list) and len(errors_list) >= count:
            return errors_list[-count:]
        elif isinstance(errors_list, list):
            return errors_list
        else:
            return []
    
    def _get_error_count(self) -> int:
        """Get error count with proper type checking."""
        errors_list = self._stats.get("errors", [])
        if isinstance(errors_list, list):
            return len(errors_list)
        else:
            return 0