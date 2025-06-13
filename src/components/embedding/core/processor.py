"""
Core Embedding Component

This module provides the core embedding component that implements the
Embedder protocol. It acts as a factory and coordinator for different
embedding strategies in the new component architecture.
"""

import logging
import psutil
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timezone
from dataclasses import dataclass, field

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

# Import embedder implementations
from ..cpu.processor import CPUEmbedder
from ..gpu.processor import GPUEmbedder
from ..encoder.processor import EncoderEmbedder


@dataclass
class EmbeddingStats:
    """Typed statistics for embedding operations."""
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    delegated_operations: int = 0
    total_processing_time: float = 0.0
    total_embeddings_created: int = 0
    last_processing_time: Optional[float] = None
    initialization_count: int = 1
    errors: List[Dict[str, Any]] = field(default_factory=list)
    embedder_usage_counts: Dict[str, int] = field(default_factory=dict)


class CoreEmbedder(Embedder):
    """
    Core embedder component implementing Embedder protocol.
    
    This component acts as a factory and coordinator for different embedding
    strategies, selecting the appropriate embedder based on configuration.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize core embedder.
        
        Args:
            config: Optional configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self._config = config or {}
        
        # Component metadata
        self._metadata = ComponentMetadata(
            component_type=ComponentType.EMBEDDING,
            component_name="core",
            component_version="1.0.0",
            config=self._config
        )
        
        # Get default embedder type
        self._default_embedder_type = self._config.get('default_embedder', 'cpu')
        
        # Available embedder types
        self._embedder_types = {
            'cpu': CPUEmbedder,
            'gpu': GPUEmbedder,
            'encoder': EncoderEmbedder
        }
        
        # Cache for embedder instances
        self._embedder_cache: Dict[str, Embedder] = {}
        
        # Monitoring integration - standardized stats tracking
        self._startup_time = datetime.now(timezone.utc)
        self._stats = EmbeddingStats()
        
        self.logger.info(f"Initialized core embedder with default embedder: {self._default_embedder_type}")
    
    @property
    def name(self) -> str:
        """Component name for identification."""
        return "core"
    
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
        
        # Update default embedder if specified
        if 'default_embedder' in config:
            self._default_embedder_type = config['default_embedder']
        
        # Clear embedder cache to force re-initialization
        self._embedder_cache.clear()
        
        self.logger.info(f"Updated core embedder configuration")
    
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
        
        # Validate default embedder if provided
        if 'default_embedder' in config:
            embedder_type = config['default_embedder']
            if not isinstance(embedder_type, str) or embedder_type not in self._embedder_types:
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
                "default_embedder": {
                    "type": "string",
                    "description": "Default embedder type to use",
                    "enum": list(self._embedder_types.keys()),
                    "default": "cpu"
                },
                "model_settings": {
                    "type": "object",
                    "description": "Settings for embedding models",
                    "properties": {
                        "model_name": {
                            "type": "string",
                            "description": "Name of the embedding model",
                            "default": "all-MiniLM-L6-v2"
                        },
                        "embedding_dimension": {
                            "type": "integer",
                            "description": "Expected embedding dimension",
                            "minimum": 1,
                            "default": 384
                        },
                        "batch_size": {
                            "type": "integer",
                            "description": "Batch size for processing",
                            "minimum": 1,
                            "default": 32
                        }
                    }
                },
                "processing_options": {
                    "type": "object",
                    "description": "Default processing options",
                    "properties": {
                        "normalize": {
                            "type": "boolean",
                            "description": "Whether to normalize embeddings",
                            "default": True
                        },
                        "pooling": {
                            "type": "string",
                            "description": "Pooling strategy",
                            "enum": ["mean", "max", "cls"],
                            "default": "mean"
                        }
                    }
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
            # Check if we can get an embedder instance
            embedder = self._get_embedder(self._default_embedder_type)
            if not embedder:
                return False
            
            # Check embedder health
            return embedder.health_check()
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get component performance metrics.
        
        Returns:
            Dictionary containing performance metrics
        """
        return {
            "component_name": self.name,
            "component_version": self.version,
            "default_embedder": self._default_embedder_type,
            "available_embedders": list(self._embedder_types.keys()),
            "cached_embedders": list(self._embedder_cache.keys()),
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
        try:
            start_time = datetime.utcnow()
            
            # Get embedder type from options or use default
            embedder_type = input_data.embedding_options.get('embedder_type', self._default_embedder_type)
            
            # Get appropriate embedder
            embedder = self._get_embedder(embedder_type)
            if not embedder:
                raise ValueError(f"Could not get embedder: {embedder_type}")
            
            # Delegate to specific embedder
            result = embedder.embed(input_data)
            
            # Update metadata with core processing info
            core_processing_time = (datetime.utcnow() - start_time).total_seconds()
            result.metadata.processing_time = core_processing_time
            result.embedding_stats["core_processing_time"] = core_processing_time
            result.embedding_stats["delegated_to"] = embedder_type
            
            # Update stats
            self._stats.total_operations += 1
            self._stats.successful_operations += 1
            self._stats.delegated_operations += 1
            self._stats.total_processing_time += core_processing_time
            self._stats.total_embeddings_created += len(result.embeddings)
            self._stats.last_processing_time = core_processing_time
            
            # Track embedder usage
            self._stats.embedder_usage_counts[embedder_type] = self._stats.embedder_usage_counts.get(embedder_type, 0) + 1
            
            return result
            
        except Exception as e:
            error_msg = f"Core embedding failed: {str(e)}"
            self.logger.error(error_msg)
            
            # Update error stats
            self._stats.total_operations += 1
            self._stats.failed_operations += 1
            error_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": error_msg
            }
            self._stats.errors.append(error_entry)
            
            # Keep only last 50 errors to prevent memory bloat
            if len(self._stats.errors) > 50:
                self._stats.errors = self._stats.errors[-50:]
            
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
                errors=[error_msg]
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
            # Get embedder type from options or use default
            embedder_type = input_data.embedding_options.get('embedder_type', self._default_embedder_type)
            
            # Get appropriate embedder
            embedder = self._get_embedder(embedder_type)
            if embedder:
                return embedder.estimate_tokens(input_data)
            
            # Fallback estimation
            total_chars = sum(len(chunk.content) for chunk in input_data.chunks)
            return max(1, total_chars // 4)  # Simple estimation: ~4 chars per token
            
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
        try:
            # Check with all available embedders
            for embedder_type in self._embedder_types.keys():
                embedder = self._get_embedder(embedder_type)
                if embedder and embedder.supports_model(model_name):
                    return True
            
            return False
            
        except Exception:
            return False
    
    def get_embedding_dimension(self, model_name: str) -> int:
        """
        Get the embedding dimension for a given model.
        
        Args:
            model_name: Model name
            
        Returns:
            Embedding dimension
        """
        try:
            # Try with default embedder first
            embedder = self._get_embedder(self._default_embedder_type)
            if embedder:
                return embedder.get_embedding_dimension(model_name)
            
            # Common model dimensions as fallback
            model_dims = {
                'all-MiniLM-L6-v2': 384,
                'all-mpnet-base-v2': 768,
                'text-embedding-ada-002': 1536,
                'sentence-transformers': 384
            }
            
            for model, dim in model_dims.items():
                if model in model_name.lower():
                    return dim
            
            return 384  # Default dimension
            
        except Exception:
            return 384  # Default fallback
    
    def get_supported_embedders(self) -> List[str]:
        """
        Get list of supported embedder types.
        
        Returns:
            List of supported embedder names
        """
        return list(self._embedder_types.keys())
    
    @property
    def embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this embedder."""
        model_name = self._config.get('model_settings', {}).get('model_name', 'all-MiniLM-L6-v2')
        return self.get_embedding_dimension(model_name)
    
    @property
    def max_sequence_length(self) -> int:
        """Get the maximum sequence length this embedder can handle."""
        return self._config.get('model_settings', {}).get('max_length', 512)
    
    def supports_batch_processing(self) -> bool:
        """Check if embedder supports batch processing."""
        return True  # Core embedder always supports batch processing through delegation
    
    def get_optimal_batch_size(self) -> int:
        """Get the optimal batch size for this embedder."""
        return self._config.get('model_settings', {}).get('batch_size', 32)
    
    def estimate_embedding_time(self, input_data: EmbeddingInput) -> float:
        """
        Estimate time to generate embeddings.
        
        Args:
            input_data: Input data to estimate for
            
        Returns:
            Estimated processing time in seconds
        """
        try:
            # Get embedder type from options or use default
            embedder_type = input_data.embedding_options.get('embedder_type', self._default_embedder_type)
            
            # Get appropriate embedder and delegate estimation
            embedder = self._get_embedder(embedder_type)
            if embedder and hasattr(embedder, 'estimate_embedding_time'):
                return embedder.estimate_embedding_time(input_data)
            
            # Fallback estimation based on chunk count and embedder type
            num_chunks = len(input_data.chunks)
            
            # Base time estimates by embedder type
            time_per_chunk = {
                'cpu': 0.1,      # 100ms per chunk
                'gpu': 0.01,     # 10ms per chunk  
                'encoder': 0.2   # 200ms per chunk
            }.get(embedder_type, 0.1)
            
            estimated_time = num_chunks * time_per_chunk
            
            # Add small overhead for core processing
            core_overhead = 0.01  # 10ms overhead
            
            return max(0.01, estimated_time + core_overhead)
            
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
            "default_embedder": self._default_embedder_type,
            "available_embedders": list(self._embedder_types.keys()),
            "cached_embedders": list(self._embedder_cache.keys()),
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
        total_operations = self._stats.successful_operations + self._stats.failed_operations
        success_rate = (self._stats.successful_operations / max(total_operations, 1)) * 100
        
        avg_processing_time = (
            self._stats.total_processing_time / max(self._stats.successful_operations, 1)
        )
        
        uptime_seconds = (datetime.now(timezone.utc) - self._startup_time).total_seconds()
        operations_per_second = total_operations / max(uptime_seconds, 1)
        
        return {
            "component_name": self.name,
            "total_embeddings_created": self._stats.total_embeddings_created,
            "successful_operations": self._stats.successful_operations,
            "failed_operations": self._stats.failed_operations,
            "delegated_operations": self._stats.delegated_operations,
            "success_rate_percent": success_rate,
            "operations_per_second": operations_per_second,
            "average_processing_time": avg_processing_time,
            "total_processing_time": self._stats.total_processing_time,
            "last_processing_time": self._stats.last_processing_time,
            "initialization_count": self._stats.initialization_count,
            "embedder_usage_distribution": self._stats.embedder_usage_counts,
            "recent_errors": self._stats.errors[-10:] if self._stats.errors else [],
            "error_count": len(self._stats.errors),
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
            "# HELP hades_embedding_core_info Component information",
            "# TYPE hades_embedding_core_info gauge",
            f'hades_embedding_core_info{{component="{infra_metrics["component_name"]}",version="{infra_metrics["component_version"]}",default_embedder="{infra_metrics["default_embedder"]}"}} 1',
            "",
            "# HELP hades_embedding_core_uptime_seconds Component uptime in seconds",
            "# TYPE hades_embedding_core_uptime_seconds gauge",
            f'hades_embedding_core_uptime_seconds{{component="{infra_metrics["component_name"]}"}} {infra_metrics["uptime_seconds"]:.2f}',
            "",
            "# HELP hades_embedding_core_memory_rss_bytes Resident Set Size memory usage",
            "# TYPE hades_embedding_core_memory_rss_bytes gauge",
            f'hades_embedding_core_memory_rss_bytes{{component="{infra_metrics["component_name"]}"}} {infra_metrics["memory_usage"]["rss_mb"] * 1024 * 1024:.0f}',
            "",
            "# HELP hades_embedding_core_memory_vms_bytes Virtual Memory Size",
            "# TYPE hades_embedding_core_memory_vms_bytes gauge",
            f'hades_embedding_core_memory_vms_bytes{{component="{infra_metrics["component_name"]}"}} {infra_metrics["memory_usage"]["vms_mb"] * 1024 * 1024:.0f}',
            "",
            "# HELP hades_embedding_core_embeddings_created_total Total number of embeddings created",
            "# TYPE hades_embedding_core_embeddings_created_total counter",
            f'hades_embedding_core_embeddings_created_total{{component="{perf_metrics["component_name"]}"}} {perf_metrics["total_embeddings_created"]}',
            "",
            "# HELP hades_embedding_core_operations_total Total operations",
            "# TYPE hades_embedding_core_operations_total counter",
            f'hades_embedding_core_operations_total{{component="{perf_metrics["component_name"]}",status="success"}} {perf_metrics["successful_operations"]}',
            f'hades_embedding_core_operations_total{{component="{perf_metrics["component_name"]}",status="failed"}} {perf_metrics["failed_operations"]}',
            f'hades_embedding_core_operations_total{{component="{perf_metrics["component_name"]}",status="delegated"}} {perf_metrics["delegated_operations"]}',
            "",
            "# HELP hades_embedding_core_success_rate_percent Success rate percentage",
            "# TYPE hades_embedding_core_success_rate_percent gauge",
            f'hades_embedding_core_success_rate_percent{{component="{perf_metrics["component_name"]}"}} {perf_metrics["success_rate_percent"]:.2f}',
            "",
            "# HELP hades_embedding_core_processing_time_seconds Total processing time",
            "# TYPE hades_embedding_core_processing_time_seconds counter",
            f'hades_embedding_core_processing_time_seconds{{component="{perf_metrics["component_name"]}"}} {perf_metrics["total_processing_time"]:.6f}',
            "",
            "# HELP hades_embedding_core_throughput_operations_per_second Current throughput",
            "# TYPE hades_embedding_core_throughput_operations_per_second gauge",
            f'hades_embedding_core_throughput_operations_per_second{{component="{perf_metrics["component_name"]}"}} {perf_metrics["operations_per_second"]:.2f}',
            "",
            "# HELP hades_embedding_core_errors_total Total number of errors",
            "# TYPE hades_embedding_core_errors_total counter",
            f'hades_embedding_core_errors_total{{component="{perf_metrics["component_name"]}"}} {perf_metrics["error_count"]}',
            ""
        ]
        
        # Add embedder usage distribution metrics
        for embedder_type, count in perf_metrics["embedder_usage_distribution"].items():
            lines.extend([
                "# HELP hades_embedding_core_embedder_usage_total Usage count by embedder type",
                "# TYPE hades_embedding_core_embedder_usage_total counter",
                f'hades_embedding_core_embedder_usage_total{{component="{perf_metrics["component_name"]}",embedder_type="{embedder_type}"}} {count}',
                ""
            ])
        
        # Add cached embedders count
        lines.extend([
            "# HELP hades_embedding_core_cached_embedders_count Number of cached embedder instances",
            "# TYPE hades_embedding_core_cached_embedders_count gauge",
            f'hades_embedding_core_cached_embedders_count{{component="{infra_metrics["component_name"]}"}} {len(infra_metrics["cached_embedders"])}',
            ""
        ])
        
        return "\n".join(lines)
    
    def _get_embedder(self, embedder_type: str) -> Optional[Embedder]:
        """Get an embedder instance of the specified type."""
        if embedder_type in self._embedder_cache:
            return self._embedder_cache[embedder_type]
        
        if embedder_type not in self._embedder_types:
            self.logger.error(f"Unknown embedder type: {embedder_type}")
            return None
        
        try:
            # Create embedder instance with appropriate config
            embedder_config = self._config.copy()
            
            # Remove core-specific configs that shouldn't be passed to specific embedders
            embedder_config.pop('default_embedder', None)
            
            embedder_cls = self._embedder_types[embedder_type]
            embedder = embedder_cls(config=embedder_config)
            
            # Cache the embedder
            self._embedder_cache[embedder_type] = embedder
            
            return embedder
            
        except Exception as e:
            self.logger.error(f"Could not create embedder {embedder_type}: {e}")
            
            # Track embedder creation error
            error_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": f"Failed to create embedder {embedder_type}: {str(e)}"
            }
            self._stats.errors.append(error_entry)
            if len(self._stats.errors) > 50:
                self._stats.errors = self._stats.errors[-50:]
            
            return None