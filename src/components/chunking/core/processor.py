"""
Core Chunking Component

This module provides the core chunking component that implements the
Chunker protocol. It acts as a factory and coordinator for different
chunking strategies in the new component architecture.
"""

import logging
import psutil
import time
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timezone

# Import component contracts and protocols
from src.types.components.contracts import (
    ComponentType,
    ComponentMetadata,
    ChunkingInput,
    ChunkingOutput,
    TextChunk,
    ProcessingStatus
)
from src.types.components.protocols import Chunker

# Import chunker implementations
from ..chunkers.cpu.processor import CPUChunker
from ..chunkers.chonky.processor import ChonkyChunker
from ..chunkers.code.processor import CodeChunker
from ..chunkers.text.processor import TextChunker
from ..chunkers.ast.processor import ASTAwareCodeChunker


class CoreChunker(Chunker):
    """
    Core chunker component implementing Chunker protocol.
    
    This component acts as a factory and coordinator for different chunking
    strategies, selecting the appropriate chunker based on configuration.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize core chunker.
        
        Args:
            config: Optional configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self._config = config or {}
        
        # Component metadata
        self._metadata = ComponentMetadata(
            component_type=ComponentType.CHUNKING,
            component_name="core",
            component_version="1.0.0",
            config=self._config
        )
        
        # Get default chunker type
        self._default_chunker_type = self._config.get('default_chunker', 'cpu')
        
        # Available chunker types
        self._chunker_types = {
            'cpu': CPUChunker,
            'chonky': ChonkyChunker,
            'code': CodeChunker,
            'text': TextChunker,
            'ast': ASTAwareCodeChunker
        }
        
        # Cache for chunker instances
        self._chunker_cache: Dict[str, Chunker] = {}
        
        # Monitoring and metrics tracking
        self._stats: Dict[str, Any] = {
            "total_chunks_created": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "total_processing_time": 0.0,
            "delegated_operations": 0,
            "last_processing_time": None,
            "initialization_count": 1,
            "errors": [],
            "chunker_usage": {}  # Track usage of different chunker types
        }
        self._startup_time = datetime.now(timezone.utc)
        
        self.logger.info(f"Initialized core chunker with default chunker: {self._default_chunker_type}")
    
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
        return ComponentType.CHUNKING
    
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
        self._metadata.processed_at = datetime.now(timezone.utc)
        
        # Update default chunker if specified
        if 'default_chunker' in config:
            self._default_chunker_type = config['default_chunker']
        
        # Clear chunker cache to force re-initialization
        self._chunker_cache.clear()
        
        self.logger.info(f"Updated core chunker configuration")
    
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
        
        # Validate default chunker if provided
        if 'default_chunker' in config:
            chunker_type = config['default_chunker']
            if not isinstance(chunker_type, str) or chunker_type not in self._chunker_types:
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
                "default_chunker": {
                    "type": "string",
                    "description": "Default chunker type to use",
                    "enum": list(self._chunker_types.keys()),
                    "default": "cpu"
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
            # Check if we can get a chunker instance
            chunker = self._get_chunker(self._default_chunker_type)
            if not chunker:
                return False
            
            # Test basic chunking functionality
            test_input = ChunkingInput(
                text="This is a test document with some content for chunking.",
                document_id="health_check",
                chunk_size=50,
                chunk_overlap=10
            )
            
            result = chunker.chunk(test_input)
            return len(result.chunks) > 0
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False
    
    def get_infrastructure_metrics(self) -> Dict[str, Any]:
        """
        Get infrastructure and resource inventory metrics.
        
        Returns:
            Dictionary containing infrastructure metrics
        """
        try:
            # Get memory usage
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                "component_name": self.name,
                "component_version": self.version,
                "default_chunker": self._default_chunker_type,
                "available_chunkers": list(self._chunker_types.keys()),
                "cached_chunkers": list(self._chunker_cache.keys()),
                "memory_usage": {
                    "rss_mb": round(memory_info.rss / 1024 / 1024, 2),
                    "vms_mb": round(memory_info.vms / 1024 / 1024, 2)
                },
                "startup_time": self._startup_time.isoformat(),
                "uptime_seconds": (datetime.now(timezone.utc) - self._startup_time).total_seconds()
            }
        except Exception as e:
            self.logger.error(f"Failed to get infrastructure metrics: {e}")
            return {"error": str(e)}
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get runtime performance statistics.
        
        Returns:
            Dictionary containing performance metrics
        """
        try:
            current_time = datetime.now(timezone.utc)
            uptime = (current_time - self._startup_time).total_seconds()
            
            # Calculate rates
            operations_per_second = self._stats["successful_operations"] / max(uptime, 1.0)
            avg_processing_time = (
                self._stats["total_processing_time"] / max(self._stats["successful_operations"], 1)
            )
            success_rate = (
                self._stats["successful_operations"] / max(self._stats["successful_operations"] + self._stats["failed_operations"], 1) * 100
            )
            
            return {
                "component_name": self.name,
                "total_chunks_created": self._stats["total_chunks_created"],
                "successful_operations": self._stats["successful_operations"],
                "failed_operations": self._stats["failed_operations"],
                "delegated_operations": self._stats["delegated_operations"],
                "success_rate_percent": round(success_rate, 2),
                "operations_per_second": round(operations_per_second, 3),
                "average_processing_time": round(avg_processing_time, 3),
                "total_processing_time": round(
                    self._get_safe_float_value("total_processing_time", 0.0), 3
                ),
                "last_processing_time": self._stats["last_processing_time"],
                "initialization_count": self._stats["initialization_count"],
                "chunker_usage_distribution": self._stats["chunker_usage"],
                "recent_errors": self._get_recent_errors(5),  # Last 5 errors
                "error_count": self._get_error_count(),
                "uptime_seconds": round(uptime, 2)
            }
        except Exception as e:
            self.logger.error(f"Failed to get performance metrics: {e}")
            return {"error": str(e)}

    def _get_safe_float_value(self, key: str, default: float) -> float:
        """Safely get a float value from stats with type checking."""
        value = self._stats.get(key, default)
        if isinstance(value, (int, float)):
            return float(value)
        else:
            return default

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get component performance metrics (legacy method for compatibility).
        
        Returns:
            Dictionary containing performance metrics
        """
        # Combine infrastructure and performance metrics for compatibility
        infra_metrics = self.get_infrastructure_metrics()
        perf_metrics = self.get_performance_metrics()
        
        return {
            **infra_metrics,
            **perf_metrics,
            "last_health_check": datetime.now(timezone.utc).isoformat()
        }
    
    def export_metrics_prometheus(self) -> str:
        """
        Export metrics in Prometheus format.
        
        Returns:
            Prometheus-compatible metrics string
        """
        try:
            infra_metrics = self.get_infrastructure_metrics()
            perf_metrics = self.get_performance_metrics()
            
            metrics_lines = []
            
            # Infrastructure metrics
            metrics_lines.append(f"# HELP hades_component_uptime_seconds Component uptime in seconds")
            metrics_lines.append(f"# TYPE hades_component_uptime_seconds gauge")
            metrics_lines.append(f'hades_component_uptime_seconds{{component="core_chunker"}} {infra_metrics.get("uptime_seconds", 0)}')
            
            metrics_lines.append(f"# HELP hades_component_memory_rss_mb Memory RSS usage in MB")
            metrics_lines.append(f"# TYPE hades_component_memory_rss_mb gauge")
            memory_rss = infra_metrics.get("memory_usage", {}).get("rss_mb", 0)
            metrics_lines.append(f'hades_component_memory_rss_mb{{component="core_chunker"}} {memory_rss}')
            
            # Performance metrics
            metrics_lines.append(f"# HELP hades_component_operations_total Total number of operations")
            metrics_lines.append(f"# TYPE hades_component_operations_total counter")
            metrics_lines.append(f'hades_component_operations_total{{component="core_chunker"}} {perf_metrics.get("successful_operations", 0)}')
            
            metrics_lines.append(f"# HELP hades_component_operations_failed_total Total number of failed operations")
            metrics_lines.append(f"# TYPE hades_component_operations_failed_total counter")
            metrics_lines.append(f'hades_component_operations_failed_total{{component="core_chunker"}} {perf_metrics.get("failed_operations", 0)}')
            
            metrics_lines.append(f"# HELP hades_component_chunks_total Total number of chunks created")
            metrics_lines.append(f"# TYPE hades_component_chunks_total counter")
            metrics_lines.append(f'hades_component_chunks_total{{component="core_chunker"}} {perf_metrics.get("total_chunks_created", 0)}')
            
            metrics_lines.append(f"# HELP hades_component_success_rate_percent Success rate percentage")
            metrics_lines.append(f"# TYPE hades_component_success_rate_percent gauge")
            metrics_lines.append(f'hades_component_success_rate_percent{{component="core_chunker"}} {perf_metrics.get("success_rate_percent", 0)}')
            
            metrics_lines.append(f"# HELP hades_component_operations_per_second Operations processed per second")
            metrics_lines.append(f"# TYPE hades_component_operations_per_second gauge")
            metrics_lines.append(f'hades_component_operations_per_second{{component="core_chunker"}} {perf_metrics.get("operations_per_second", 0)}')
            
            # Chunker usage distribution metrics
            usage_distribution = perf_metrics.get("chunker_usage_distribution", {})
            if usage_distribution:
                metrics_lines.append(f"# HELP hades_component_chunker_usage Number of operations by chunker type")
                metrics_lines.append(f"# TYPE hades_component_chunker_usage counter")
                for chunker_type, count in usage_distribution.items():
                    metrics_lines.append(f'hades_component_chunker_usage{{component="core_chunker",chunker_type="{chunker_type}"}} {count}')
            
            return "\n".join(metrics_lines) + "\n"
            
        except Exception as e:
            self.logger.error(f"Failed to export Prometheus metrics: {e}")
            return f"# Error exporting metrics: {str(e)}\n"
    
    def chunk(self, input_data: ChunkingInput) -> ChunkingOutput:
        """
        Process text according to the chunking contract.
        
        Args:
            input_data: Input data conforming to ChunkingInput contract
            
        Returns:
            Output data conforming to ChunkingOutput contract
        """
        try:
            start_time = datetime.now(timezone.utc)
            
            # Update request statistics
            self._stats["last_processing_time"] = start_time.isoformat()
            
            # Get chunker type from options or use default
            chunker_type = input_data.processing_options.get('chunker_type', self._default_chunker_type)
            
            # Get chunker instance
            chunker = self._get_chunker(chunker_type)
            if not chunker:
                raise ValueError(f"Could not get chunker of type: {chunker_type}")
            
            # Delegate to specific chunker
            result = chunker.chunk(input_data)
            
            # Calculate processing time and update statistics
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            self._stats["successful_operations"] += 1
            self._stats["delegated_operations"] += 1
            self._stats["total_processing_time"] += processing_time
            self._stats["total_chunks_created"] += len(result.chunks)
            
            # Track chunker usage
            chunker_usage = self._stats.get("chunker_usage", {})
            if isinstance(chunker_usage, dict):
                chunker_usage[chunker_type] = chunker_usage.get(chunker_type, 0) + 1
                self._stats["chunker_usage"] = chunker_usage
            
            # Update result metadata to indicate core orchestration
            result.metadata.component_name = "core"
            result.processing_stats["delegated_to"] = chunker_type
            result.processing_stats["core_processing_time"] = processing_time
            
            return result
            
        except Exception as e:
            error_msg = f"Core chunking failed: {str(e)}"
            self.logger.error(error_msg)
            
            # Track error statistics
            self._stats["failed_operations"] += 1
            self._track_error(error_msg)
            
            metadata = ComponentMetadata(
                component_type=self.component_type,
                component_name=self.name,
                component_version=self.version,
                processed_at=datetime.now(timezone.utc),
                config=self._config,
                status=ProcessingStatus.ERROR
            )
            
            return ChunkingOutput(
                chunks=[],
                metadata=metadata,
                processing_stats={},
                errors=[error_msg]
            )
    
    def estimate_chunks(self, input_data: ChunkingInput) -> int:
        """
        Estimate number of chunks that will be generated.
        
        Args:
            input_data: Input data to estimate for
            
        Returns:
            Estimated number of chunks
        """
        try:
            chunker_type = input_data.processing_options.get('chunker_type', self._default_chunker_type)
            chunker = self._get_chunker(chunker_type)
            
            if chunker:
                return chunker.estimate_chunks(input_data)
            
            # Fallback estimation
            text_length = len(input_data.text)
            chunk_size = input_data.chunk_size
            chunk_overlap = input_data.chunk_overlap
            
            effective_chunk_size = max(1, chunk_size - chunk_overlap)
            estimated_chunks = max(1, (text_length + effective_chunk_size - 1) // effective_chunk_size)
            
            return estimated_chunks
            
        except Exception:
            return 1
    
    def supports_content_type(self, content_type: str) -> bool:
        """
        Check if chunker supports the given content type.
        
        Args:
            content_type: Content type to check (e.g., 'text', 'code')
            
        Returns:
            True if content type is supported, False otherwise
        """
        # Core chunker supports all content types through delegation
        supported_types = ['text', 'code', 'markdown', 'json', 'yaml', 'python', 'document']
        return content_type.lower() in supported_types
    
    def get_optimal_chunk_size(self, content_type: str) -> int:
        """
        Get the optimal chunk size for a given content type.
        
        Args:
            content_type: Content type
            
        Returns:
            Optimal chunk size in characters
        """
        # Different content types have different optimal sizes
        optimal_sizes = {
            'text': 512,
            'code': 1024,
            'python': 1024,
            'markdown': 768,
            'json': 512,
            'yaml': 512,
            'document': 1024
        }
        
        return optimal_sizes.get(content_type.lower(), 512)
    
    def get_supported_chunkers(self) -> List[str]:
        """
        Get list of supported chunker types.
        
        Returns:
            List of supported chunker names
        """
        return list(self._chunker_types.keys())
    
    def _get_chunker(self, chunker_type: str) -> Optional[Chunker]:
        """Get a chunker instance of the specified type."""
        if chunker_type in self._chunker_cache:
            return self._chunker_cache[chunker_type]
        
        try:
            if chunker_type not in self._chunker_types:
                self.logger.error(f"Unknown chunker type: {chunker_type}")
                return None
            
            # Create new chunker instance
            chunker_class = self._chunker_types[chunker_type]
            chunker = chunker_class(self._config.get(f'{chunker_type}_config', {}))
            
            # Cache the chunker instance
            self._chunker_cache[chunker_type] = chunker
            
            self.logger.debug(f"Created chunker instance: {chunker_type}")
            return chunker
            
        except Exception as e:
            self.logger.error(f"Could not create chunker {chunker_type}: {e}")
            return None
    
    def _track_error(self, error_msg: str) -> None:
        """Track an error in statistics."""
        errors_list = self._stats.get("errors", [])
        if isinstance(errors_list, list):
            errors_list.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": error_msg
            })
            
            # Keep only last 50 errors to prevent memory growth
            if len(errors_list) > 50:
                errors_list = errors_list[-50:]
            
            self._stats["errors"] = errors_list
    
    def _get_recent_errors(self, count: int = 5) -> List[Any]:
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