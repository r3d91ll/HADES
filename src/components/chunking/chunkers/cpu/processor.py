"""
CPU Chunking Component

This module provides the CPU-optimized chunking component that implements
the Chunker protocol. It focuses on efficient text chunking using CPU-based
algorithms without GPU dependencies.
"""

import logging
import psutil
import re
import time
from typing import Dict, Any, List, Optional
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


class CPUChunker(Chunker):
    """
    CPU-optimized chunking component implementing Chunker protocol.
    
    This component provides efficient text chunking using CPU-based algorithms,
    optimized for scenarios where GPU acceleration is not available or not needed.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize CPU chunker component.
        
        Args:
            config: Optional configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self._config = config or {}
        
        # Component metadata
        self._metadata = ComponentMetadata(
            component_type=ComponentType.CHUNKING,
            component_name="cpu",
            component_version="1.0.0",
            config=self._config
        )
        
        # Configuration
        self._chunk_size = self._config.get('chunk_size', 512)
        self._chunk_overlap = self._config.get('chunk_overlap', 50)
        self._chunking_method = self._config.get('chunking_method', 'sentence_aware')
        self._preserve_sentence_boundaries = self._config.get('preserve_sentence_boundaries', True)
        self._preserve_paragraph_boundaries = self._config.get('preserve_paragraph_boundaries', True)
        self._min_chunk_size = self._config.get('min_chunk_size', 100)
        self._max_chunk_size = self._config.get('max_chunk_size', 2048)
        
        # Language processing settings
        self._language = self._config.get('language', 'en')
        self._use_nltk = self._config.get('use_nltk', False)
        
        # Monitoring and metrics tracking
        self._stats: Dict[str, Any] = {
            "total_chunks_created": 0,
            "successful_chunks": 0,
            "failed_chunks": 0,
            "total_processing_time": 0.0,
            "total_characters_processed": 0,
            "last_processing_time": None,
            "initialization_count": 1,
            "errors": [],
            "method_counts": {}  # Track usage of different chunking methods
        }
        self._startup_time = datetime.now(timezone.utc)
        
        # Sentence splitters (lazy loaded)
        self._sentence_splitter = None
        self._nltk_initialized = False
        
        self.logger.info(f"Initialized CPU chunker with method: {self._chunking_method}")
    
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
        
        # Update configuration parameters
        if 'chunk_size' in config:
            self._chunk_size = config['chunk_size']
        
        if 'chunk_overlap' in config:
            self._chunk_overlap = config['chunk_overlap']
        
        if 'chunking_method' in config:
            self._chunking_method = config['chunking_method']
        
        if 'preserve_sentence_boundaries' in config:
            self._preserve_sentence_boundaries = config['preserve_sentence_boundaries']
        
        if 'preserve_paragraph_boundaries' in config:
            self._preserve_paragraph_boundaries = config['preserve_paragraph_boundaries']
        
        if 'language' in config:
            self._language = config['language']
            # Reset sentence splitter to reload with new language
            self._sentence_splitter = None
            self._nltk_initialized = False
        
        self.logger.info("Updated CPU chunker configuration")
    
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
        
        # Validate chunk size
        if 'chunk_size' in config:
            chunk_size = config['chunk_size']
            if not isinstance(chunk_size, int) or chunk_size < 1:
                return False
        
        # Validate chunk overlap
        if 'chunk_overlap' in config:
            overlap = config['chunk_overlap']
            if not isinstance(overlap, int) or overlap < 0:
                return False
            
            # Check overlap is less than chunk size
            chunk_size = config.get('chunk_size', self._chunk_size)
            if overlap >= chunk_size:
                return False
        
        # Validate chunking method
        if 'chunking_method' in config:
            method = config['chunking_method']
            valid_methods = ['fixed', 'sentence_aware', 'paragraph_aware', 'adaptive']
            if method not in valid_methods:
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
                "chunk_size": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 8192,
                    "default": 512,
                    "description": "Target chunk size in characters"
                },
                "chunk_overlap": {
                    "type": "integer",
                    "minimum": 0,
                    "default": 50,
                    "description": "Overlap between chunks in characters"
                },
                "chunking_method": {
                    "type": "string",
                    "enum": ["fixed", "sentence_aware", "paragraph_aware", "adaptive"],
                    "default": "sentence_aware",
                    "description": "Chunking method to use"
                },
                "preserve_sentence_boundaries": {
                    "type": "boolean",
                    "default": True,
                    "description": "Preserve sentence boundaries when chunking"
                },
                "preserve_paragraph_boundaries": {
                    "type": "boolean",
                    "default": True,
                    "description": "Preserve paragraph boundaries when chunking"
                },
                "min_chunk_size": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 100,
                    "description": "Minimum chunk size in characters"
                },
                "max_chunk_size": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 2048,
                    "description": "Maximum chunk size in characters"
                },
                "language": {
                    "type": "string",
                    "default": "en",
                    "description": "Language for text processing"
                },
                "use_nltk": {
                    "type": "boolean",
                    "default": False,
                    "description": "Use NLTK for sentence segmentation"
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
            # Test basic chunking functionality
            test_text = "This is a test sentence. This is another test sentence."
            test_input = ChunkingInput(
                text=test_text,
                document_id="health_check",
                chunk_size=50,
                chunk_overlap=10
            )
            
            # Try to chunk the test text
            chunks = self._perform_chunking(test_input)
            
            return len(chunks) > 0
            
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
                "chunking_method": self._chunking_method,
                "chunk_size": self._chunk_size,
                "chunk_overlap": self._chunk_overlap,
                "language": self._language,
                "use_nltk": self._use_nltk,
                "nltk_initialized": self._nltk_initialized,
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
            total_chunks = int(self._stats["total_chunks_created"])
            successful_chunks = int(self._stats["successful_chunks"])
            total_time = float(self._stats["total_processing_time"])
            
            chunks_per_second = total_chunks / max(uptime, 1.0)
            avg_processing_time = total_time / max(total_chunks, 1)
            success_rate = successful_chunks / max(total_chunks, 1) * 100
            total_chars = int(self._stats["total_characters_processed"])
            avg_chars_per_chunk = total_chars / max(total_chunks, 1)
            
            return {
                "component_name": self.name,
                "total_chunks_created": self._stats["total_chunks_created"],
                "successful_chunks": self._stats["successful_chunks"],
                "failed_chunks": self._stats["failed_chunks"],
                "success_rate_percent": round(success_rate, 2),
                "chunks_per_second": round(chunks_per_second, 3),
                "average_processing_time": round(avg_processing_time, 3),
                "total_processing_time": round(self._stats["total_processing_time"], 3),
                "total_characters_processed": self._stats["total_characters_processed"],
                "avg_chars_per_chunk": round(avg_chars_per_chunk, 2),
                "last_processing_time": self._stats["last_processing_time"],
                "initialization_count": self._stats["initialization_count"],
                "method_distribution": self._stats["method_counts"],
                "recent_errors": self._stats["errors"][-5:],  # Last 5 errors
                "error_count": len(self._stats["errors"]),
                "uptime_seconds": round(uptime, 2)
            }
        except Exception as e:
            self.logger.error(f"Failed to get performance metrics: {e}")
            return {"error": str(e)}

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
            metrics_lines.append(f'hades_component_uptime_seconds{{component="cpu_chunker"}} {infra_metrics.get("uptime_seconds", 0)}')
            
            metrics_lines.append(f"# HELP hades_component_memory_rss_mb Memory RSS usage in MB")
            metrics_lines.append(f"# TYPE hades_component_memory_rss_mb gauge")
            memory_rss = infra_metrics.get("memory_usage", {}).get("rss_mb", 0)
            metrics_lines.append(f'hades_component_memory_rss_mb{{component="cpu_chunker"}} {memory_rss}')
            
            # Performance metrics
            metrics_lines.append(f"# HELP hades_component_chunks_total Total number of processed chunks")
            metrics_lines.append(f"# TYPE hades_component_chunks_total counter")
            metrics_lines.append(f'hades_component_chunks_total{{component="cpu_chunker"}} {perf_metrics.get("total_chunks_created", 0)}')
            
            metrics_lines.append(f"# HELP hades_component_chunks_successful_total Total number of successful chunks")
            metrics_lines.append(f"# TYPE hades_component_chunks_successful_total counter")
            metrics_lines.append(f'hades_component_chunks_successful_total{{component="cpu_chunker"}} {perf_metrics.get("successful_chunks", 0)}')
            
            metrics_lines.append(f"# HELP hades_component_chunks_failed_total Total number of failed chunks")
            metrics_lines.append(f"# TYPE hades_component_chunks_failed_total counter")
            metrics_lines.append(f'hades_component_chunks_failed_total{{component="cpu_chunker"}} {perf_metrics.get("failed_chunks", 0)}')
            
            metrics_lines.append(f"# HELP hades_component_success_rate_percent Success rate percentage")
            metrics_lines.append(f"# TYPE hades_component_success_rate_percent gauge")
            metrics_lines.append(f'hades_component_success_rate_percent{{component="cpu_chunker"}} {perf_metrics.get("success_rate_percent", 0)}')
            
            metrics_lines.append(f"# HELP hades_component_chunks_per_second Chunks processed per second")
            metrics_lines.append(f"# TYPE hades_component_chunks_per_second gauge")
            metrics_lines.append(f'hades_component_chunks_per_second{{component="cpu_chunker"}} {perf_metrics.get("chunks_per_second", 0)}')
            
            metrics_lines.append(f"# HELP hades_component_avg_processing_time_seconds Average processing time in seconds")
            metrics_lines.append(f"# TYPE hades_component_avg_processing_time_seconds gauge")
            metrics_lines.append(f'hades_component_avg_processing_time_seconds{{component="cpu_chunker"}} {perf_metrics.get("average_processing_time", 0)}')
            
            metrics_lines.append(f"# HELP hades_component_avg_chars_per_chunk Average characters per chunk")
            metrics_lines.append(f"# TYPE hades_component_avg_chars_per_chunk gauge")
            metrics_lines.append(f'hades_component_avg_chars_per_chunk{{component="cpu_chunker"}} {perf_metrics.get("avg_chars_per_chunk", 0)}')
            
            # Method distribution metrics
            method_counts = perf_metrics.get("method_distribution", {})
            if method_counts:
                metrics_lines.append(f"# HELP hades_component_method_count Number of chunks processed by method")
                metrics_lines.append(f"# TYPE hades_component_method_count counter")
                for method_name, count in method_counts.items():
                    metrics_lines.append(f'hades_component_method_count{{component="cpu_chunker",method="{method_name}"}} {count}')
            
            return "\n".join(metrics_lines) + "\n"
            
        except Exception as e:
            self.logger.error(f"Failed to export Prometheus metrics: {e}")
            return f"# Error exporting metrics: {str(e)}\n"
    
    def chunk(self, input_data: ChunkingInput) -> ChunkingOutput:
        """
        Chunk text according to the contract using CPU algorithms.
        
        Args:
            input_data: Input data conforming to ChunkingInput contract
            
        Returns:
            Output data conforming to ChunkingOutput contract
            
        Raises:
            ChunkingError: If chunking fails
        """
        errors: List[str] = []
        chunks: List[TextChunk] = []
        
        try:
            start_time = datetime.now(timezone.utc)
            
            # Update request statistics
            self._stats["last_processing_time"] = start_time.isoformat()
            
            # Override config with input parameters
            self._chunk_size = input_data.chunk_size
            self._chunk_overlap = input_data.chunk_overlap
            
            # Perform chunking
            chunks = self._perform_chunking(input_data)
            
            # Calculate processing time
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            # Update statistics
            self._stats["total_chunks_created"] += len(chunks)
            self._stats["successful_chunks"] += len(chunks)
            self._stats["total_processing_time"] += processing_time
            self._stats["total_characters_processed"] += len(input_data.text)
            
            # Track method usage
            method_key = self._chunking_method
            self._stats["method_counts"][method_key] = self._stats["method_counts"].get(method_key, 0) + 1
            
            # Update metadata
            metadata = ComponentMetadata(
                component_type=self.component_type,
                component_name=self.name,
                component_version=self.version,
                processing_time=processing_time,
                processed_at=datetime.now(timezone.utc),
                config=self._config,
                status=ProcessingStatus.SUCCESS if not errors else ProcessingStatus.ERROR
            )
            
            return ChunkingOutput(
                chunks=chunks,
                metadata=metadata,
                processing_stats={
                    "processing_time": processing_time,
                    "chunks_created": len(chunks),
                    "characters_processed": len(input_data.text),
                    "chunking_method": self._chunking_method,
                    "chunk_size_used": self._chunk_size,
                    "chunk_overlap_used": self._chunk_overlap,
                    "throughput_chars_per_second": len(input_data.text) / max(processing_time, 0.001),
                    "throughput_chunks_per_second": len(chunks) / max(processing_time, 0.001)
                },
                errors=errors
            )
            
        except Exception as e:
            error_msg = f"CPU chunking failed: {str(e)}"
            errors.append(error_msg)
            self.logger.error(error_msg)
            
            # Track error statistics
            self._stats["failed_chunks"] += 1
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
                errors=errors
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
            text_length = len(input_data.text)
            chunk_size = input_data.chunk_size
            chunk_overlap = input_data.chunk_overlap
            
            effective_chunk_size = chunk_size - chunk_overlap
            
            if effective_chunk_size <= 0:
                return 1
            
            estimated_chunks = max(1, text_length // effective_chunk_size)
            
            return estimated_chunks
            
        except Exception as e:
            self.logger.error(f"Failed to estimate chunks: {e}")
            return 1
    
    def supports_content_type(self, content_type: str) -> bool:
        """
        Check if chunker supports the given content type.
        
        Args:
            content_type: Content type to check
            
        Returns:
            True if content type is supported, False otherwise
        """
        supported_types = ['text', 'markdown', 'document', 'plain']
        return content_type.lower() in supported_types
    
    def get_optimal_chunk_size(self, content_type: str) -> int:
        """
        Get the optimal chunk size for a given content type.
        
        Args:
            content_type: Content type
            
        Returns:
            Optimal chunk size in characters
        """
        base_size = self._chunk_size
        
        # Adjust based on content type
        content_multipliers = {
            'text': 1.0,
            'markdown': 0.9,    # Slightly smaller for markdown formatting
            'document': 1.2,    # Larger for document content
            'plain': 1.0
        }
        
        multiplier = content_multipliers.get(content_type.lower(), 1.0)
        return int(base_size * multiplier)
    
    def _perform_chunking(self, input_data: ChunkingInput) -> List[TextChunk]:
        """Perform the actual chunking based on the configured method."""
        text = input_data.text
        
        if self._chunking_method == 'fixed':
            return self._fixed_chunking(text, input_data)
        elif self._chunking_method == 'sentence_aware':
            return self._sentence_aware_chunking(text, input_data)
        elif self._chunking_method == 'paragraph_aware':
            return self._paragraph_aware_chunking(text, input_data)
        elif self._chunking_method == 'adaptive':
            return self._adaptive_chunking(text, input_data)
        else:
            # Fallback to fixed chunking
            return self._fixed_chunking(text, input_data)
    
    def _fixed_chunking(self, text: str, input_data: ChunkingInput) -> List[TextChunk]:
        """Fixed-size chunking with character boundaries."""
        chunks: List[TextChunk] = []
        chunk_id = 0
        start_idx = 0
        
        while start_idx < len(text):
            end_idx = min(start_idx + self._chunk_size, len(text))
            chunk_text = text[start_idx:end_idx]
            
            if len(chunk_text.strip()) < self._min_chunk_size and start_idx > 0:
                # Merge small chunks with previous one
                if chunks:
                    chunks[-1].text += " " + chunk_text
                    chunks[-1].end_index = end_idx
                break
            
            chunk = TextChunk(
                id=f"{input_data.document_id}_chunk_{chunk_id}",
                text=chunk_text,
                start_index=start_idx,
                end_index=end_idx,
                chunk_index=chunk_id,
                metadata={
                    "chunking_method": "fixed",
                    "chunk_size": self._chunk_size,
                    "overlap": self._chunk_overlap
                }
            )
            
            chunks.append(chunk)
            chunk_id += 1
            
            # Move to next chunk with overlap
            start_idx += (self._chunk_size - self._chunk_overlap)
            
            if end_idx >= len(text):
                break
        
        return chunks
    
    def _sentence_aware_chunking(self, text: str, input_data: ChunkingInput) -> List[TextChunk]:
        """Sentence-aware chunking that respects sentence boundaries."""
        sentences = self._split_sentences(text)
        
        chunks: List[TextChunk] = []
        chunk_id = 0
        current_chunk_sentences: List[str] = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # Check if adding this sentence would exceed chunk size
            if current_length + sentence_length > self._chunk_size and current_chunk_sentences:
                # Create chunk from current sentences
                chunk_text = " ".join(current_chunk_sentences)
                start_idx = text.find(current_chunk_sentences[0])
                end_idx = start_idx + len(chunk_text)
                
                chunk = TextChunk(
                    id=f"{input_data.document_id}_chunk_{chunk_id}",
                    text=chunk_text,
                    start_index=start_idx,
                    end_index=end_idx,
                    chunk_index=chunk_id,
                    metadata={
                        "chunking_method": "sentence_aware",
                        "sentence_count": len(current_chunk_sentences),
                        "preserves_sentence_boundaries": True
                    }
                )
                
                chunks.append(chunk)
                chunk_id += 1
                
                # Start new chunk with overlap
                overlap_sentences = self._get_overlap_sentences(current_chunk_sentences)
                current_chunk_sentences = overlap_sentences + [sentence]
                current_length = sum(len(s) for s in current_chunk_sentences)
            else:
                current_chunk_sentences.append(sentence)
                current_length += sentence_length
        
        # Add remaining sentences as final chunk
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            start_idx = text.find(current_chunk_sentences[0])
            end_idx = start_idx + len(chunk_text)
            
            chunk = TextChunk(
                id=f"{input_data.document_id}_chunk_{chunk_id}",
                text=chunk_text,
                start_index=start_idx,
                end_index=end_idx,
                chunk_index=chunk_id,
                metadata={
                    "chunking_method": "sentence_aware",
                    "sentence_count": len(current_chunk_sentences),
                    "preserves_sentence_boundaries": True
                }
            )
            
            chunks.append(chunk)
        
        return chunks
    
    def _paragraph_aware_chunking(self, text: str, input_data: ChunkingInput) -> List[TextChunk]:
        """Paragraph-aware chunking that respects paragraph boundaries."""
        paragraphs = self._split_paragraphs(text)
        
        chunks: List[TextChunk] = []
        chunk_id = 0
        current_chunk_paragraphs: List[str] = []
        current_length = 0
        
        for paragraph in paragraphs:
            paragraph_length = len(paragraph)
            
            # Check if adding this paragraph would exceed chunk size
            if current_length + paragraph_length > self._chunk_size and current_chunk_paragraphs:
                # Create chunk from current paragraphs
                chunk_text = "\n\n".join(current_chunk_paragraphs)
                
                chunk = TextChunk(
                    id=f"{input_data.document_id}_chunk_{chunk_id}",
                    text=chunk_text,
                    start_index=0,  # Simplified for paragraph chunking
                    end_index=len(chunk_text),
                    chunk_index=chunk_id,
                    metadata={
                        "chunking_method": "paragraph_aware",
                        "paragraph_count": len(current_chunk_paragraphs),
                        "preserves_paragraph_boundaries": True
                    }
                )
                
                chunks.append(chunk)
                chunk_id += 1
                
                # Start new chunk
                current_chunk_paragraphs = [paragraph]
                current_length = paragraph_length
            else:
                current_chunk_paragraphs.append(paragraph)
                current_length += paragraph_length
        
        # Add remaining paragraphs as final chunk
        if current_chunk_paragraphs:
            chunk_text = "\n\n".join(current_chunk_paragraphs)
            
            chunk = TextChunk(
                id=f"{input_data.document_id}_chunk_{chunk_id}",
                text=chunk_text,
                start_index=0,
                end_index=len(chunk_text),
                chunk_index=chunk_id,
                metadata={
                    "chunking_method": "paragraph_aware",
                    "paragraph_count": len(current_chunk_paragraphs),
                    "preserves_paragraph_boundaries": True
                }
            )
            
            chunks.append(chunk)
        
        return chunks
    
    def _adaptive_chunking(self, text: str, input_data: ChunkingInput) -> List[TextChunk]:
        """Adaptive chunking that combines multiple methods."""
        # Start with sentence-aware chunking
        chunks = self._sentence_aware_chunking(text, input_data)
        
        # Post-process to merge very small chunks
        merged_chunks = []
        current_chunk = None
        
        for chunk in chunks:
            if current_chunk is None:
                current_chunk = chunk
            elif len(chunk.text) < self._min_chunk_size:
                # Merge with current chunk
                current_chunk.text += " " + chunk.text
                current_chunk.end_index = chunk.end_index
                current_chunk.metadata["merged_chunks"] = current_chunk.metadata.get("merged_chunks", 0) + 1
            else:
                merged_chunks.append(current_chunk)
                current_chunk = chunk
        
        if current_chunk:
            merged_chunks.append(current_chunk)
        
        return merged_chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        if self._use_nltk and self._initialize_nltk():
            return self._nltk_sentence_split(text)
        else:
            return self._simple_sentence_split(text)
    
    def _split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        # Split by double newlines or more
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _simple_sentence_split(self, text: str) -> List[str]:
        """Simple sentence splitting using regex."""
        # Basic sentence boundary detection
        sentences = re.split(r'[.!?]+\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _nltk_sentence_split(self, text: str) -> List[str]:
        """NLTK-based sentence splitting."""
        try:
            from nltk.tokenize import sent_tokenize
            sentences = sent_tokenize(text, language=self._language)
            return [s for s in sentences if s.strip()]
        except ImportError:
            self.logger.warning("NLTK not available, falling back to simple splitting")
            return self._simple_sentence_split(text)
    
    def _initialize_nltk(self) -> bool:
        """Initialize NLTK if requested."""
        if self._nltk_initialized:
            return True
        
        try:
            import nltk
            # Download required data if not present
            nltk.download('punkt', quiet=True)
            self._nltk_initialized = True
            return True
        except ImportError:
            self.logger.warning("NLTK not available")
            return False
        except Exception as e:
            self.logger.warning(f"Failed to initialize NLTK: {e}")
            return False
    
    def _get_overlap_sentences(self, sentences: List[str]) -> List[str]:
        """Get sentences for overlap based on overlap size."""
        if not sentences:
            return []
        
        # Calculate how many sentences to include in overlap
        total_length = sum(len(s) for s in sentences)
        overlap_length = 0
        overlap_sentences: List[str] = []
        
        # Start from the end and work backwards
        for sentence in reversed(sentences):
            if overlap_length + len(sentence) <= self._chunk_overlap:
                overlap_sentences.insert(0, sentence)
                overlap_length += len(sentence)
            else:
                break
        
        return overlap_sentences
    
    def _track_error(self, error_msg: str) -> None:
        """Track an error in statistics."""
        self._stats["errors"].append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": error_msg
        })
        
        # Keep only last 50 errors to prevent memory growth
        if len(self._stats["errors"]) > 50:
            self._stats["errors"] = self._stats["errors"][-50:]