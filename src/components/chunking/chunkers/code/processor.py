"""
Code Chunking Component

This module provides specialized code chunking component that implements the
Chunker protocol specifically for source code content.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

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

# Import CPU chunker for delegation
from ..cpu.processor import CPUChunker


class CodeChunker(Chunker):
    """
    Code-specific chunking component implementing Chunker protocol.
    
    This component provides code-optimized chunking strategies,
    currently delegating to CPU chunker with code-specific configurations.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize code chunker component.
        
        Args:
            config: Optional configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self._config = config or {}
        
        # Code-specific defaults
        code_defaults = {
            'chunking_method': 'fixed',  # Fixed chunking better for code
            'preserve_sentence_boundaries': False,  # Not relevant for code
            'preserve_paragraph_boundaries': False, # Not relevant for code
            'chunk_size': 1024,  # Larger chunks for code
            'chunk_overlap': 100,  # Larger overlap to preserve context
            'min_chunk_size': 200,  # Larger minimum for meaningful code chunks
            'language': 'en'
        }
        
        # Merge with provided config
        code_config = {**code_defaults, **self._config}
        
        # Initialize CPU chunker with code-optimized config
        self._cpu_chunker = CPUChunker(code_config)
        
        # Component metadata
        self._metadata = ComponentMetadata(
            component_type=ComponentType.CHUNKING,
            component_name="code",
            component_version="1.0.0",
            config=self._config
        )
        
        self.logger.info("Initialized code chunker")
    
    @property
    def name(self) -> str:
        """Component name for identification."""
        return "code"
    
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
        self._config.update(config)
        self._metadata.config = self._config
        self._metadata.processed_at = datetime.utcnow()
        
        # Update CPU chunker configuration
        self._cpu_chunker.configure(config)
        
        self.logger.info("Updated code chunker configuration")
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate configuration parameters.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            True if configuration is valid, False otherwise
        """
        return self._cpu_chunker.validate_config(config)
    
    def get_config_schema(self) -> Dict[str, Any]:
        """
        Get the JSON schema for component configuration.
        
        Returns:
            JSON schema dictionary describing valid configuration
        """
        schema = self._cpu_chunker.get_config_schema()
        schema["description"] = "Configuration for code-optimized chunking"
        return schema
    
    def health_check(self) -> bool:
        """
        Check if component is healthy and ready to process data.
        
        Returns:
            True if component is healthy, False otherwise
        """
        return self._cpu_chunker.health_check()
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get component performance metrics.
        
        Returns:
            Dictionary containing performance metrics
        """
        metrics = self._cpu_chunker.get_metrics()
        metrics["component_name"] = self.name
        metrics["specialization"] = "code_optimized"
        return metrics
    
    def chunk(self, input_data: ChunkingInput) -> ChunkingOutput:
        """
        Chunk code according to the contract using code-optimized strategies.
        
        Args:
            input_data: Input data conforming to ChunkingInput contract
            
        Returns:
            Output data conforming to ChunkingOutput contract
        """
        try:
            # Delegate to CPU chunker
            result = self._cpu_chunker.chunk(input_data)
            
            # Update metadata to indicate code specialization
            result.metadata.component_name = "code"
            result.processing_stats["specialization"] = "code_optimized"
            
            return result
            
        except Exception as e:
            error_msg = f"Code chunking failed: {str(e)}"
            self.logger.error(error_msg)
            
            metadata = ComponentMetadata(
                component_type=self.component_type,
                component_name=self.name,
                component_version=self.version,
                processed_at=datetime.utcnow(),
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
        return self._cpu_chunker.estimate_chunks(input_data)
    
    def supports_content_type(self, content_type: str) -> bool:
        """
        Check if chunker supports the given content type.
        
        Args:
            content_type: Content type to check
            
        Returns:
            True if content type is supported, False otherwise
        """
        code_types = ['code', 'python', 'javascript', 'java', 'cpp', 'c', 'json', 'yaml', 'xml', 'html', 'css']
        return content_type.lower() in code_types
    
    def get_optimal_chunk_size(self, content_type: str) -> int:
        """
        Get the optimal chunk size for a given content type.
        
        Args:
            content_type: Content type
            
        Returns:
            Optimal chunk size in characters
        """
        # Code-optimized sizes
        code_sizes = {
            'code': 1024,
            'python': 1024,
            'javascript': 1024,
            'java': 1536,      # Java tends to be more verbose
            'cpp': 1024,
            'c': 1024,
            'json': 512,       # JSON can be more compact
            'yaml': 512,       # YAML can be more compact
            'xml': 768,
            'html': 768,
            'css': 512
        }
        
        return code_sizes.get(content_type.lower(), 1024)