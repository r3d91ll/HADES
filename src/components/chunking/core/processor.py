"""
Core Chunking Component

This module provides the core chunking component that implements the
Chunker protocol. It acts as a factory and coordinator for different
chunking strategies in the new component architecture.
"""

import logging
from typing import Dict, Any, List, Optional, Union
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

# Import chunker implementations
from ..chunkers.cpu.processor import CPUChunker
from ..chunkers.chonky.processor import ChonkyChunker
from ..chunkers.code.processor import CodeChunker
from ..chunkers.text.processor import TextChunker


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
            'text': TextChunker
        }
        
        # Cache for chunker instances
        self._chunker_cache: Dict[str, Chunker] = {}
        
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
        self._metadata.processed_at = datetime.utcnow()
        
        # Update default chunker if specified
        if 'default_chunker' in config:
            self._default_chunker_type = config['default_chunker']
        
        # Clear chunker cache to force re-initialization
        self._chunker_cache.clear()
        
        self.logger.info(f"Updated core chunker configuration")
    
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
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get component performance metrics.
        
        Returns:
            Dictionary containing performance metrics
        """
        return {
            "component_name": self.name,
            "component_version": self.version,
            "default_chunker": self._default_chunker_type,
            "available_chunkers": list(self._chunker_types.keys()),
            "cached_chunkers": list(self._chunker_cache.keys()),
            "last_health_check": datetime.utcnow().isoformat()
        }
    
    def chunk(self, input_data: ChunkingInput) -> ChunkingOutput:
        """
        Process text according to the chunking contract.
        
        Args:
            input_data: Input data conforming to ChunkingInput contract
            
        Returns:
            Output data conforming to ChunkingOutput contract
        """
        try:
            start_time = datetime.utcnow()
            
            # Get chunker type from options or use default
            chunker_type = input_data.processing_options.get('chunker_type', self._default_chunker_type)
            
            # Get chunker instance
            chunker = self._get_chunker(chunker_type)
            if not chunker:
                raise ValueError(f"Could not get chunker of type: {chunker_type}")
            
            # Delegate to specific chunker
            result = chunker.chunk(input_data)
            
            # Update result metadata to indicate core orchestration
            result.metadata.component_name = "core"
            result.processing_stats["delegated_to"] = chunker_type
            result.processing_stats["core_processing_time"] = (datetime.utcnow() - start_time).total_seconds()
            
            return result
            
        except Exception as e:
            error_msg = f"Core chunking failed: {str(e)}"
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