"""
Core Embedding Component

This module provides the core embedding component that implements the
Embedder protocol. It acts as a factory and coordinator for different
embedding strategies in the new component architecture.
"""

import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

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
            result.metadata.processing_time = (datetime.utcnow() - start_time).total_seconds()
            result.embedding_stats["core_processing_time"] = result.metadata.processing_time
            result.embedding_stats["delegated_to"] = embedder_type
            
            return result
            
        except Exception as e:
            error_msg = f"Core embedding failed: {str(e)}"
            self.logger.error(error_msg)
            
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
            return None