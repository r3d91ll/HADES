"""
Factory for creating Jina v4 processor instances.

This module provides a factory pattern for creating and managing JinaV4Processor instances.
"""

import logging
from typing import Dict, Any, Optional
from .jina_processor import JinaV4Processor

logger = logging.getLogger(__name__)


class JinaV4Factory:
    """
    Factory for creating Jina v4 processor instances.
    
    This factory ensures consistent configuration and singleton pattern
    for the Jina v4 processor across the application.
    """
    
    _instance: Optional['JinaV4Factory'] = None
    _processor: Optional[JinaV4Processor] = None
    
    def __new__(cls) -> 'JinaV4Factory':
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self) -> None:
        """Initialize the factory."""
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._default_config = {
                'model': 'jinaai/jina-embeddings-v4',
                'device': 'cuda',
                'batch_size': 32,
                'max_input_tokens': 8192,
                'late_chunking': True,
                'extract_keywords': True,
                'output_mode': 'multi-vector',
                'features': {
                    'ast_analysis': {'enabled': True},
                    'keyword_extraction': {'enabled': True, 'max_keywords': 10},
                    'semantic_chunking': {'enabled': True},
                    'multimodal': {'enabled': True}
                },
                'vllm': {
                    'tensor_parallel_size': 1,
                    'gpu_memory_utilization': 0.9,
                    'max_model_len': 8192,
                    'dtype': 'float16'
                },
                'lora_adapter': 'retrieval'
            }
    
    def create_processor(self, config: Optional[Dict[str, Any]] = None) -> JinaV4Processor:
        """
        Create or return existing Jina v4 processor instance.
        
        Args:
            config: Optional configuration dictionary. If not provided,
                   uses default configuration.
                   
        Returns:
            JinaV4Processor instance
        """
        if self._processor is None:
            # Merge provided config with defaults
            final_config = self._default_config.copy()
            if config:
                final_config.update(config)
            
            logger.info("Creating new JinaV4Processor instance")
            self._processor = JinaV4Processor(final_config)
        else:
            logger.debug("Returning existing JinaV4Processor instance")
            
        return self._processor
    
    def get_processor(self) -> Optional[JinaV4Processor]:
        """Get the current processor instance if it exists."""
        return self._processor
    
    def reset(self) -> None:
        """Reset the factory, clearing the processor instance."""
        if self._processor:
            logger.info("Resetting JinaV4Factory")
        self._processor = None
    
    def update_config(self, config: Dict[str, Any]) -> None:
        """
        Update the default configuration.
        
        Note: This will not affect already created processor instances.
        """
        self._default_config.update(config)
        logger.info(f"Updated default configuration: {list(config.keys())}")


# Convenience functions
def get_jina_processor(config: Optional[Dict[str, Any]] = None) -> JinaV4Processor:
    """
    Get a Jina v4 processor instance.
    
    This is a convenience function that uses the factory pattern.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        JinaV4Processor instance
    """
    factory = JinaV4Factory()
    return factory.create_processor(config)


def create_jina_component(config: Optional[Dict[str, Any]] = None) -> JinaV4Processor:
    """
    Create a Jina v4 component (alias for get_jina_processor).
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        JinaV4Processor instance
    """
    return get_jina_processor(config)