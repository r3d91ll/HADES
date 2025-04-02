"""
Initialization module for the HADES modular pipeline architecture.

This module handles the registration of built-in components with the plugin registry.
"""

import logging
from typing import List

from hades.pipeline import registry
from hades.storage.redis.storage import RedisDocumentStore, RedisChunkStore, RedisEmbeddingStore
from hades.ingest.processors.basic_processor import BasicDocumentProcessor
from hades.ingest.chunkers.text_chunker import TextChunker
from hades.models.embedding.openai_embeddings import OpenAIEmbeddingModel
from hades.models.embedding.local_embeddings import LocalEmbeddingModel
from hades.models.embedding.vllm_embeddings import VLLMEmbeddingModel
from hades.ingest.pipeline import ModularIngestPipeline


def register_built_in_components() -> List[str]:
    """Register all built-in components with the plugin registry.
    
    Returns:
        List of registered component names
    """
    logger = logging.getLogger(__name__)
    registered = []
    
    # Register storage components
    try:
        registry.register_plugin('document_store', 'redis', RedisDocumentStore)
        registered.append('document_store.redis')
        
        registry.register_plugin('chunk_store', 'redis', RedisChunkStore)
        registered.append('chunk_store.redis')
        
        registry.register_plugin('embedding_store', 'redis', RedisEmbeddingStore)
        registered.append('embedding_store.redis')
    except Exception as e:
        logger.error(f"Error registering storage components: {e}")
    
    # Register ingest components
    try:
        registry.register_plugin('document_processor', 'basic', BasicDocumentProcessor)
        registered.append('document_processor.basic')
        
        registry.register_plugin('chunker', 'text', TextChunker)
        registered.append('chunker.text')
    except Exception as e:
        logger.error(f"Error registering ingest components: {e}")
    
    # Register model components
    try:
        # Register vLLM embedding model (default)
        registry.register_plugin('embedding_model', 'vllm', VLLMEmbeddingModel)
        registered.append('embedding_model.vllm')
        
        # Register local embedding model (alternative)
        registry.register_plugin('embedding_model', 'local', LocalEmbeddingModel)
        registered.append('embedding_model.local')
        
        # Register OpenAI embedding model (alternative)
        registry.register_plugin('embedding_model', 'openai', OpenAIEmbeddingModel)
        registered.append('embedding_model.openai')
    except Exception as e:
        logger.error(f"Error registering model components: {e}")
    
    # Register pipeline components
    try:
        registry.register_plugin('ingest_pipeline', 'modular', ModularIngestPipeline)
        registered.append('ingest_pipeline.modular')
    except Exception as e:
        logger.error(f"Error registering pipeline components: {e}")
    
    logger.info(f"Registered {len(registered)} built-in components")
    return registered


# Register components when module is imported
register_built_in_components()
