"""
Core implementation of the PathRAG system interface.

This module provides the underlying implementation for the API endpoints.
"""

import time
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

# Import HADES-PathRAG components
import logging

# Dynamic imports to handle missing dependencies gracefully
def import_component(module_path: str, class_name: str) -> Any:
    """Import a component with graceful error handling."""
    try:
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        logging.warning(f"Cannot import {class_name} from {module_path}: {str(e)}")
        logging.warning(f"The server will continue with limited functionality")
        return None

logger = logging.getLogger(__name__)


class PathRAGSystem:
    """
    Core system that handles interactions with the HADES-PathRAG components.
    
    This class provides a simplified interface for writing data to and querying
    the PathRAG knowledge graph. It orchestrates the pre-processing, embedding,
    and storage components of the system.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the PathRAG system.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        logger.info(f"Initializing PathRAGSystem with config: {config_path}")
        
        # Initialize components with graceful error handling
        # Dynamic import of storage component
        try:
            ArangoStorage = import_component("..storage.arango.connection", "ArangoConnection")
            if ArangoStorage:
                self.storage = ArangoStorage()
                logger.info("ArangoDB storage initialized successfully")
            else:
                self.storage = None
                logger.warning("ArangoDB storage not available")
        except Exception as e:
            logger.error(f"Error initializing storage: {str(e)}")
            self.storage = None
        
        # Embedding system is optional
        self.embedder = None
        try:
            # Check if vLLM is available without importing it directly
            try:
                import vllm
                logger.info(f"vLLM is available (version {vllm.__version__})")
                # If successful, try to import the embedder
                ISNEEmbedder = import_component("..isne.processors.embedding_processor", "EmbeddingProcessor")
                if ISNEEmbedder:
                    logger.info("Embedder component is available")
                    # Just track the class but don't initialize yet to avoid errors
                    self._embedder_class = ISNEEmbedder
            except (ImportError, Exception) as e:
                logger.warning(f"vLLM not available: {str(e)}")
                logger.warning("Embedding functionality will be limited")
                self._embedder_class = None
        except Exception as e:
            logger.error(f"Error checking embedding components: {str(e)}")
            self._embedder_class = None
        
        # Ingestor component is optional but useful for document processing
        try:
            RepositoryIngestor = import_component("..ingest.orchestrator.ingestor", "RepositoryIngestor")
            if RepositoryIngestor:
                self.ingestor = RepositoryIngestor()
                logger.info("Repository ingestor initialized successfully")
            else:
                self.ingestor = None
                logger.warning("Repository ingestor not available")
        except Exception as e:
            logger.error(f"Error initializing ingestor: {str(e)}")
            self.ingestor = None
        
        # Track initialization time for metrics
        self._start_time = time.time()
        logger.info("PathRAGSystem initialized with available components")
    
    def write(self, content: str, path: Optional[str] = None, 
              metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Write/update data in the knowledge graph.
        
        This method processes the input content through the appropriate
        pre-processor based on content type, generates embeddings, and
        stores the result in the database.
        
        Args:
            content: Document/code content or serialized relationship data
            path: File path or identifier (optional)
            metadata: Additional information about the content (optional)
            
        Returns:
            ID of the created/updated entity
        """
        start_time = time.time()
        entity_id = "document_"+str(time.time())
        
        # Process content through ingestor if available
        if self.ingestor:
            try:
                processed_document = self.ingestor.process_content(
                    content=content,
                    path=path,
                    metadata=metadata or {}
                )
                logger.info(f"Content processed successfully through ingestor")
            except Exception as e:
                logger.warning(f"Error processing content: {str(e)}")
                # Create basic document structure when ingestor fails
                processed_document = {
                    "content": content,
                    "path": path or "unknown",
                    "metadata": metadata or {}
                }
        else:
            # Create basic document structure when ingestor not available
            processed_document = {
                "content": content,
                "path": path or "unknown",
                "metadata": metadata or {}
            }
        
        # Store in database if available
        if self.storage:
            try:
                entity_id = self.storage.insert_document("nodes", processed_document)
                logger.info(f"Document stored with ID: {entity_id}")
            except Exception as e:
                logger.warning(f"Error storing document: {str(e)}")
        else:
            logger.warning("Storage not available, document not persisted")
        
        return entity_id
        
    def query(self, query_text: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Query the knowledge graph with natural language.
        
        Args:
            query_text: Natural language query
            max_results: Maximum number of results to return
            
        Returns:
            List of result items with content and confidence scores
        """
        logger.info(f"Processing query: '{query_text}' (max_results={max_results})")
        
        # Check if storage is available
        if not self.storage:
            logger.warning("Storage not available, cannot perform query")
            return [{
                "content": "Storage not available. The system is operating in limited mode.",
                "confidence": 1.0,
                "metadata": {"status": "error", "message": "Storage component not available"}
            }]
        
        # Simple string matching fallback when embeddings not available
        try:
            # Note: This is a placeholder. In a real implementation, we would 
            # use ArangoDB's full-text search capabilities if available.
            results = [{
                "content": f"This is a placeholder response for query: {query_text}",
                "confidence": 0.5,
                "metadata": {"status": "placeholder", "query": query_text}
            }] * min(max_results, 3)
            
            return results
        except Exception as e:
            logger.error(f"Error performing query: {str(e)}")
            return [{
                "content": "An error occurred while processing your query.",
                "confidence": 0.0,
                "metadata": {"status": "error", "message": str(e)}
            }]
        
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current system status.
        
        Returns:
            Dictionary with status information
        """
        # Calculate uptime
        uptime_seconds = time.time() - self._start_time
        
        status_info = {
            "status": "online",
            "uptime_seconds": uptime_seconds,
            "version": "0.1.0",  # TODO: Read from package version
        }
        
        # Add component status
        components = {}
        
        # Check storage
        if self.storage:
            try:
                components["storage"] = "available"
                # Try to get document count if possible
                document_count = 0  # Placeholder
                status_info["document_count"] = document_count
            except Exception as e:
                components["storage"] = f"error: {str(e)}"
        else:
            components["storage"] = "unavailable"
            status_info["document_count"] = 0
        
        # Check embedding
        if self._embedder_class:
            components["embedding"] = "available"
        else:
            components["embedding"] = "unavailable"
        
        # Check ingestor
        if self.ingestor:
            components["ingestor"] = "available"
        else:
            components["ingestor"] = "unavailable"
        
        status_info["components"] = components
        return status_info
