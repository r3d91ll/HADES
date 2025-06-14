"""
In-Memory Storage Component

This module provides an in-memory storage component that implements the Storage
protocol. It's useful for testing and small-scale deployments.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid

# Import component contracts and protocols
from src.types.components.contracts import (
    ComponentType,
    ComponentMetadata,
    StorageInput,
    StorageOutput,
    StoredItem,
    QueryInput,
    QueryOutput,
    RetrievalResult,
    ProcessingStatus
)
from src.types.components.protocols import Storage


class MemoryStorage(Storage):
    """
    In-memory storage component implementing Storage protocol.
    
    This component provides basic storage operations using in-memory dictionaries.
    Data is not persisted between restarts.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize memory storage component.
        
        Args:
            config: Optional configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self._config = config or {}
        
        # Component metadata
        self._metadata = ComponentMetadata(
            component_type=ComponentType.STORAGE,
            component_name="memory",
            component_version="1.0.0",
            config=self._config
        )
        
        # In-memory storage
        self._items: Dict[str, Dict[str, Any]] = {}
        self._embeddings: Dict[str, List[float]] = {}
        
        self.logger.info(f"Initialized memory storage component")
    
    @property
    def name(self) -> str:
        """Component name for identification."""
        return "memory"
    
    @property
    def version(self) -> str:
        """Component version string."""
        return "1.0.0"
    
    @property
    def component_type(self) -> ComponentType:
        """Type of component."""
        return ComponentType.STORAGE
    
    def configure(self, config: Dict[str, Any]) -> None:
        """Configure component with parameters."""
        self._config.update(config)
        self._metadata.config = self._config
        self._metadata.processed_at = datetime.utcnow()
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration parameters."""
        return isinstance(config, dict)
    
    def get_config_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for component configuration."""
        return {
            "type": "object",
            "properties": {
                "max_items": {
                    "type": "integer",
                    "description": "Maximum number of items to store",
                    "minimum": 1,
                    "default": 100000
                }
            }
        }
    
    def health_check(self) -> bool:
        """Check if component is healthy."""
        return True
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get component performance metrics."""
        return {
            "component_name": self.name,
            "component_version": self.version,
            "item_count": len(self._items),
            "embedding_count": len(self._embeddings),
            "memory_usage_estimate": len(str(self._items)) + len(str(self._embeddings))
        }
    
    def store(self, input_data: StorageInput) -> StorageOutput:
        """Store data according to the contract."""
        stored_items = []
        errors = []
        
        try:
            start_time = datetime.utcnow()
            
            for embedding in input_data.enhanced_embeddings:
                try:
                    item_id = embedding.chunk_id or str(uuid.uuid4())
                    
                    # Store the item
                    self._items[item_id] = {
                        "id": item_id,
                        "chunk_id": embedding.chunk_id,
                        "original_embedding": embedding.original_embedding,
                        "enhanced_embedding": embedding.enhanced_embedding,
                        "graph_features": embedding.graph_features,
                        "enhancement_score": embedding.enhancement_score,
                        "metadata": embedding.metadata,
                        "stored_at": datetime.utcnow().isoformat()
                    }
                    
                    # Store embedding for similarity search
                    self._embeddings[item_id] = embedding.enhanced_embedding
                    
                    stored_item = StoredItem(
                        item_id=item_id,
                        storage_location=f"memory://{item_id}",
                        storage_timestamp=datetime.utcnow(),
                        index_status=ProcessingStatus.SUCCESS
                    )
                    stored_items.append(stored_item)
                    
                except Exception as e:
                    error_msg = f"Failed to store embedding {embedding.chunk_id}: {str(e)}"
                    errors.append(error_msg)
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            metadata = ComponentMetadata(
                component_type=self.component_type,
                component_name=self.name,
                component_version=self.version,
                processing_time=processing_time,
                processed_at=datetime.utcnow(),
                config=self._config
            )
            
            return StorageOutput(
                stored_items=stored_items,
                metadata=metadata,
                storage_stats={"items_stored": len(stored_items)},
                errors=errors
            )
            
        except Exception as e:
            errors.append(f"Storage operation failed: {str(e)}")
            
            metadata = ComponentMetadata(
                component_type=self.component_type,
                component_name=self.name,
                component_version=self.version,
                processed_at=datetime.utcnow(),
                config=self._config
            )
            
            return StorageOutput(
                stored_items=stored_items,
                metadata=metadata,
                errors=errors
            )
    
    def query(self, query_data: QueryInput) -> QueryOutput:
        """Query stored data according to the contract."""
        results: List[RetrievalResult] = []
        errors: List[str] = []
        
        try:
            start_time = datetime.utcnow()
            
            if query_data.query_embedding:
                # Vector similarity search
                results = self._similarity_search(
                    query_data.query_embedding,
                    query_data.top_k
                )
            else:
                # Text search
                results = self._text_search(
                    query_data.query,
                    query_data.top_k
                )
            
            search_time = (datetime.utcnow() - start_time).total_seconds()
            
            metadata = ComponentMetadata(
                component_type=self.component_type,
                component_name=self.name,
                component_version=self.version,
                processing_time=search_time,
                processed_at=datetime.utcnow(),
                config=self._config
            )
            
            return QueryOutput(
                results=results,
                metadata=metadata,
                search_stats={"search_time": search_time},
                errors=errors,
                search_time=search_time
            )
            
        except Exception as e:
            errors.append(f"Query operation failed: {str(e)}")
            
            metadata = ComponentMetadata(
                component_type=self.component_type,
                component_name=self.name,
                component_version=self.version,
                processed_at=datetime.utcnow(),
                config=self._config
            )
            
            return QueryOutput(
                results=results,
                metadata=metadata,
                errors=errors
            )
    
    def delete(self, item_ids: List[str]) -> bool:
        """Delete items by their IDs."""
        try:
            for item_id in item_ids:
                if item_id in self._items:
                    del self._items[item_id]
                if item_id in self._embeddings:
                    del self._embeddings[item_id]
            return True
        except Exception:
            return False
    
    def update(self, item_id: str, data: Dict[str, Any]) -> bool:
        """Update an existing item."""
        try:
            if item_id in self._items:
                self._items[item_id].update(data)
                return True
            return False
        except Exception:
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics."""
        return {
            "item_count": len(self._items),
            "embedding_count": len(self._embeddings),
            "storage_type": "memory"
        }
    
    def get_capacity_info(self) -> Dict[str, Any]:
        """Get storage capacity information."""
        return {
            "current_items": len(self._items),
            "max_items": self._config.get("max_items", 100000),
            "storage_type": "memory"
        }
    
    def supports_transactions(self) -> bool:
        """Check if storage supports transactions."""
        return False  # Memory storage doesn't support transactions
    
    def _similarity_search(self, query_embedding: List[float], top_k: int) -> List[RetrievalResult]:
        """Perform similarity search using cosine similarity."""
        import numpy as np
        
        results = []
        query_vec = np.array(query_embedding)
        
        for item_id, embedding in self._embeddings.items():
            if item_id in self._items:
                item = self._items[item_id]
                embed_vec = np.array(embedding)
                
                # Cosine similarity
                similarity = np.dot(query_vec, embed_vec) / (
                    np.linalg.norm(query_vec) * np.linalg.norm(embed_vec)
                )
                
                result = RetrievalResult(
                    item_id=item_id,
                    content=item.get("metadata", {}).get("content", ""),
                    score=float(similarity),
                    metadata=item.get("metadata", {})
                )
                results.append(result)
        
        # Sort by similarity score and return top_k
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
    
    def _text_search(self, query: str, top_k: int) -> List[RetrievalResult]:
        """Perform simple text search."""
        results = []
        query_lower = query.lower()
        
        for item_id, item in self._items.items():
            content = item.get("metadata", {}).get("content", "")
            
            if query_lower in content.lower():
                # Simple relevance scoring based on query term frequency
                score = content.lower().count(query_lower) / max(len(content.split()), 1)
                
                result = RetrievalResult(
                    item_id=item_id,
                    content=content,
                    score=min(score, 1.0),
                    metadata=item.get("metadata", {})
                )
                results.append(result)
        
        # Sort by score and return top_k
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]