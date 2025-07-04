"""
Nano VectorDB storage implementation.

This module provides a lightweight in-memory vector database implementation
that can serve as a simple alternative to more complex vector stores.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
from pathlib import Path
import json
import pickle

logger = logging.getLogger(__name__)


class NanoVectorDB:
    """Simple in-memory vector database with persistence support."""
    
    def __init__(self, dimension: int = 768, persistence_path: Optional[Path] = None):
        """Initialize nano vector database.
        
        Args:
            dimension: Vector dimension
            persistence_path: Optional path for saving/loading the database
        """
        self.dimension = dimension
        self.vectors: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self.persistence_path = persistence_path
        
        # Load from persistence if path provided and exists
        if persistence_path and persistence_path.exists():
            self.load()
    
    def add(self, key: str, vector: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a vector to the database."""
        if vector.shape[0] != self.dimension:
            raise ValueError(f"Vector dimension {vector.shape[0]} != {self.dimension}")
        
        self.vectors[key] = vector
        if metadata:
            self.metadata[key] = metadata
            
        logger.debug(f"Added vector with key: {key}")
    
    def add_batch(self, items: List[Tuple[str, np.ndarray, Optional[Dict[str, Any]]]]) -> None:
        """Add multiple vectors at once."""
        for key, vector, metadata in items:
            self.add(key, vector, metadata)
    
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        threshold: Optional[float] = None,
        metric: str = "cosine"
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar vectors.
        
        Args:
            query_vector: Query vector
            top_k: Number of results to return
            threshold: Optional similarity threshold
            metric: Distance metric ('cosine', 'euclidean', 'dot')
            
        Returns:
            List of (key, similarity, metadata) tuples
        """
        if not self.vectors:
            return []
        
        if query_vector.shape[0] != self.dimension:
            raise ValueError(f"Query dimension {query_vector.shape[0]} != {self.dimension}")
        
        # Compute similarities
        similarities = []
        for key, vector in self.vectors.items():
            if metric == "cosine":
                # Cosine similarity
                sim = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
            elif metric == "euclidean":
                # Negative euclidean distance (so higher is better)
                sim = -np.linalg.norm(query_vector - vector)
            elif metric == "dot":
                # Dot product
                sim = np.dot(query_vector, vector)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            if threshold is None or sim >= threshold:
                similarities.append((key, float(sim), self.metadata.get(key, {})))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def get(self, key: str) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """Get a vector by key."""
        if key not in self.vectors:
            return None
        return self.vectors[key], self.metadata.get(key, {})
    
    def delete(self, key: str) -> bool:
        """Delete a vector by key."""
        if key in self.vectors:
            del self.vectors[key]
            if key in self.metadata:
                del self.metadata[key]
            return True
        return False
    
    def clear(self) -> None:
        """Clear all vectors from the database."""
        self.vectors.clear()
        self.metadata.clear()
    
    def save(self, path: Optional[Path] = None) -> None:
        """Save the database to disk."""
        save_path = path or self.persistence_path
        if not save_path:
            raise ValueError("No save path provided")
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "dimension": self.dimension,
            "vectors": self.vectors,
            "metadata": self.metadata
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Saved {len(self.vectors)} vectors to {save_path}")
    
    def load(self, path: Optional[Path] = None) -> None:
        """Load the database from disk."""
        load_path = path or self.persistence_path
        if not load_path:
            raise ValueError("No load path provided")
        
        if not load_path.exists():
            raise FileNotFoundError(f"Database file not found: {load_path}")
        
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        
        self.dimension = data["dimension"]
        self.vectors = data["vectors"]
        self.metadata = data["metadata"]
        
        logger.info(f"Loaded {len(self.vectors)} vectors from {load_path}")
    
    def size(self) -> int:
        """Get the number of vectors in the database."""
        return len(self.vectors)
    
    def __len__(self) -> int:
        """Get the number of vectors in the database."""
        return len(self.vectors)
    
    def __contains__(self, key: str) -> bool:
        """Check if a key exists in the database."""
        return key in self.vectors
    
    def keys(self) -> List[str]:
        """Get all keys in the database."""
        return list(self.vectors.keys())
    
    def stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        if not self.vectors:
            return {
                "num_vectors": 0,
                "dimension": self.dimension,
                "memory_usage_mb": 0
            }
        
        # Calculate memory usage
        vector_memory = sum(v.nbytes for v in self.vectors.values())
        metadata_memory = len(json.dumps(self.metadata).encode())
        total_memory_mb = (vector_memory + metadata_memory) / (1024 * 1024)
        
        # Calculate vector statistics
        all_vectors = np.array(list(self.vectors.values()))
        
        return {
            "num_vectors": len(self.vectors),
            "dimension": self.dimension,
            "memory_usage_mb": round(total_memory_mb, 2),
            "vector_stats": {
                "mean_norm": float(np.mean(np.linalg.norm(all_vectors, axis=1))),
                "std_norm": float(np.std(np.linalg.norm(all_vectors, axis=1))),
                "min_norm": float(np.min(np.linalg.norm(all_vectors, axis=1))),
                "max_norm": float(np.max(np.linalg.norm(all_vectors, axis=1)))
            }
        }


class NanoVectorStorage:
    """Storage adapter for nano vector database."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize storage with configuration."""
        self.config = config
        self.dimension = config.get("dimension", 768)
        self.persistence_path = config.get("persistence_path")
        
        if self.persistence_path:
            self.persistence_path = Path(self.persistence_path)
        
        self.db = NanoVectorDB(
            dimension=self.dimension,
            persistence_path=self.persistence_path
        )
    
    def store_vectors(
        self,
        vectors: Union[np.ndarray, List[np.ndarray]],
        keys: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """Store vectors with keys and optional metadata."""
        try:
            if isinstance(vectors, np.ndarray):
                vectors = list(vectors)
            
            if metadata is None:
                metadata = [{}] * len(vectors)
            
            items = list(zip(keys, vectors, metadata))
            self.db.add_batch(items)
            
            # Auto-save if persistence is enabled
            if self.persistence_path:
                self.db.save()
            
            return True
        except Exception as e:
            logger.error(f"Failed to store vectors: {e}")
            return False
    
    def search_vectors(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        try:
            results = self.db.search(
                query_vector,
                top_k=top_k,
                metric=self.config.get("metric", "cosine")
            )
            
            # Convert to expected format
            formatted_results = []
            for key, score, metadata in results:
                # Apply filters if provided
                if filters:
                    match = all(
                        metadata.get(k) == v
                        for k, v in filters.items()
                    )
                    if not match:
                        continue
                
                formatted_results.append({
                    "id": key,
                    "score": score,
                    "metadata": metadata
                })
            
            return formatted_results[:top_k]
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def get_vector(self, key: str) -> Optional[Dict[str, Any]]:
        """Get a vector by key."""
        result = self.db.get(key)
        if result is None:
            return None
        
        vector, metadata = result
        return {
            "id": key,
            "vector": vector,
            "metadata": metadata
        }
    
    def delete_vector(self, key: str) -> bool:
        """Delete a vector by key."""
        success = self.db.delete(key)
        
        # Auto-save if persistence is enabled
        if success and self.persistence_path:
            self.db.save()
        
        return success
    
    def health_check(self) -> bool:
        """Check if storage is healthy."""
        try:
            # Try a simple operation
            test_vector = np.random.rand(self.dimension)
            self.db.search(test_vector, top_k=1)
            return True
        except Exception:
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        return self.db.stats()


__all__ = ["NanoVectorDB", "NanoVectorStorage"]