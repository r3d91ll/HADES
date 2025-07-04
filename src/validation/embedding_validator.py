"""
Embedding validation utilities.

This module provides validation functions for embeddings to ensure
quality and consistency.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingValidator:
    """Validates embeddings for quality and consistency."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize embedding validator."""
        self.config = config or {}
        self.expected_dim = self.config.get("expected_dimension", 768)
        self.min_norm = self.config.get("min_norm", 0.1)
        self.max_norm = self.config.get("max_norm", 100.0)
        self.allow_nan = self.config.get("allow_nan", False)
        self.allow_inf = self.config.get("allow_inf", False)
    
    def validate_single(self, embedding: np.ndarray, embedding_id: str = "unknown") -> Tuple[bool, List[str]]:
        """Validate a single embedding."""
        errors = []
        
        # Check dimension
        if embedding.shape[0] != self.expected_dim:
            errors.append(f"Dimension mismatch: {embedding.shape[0]} != {self.expected_dim}")
        
        # Check for NaN values
        if not self.allow_nan and np.isnan(embedding).any():
            errors.append("Contains NaN values")
        
        # Check for infinite values
        if not self.allow_inf and np.isinf(embedding).any():
            errors.append("Contains infinite values")
        
        # Check norm
        norm = np.linalg.norm(embedding)
        if norm < self.min_norm:
            errors.append(f"Norm too small: {norm:.4f} < {self.min_norm}")
        if norm > self.max_norm:
            errors.append(f"Norm too large: {norm:.4f} > {self.max_norm}")
        
        # Check if all zeros
        if np.allclose(embedding, 0):
            errors.append("Embedding is all zeros")
        
        is_valid = len(errors) == 0
        
        if not is_valid:
            logger.warning(f"Invalid embedding {embedding_id}: {', '.join(errors)}")
        
        return is_valid, errors
    
    def validate_batch(
        self,
        embeddings: List[np.ndarray],
        embedding_ids: Optional[List[str]] = None
    ) -> Tuple[List[bool], Dict[str, List[str]]]:
        """Validate a batch of embeddings."""
        if embedding_ids is None:
            embedding_ids = [f"embedding_{i}" for i in range(len(embeddings))]
        
        results = []
        all_errors = {}
        
        for embedding, emb_id in zip(embeddings, embedding_ids):
            is_valid, errors = self.validate_single(embedding, emb_id)
            results.append(is_valid)
            if errors:
                all_errors[emb_id] = errors
        
        valid_count = sum(results)
        logger.info(f"Validated {len(embeddings)} embeddings: {valid_count} valid, {len(embeddings) - valid_count} invalid")
        
        return results, all_errors
    
    def compute_statistics(self, embeddings: List[np.ndarray]) -> Dict[str, Any]:
        """Compute statistics for a set of embeddings."""
        if not embeddings:
            return {}
        
        embeddings_array = np.array(embeddings)
        
        stats = {
            "count": len(embeddings),
            "dimension": embeddings[0].shape[0],
            "mean_norm": float(np.mean([np.linalg.norm(e) for e in embeddings])),
            "std_norm": float(np.std([np.linalg.norm(e) for e in embeddings])),
            "min_norm": float(np.min([np.linalg.norm(e) for e in embeddings])),
            "max_norm": float(np.max([np.linalg.norm(e) for e in embeddings])),
            "mean_values": float(np.mean(embeddings_array)),
            "std_values": float(np.std(embeddings_array)),
            "has_nan": bool(np.isnan(embeddings_array).any()),
            "has_inf": bool(np.isinf(embeddings_array).any()),
            "zero_embeddings": sum(1 for e in embeddings if np.allclose(e, 0))
        }
        
        return stats


def create_embedding_validator(config: Optional[Dict[str, Any]] = None) -> EmbeddingValidator:
    """Factory function to create embedding validator."""
    return EmbeddingValidator(config)


__all__ = [
    "EmbeddingValidator",
    "create_embedding_validator"
]