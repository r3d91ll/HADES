"""
Embedding validation utilities.

This module provides validation functions for embeddings to ensure
quality and consistency.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch

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
            errors.append(f"Norm too small: {norm} < {self.min_norm}")
        if norm > self.max_norm:
            errors.append(f"Norm too large: {norm} > {self.max_norm}")
        
        # Check variance
        if np.var(embedding) < 1e-6:
            errors.append("Very low variance - possible constant embedding")
        
        is_valid = len(errors) == 0
        if not is_valid:
            logger.warning(f"Embedding {embedding_id} validation failed: {errors}")
        
        return is_valid, errors
    
    def validate_batch(self, embeddings: np.ndarray, embedding_ids: Optional[List[str]] = None) -> Tuple[bool, Dict[str, List[str]]]:
        """Validate a batch of embeddings."""
        if embedding_ids is None:
            embedding_ids = [f"embedding_{i}" for i in range(len(embeddings))]
        
        all_errors = {}
        all_valid = True
        
        for i, (embedding, embedding_id) in enumerate(zip(embeddings, embedding_ids)):
            is_valid, errors = self.validate_single(embedding, embedding_id)
            if not is_valid:
                all_valid = False
                all_errors[embedding_id] = errors
        
        return all_valid, all_errors
    
    def compute_statistics(self, embeddings: np.ndarray) -> Dict[str, Any]:
        """Compute statistics for a batch of embeddings."""
        stats = {
            "num_embeddings": len(embeddings),
            "mean_norm": float(np.mean([np.linalg.norm(e) for e in embeddings])),
            "std_norm": float(np.std([np.linalg.norm(e) for e in embeddings])),
            "mean_variance": float(np.mean([np.var(e) for e in embeddings])),
            "num_nan": int(np.sum([np.isnan(e).any() for e in embeddings])),
            "num_inf": int(np.sum([np.isinf(e).any() for e in embeddings])),
        }
        
        # Compute pairwise similarities
        if len(embeddings) > 1:
            normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            similarities = np.dot(normalized, normalized.T)
            np.fill_diagonal(similarities, 0)  # Exclude self-similarity
            
            stats.update({
                "mean_similarity": float(np.mean(similarities)),
                "max_similarity": float(np.max(similarities)),
                "min_similarity": float(np.min(similarities)),
            })
        
        return stats
    
    def check_quality(self, embeddings: np.ndarray) -> Tuple[bool, str]:
        """Check overall quality of embeddings."""
        stats = self.compute_statistics(embeddings)
        
        # Quality checks
        if stats["num_nan"] > 0:
            return False, f"Found {stats['num_nan']} embeddings with NaN values"
        
        if stats["num_inf"] > 0:
            return False, f"Found {stats['num_inf']} embeddings with infinite values"
        
        if stats["mean_variance"] < 1e-6:
            return False, "Embeddings have very low variance"
        
        if stats.get("max_similarity", 0) > 0.99:
            return False, "Found near-duplicate embeddings"
        
        if stats["mean_norm"] < self.min_norm:
            return False, f"Average norm too low: {stats['mean_norm']}"
        
        return True, "Embeddings pass quality checks"


def validate_embeddings(embeddings: np.ndarray, 
                       expected_dim: int = 768,
                       check_normalized: bool = True) -> Dict[str, Any]:
    """Validate embeddings array.
    
    Args:
        embeddings: Numpy array of embeddings
        expected_dim: Expected embedding dimension
        check_normalized: Whether to check if embeddings are normalized
        
    Returns:
        Validation results dictionary
    """
    results: Dict[str, Any] = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "stats": {}
    }
    
    # Basic shape validation
    if len(embeddings.shape) != 2:
        results["valid"] = False
        results["errors"].append(f"Expected 2D array, got {len(embeddings.shape)}D")
        return results
    
    # Dimension check
    if embeddings.shape[1] != expected_dim:
        results["valid"] = False
        results["errors"].append(f"Dimension mismatch: expected {expected_dim}, got {embeddings.shape[1]}")
    
    # NaN/Inf check
    if np.isnan(embeddings).any():
        results["valid"] = False
        results["errors"].append("Contains NaN values")
    
    if np.isinf(embeddings).any():
        results["valid"] = False
        results["errors"].append("Contains infinite values")
    
    # Normalization check
    if check_normalized:
        norms = np.linalg.norm(embeddings, axis=1)
        if not np.allclose(norms, 1.0, rtol=1e-3):
            results["warnings"].append("Embeddings are not normalized")
            results["stats"]["norm_stats"] = {
                "mean": float(np.mean(norms)),
                "std": float(np.std(norms)),
                "min": float(np.min(norms)),
                "max": float(np.max(norms))
            }
    
    # Diversity check
    if len(embeddings) > 1:
        pairwise_sims = np.dot(embeddings, embeddings.T)
        np.fill_diagonal(pairwise_sims, 0)
        max_sim = np.max(pairwise_sims)
        
        if max_sim > 0.99:
            results["warnings"].append(f"Found near-duplicate embeddings (max similarity: {max_sim:.3f})")
        
        results["stats"]["similarity_stats"] = {
            "mean": float(np.mean(pairwise_sims)),
            "max": float(max_sim),
            "min": float(np.min(pairwise_sims))
        }
    
    return results


def validate_torch_embeddings(embeddings: torch.Tensor,
                            expected_dim: int = 768,
                            device: Optional[str] = None) -> Dict[str, Any]:
    """Validate PyTorch tensor embeddings.
    
    Args:
        embeddings: PyTorch tensor of embeddings
        expected_dim: Expected embedding dimension
        device: Expected device (e.g., 'cuda', 'cpu')
        
    Returns:
        Validation results
    """
    results: Dict[str, Any] = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "device": str(embeddings.device)
    }
    
    # Device check
    if device and str(embeddings.device) != device:
        results["warnings"].append(f"Embeddings on {embeddings.device}, expected {device}")
    
    # Convert to numpy for validation
    embeddings_np = embeddings.detach().cpu().numpy()
    
    # Use numpy validation
    numpy_results = validate_embeddings(embeddings_np, expected_dim, check_normalized=True)
    
    # Merge results
    results["valid"] = numpy_results["valid"]
    results["errors"].extend(numpy_results["errors"])
    results["warnings"].extend(numpy_results["warnings"])
    results["stats"] = numpy_results.get("stats", {})
    
    return results


__all__ = [
    "EmbeddingValidator",
    "validate_embeddings",
    "validate_torch_embeddings"
]