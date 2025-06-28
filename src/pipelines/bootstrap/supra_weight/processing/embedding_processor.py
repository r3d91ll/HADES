"""
Embedding processor for bootstrap pipeline.
"""

from typing import Dict, List, Any, Optional, Iterator, Tuple
import numpy as np
import logging
from pathlib import Path
import torch
from torch.nn.functional import cosine_similarity

logger = logging.getLogger(__name__)


class EmbeddingProcessor:
    """
    Processes embeddings for similarity computation and edge creation.
    
    This processor handles embedding operations including similarity
    computation, batch processing, and optimization.
    """
    
    def __init__(self,
                 embedding_dim: Optional[int] = None,
                 similarity_threshold: float = 0.5,
                 use_gpu: bool = True,
                 batch_size: int = 1000):
        """
        Initialize processor.
        
        Args:
            embedding_dim: Expected embedding dimension (for validation)
            similarity_threshold: Minimum similarity for edge creation
            use_gpu: Whether to use GPU for computations
            batch_size: Batch size for similarity computation
        """
        self.embedding_dim = embedding_dim
        self.similarity_threshold = similarity_threshold
        self.batch_size = batch_size
        
        # Set device
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        logger.info(f"Initialized EmbeddingProcessor on device: {self.device}")
        
    def validate_embeddings(self, nodes: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        """
        Validate embeddings in nodes.
        
        Args:
            nodes: List of nodes with embeddings
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        embedding_dims = set()
        
        for idx, node in enumerate(nodes):
            if 'embedding' not in node:
                continue
                
            embedding = node['embedding']
            
            # Convert to numpy if needed
            if isinstance(embedding, list):
                embedding = np.array(embedding)
                
            # Check dimension
            if embedding.ndim != 1:
                errors.append(f"Node {node.get('node_id', idx)}: embedding has {embedding.ndim} dimensions, expected 1")
                
            # Track embedding size
            embedding_dims.add(embedding.shape[0])
            
            # Check for NaN or Inf
            if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
                errors.append(f"Node {node.get('node_id', idx)}: embedding contains NaN or Inf values")
                
        # Check consistency
        if len(embedding_dims) > 1:
            errors.append(f"Inconsistent embedding dimensions: {embedding_dims}")
            
        # Check against expected dimension
        if self.embedding_dim and embedding_dims:
            actual_dim = next(iter(embedding_dims))
            if actual_dim != self.embedding_dim:
                errors.append(f"Expected embedding dimension {self.embedding_dim}, got {actual_dim}")
                
        return len(errors) == 0, errors
        
    def compute_similarities_batch(self,
                                 nodes: List[Dict[str, Any]],
                                 indices: Optional[List[Tuple[int, int]]] = None) -> Iterator[Tuple[int, int, float]]:
        """
        Compute similarities in batches.
        
        Args:
            nodes: List of nodes with embeddings
            indices: Optional specific pairs to compute (if None, computes all pairs)
            
        Yields:
            Tuples of (idx1, idx2, similarity)
        """
        # Extract embeddings
        embeddings = []
        valid_indices = []
        
        for idx, node in enumerate(nodes):
            if 'embedding' in node:
                emb = node['embedding']
                if isinstance(emb, list):
                    emb = np.array(emb)
                embeddings.append(emb)
                valid_indices.append(idx)
                
        if not embeddings:
            logger.warning("No embeddings found in nodes")
            return
            
        # Convert to torch tensors
        embeddings_tensor = torch.tensor(np.array(embeddings), device=self.device, dtype=torch.float32)
        
        # Normalize embeddings
        embeddings_tensor = torch.nn.functional.normalize(embeddings_tensor, p=2, dim=1)
        
        if indices:
            # Compute specific pairs
            for idx1, idx2 in indices:
                if idx1 in valid_indices and idx2 in valid_indices:
                    pos1 = valid_indices.index(idx1)
                    pos2 = valid_indices.index(idx2)
                    
                    similarity = cosine_similarity(
                        embeddings_tensor[pos1].unsqueeze(0),
                        embeddings_tensor[pos2].unsqueeze(0)
                    ).item()
                    
                    if similarity >= self.similarity_threshold:
                        yield (idx1, idx2, similarity)
        else:
            # Compute all pairs in batches
            n = len(embeddings_tensor)
            
            for i in range(0, n, self.batch_size):
                batch_end = min(i + self.batch_size, n)
                batch = embeddings_tensor[i:batch_end]
                
                # Compute similarities for this batch against all embeddings
                similarities = torch.mm(batch, embeddings_tensor.t())
                
                # Process results
                for local_idx in range(batch_end - i):
                    global_idx1 = i + local_idx
                    
                    for global_idx2 in range(global_idx1 + 1, n):
                        sim = similarities[local_idx, global_idx2].item()
                        
                        if sim >= self.similarity_threshold:
                            yield (
                                valid_indices[global_idx1],
                                valid_indices[global_idx2],
                                sim
                            )
                            
    def compute_embedding_statistics(self, nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute statistics about embeddings.
        
        Args:
            nodes: List of nodes with embeddings
            
        Returns:
            Dictionary of statistics
        """
        embeddings = []
        
        for node in nodes:
            if 'embedding' in node:
                emb = node['embedding']
                if isinstance(emb, list):
                    emb = np.array(emb)
                embeddings.append(emb)
                
        if not embeddings:
            return {'count': 0}
            
        embeddings_array = np.array(embeddings)
        
        # Compute statistics
        stats: Dict[str, Any] = {
            'count': len(embeddings),
            'dimension': embeddings_array.shape[1],
            'mean_norm': float(np.mean(np.linalg.norm(embeddings_array, axis=1))),
            'std_norm': float(np.std(np.linalg.norm(embeddings_array, axis=1))),
            'min_norm': float(np.min(np.linalg.norm(embeddings_array, axis=1))),
            'max_norm': float(np.max(np.linalg.norm(embeddings_array, axis=1))),
            'sparsity': float(np.mean(embeddings_array == 0))
        }
        
        # Sample similarity distribution
        if len(embeddings) > 1:
            sample_size = min(1000, len(embeddings) * (len(embeddings) - 1) // 2)
            sample_sims = []
            
            for _ in range(sample_size):
                idx1, idx2 = np.random.choice(len(embeddings), 2, replace=False)
                sim = np.dot(embeddings[idx1], embeddings[idx2]) / (
                    np.linalg.norm(embeddings[idx1]) * np.linalg.norm(embeddings[idx2])
                )
                sample_sims.append(sim)
                
            similarity_distribution = {
                'mean': float(np.mean(sample_sims)),
                'std': float(np.std(sample_sims)),
                'min': float(np.min(sample_sims)),
                'max': float(np.max(sample_sims)),
                'percentiles': {
                    '25': float(np.percentile(sample_sims, 25)),
                    '50': float(np.percentile(sample_sims, 50)),
                    '75': float(np.percentile(sample_sims, 75)),
                    '95': float(np.percentile(sample_sims, 95))
                }
            }
            stats['similarity_distribution'] = similarity_distribution
            
        return stats
        
    def create_embedding_index(self, nodes: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """
        Create an index of node embeddings for fast lookup.
        
        Args:
            nodes: List of nodes with embeddings
            
        Returns:
            Dictionary mapping node_id to embedding
        """
        index = {}
        
        for node in nodes:
            if 'embedding' in node and 'node_id' in node:
                emb = node['embedding']
                if isinstance(emb, list):
                    emb = np.array(emb)
                index[node['node_id']] = emb
                
        return index