"""
ISNE Retriever implementation for retrieving documents based on embeddings.

This module provides retrieval capabilities for ISNE embeddings.
"""

import torch
from torch import Tensor
import numpy as np
from typing import Tuple, List, Optional, Dict, Any

from src.isne.models.isne_model import ISNEModel


class ISNERetriever:
    """
    Retriever for ISNE document embeddings.
    
    This class provides methods for retrieving similar documents
    based on ISNE embeddings.
    """
    
    def __init__(self, model: Optional[ISNEModel] = None, device: str = "cpu"):
        """
        Initialize the ISNERetriever.
        
        Args:
            model: Optional ISNE model for generating embeddings
            device: Device to use for tensor operations
        """
        self.model = model
        self.device = device
    
    def retrieve(
        self,
        query_embedding: Tensor,
        document_embeddings: Tensor,
        k: int = 5
    ) -> Tuple[List[List[int]], List[List[float]]]:
        """
        Retrieve top-k similar documents based on embeddings.
        
        Args:
            query_embedding: Query embedding tensor of shape [batch_size, embedding_dim]
            document_embeddings: Document embeddings of shape [num_docs, embedding_dim]
            k: Number of similar documents to retrieve
            
        Returns:
            Tuple of (indices, scores) where:
              - indices: List of lists of document indices for each query
              - scores: List of lists of similarity scores for each query
        """
        # Ensure tensors are on the correct device
        query_embedding = query_embedding.to(self.device)
        document_embeddings = document_embeddings.to(self.device)
        
        # Compute cosine similarity
        # Normalize embeddings for cosine similarity
        query_embedding_norm = torch.nn.functional.normalize(query_embedding, p=2, dim=1)
        document_embeddings_norm = torch.nn.functional.normalize(document_embeddings, p=2, dim=1)
        
        # Compute similarity matrix (batch_size, num_docs)
        similarity = torch.mm(query_embedding_norm, document_embeddings_norm.t())
        
        # Get top-k similar documents for each query
        k = min(k, document_embeddings.shape[0])  # Ensure k is not larger than number of docs
        top_k_values, top_k_indices = torch.topk(similarity, k=k, dim=1)
        
        # Convert to lists
        indices = top_k_indices.cpu().numpy().tolist()
        scores = top_k_values.cpu().numpy().tolist()
        
        return indices, scores
