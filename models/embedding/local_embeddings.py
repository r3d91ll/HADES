"""
Local Embedding Model implementation using sentence-transformers.

This module provides a local embedding model implementation that uses
sentence-transformers to generate embeddings without requiring external API calls.
"""

import os
import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
from sentence_transformers import SentenceTransformer

from hades.models.interface import EmbeddingModel


class LocalEmbeddingModel(EmbeddingModel):
    """Local embedding model using sentence-transformers."""
    
    def __init__(self) -> None:
        """Initialize the local embedding model."""
        super().__init__()
        self.model = None
        self.model_name = None
        self.dimensions = None
        self.device = None
        self.batch_size = 32
        self.logger = logging.getLogger(__name__)
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the embedding model with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.model_name = config.get('model_name', 'all-MiniLM-L6-v2')
        self.dimensions = config.get('dimensions', 384)  # Default for all-MiniLM-L6-v2
        self.device = config.get('device', 'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu')
        self.batch_size = config.get('batch_size', 32)
        
        # Map common model names to their dimensions
        model_dimensions = {
            'all-MiniLM-L6-v2': 384,
            'all-mpnet-base-v2': 768,
            'multi-qa-mpnet-base-dot-v1': 768,
            'all-distilroberta-v1': 768,
            'paraphrase-multilingual-mpnet-base-v2': 768,
            'paraphrase-albert-small-v2': 768,
            'msmarco-distilbert-base-v4': 768
        }
        
        # Update dimensions if the model is known
        if self.model_name in model_dimensions and 'dimensions' not in config:
            self.dimensions = model_dimensions[self.model_name]
            self.logger.info(f"Using {self.dimensions} dimensions for model {self.model_name}")
    
    def load(self) -> bool:
        """Load the embedding model.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info(f"Loading model {self.model_name} on {self.device}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            return True
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return False
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings.
        
        Returns:
            Embedding dimension
        """
        return self.dimensions
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to generate embeddings for
            
        Returns:
            List of embeddings (as lists of floats)
        """
        if not self.model:
            if not self.load():
                raise RuntimeError("Failed to load embedding model")
        
        try:
            # Process in batches to avoid memory issues
            all_embeddings = []
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                self.logger.debug(f"Processing batch {i//self.batch_size + 1} with {len(batch)} texts")
                
                # Generate embeddings
                embeddings = self.model.encode(batch, convert_to_numpy=True)
                
                # Convert to list of lists
                embeddings_list = embeddings.tolist()
                all_embeddings.extend(embeddings_list)
            
            return all_embeddings
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {e}")
            raise
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            Embedding as a list of floats
        """
        embeddings = self.generate_embeddings([text])
        return embeddings[0]
    
    @classmethod
    def get_type(cls) -> str:
        """Get the type of the component.
        
        Returns:
            Component type
        """
        return 'embedding_model.local'
