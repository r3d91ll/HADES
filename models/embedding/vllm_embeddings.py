"""
vLLM-based Embedding Model implementation.

This module provides an embedding model implementation that uses vLLM
to generate embeddings from the ModernBERT model locally.
"""

import os
import logging
from typing import Any, Dict, List, Optional, Union
import numpy as np

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

from hades.models.interface import EmbeddingModel


class VLLMEmbeddingModel(EmbeddingModel):
    """vLLM-based embedding model for local embedding generation."""
    
    def __init__(self) -> None:
        """Initialize the vLLM embedding model."""
        super().__init__()
        self.model = None
        self.model_name = None
        self.dimensions = None
        self.device = None
        self.batch_size = 32
        self.logger = logging.getLogger(__name__)
        
        if not VLLM_AVAILABLE:
            self.logger.warning("vLLM is not available. Please install it with 'pip install vllm'.")
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the embedding model with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.model_name = config.get('model_name', 'answerdotai/ModernBERT-base')
        self.dimensions = config.get('dimensions', 768)  # Default for ModernBERT-base
        self.device = config.get('device', 'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu')
        self.batch_size = config.get('batch_size', 32)
        self.tensor_parallel_size = config.get('tensor_parallel_size', 1)
        self.max_model_len = config.get('max_model_len', 512)
        
        # Additional vLLM-specific configuration
        self.vllm_config = {
            'tensor_parallel_size': self.tensor_parallel_size,
            'max_model_len': self.max_model_len,
            'gpu_memory_utilization': config.get('gpu_memory_utilization', 0.9),
            'dtype': config.get('dtype', 'half'),
            'trust_remote_code': config.get('trust_remote_code', True)
        }
    
    def load(self) -> bool:
        """Load the embedding model.
        
        Returns:
            True if successful, False otherwise
        """
        if not VLLM_AVAILABLE:
            self.logger.error("vLLM is not available. Please install it with 'pip install vllm'.")
            return False
        
        try:
            self.logger.info(f"Loading model {self.model_name} with vLLM on {self.device}")
            
            # Initialize vLLM model
            self.model = LLM(
                model=self.model_name,
                tensor_parallel_size=self.vllm_config['tensor_parallel_size'],
                max_model_len=self.vllm_config['max_model_len'],
                gpu_memory_utilization=self.vllm_config['gpu_memory_utilization'],
                dtype=self.vllm_config['dtype'],
                trust_remote_code=self.vllm_config['trust_remote_code']
            )
            
            return True
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return False
    
    def _extract_embeddings_from_outputs(self, outputs) -> List[List[float]]:
        """Extract embeddings from vLLM outputs.
        
        Args:
            outputs: vLLM model outputs
            
        Returns:
            List of embeddings (as lists of floats)
        """
        # For ModernBERT, we use the CLS token embedding from the last hidden state
        # This is a simplified approach - production code would need to match the exact
        # embedding extraction method used by the model authors
        embeddings = []
        
        for output in outputs:
            # Get the hidden states from the model output
            # The exact implementation depends on how vLLM exposes the embeddings
            # This is a placeholder - actual implementation will depend on vLLM's API
            hidden_states = output.hidden_states
            
            # Extract the CLS token embedding (first token)
            cls_embedding = hidden_states[0].tolist()
            embeddings.append(cls_embedding)
        
        return embeddings
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to generate embeddings for
            
        Returns:
            List of embeddings (as lists of floats)
        """
        if not VLLM_AVAILABLE:
            raise RuntimeError("vLLM is not available. Please install it with 'pip install vllm'.")
        
        if not self.model:
            if not self.load():
                raise RuntimeError("Failed to load embedding model")
        
        try:
            # Process in batches to avoid memory issues
            all_embeddings = []
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                self.logger.debug(f"Processing batch {i//self.batch_size + 1} with {len(batch)} texts")
                
                # Set up sampling parameters for embedding extraction
                sampling_params = SamplingParams(
                    temperature=0.0,  # Deterministic
                    max_tokens=0,     # We only need the embeddings, not generation
                )
                
                # Generate outputs with vLLM
                outputs = self.model.generate(batch, sampling_params)
                
                # Extract embeddings from outputs
                embeddings = self._extract_embeddings_from_outputs(outputs)
                all_embeddings.extend(embeddings)
            
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
        return 'embedding_model.vllm'
