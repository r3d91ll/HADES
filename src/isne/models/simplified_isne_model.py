"""
Simplified ISNE model adapter for code-enabled training.

This module provides a simplified version of the ISNE model that works with 
the code-enabled training outputs, adapting them to the interface expected
by the ingestion pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Any, Optional, List, Union

logger = logging.getLogger(__name__)

class SimplifiedISNEModel(nn.Module):
    """
    A simplified ISNE model that wraps a single linear projection layer
    but presents the same interface as the full ISNE model expected by the pipeline.
    
    This adapter enables the use of simpler trained models with the full pipeline.
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        *args: Any,
        **kwargs: Any
    ) -> None:
        """
        Initialize the simplified ISNE model adapter.
        
        Args:
            in_features: Dimensionality of input features (e.g., 768 for ModernBERT)
            hidden_features: Not used in this model but kept for compatibility
            out_features: Dimensionality of output embeddings (e.g., 64)
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)
        
        logger.info(f"Initialized SimplifiedISNEModel with input={in_features}, output={out_features}")
        
    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """
        Forward pass for the simplified ISNE model.
        
        Args:
            x: Input tensor of shape [batch_size, in_features]
            
        Returns:
            Output tensor of shape [batch_size, out_features]
        """
        return self.linear(x)
    
    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True) -> None:
        """
        Load a state dictionary from various formats.
        
        This method handles both the simple linear layer format from our
        training and the nested 'model_state_dict' format expected by the pipeline.
        
        Args:
            state_dict: State dictionary to load
            strict: Whether to strictly enforce that the keys match
            
        Returns:
            Result of the load operation
        """
        # Handle case where state_dict is already in the right format
        if "linear.weight" in state_dict and "linear.bias" in state_dict:
            # Just return original dict if already in right format
            super().load_state_dict(state_dict, strict)
            return None
        
        # Handle case where state_dict is nested in 'model_state_dict'
        elif 'model_state_dict' in state_dict:
            super().load_state_dict(state_dict['model_state_dict'], strict)
            return None
        
        # If we get here, we couldn't find the right format
        logger.error(f"Could not load state_dict with keys: {list(state_dict.keys())}")
        if "model_state_dict" in state_dict:
            logger.error(f"model_state_dict has keys: {list(state_dict['model_state_dict'].keys())}")
        
        # Try to load whatever is available, non-strictly
        logger.warning("Attempting to load state dictionary in unknown format")
        super().load_state_dict(state_dict, strict=False)
        return None
