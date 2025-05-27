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
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

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
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape [batch_size, in_features]
            
        Returns:
            Output tensor of shape [batch_size, out_features]
        """
        # Explicitly cast the return value to Tensor to satisfy type checking
        result: torch.Tensor = self.linear(x)
        return result
    
    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False) -> Any:
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
            # Return the result of the parent's load_state_dict call
            return super().load_state_dict(state_dict, strict=strict, assign=assign)
        
        # Handle case where state_dict is nested in 'model_state_dict'
        elif 'model_state_dict' in state_dict:
            inner_dict = state_dict['model_state_dict']
            if isinstance(inner_dict, dict):
                # Direct assignment for compatibility
                if 'linear.weight' in inner_dict:
                    return super().load_state_dict(inner_dict, strict=strict, assign=assign)
                
                # Try to adapt based on keys
                adapted_dict = {}
                for k, v in inner_dict.items():
                    if k.startswith('linear.'):
                        adapted_dict[k] = v
                
                if adapted_dict:
                    return super().load_state_dict(adapted_dict, strict=False, assign=assign)
        
        # Handle case where weights/biases are exposed directly
        # This is common in exported/simplified models
        if 'weight' in state_dict and 'bias' in state_dict:
            adapted_dict = {
                'linear.weight': state_dict['weight'],
                'linear.bias': state_dict['bias']
            }
            return super().load_state_dict(adapted_dict, strict=False, assign=assign)
        
        # Fall back to default behavior
        return super().load_state_dict(state_dict, strict=False, assign=assign)
