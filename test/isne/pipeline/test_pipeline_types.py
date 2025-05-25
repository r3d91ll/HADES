"""
Tests for type safety and behavior of the ISNE pipeline.

This module tests the type safety of the ISNE pipeline implementation,
focusing on proper model handling, input validation, and type conversion.
"""

import unittest
from typing import Dict, Any, Optional, Union, cast
import torch
import numpy as np
from pathlib import Path

from src.isne.pipeline.isne_pipeline import ISNEPipeline, ISNEModelTypes
from src.isne.models.isne_model import ISNEModel
from src.isne.models.simplified_isne_model import SimplifiedISNEModel


class TestPipelineTypeSafety(unittest.TestCase):
    """Tests for type safety in the ISNE pipeline."""

    def setUp(self):
        """Set up test fixtures."""
        # Create minimal tensors for testing
        self.dummy_embedding = torch.randn(10, 768)
        self.dummy_edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        
        # Create mock model data for testing
        self.mock_model_data = {
            "model_state_dict": {},
            "model_type": "simplified",
            "embedding_dim": 768,
            "hidden_dim": 256,
            "output_dim": 128,
            "num_layers": 2,
            "metadata": {
                "creation_date": "2025-05-25",
                "version": "1.0.0"
            }
        }

    def test_model_type_union(self):
        """Test the ISNEModelTypes union type works correctly."""
        # Create both model types
        simple_model = SimplifiedISNEModel(in_features=768, hidden_features=256, out_features=128)
        full_model = ISNEModel(in_features=768, hidden_features=256, out_features=128, num_layers=2)
        
        # Test type annotation with both model types
        model: ISNEModelTypes
        
        # Should work with SimplifiedISNEModel
        model = simple_model
        self.assertIsInstance(model, SimplifiedISNEModel)
        
        # Should also work with ISNEModel
        model = full_model
        self.assertIsInstance(model, ISNEModel)
        
        # Test with the cast function
        model = cast(ISNEModelTypes, simple_model)
        self.assertIsInstance(model, SimplifiedISNEModel)

    def test_tensor_type_handling(self):
        """Test proper tensor type handling in pipeline code."""
        # Create tensor with different dtypes
        float_tensor = torch.randn(10, 10)
        int_tensor = torch.ones(10, 10, dtype=torch.int)
        long_tensor = torch.ones(10, 10, dtype=torch.long)
        
        # Test the type checking logic directly
        self.assertTrue(isinstance(float_tensor, torch.Tensor) and float_tensor.dtype == torch.float)
        self.assertTrue(isinstance(int_tensor, torch.Tensor) and int_tensor.dtype != torch.float)
        self.assertTrue(isinstance(long_tensor, torch.Tensor) and long_tensor.dtype != torch.float)

    def test_get_embeddings_return_type(self):
        """Test that get_embeddings always returns a tensor."""
        # Create a test model
        model = ISNEModel(in_features=768, hidden_features=256, out_features=128)
        
        # Call get_embeddings with some test data
        embeddings = model.get_embeddings(self.dummy_embedding, self.dummy_edge_index)
        
        # Verify return type
        self.assertIsInstance(embeddings, torch.Tensor)
        self.assertEqual(embeddings.shape[1], 128)  # Should match output dimension

    def test_project_features_return_type(self):
        """Test that project_features always returns a tensor."""
        # Create a test model
        model = ISNEModel(in_features=768, hidden_features=256, out_features=128)
        
        # Call project_features with some test data
        projected = model.project_features(self.dummy_embedding)
        
        # Verify return type
        self.assertIsInstance(projected, torch.Tensor)
        self.assertEqual(projected.shape[1], 128)  # Should match output dimension


if __name__ == "__main__":
    unittest.main()
