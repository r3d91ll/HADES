"""
Unit tests for Inductive ISNE Processor.

Tests the InductiveISNE component implementation including neural network training,
inductive enhancement, and PyTorch model management.
"""

import pytest
import numpy as np
from typing import Dict, Any, List
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock
import tempfile

from src.components.graph_enhancement.isne.inductive.processor import InductiveISNE
from src.types.components.contracts import (
    ComponentType,
    GraphEnhancementInput,
    GraphEnhancementOutput,
    ChunkEmbedding,
    EnhancedEmbedding,
    ProcessingStatus
)


class TestInductiveISNE:
    """Test suite for InductiveISNE processor."""
    
    @pytest.fixture
    def sample_config(self) -> Dict[str, Any]:
        """Sample configuration for testing."""
        return {
            "hidden_dim": 8,
            "num_layers": 2,
            "dropout": 0.1,
            "device": "cpu",  # Force CPU for testing
            "learning_rate": 0.01,
            "max_epochs": 5,  # Small number for fast testing
            "early_stopping_patience": 3,
            "input_dim": 4
        }
    
    @pytest.fixture
    def inductive_isne(self, sample_config: Dict[str, Any]) -> InductiveISNE:
        """Create InductiveISNE instance for testing."""
        return InductiveISNE(config=sample_config)
    
    @pytest.fixture
    def sample_embeddings(self) -> List[List[float]]:
        """Sample embeddings for testing."""
        return [
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
            [0.2, 0.3, 0.4, 0.5],
            [0.6, 0.7, 0.8, 0.9],
            [0.1, 0.1, 0.1, 0.1],
            [0.9, 0.8, 0.7, 0.6],
            [0.3, 0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9, 1.0]
        ]
    
    @pytest.fixture
    def sample_input(self, sample_embeddings: List[List[float]]) -> GraphEnhancementInput:
        """Sample input data for testing."""
        chunk_embeddings = []
        for i, embedding in enumerate(sample_embeddings):
            chunk_embeddings.append(ChunkEmbedding(
                chunk_id=f"chunk_{i}",
                embedding=embedding,
                metadata={"index": i}
            ))
        
        return GraphEnhancementInput(
            embeddings=chunk_embeddings,
            metadata={"test": True},
            enhancement_options={"method": "inductive"}
        )
    
    def test_initialization(self, sample_config: Dict[str, Any]):
        """Test InductiveISNE initialization."""
        enhancer = InductiveISNE(config=sample_config)
        
        assert enhancer.name == "inductive"
        assert enhancer.version == "1.0.0"
        assert enhancer.component_type == ComponentType.GRAPH_ENHANCEMENT
        assert enhancer._hidden_dim == 8
        assert enhancer._num_layers == 2
        assert enhancer._dropout == 0.1
        assert enhancer._device == "cpu"
        assert not enhancer._is_trained
        assert enhancer._model is None
    
    def test_initialization_default_config(self):
        """Test InductiveISNE initialization with default config."""
        enhancer = InductiveISNE()
        
        assert enhancer.name == "inductive"
        assert enhancer._hidden_dim == 256  # Default
        assert enhancer._num_layers == 2  # Default
        assert enhancer._dropout == 0.1  # Default
        # Device should be determined automatically
        assert enhancer._device in ["cuda", "cpu"]
    
    def test_configure(self, inductive_isne: InductiveISNE):
        """Test configuration updates."""
        new_config = {
            "hidden_dim": 16,
            "num_layers": 3,
            "dropout": 0.2,
            "device": "cpu"
        }
        
        inductive_isne.configure(new_config)
        
        assert inductive_isne._hidden_dim == 16
        assert inductive_isne._num_layers == 3
        assert inductive_isne._dropout == 0.2
        assert inductive_isne._device == "cpu"
    
    def test_configure_invalid_config(self, inductive_isne: InductiveISNE):
        """Test configuration with invalid parameters."""
        invalid_configs = [
            {"hidden_dim": 0},
            {"num_layers": -1},
            {"dropout": 1.5},  # Out of range
            {"dropout": -0.1},  # Negative
            "not_a_dict",
        ]
        
        for invalid_config in invalid_configs:
            with pytest.raises(ValueError, match="Invalid configuration"):
                inductive_isne.configure(invalid_config)
    
    def test_validate_config(self, inductive_isne: InductiveISNE):
        """Test configuration validation."""
        valid_configs = [
            {"hidden_dim": 128, "num_layers": 3},
            {"dropout": 0.5},
            {"device": "cuda"},
            {},  # Empty config should be valid
        ]
        
        for config in valid_configs:
            assert inductive_isne.validate_config(config) is True
        
        invalid_configs = [
            {"hidden_dim": 0},
            {"num_layers": -1},
            {"dropout": 2.0},
            "not_a_dict"
        ]
        
        for config in invalid_configs:
            assert inductive_isne.validate_config(config) is False
    
    def test_get_config_schema(self, inductive_isne: InductiveISNE):
        """Test configuration schema retrieval."""
        schema = inductive_isne.get_config_schema()
        
        assert isinstance(schema, dict)
        assert "type" in schema
        assert schema["type"] == "object"
        assert "properties" in schema
        
        properties = schema["properties"]
        assert "hidden_dim" in properties
        assert "num_layers" in properties
        assert "dropout" in properties
        assert "device" in properties
        assert "learning_rate" in properties
        assert "max_epochs" in properties
    
    @patch('torch.cuda.is_available')
    def test_health_check_cpu(self, mock_cuda, inductive_isne: InductiveISNE):
        """Test health check functionality with CPU."""
        mock_cuda.return_value = False
        
        # Should pass with CPU device
        assert inductive_isne.health_check() is True
        assert inductive_isne._model is not None
    
    @patch('torch.cuda.is_available')
    def test_health_check_cuda_not_available(self, mock_cuda):
        """Test health check when CUDA is requested but not available."""
        mock_cuda.return_value = False
        
        config = {"device": "cuda"}
        enhancer = InductiveISNE(config=config)
        
        # Should fail when CUDA is requested but not available
        assert enhancer.health_check() is False
    
    def test_get_metrics(self, inductive_isne: InductiveISNE):
        """Test metrics retrieval."""
        metrics = inductive_isne.get_metrics()
        
        assert isinstance(metrics, dict)
        assert "component_name" in metrics
        assert "component_version" in metrics
        assert "hidden_dim" in metrics
        assert "num_layers" in metrics
        assert "device" in metrics
        assert "is_trained" in metrics
        assert "training_epochs" in metrics
        
        assert metrics["component_name"] == "inductive"
        assert metrics["component_version"] == "1.0.0"
        assert metrics["is_trained"] is False  # Initially not trained
        assert metrics["training_epochs"] == 0
    
    def test_enhance_without_training(self, inductive_isne: InductiveISNE, sample_input: GraphEnhancementInput):
        """Test enhancement without training (should use random initialization)."""
        result = inductive_isne.enhance(sample_input)
        
        assert isinstance(result, GraphEnhancementOutput)
        assert len(result.enhanced_embeddings) == len(sample_input.embeddings)
        assert result.metadata.status == ProcessingStatus.SUCCESS
        
        # Check enhanced embeddings
        for enhanced_emb in result.enhanced_embeddings:
            assert isinstance(enhanced_emb, EnhancedEmbedding)
            assert enhanced_emb.chunk_id is not None
            assert len(enhanced_emb.enhanced_embedding) > 0
            assert enhanced_emb.enhancement_score >= 0.0
            assert enhanced_emb.graph_features["enhancement_method"] == "inductive_neural"
    
    def test_train_method(self, inductive_isne: InductiveISNE, sample_input: GraphEnhancementInput):
        """Test explicit training method."""
        assert not inductive_isne.is_trained()
        
        inductive_isne.train(sample_input)
        
        assert inductive_isne.is_trained()
        assert len(inductive_isne._training_history) > 0
        
        # Check training history
        history_entry = inductive_isne._training_history[0]
        assert "epoch" in history_entry
        assert "loss" in history_entry
        assert "timestamp" in history_entry
    
    def test_train_method_no_data(self, inductive_isne: InductiveISNE):
        """Test training method with no data."""
        empty_input = GraphEnhancementInput(
            embeddings=[],
            metadata={},
            enhancement_options={}
        )
        
        with pytest.raises(RuntimeError, match="Training failed"):
            inductive_isne.train(empty_input)
    
    def test_is_trained(self, inductive_isne: InductiveISNE, sample_input: GraphEnhancementInput):
        """Test is_trained method."""
        assert not inductive_isne.is_trained()
        
        inductive_isne.train(sample_input)
        
        assert inductive_isne.is_trained()
    
    def test_save_model(self, inductive_isne: InductiveISNE, sample_input: GraphEnhancementInput):
        """Test model saving functionality."""
        # Train the model first
        inductive_isne.train(sample_input)
        
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            temp_path = f.name
        
        try:
            inductive_isne.save_model(temp_path)
            
            # File should exist
            import os
            assert os.path.exists(temp_path)
            assert os.path.getsize(temp_path) > 0
            
        finally:
            import os
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_save_model_no_model(self, inductive_isne: InductiveISNE):
        """Test saving when no model exists."""
        with pytest.raises(IOError, match="Failed to save model"):
            inductive_isne.save_model("/tmp/test_model.pth")
    
    def test_load_model_file_not_found(self, inductive_isne: InductiveISNE):
        """Test loading a model file that doesn't exist."""
        with pytest.raises(IOError, match="Failed to load model"):
            inductive_isne.load_model("/non/existent/path.pth")
    
    def test_get_model_info(self, inductive_isne: InductiveISNE):
        """Test model info retrieval."""
        info = inductive_isne.get_model_info()
        
        assert isinstance(info, dict)
        assert info["model_type"] == "inductive_isne"
        assert info["hidden_dim"] == inductive_isne._hidden_dim
        assert info["num_layers"] == inductive_isne._num_layers
        assert info["device"] == inductive_isne._device
        assert info["is_trained"] is False
        assert info["parameter_count"] >= 0
        assert info["training_epochs"] == 0
    
    def test_supports_incremental_training(self, inductive_isne: InductiveISNE):
        """Test incremental training support."""
        assert inductive_isne.supports_incremental_training() is True
    
    def test_initialize_model(self, inductive_isne: InductiveISNE):
        """Test model initialization."""
        assert inductive_isne._model is None
        
        inductive_isne._initialize_model()
        
        assert inductive_isne._model is not None
        
        # Check model structure
        import torch.nn as nn
        assert isinstance(inductive_isne._model, nn.Sequential)
        
        # Count parameters
        param_count = inductive_isne._count_parameters()
        assert param_count > 0
    
    def test_enhance_single_embedding(self, inductive_isne: InductiveISNE):
        """Test enhancement of a single embedding."""
        chunk_emb = ChunkEmbedding(
            chunk_id="test_chunk",
            embedding=[0.1, 0.2, 0.3, 0.4],
            metadata={}
        )
        
        # Initialize model
        inductive_isne._initialize_model()
        
        enhanced_emb = inductive_isne._enhance_single_embedding(chunk_emb, {})
        
        assert isinstance(enhanced_emb, EnhancedEmbedding)
        assert enhanced_emb.chunk_id == "test_chunk"
        assert len(enhanced_emb.enhanced_embedding) == 4
        assert enhanced_emb.enhancement_score >= 0.0
        assert enhanced_emb.graph_features["enhancement_method"] == "inductive_neural"
    
    def test_calculate_enhancement_score(self, inductive_isne: InductiveISNE):
        """Test enhancement score calculation."""
        original = [1.0, 0.0, 0.0]
        enhanced = [0.8, 0.2, 0.1]
        
        score = inductive_isne._calculate_enhancement_score(original, enhanced)
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
    
    def test_calculate_quality_score(self, inductive_isne: InductiveISNE):
        """Test overall quality score calculation."""
        enhanced_embeddings = [
            EnhancedEmbedding(
                chunk_id="chunk_1",
                original_embedding=[1.0, 0.0],
                enhanced_embedding=[0.8, 0.2],
                graph_features={},
                enhancement_score=0.8,
                metadata={}
            ),
            EnhancedEmbedding(
                chunk_id="chunk_2",
                original_embedding=[0.0, 1.0],
                enhanced_embedding=[0.1, 0.9],
                graph_features={},
                enhancement_score=0.9,
                metadata={}
            )
        ]
        
        quality_score = inductive_isne._calculate_quality_score(enhanced_embeddings)
        
        assert isinstance(quality_score, float)
        assert quality_score == 0.85  # Average of 0.8 and 0.9
    
    def test_calculate_quality_score_empty(self, inductive_isne: InductiveISNE):
        """Test quality score calculation with empty list."""
        quality_score = inductive_isne._calculate_quality_score([])
        assert quality_score == 0.0
    
    def test_count_parameters(self, inductive_isne: InductiveISNE):
        """Test parameter counting."""
        # Without model
        assert inductive_isne._count_parameters() == 0
        
        # With model
        inductive_isne._initialize_model()
        param_count = inductive_isne._count_parameters()
        assert param_count > 0
    
    def test_training_metrics_update(self, inductive_isne: InductiveISNE, sample_input: GraphEnhancementInput):
        """Test that training updates metrics properly."""
        initial_metrics = inductive_isne.get_metrics()
        assert initial_metrics["training_epochs"] == 0
        assert "last_training_loss" not in initial_metrics
        
        inductive_isne.train(sample_input)
        
        updated_metrics = inductive_isne.get_metrics()
        assert updated_metrics["training_epochs"] > 0
        assert "last_training_loss" in updated_metrics
        assert "best_validation_score" in updated_metrics
    
    def test_enhance_with_trained_model(self, inductive_isne: InductiveISNE, sample_input: GraphEnhancementInput):
        """Test enhancement after training."""
        # Train the model
        inductive_isne.train(sample_input)
        assert inductive_isne.is_trained()
        
        # Test enhancement
        result = inductive_isne.enhance(sample_input)
        
        assert isinstance(result, GraphEnhancementOutput)
        assert len(result.enhanced_embeddings) == len(sample_input.embeddings)
        assert result.metadata.status == ProcessingStatus.SUCCESS
        
        # Model info should reflect trained state
        assert result.model_info["model_type"] == "inductive_isne"
        assert result.graph_stats["model_trained"] is True
    
    def test_early_stopping(self, inductive_isne: InductiveISNE, sample_input: GraphEnhancementInput):
        """Test early stopping during training."""
        # Configure for early stopping
        inductive_isne.configure({
            "max_epochs": 100,  # High number
            "early_stopping_patience": 2  # Low patience for quick stopping
        })
        
        inductive_isne.train(sample_input)
        
        # Should have stopped early
        assert len(inductive_isne._training_history) < 100
        assert inductive_isne.is_trained()
    
    def test_training_loss_decreases(self, inductive_isne: InductiveISNE, sample_input: GraphEnhancementInput):
        """Test that training loss generally decreases."""
        inductive_isne.train(sample_input)
        
        history = inductive_isne._training_history
        assert len(history) > 1
        
        # Loss should generally decrease (allow some fluctuation)
        first_loss = history[0]["loss"]
        last_loss = history[-1]["loss"]
        assert last_loss <= first_loss  # Should improve or stay same
    
    def test_device_handling(self):
        """Test device configuration and handling."""
        # Test auto device selection
        auto_config = {"device": "auto"}
        enhancer_auto = InductiveISNE(config=auto_config)
        assert enhancer_auto._device in ["cuda", "cpu"]
        
        # Test explicit CPU
        cpu_config = {"device": "cpu"}
        enhancer_cpu = InductiveISNE(config=cpu_config)
        assert enhancer_cpu._device == "cpu"
    
    def test_error_handling_in_enhancement(self, inductive_isne: InductiveISNE):
        """Test error handling during enhancement."""
        # Create input with problematic data
        problematic_input = GraphEnhancementInput(
            embeddings=[ChunkEmbedding(
                chunk_id="problematic_chunk",
                embedding=[],  # Empty embedding might cause issues
                metadata={}
            )],
            metadata={},
            enhancement_options={}
        )
        
        result = inductive_isne.enhance(problematic_input)
        
        # Should handle gracefully
        assert isinstance(result, GraphEnhancementOutput)
        # Might have errors or be processed depending on implementation
        assert result.metadata.status in [ProcessingStatus.SUCCESS, ProcessingStatus.ERROR]
    
    def test_enhancement_consistency(self, inductive_isne: InductiveISNE, sample_input: GraphEnhancementInput):
        """Test that enhancement produces consistent results for same input."""
        # Initialize model to ensure deterministic behavior
        inductive_isne._initialize_model()
        
        # Set model to eval mode for consistent results
        inductive_isne._model.eval()
        
        result1 = inductive_isne.enhance(sample_input)
        result2 = inductive_isne.enhance(sample_input)
        
        assert len(result1.enhanced_embeddings) == len(result2.enhanced_embeddings)
        
        # Results should be very close (allowing for minor floating point differences)
        for emb1, emb2 in zip(result1.enhanced_embeddings, result2.enhanced_embeddings):
            assert emb1.chunk_id == emb2.chunk_id
            np.testing.assert_array_almost_equal(
                emb1.enhanced_embedding,
                emb2.enhanced_embedding,
                decimal=3  # Allow some variance due to PyTorch operations
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])