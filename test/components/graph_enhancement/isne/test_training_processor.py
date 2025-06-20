"""
Unit tests for ISNE Training Processor.

Tests the ISNETrainingEnhancer component implementation including training functionality,
model state management, and enhancement with trained models.
"""

import pytest
import numpy as np
from typing import Dict, Any, List
from datetime import datetime, timezone

from src.components.graph_enhancement.isne.training.processor import ISNETrainingEnhancer
from src.types.components.contracts import (
    ComponentType,
    GraphEnhancementInput,
    GraphEnhancementOutput,
    ChunkEmbedding,
    EnhancedEmbedding,
    ProcessingStatus
)


class TestISNETrainingEnhancer:
    """Test suite for ISNETrainingEnhancer processor."""
    
    @pytest.fixture
    def sample_config(self) -> Dict[str, Any]:
        """Sample configuration for testing."""
        return {
            "training_config": {
                "epochs": 10,
                "batch_size": 32,
                "learning_rate": 0.001,
                "patience": 5,
                "validation_split": 0.2
            },
            "model_config": {
                "embedding_dim": 4,
                "hidden_dim": 8,
                "num_layers": 2,
                "dropout": 0.1
            }
        }
    
    @pytest.fixture
    def training_enhancer(self, sample_config: Dict[str, Any]) -> ISNETrainingEnhancer:
        """Create ISNETrainingEnhancer instance for testing."""
        return ISNETrainingEnhancer(config=sample_config)
    
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
            [0.7, 0.8, 0.9, 1.0],
            [0.0, 0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6, 0.7],
            [0.8, 0.9, 1.0, 0.9],
            [0.2, 0.1, 0.3, 0.2]  # 12 embeddings for sufficient training data
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
            enhancement_options={"method": "isne_training"}
        )
    
    def test_initialization(self, sample_config: Dict[str, Any]):
        """Test ISNETrainingEnhancer initialization."""
        enhancer = ISNETrainingEnhancer(config=sample_config)
        
        assert enhancer.name == "isne_training"
        assert enhancer.version == "1.0.0"
        assert enhancer.component_type == ComponentType.GRAPH_ENHANCEMENT
        assert not enhancer._is_trained
        assert enhancer._model_parameters is None
        assert len(enhancer._training_history) == 0
    
    def test_initialization_default_config(self):
        """Test ISNETrainingEnhancer initialization with default config."""
        enhancer = ISNETrainingEnhancer()
        
        assert enhancer.name == "isne_training"
        assert enhancer._training_config == {}
        assert enhancer._model_config == {}
        assert not enhancer._is_trained
    
    def test_configure(self, training_enhancer: ISNETrainingEnhancer):
        """Test configuration updates."""
        new_config = {
            "training_config": {
                "epochs": 20,
                "learning_rate": 0.002
            },
            "model_config": {
                "embedding_dim": 6,
                "hidden_dim": 12
            }
        }
        
        training_enhancer.configure(new_config)
        
        assert training_enhancer._training_config["epochs"] == 20
        assert training_enhancer._training_config["learning_rate"] == 0.002
        assert training_enhancer._model_config["embedding_dim"] == 6
        assert training_enhancer._model_config["hidden_dim"] == 12
        
        # Training state should be reset
        assert not training_enhancer._is_trained
        assert training_enhancer._model_parameters is None
    
    def test_configure_invalid_config(self, training_enhancer: ISNETrainingEnhancer):
        """Test configuration with invalid parameters."""
        invalid_configs = [
            {"training_config": {"epochs": -1}},
            {"training_config": {"learning_rate": -0.1}},
            {"model_config": {"embedding_dim": 0}},
            {"training_config": "not_dict"},
            {"model_config": {"embedding_dim": "not_int"}},
        ]
        
        for invalid_config in invalid_configs:
            with pytest.raises(ValueError, match="Invalid configuration"):
                training_enhancer.configure(invalid_config)
    
    def test_validate_config(self, training_enhancer: ISNETrainingEnhancer):
        """Test configuration validation."""
        valid_configs = [
            {"training_config": {"epochs": 50, "learning_rate": 0.01}},
            {"model_config": {"embedding_dim": 768, "hidden_dim": 256}},
            {},  # Empty config should be valid
            {"training_config": {"batch_size": 64}}
        ]
        
        for config in valid_configs:
            assert training_enhancer.validate_config(config) is True
        
        invalid_configs = [
            {"training_config": {"epochs": 0}},
            {"training_config": {"learning_rate": 0}},
            {"model_config": {"embedding_dim": -1}},
            "not_a_dict",
            {"training_config": "not_dict"}
        ]
        
        for config in invalid_configs:
            assert training_enhancer.validate_config(config) is False
    
    def test_get_config_schema(self, training_enhancer: ISNETrainingEnhancer):
        """Test configuration schema retrieval."""
        schema = training_enhancer.get_config_schema()
        
        assert isinstance(schema, dict)
        assert "type" in schema
        assert schema["type"] == "object"
        assert "properties" in schema
        
        properties = schema["properties"]
        assert "training_config" in properties
        assert "model_config" in properties
        
        # Check training config schema
        training_schema = properties["training_config"]["properties"]
        assert "epochs" in training_schema
        assert "learning_rate" in training_schema
        assert "batch_size" in training_schema
        
        # Check model config schema
        model_schema = properties["model_config"]["properties"]
        assert "embedding_dim" in model_schema
        assert "hidden_dim" in model_schema
        assert "num_layers" in model_schema
    
    def test_health_check(self, training_enhancer: ISNETrainingEnhancer):
        """Test health check functionality."""
        assert training_enhancer.health_check() is True
    
    def test_get_metrics(self, training_enhancer: ISNETrainingEnhancer):
        """Test metrics retrieval."""
        metrics = training_enhancer.get_metrics()
        
        assert isinstance(metrics, dict)
        assert "component_name" in metrics
        assert "component_version" in metrics
        assert "is_trained" in metrics
        assert "training_epochs_completed" in metrics
        assert "total_enhancements" in metrics
        assert "avg_processing_time" in metrics
        
        assert metrics["component_name"] == "isne_training"
        assert metrics["component_version"] == "1.0.0"
        assert metrics["is_trained"] is False  # Initially not trained
        assert metrics["training_epochs_completed"] == 0
    
    def test_train_on_data(self, training_enhancer: ISNETrainingEnhancer, sample_embeddings: List[List[float]]):
        """Test training on embedding data."""
        assert not training_enhancer._is_trained
        assert training_enhancer._model_parameters is None
        
        training_enhancer._train_on_data(sample_embeddings)
        
        assert training_enhancer._is_trained
        assert training_enhancer._model_parameters is not None
        assert len(training_enhancer._training_history) > 0
        
        # Check model parameters structure
        params = training_enhancer._model_parameters
        assert "weight_matrix" in params
        assert "bias" in params
        assert "output_weight" in params
        assert "output_bias" in params
        
        # Check parameter shapes
        embedding_dim = len(sample_embeddings[0])
        hidden_dim = training_enhancer._model_config.get('hidden_dim', 256)
        
        assert params["weight_matrix"].shape == (embedding_dim, hidden_dim)
        assert params["bias"].shape == (hidden_dim,)
        assert params["output_weight"].shape == (hidden_dim, embedding_dim)
        assert params["output_bias"].shape == (embedding_dim,)
    
    def test_enhance_with_training(self, training_enhancer: ISNETrainingEnhancer, sample_input: GraphEnhancementInput):
        """Test enhancement that triggers training and uses trained model."""
        assert not training_enhancer._is_trained
        
        result = training_enhancer.enhance(sample_input)
        
        assert isinstance(result, GraphEnhancementOutput)
        assert len(result.enhanced_embeddings) == len(sample_input.embeddings)
        assert result.metadata.status == ProcessingStatus.SUCCESS
        assert len(result.errors) == 0
        
        # Should have triggered training
        assert training_enhancer._is_trained
        assert training_enhancer._model_parameters is not None
        
        # Check enhanced embeddings
        for enhanced_emb in result.enhanced_embeddings:
            assert isinstance(enhanced_emb, EnhancedEmbedding)
            assert enhanced_emb.chunk_id is not None
            assert len(enhanced_emb.enhanced_embedding) > 0
            assert enhanced_emb.enhancement_score >= 0.0
            assert enhanced_emb.graph_features["enhancement_method"] == "isne_training"
            assert enhanced_emb.graph_features["is_trained"] is True
    
    def test_enhance_without_sufficient_data(self, training_enhancer: ISNETrainingEnhancer):
        """Test enhancement with insufficient data for training."""
        small_input = GraphEnhancementInput(
            embeddings=[ChunkEmbedding(
                chunk_id="chunk_0",
                embedding=[0.1, 0.2, 0.3],
                metadata={}
            )],
            metadata={},
            enhancement_options={}
        )
        
        result = training_enhancer.enhance(small_input)
        
        assert isinstance(result, GraphEnhancementOutput)
        assert len(result.enhanced_embeddings) == 1
        assert result.metadata.status == ProcessingStatus.SUCCESS
        
        # Should not have triggered training (insufficient data)
        assert not training_enhancer._is_trained
        assert training_enhancer._model_parameters is None
    
    def test_enhance_with_pretrained_model(self, training_enhancer: ISNETrainingEnhancer, sample_embeddings: List[List[float]]):
        """Test enhancement using pre-trained model."""
        # Pre-train the model
        training_enhancer._train_on_data(sample_embeddings)
        assert training_enhancer._is_trained
        
        # Create new input for enhancement
        test_input = GraphEnhancementInput(
            embeddings=[ChunkEmbedding(
                chunk_id="test_chunk",
                embedding=[0.5, 0.5, 0.5, 0.5],
                metadata={}
            )],
            metadata={},
            enhancement_options={}
        )
        
        result = training_enhancer.enhance(test_input)
        
        assert isinstance(result, GraphEnhancementOutput)
        assert len(result.enhanced_embeddings) == 1
        assert result.metadata.status == ProcessingStatus.SUCCESS
        
        enhanced_emb = result.enhanced_embeddings[0]
        assert enhanced_emb.graph_features["is_trained"] is True
    
    def test_train_method(self, training_enhancer: ISNETrainingEnhancer, sample_input: GraphEnhancementInput):
        """Test explicit training method."""
        assert not training_enhancer.is_trained()
        
        training_enhancer.train(sample_input)
        
        assert training_enhancer.is_trained()
        assert training_enhancer._model_parameters is not None
        assert len(training_enhancer._training_history) > 0
    
    def test_train_method_no_data(self, training_enhancer: ISNETrainingEnhancer):
        """Test training method with no data."""
        empty_input = GraphEnhancementInput(
            embeddings=[],
            metadata={},
            enhancement_options={}
        )
        
        with pytest.raises(RuntimeError, match="Training failed"):
            training_enhancer.train(empty_input)
    
    def test_is_trained(self, training_enhancer: ISNETrainingEnhancer, sample_embeddings: List[List[float]]):
        """Test is_trained method."""
        assert not training_enhancer.is_trained()
        
        training_enhancer._train_on_data(sample_embeddings)
        
        assert training_enhancer.is_trained()
    
    def test_estimate_enhancement_time(self, training_enhancer: ISNETrainingEnhancer, sample_input: GraphEnhancementInput):
        """Test enhancement time estimation."""
        # Test untrained model (should include training time)
        time_estimate = training_enhancer.estimate_enhancement_time(sample_input)
        assert isinstance(time_estimate, float)
        assert time_estimate > 0.0
        
        # Train the model
        training_enhancer._train_on_data([emb.embedding for emb in sample_input.embeddings])
        
        # Test trained model (should be faster)
        trained_estimate = training_enhancer.estimate_enhancement_time(sample_input)
        assert trained_estimate < time_estimate  # Should be faster when trained
    
    def test_supports_enhancement_method(self, training_enhancer: ISNETrainingEnhancer):
        """Test enhancement method support checking."""
        supported_methods = ["isne_training", "isne", "training"]
        unsupported_methods = ["inference", "similarity", "unknown"]
        
        for method in supported_methods:
            assert training_enhancer.supports_enhancement_method(method) is True
        
        for method in unsupported_methods:
            assert training_enhancer.supports_enhancement_method(method) is False
    
    def test_get_required_graph_features(self, training_enhancer: ISNETrainingEnhancer):
        """Test required graph features listing."""
        features = training_enhancer.get_required_graph_features()
        
        assert isinstance(features, list)
        assert "node_embeddings" in features
        assert "edge_index" in features
        assert "edge_weights" in features
        assert "node_features" in features
        assert "training_labels" in features
    
    def test_enhance_with_trained_model(self, training_enhancer: ISNETrainingEnhancer, sample_embeddings: List[List[float]]):
        """Test enhancement with trained model."""
        # Pre-train the model
        training_enhancer._train_on_data(sample_embeddings)
        
        enhanced = training_enhancer._enhance_with_trained_model(sample_embeddings[:3])
        
        assert len(enhanced) == 3
        assert all(len(emb) == len(sample_embeddings[0]) for emb in enhanced)
        
        # Enhanced embeddings should be different from originals (residual connection)
        for orig, enh in zip(sample_embeddings[:3], enhanced):
            # Should not be identical due to enhancement
            assert not np.allclose(orig, enh, atol=1e-6)
    
    def test_enhance_embeddings_fallback(self, training_enhancer: ISNETrainingEnhancer, sample_embeddings: List[List[float]]):
        """Test fallback enhancement method."""
        enhanced = training_enhancer._enhance_embeddings_fallback(sample_embeddings)
        
        assert len(enhanced) == len(sample_embeddings)
        assert all(len(emb) == len(sample_embeddings[0]) for emb in enhanced)
    
    def test_enhancement_score_calculation(self, training_enhancer: ISNETrainingEnhancer):
        """Test enhancement score calculation."""
        original = [1.0, 0.0, 0.0]
        enhanced = [0.8, 0.2, 0.1]
        
        score = training_enhancer._calculate_enhancement_score(original, enhanced)
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
    
    def test_training_history_tracking(self, training_enhancer: ISNETrainingEnhancer, sample_embeddings: List[List[float]]):
        """Test that training history is properly tracked."""
        assert len(training_enhancer._training_history) == 0
        
        training_enhancer._train_on_data(sample_embeddings)
        
        assert len(training_enhancer._training_history) > 0
        
        history_entry = training_enhancer._training_history[0]
        assert "timestamp" in history_entry
        assert "epochs" in history_entry
        assert "final_loss" in history_entry
        assert "embedding_count" in history_entry
    
    def test_performance_metrics_update(self, training_enhancer: ISNETrainingEnhancer, sample_input: GraphEnhancementInput):
        """Test that performance metrics are updated after enhancement."""
        initial_metrics = training_enhancer.get_metrics()
        initial_count = initial_metrics["total_enhancements"]
        initial_time = initial_metrics["total_processing_time"]
        
        training_enhancer.enhance(sample_input)
        
        updated_metrics = training_enhancer.get_metrics()
        assert updated_metrics["total_enhancements"] > initial_count
        assert updated_metrics["total_processing_time"] >= initial_time
        assert updated_metrics["is_trained"] is True  # Should be trained after enhancement
    
    def test_empty_input_handling(self, training_enhancer: ISNETrainingEnhancer):
        """Test handling of empty input."""
        empty_input = GraphEnhancementInput(
            embeddings=[],
            metadata={},
            enhancement_options={}
        )
        
        result = training_enhancer.enhance(empty_input)
        
        assert isinstance(result, GraphEnhancementOutput)
        assert len(result.enhanced_embeddings) == 0
        assert result.metadata.status == ProcessingStatus.SUCCESS
        # Should not trigger training with empty input
        assert not training_enhancer._is_trained


if __name__ == "__main__":
    pytest.main([__file__, "-v"])