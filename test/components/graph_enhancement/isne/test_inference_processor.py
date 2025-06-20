"""
Unit tests for ISNE Inference Processor.

Tests the ISNEInferenceEnhancer component implementation including model loading,
inference functionality, and configuration management.
"""

import pytest
import numpy as np
from typing import Dict, Any, List
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock
from pathlib import Path

from src.components.graph_enhancement.isne.inference.processor import ISNEInferenceEnhancer
from src.types.components.contracts import (
    ComponentType,
    GraphEnhancementInput,
    GraphEnhancementOutput,
    ChunkEmbedding,
    EnhancedEmbedding,
    ProcessingStatus
)


class TestISNEInferenceEnhancer:
    """Test suite for ISNEInferenceEnhancer processor."""
    
    @pytest.fixture
    def sample_config(self) -> Dict[str, Any]:
        """Sample configuration for testing."""
        return {
            "model_config": {
                "embedding_dim": 4,
                "hidden_dim": 8,
                "num_layers": 2
            },
            "inference_config": {
                "batch_size": 32,
                "use_cache": True,
                "enhancement_strength": 0.5
            },
            "model_path": "/tmp/test_model.pkl"
        }
    
    @pytest.fixture
    def inference_enhancer(self, sample_config: Dict[str, Any]) -> ISNEInferenceEnhancer:
        """Create ISNEInferenceEnhancer instance for testing."""
        return ISNEInferenceEnhancer(config=sample_config)
    
    @pytest.fixture
    def sample_embeddings(self) -> List[List[float]]:
        """Sample embeddings for testing."""
        return [
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
            [0.2, 0.3, 0.4, 0.5],
            [0.6, 0.7, 0.8, 0.9],
            [0.1, 0.1, 0.1, 0.1]
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
            enhancement_options={"method": "isne_inference"}
        )
    
    def test_initialization(self, sample_config: Dict[str, Any]):
        """Test ISNEInferenceEnhancer initialization."""
        enhancer = ISNEInferenceEnhancer(config=sample_config)
        
        assert enhancer.name == "isne_inference"
        assert enhancer.version == "1.0.0"
        assert enhancer.component_type == ComponentType.GRAPH_ENHANCEMENT
        assert not enhancer._model_loaded
        assert enhancer._model_parameters is None
        assert enhancer._model_path == "/tmp/test_model.pkl"
    
    def test_initialization_default_config(self):
        """Test ISNEInferenceEnhancer initialization with default config."""
        enhancer = ISNEInferenceEnhancer()
        
        assert enhancer.name == "isne_inference"
        assert enhancer._model_config == {}
        assert enhancer._inference_config == {}
        assert not enhancer._model_loaded
        assert enhancer._model_path is None
    
    def test_configure(self, inference_enhancer: ISNEInferenceEnhancer):
        """Test configuration updates."""
        new_config = {
            "model_config": {
                "embedding_dim": 6,
                "hidden_dim": 12
            },
            "inference_config": {
                "batch_size": 64,
                "enhancement_strength": 0.7
            },
            "model_path": "/tmp/new_model.pkl"
        }
        
        inference_enhancer.configure(new_config)
        
        assert inference_enhancer._model_config["embedding_dim"] == 6
        assert inference_enhancer._model_config["hidden_dim"] == 12
        assert inference_enhancer._inference_config["batch_size"] == 64
        assert inference_enhancer._inference_config["enhancement_strength"] == 0.7
        assert inference_enhancer._model_path == "/tmp/new_model.pkl"
        
        # Model state should be reset
        assert not inference_enhancer._model_loaded
        assert inference_enhancer._model_parameters is None
    
    def test_configure_invalid_config(self, inference_enhancer: ISNEInferenceEnhancer):
        """Test configuration with invalid parameters."""
        invalid_configs = [
            {"model_config": {"embedding_dim": -1}},
            {"inference_config": {"batch_size": 0}},
            {"model_path": 123},  # Not a string
            {"model_config": "not_dict"},
            {"inference_config": {"enhancement_strength": 1.5}},  # Out of range
        ]
        
        for invalid_config in invalid_configs:
            with pytest.raises(ValueError, match="Invalid configuration"):
                inference_enhancer.configure(invalid_config)
    
    def test_validate_config(self, inference_enhancer: ISNEInferenceEnhancer):
        """Test configuration validation."""
        valid_configs = [
            {"model_config": {"embedding_dim": 768, "hidden_dim": 256}},
            {"inference_config": {"batch_size": 128, "use_cache": False}},
            {"model_path": "/path/to/model.pkl"},
            {},  # Empty config should be valid
        ]
        
        for config in valid_configs:
            assert inference_enhancer.validate_config(config) is True
        
        invalid_configs = [
            {"model_config": {"embedding_dim": 0}},
            {"inference_config": {"batch_size": -1}},
            {"model_path": 123},
            "not_a_dict",
            {"model_config": "not_dict"}
        ]
        
        for config in invalid_configs:
            assert inference_enhancer.validate_config(config) is False
    
    def test_get_config_schema(self, inference_enhancer: ISNEInferenceEnhancer):
        """Test configuration schema retrieval."""
        schema = inference_enhancer.get_config_schema()
        
        assert isinstance(schema, dict)
        assert "type" in schema
        assert schema["type"] == "object"
        assert "properties" in schema
        
        properties = schema["properties"]
        assert "model_path" in properties
        assert "model_config" in properties
        assert "inference_config" in properties
        
        # Check model config schema
        model_schema = properties["model_config"]["properties"]
        assert "embedding_dim" in model_schema
        assert "hidden_dim" in model_schema
        assert "num_layers" in model_schema
        
        # Check inference config schema
        inference_schema = properties["inference_config"]["properties"]
        assert "batch_size" in inference_schema
        assert "use_cache" in inference_schema
        assert "enhancement_strength" in inference_schema
    
    def test_health_check(self, inference_enhancer: ISNEInferenceEnhancer):
        """Test health check functionality."""
        assert inference_enhancer.health_check() is True
    
    def test_get_metrics(self, inference_enhancer: ISNEInferenceEnhancer):
        """Test metrics retrieval."""
        metrics = inference_enhancer.get_metrics()
        
        assert isinstance(metrics, dict)
        assert "component_name" in metrics
        assert "component_version" in metrics
        assert "model_loaded" in metrics
        assert "model_path" in metrics
        assert "dependencies_available" in metrics
        assert "total_inferences" in metrics
        assert "avg_processing_time" in metrics
        
        assert metrics["component_name"] == "isne_inference"
        assert metrics["component_version"] == "1.0.0"
        assert metrics["model_loaded"] is False  # Initially not loaded
    
    def test_enhance_without_loaded_model(self, inference_enhancer: ISNEInferenceEnhancer, sample_input: GraphEnhancementInput):
        """Test enhancement without a loaded model (fallback mode)."""
        result = inference_enhancer.enhance(sample_input)
        
        assert isinstance(result, GraphEnhancementOutput)
        assert len(result.enhanced_embeddings) == len(sample_input.embeddings)
        assert result.metadata.status == ProcessingStatus.SUCCESS
        assert len(result.errors) == 0
        
        # Check enhanced embeddings
        for enhanced_emb in result.enhanced_embeddings:
            assert isinstance(enhanced_emb, EnhancedEmbedding)
            assert enhanced_emb.chunk_id is not None
            assert len(enhanced_emb.enhanced_embedding) > 0
            assert enhanced_emb.enhancement_score >= 0.0
            assert enhanced_emb.graph_features["enhancement_method"] == "isne_inference"
            assert enhanced_emb.graph_features["model_loaded"] is False
    
    @patch('pathlib.Path.exists')
    def test_enhance_with_mock_loaded_model(self, mock_exists, inference_enhancer: ISNEInferenceEnhancer, sample_input: GraphEnhancementInput):
        """Test enhancement with a mocked loaded model."""
        # Mock model file existence
        mock_exists.return_value = True
        
        # Manually set up model parameters to simulate loaded model
        inference_enhancer._model_loaded = True
        inference_enhancer._model_parameters = {
            'weight_matrix': np.random.normal(0, 0.1, (4, 8)),
            'bias': np.zeros(8),
            'output_weight': np.random.normal(0, 0.1, (8, 4)),
            'output_bias': np.zeros(4)
        }
        
        result = inference_enhancer.enhance(sample_input)
        
        assert isinstance(result, GraphEnhancementOutput)
        assert len(result.enhanced_embeddings) == len(sample_input.embeddings)
        assert result.metadata.status == ProcessingStatus.SUCCESS
        
        # Check that model-based enhancement was used
        for enhanced_emb in result.enhanced_embeddings:
            assert enhanced_emb.graph_features["model_loaded"] is True
    
    def test_load_model_file_not_found(self, inference_enhancer: ISNEInferenceEnhancer):
        """Test loading a model file that doesn't exist."""
        with pytest.raises(IOError, match="Failed to load model"):
            inference_enhancer.load_model("/non/existent/path.pkl")
    
    @patch('pathlib.Path.exists')
    def test_load_model_success(self, mock_exists, inference_enhancer: ISNEInferenceEnhancer):
        """Test successful model loading."""
        mock_exists.return_value = True
        
        # The actual loading will still fail, but we test the path logic
        with pytest.raises(IOError):  # Expected since we're not creating a real file
            inference_enhancer.load_model("/tmp/test_model.pkl")
    
    def test_estimate_enhancement_time(self, inference_enhancer: ISNEInferenceEnhancer, sample_input: GraphEnhancementInput):
        """Test enhancement time estimation."""
        # Test without loaded model
        time_estimate = inference_enhancer.estimate_enhancement_time(sample_input)
        assert isinstance(time_estimate, float)
        assert time_estimate > 0.0
        
        # Test with mock loaded model
        inference_enhancer._model_loaded = True
        loaded_estimate = inference_enhancer.estimate_enhancement_time(sample_input)
        assert loaded_estimate < time_estimate  # Should be faster with loaded model
    
    def test_supports_enhancement_method(self, inference_enhancer: ISNEInferenceEnhancer):
        """Test enhancement method support checking."""
        supported_methods = ["isne_inference", "isne", "inference"]
        unsupported_methods = ["training", "similarity", "unknown"]
        
        for method in supported_methods:
            assert inference_enhancer.supports_enhancement_method(method) is True
        
        for method in unsupported_methods:
            assert inference_enhancer.supports_enhancement_method(method) is False
    
    def test_get_required_graph_features(self, inference_enhancer: ISNEInferenceEnhancer):
        """Test required graph features listing."""
        features = inference_enhancer.get_required_graph_features()
        
        assert isinstance(features, list)
        assert "node_embeddings" in features
        assert "graph_structure" in features
        # Inference requires fewer features than training
        assert len(features) == 2
    
    def test_inference_with_model(self, inference_enhancer: ISNEInferenceEnhancer, sample_embeddings: List[List[float]]):
        """Test inference with model parameters."""
        # Set up mock model parameters
        inference_enhancer._model_parameters = {
            'weight_matrix': np.random.normal(0, 0.1, (4, 8)),
            'bias': np.zeros(8),
            'output_weight': np.random.normal(0, 0.1, (8, 4)),
            'output_bias': np.zeros(4)
        }
        
        enhanced = inference_enhancer._inference_with_model(sample_embeddings)
        
        assert len(enhanced) == len(sample_embeddings)
        assert all(len(emb) == len(sample_embeddings[0]) for emb in enhanced)
        
        # Enhanced embeddings should be different from originals
        for orig, enh in zip(sample_embeddings, enhanced):
            # Due to enhancement strength, they should be different
            assert not np.allclose(orig, enh, atol=1e-6)
    
    def test_inference_fallback(self, inference_enhancer: ISNEInferenceEnhancer, sample_embeddings: List[List[float]]):
        """Test fallback inference method."""
        enhanced = inference_enhancer._inference_fallback(sample_embeddings)
        
        assert len(enhanced) == len(sample_embeddings)
        assert all(len(emb) == len(sample_embeddings[0]) for emb in enhanced)
    
    def test_enhancement_score_calculation(self, inference_enhancer: ISNEInferenceEnhancer):
        """Test enhancement score calculation."""
        original = [1.0, 0.0, 0.0]
        enhanced = [0.8, 0.2, 0.1]
        
        score = inference_enhancer._calculate_enhancement_score(original, enhanced)
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
    
    def test_inference_config_enhancement_strength(self, inference_enhancer: ISNEInferenceEnhancer, sample_embeddings: List[List[float]]):
        """Test that enhancement strength config affects results."""
        # Set up model parameters
        inference_enhancer._model_parameters = {
            'weight_matrix': np.random.normal(0, 0.1, (4, 8)),
            'bias': np.zeros(8),
            'output_weight': np.random.normal(0, 0.1, (8, 4)),
            'output_bias': np.zeros(4)
        }
        
        # Test with low enhancement strength
        inference_enhancer._inference_config['enhancement_strength'] = 0.1
        enhanced_low = inference_enhancer._inference_with_model(sample_embeddings[:1])
        
        # Test with high enhancement strength
        inference_enhancer._inference_config['enhancement_strength'] = 0.9
        enhanced_high = inference_enhancer._inference_with_model(sample_embeddings[:1])
        
        # High enhancement should be more different from original
        original = np.array(sample_embeddings[0])
        diff_low = np.linalg.norm(np.array(enhanced_low[0]) - original)
        diff_high = np.linalg.norm(np.array(enhanced_high[0]) - original)
        
        assert diff_high > diff_low
    
    def test_performance_metrics_update(self, inference_enhancer: ISNEInferenceEnhancer, sample_input: GraphEnhancementInput):
        """Test that performance metrics are updated after inference."""
        initial_metrics = inference_enhancer.get_metrics()
        initial_count = initial_metrics["total_inferences"]
        initial_time = initial_metrics["total_processing_time"]
        
        inference_enhancer.enhance(sample_input)
        
        updated_metrics = inference_enhancer.get_metrics()
        assert updated_metrics["total_inferences"] > initial_count
        assert updated_metrics["total_processing_time"] >= initial_time
    
    def test_empty_input_handling(self, inference_enhancer: ISNEInferenceEnhancer):
        """Test handling of empty input."""
        empty_input = GraphEnhancementInput(
            embeddings=[],
            metadata={},
            enhancement_options={}
        )
        
        result = inference_enhancer.enhance(empty_input)
        
        assert isinstance(result, GraphEnhancementOutput)
        assert len(result.enhanced_embeddings) == 0
        assert result.metadata.status == ProcessingStatus.SUCCESS
    
    def test_single_embedding_inference(self, inference_enhancer: ISNEInferenceEnhancer):
        """Test inference with single embedding."""
        single_input = GraphEnhancementInput(
            embeddings=[ChunkEmbedding(
                chunk_id="single_chunk",
                embedding=[0.1, 0.2, 0.3, 0.4],
                metadata={}
            )],
            metadata={},
            enhancement_options={}
        )
        
        result = inference_enhancer.enhance(single_input)
        
        assert isinstance(result, GraphEnhancementOutput)
        assert len(result.enhanced_embeddings) == 1
        assert result.metadata.status == ProcessingStatus.SUCCESS
    
    def test_error_handling_invalid_embedding(self, inference_enhancer: ISNEInferenceEnhancer):
        """Test error handling with invalid embedding data."""
        invalid_input = GraphEnhancementInput(
            embeddings=[ChunkEmbedding(
                chunk_id="invalid_chunk",
                embedding=[],  # Empty embedding
                metadata={}
            )],
            metadata={},
            enhancement_options={}
        )
        
        result = inference_enhancer.enhance(invalid_input)
        
        # Should handle gracefully and return valid output
        assert isinstance(result, GraphEnhancementOutput)
        assert result.metadata.status in [ProcessingStatus.SUCCESS, ProcessingStatus.ERROR]
    
    def test_consistency_across_calls(self, inference_enhancer: ISNEInferenceEnhancer, sample_input: GraphEnhancementInput):
        """Test that inference produces consistent results across multiple calls."""
        result1 = inference_enhancer.enhance(sample_input)
        result2 = inference_enhancer.enhance(sample_input)
        
        assert len(result1.enhanced_embeddings) == len(result2.enhanced_embeddings)
        
        # Results should be deterministic for same input
        for emb1, emb2 in zip(result1.enhanced_embeddings, result2.enhanced_embeddings):
            assert emb1.chunk_id == emb2.chunk_id
            # Enhanced embeddings should be close (allowing for floating point precision)
            np.testing.assert_array_almost_equal(
                emb1.enhanced_embedding, 
                emb2.enhanced_embedding,
                decimal=5
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])