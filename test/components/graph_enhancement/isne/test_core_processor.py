"""
Unit tests for Core ISNE Processor.

Tests the CoreISNE component implementation including multiple enhancement methods,
configuration validation, and protocol compliance.
"""

import pytest
import numpy as np
from typing import Dict, Any, List
from datetime import datetime, timezone

from src.components.graph_enhancement.isne.core.processor import CoreISNE
from src.types.components.contracts import (
    ComponentType,
    GraphEnhancementInput,
    GraphEnhancementOutput,
    ChunkEmbedding,
    EnhancedEmbedding,
    ProcessingStatus
)


class TestCoreISNE:
    """Test suite for CoreISNE processor."""
    
    @pytest.fixture
    def sample_config(self) -> Dict[str, Any]:
        """Sample configuration for testing."""
        return {
            "enhancement_method": "isne",
            "neighbor_count": 5,
            "enhancement_strength": 0.3,
            "use_similarity_weights": True,
            "fallback_method": "similarity"
        }
    
    @pytest.fixture
    def core_isne(self, sample_config: Dict[str, Any]) -> CoreISNE:
        """Create CoreISNE instance for testing."""
        return CoreISNE(config=sample_config)
    
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
            enhancement_options={"method": "isne"}
        )
    
    def test_initialization(self, sample_config: Dict[str, Any]):
        """Test CoreISNE initialization."""
        core_isne = CoreISNE(config=sample_config)
        
        assert core_isne.name == "core"
        assert core_isne.version == "1.0.0"
        assert core_isne.component_type == ComponentType.GRAPH_ENHANCEMENT
        assert core_isne._enhancement_method == "isne"
        assert core_isne._neighbor_count == 5
        assert core_isne._enhancement_strength == 0.3
    
    def test_initialization_default_config(self):
        """Test CoreISNE initialization with default config."""
        core_isne = CoreISNE()
        
        assert core_isne.name == "core"
        assert core_isne._enhancement_method == "isne"
        assert core_isne._neighbor_count == 3
        assert core_isne._enhancement_strength == 0.2
    
    def test_configure(self, core_isne: CoreISNE):
        """Test configuration updates."""
        new_config = {
            "enhancement_method": "similarity",
            "neighbor_count": 7,
            "enhancement_strength": 0.5
        }
        
        core_isne.configure(new_config)
        
        assert core_isne._enhancement_method == "similarity"
        assert core_isne._neighbor_count == 7
        assert core_isne._enhancement_strength == 0.5
    
    def test_configure_invalid_config(self, core_isne: CoreISNE):
        """Test configuration with invalid parameters."""
        invalid_configs = [
            {"neighbor_count": -1},
            {"enhancement_strength": 1.5},
            {"enhancement_method": "invalid_method"},
            {"neighbor_count": "not_int"},
        ]
        
        for invalid_config in invalid_configs:
            with pytest.raises(ValueError, match="Invalid configuration"):
                core_isne.configure(invalid_config)
    
    def test_validate_config(self, core_isne: CoreISNE):
        """Test configuration validation."""
        valid_configs = [
            {"enhancement_method": "isne", "neighbor_count": 5},
            {"enhancement_strength": 0.5},
            {"use_similarity_weights": True},
            {}  # Empty config should be valid
        ]
        
        for config in valid_configs:
            assert core_isne.validate_config(config) is True
        
        invalid_configs = [
            {"neighbor_count": -1},
            {"enhancement_strength": 2.0},
            {"enhancement_method": "unknown"},
            "not_a_dict"
        ]
        
        for config in invalid_configs:
            assert core_isne.validate_config(config) is False
    
    def test_get_config_schema(self, core_isne: CoreISNE):
        """Test configuration schema retrieval."""
        schema = core_isne.get_config_schema()
        
        assert isinstance(schema, dict)
        assert "type" in schema
        assert schema["type"] == "object"
        assert "properties" in schema
        
        properties = schema["properties"]
        assert "enhancement_method" in properties
        assert "neighbor_count" in properties
        assert "enhancement_strength" in properties
    
    def test_health_check(self, core_isne: CoreISNE):
        """Test health check functionality."""
        assert core_isne.health_check() is True
    
    def test_get_metrics(self, core_isne: CoreISNE):
        """Test metrics retrieval."""
        metrics = core_isne.get_metrics()
        
        assert isinstance(metrics, dict)
        assert "component_name" in metrics
        assert "component_version" in metrics
        assert "enhancement_method" in metrics
        assert "total_enhancements" in metrics
        assert "avg_processing_time" in metrics
        
        assert metrics["component_name"] == "core"
        assert metrics["component_version"] == "1.0.0"
    
    def test_enhance_isne_method(self, core_isne: CoreISNE, sample_input: GraphEnhancementInput):
        """Test enhancement using ISNE method."""
        core_isne.configure({"enhancement_method": "isne"})
        
        result = core_isne.enhance(sample_input)
        
        assert isinstance(result, GraphEnhancementOutput)
        assert len(result.enhanced_embeddings) == len(sample_input.embeddings)
        assert result.metadata.status == ProcessingStatus.SUCCESS
        assert len(result.errors) == 0
        
        # Check that embeddings are enhanced
        for enhanced_emb in result.enhanced_embeddings:
            assert isinstance(enhanced_emb, EnhancedEmbedding)
            assert enhanced_emb.chunk_id is not None
            assert len(enhanced_emb.enhanced_embedding) > 0
            assert enhanced_emb.enhancement_score >= 0.0
    
    def test_enhance_similarity_method(self, core_isne: CoreISNE, sample_input: GraphEnhancementInput):
        """Test enhancement using similarity method."""
        core_isne.configure({"enhancement_method": "similarity"})
        
        result = core_isne.enhance(sample_input)
        
        assert isinstance(result, GraphEnhancementOutput)
        assert len(result.enhanced_embeddings) == len(sample_input.embeddings)
        assert result.metadata.status == ProcessingStatus.SUCCESS
        assert len(result.errors) == 0
    
    def test_enhance_basic_method(self, core_isne: CoreISNE, sample_input: GraphEnhancementInput):
        """Test enhancement using basic method."""
        core_isne.configure({"enhancement_method": "basic"})
        
        result = core_isne.enhance(sample_input)
        
        assert isinstance(result, GraphEnhancementOutput)
        assert len(result.enhanced_embeddings) == len(sample_input.embeddings)
        assert result.metadata.status == ProcessingStatus.SUCCESS
        assert len(result.errors) == 0
        
        # Basic method should return original embeddings
        for i, enhanced_emb in enumerate(result.enhanced_embeddings):
            original_embedding = sample_input.embeddings[i].embedding
            assert enhanced_emb.enhanced_embedding == original_embedding
    
    def test_enhance_empty_input(self, core_isne: CoreISNE):
        """Test enhancement with empty input."""
        empty_input = GraphEnhancementInput(
            embeddings=[],
            metadata={},
            enhancement_options={}
        )
        
        result = core_isne.enhance(empty_input)
        
        assert isinstance(result, GraphEnhancementOutput)
        assert len(result.enhanced_embeddings) == 0
        assert result.metadata.status == ProcessingStatus.SUCCESS
    
    def test_enhance_single_embedding(self, core_isne: CoreISNE):
        """Test enhancement with single embedding."""
        single_input = GraphEnhancementInput(
            embeddings=[ChunkEmbedding(
                chunk_id="single_chunk",
                embedding=[0.1, 0.2, 0.3],
                metadata={}
            )],
            metadata={},
            enhancement_options={}
        )
        
        result = core_isne.enhance(single_input)
        
        assert isinstance(result, GraphEnhancementOutput)
        assert len(result.enhanced_embeddings) == 1
        assert result.metadata.status == ProcessingStatus.SUCCESS
    
    def test_estimate_enhancement_time(self, core_isne: CoreISNE, sample_input: GraphEnhancementInput):
        """Test enhancement time estimation."""
        time_estimate = core_isne.estimate_enhancement_time(sample_input)
        
        assert isinstance(time_estimate, float)
        assert time_estimate > 0.0
        
        # Should scale with number of embeddings
        larger_input = GraphEnhancementInput(
            embeddings=sample_input.embeddings * 2,
            metadata={},
            enhancement_options={}
        )
        
        larger_estimate = core_isne.estimate_enhancement_time(larger_input)
        assert larger_estimate > time_estimate
    
    def test_supports_enhancement_method(self, core_isne: CoreISNE):
        """Test enhancement method support checking."""
        supported_methods = ["isne", "similarity", "basic", "core"]
        unsupported_methods = ["unknown", "invalid", ""]
        
        for method in supported_methods:
            assert core_isne.supports_enhancement_method(method) is True
        
        for method in unsupported_methods:
            assert core_isne.supports_enhancement_method(method) is False
    
    def test_get_required_graph_features(self, core_isne: CoreISNE):
        """Test required graph features listing."""
        features = core_isne.get_required_graph_features()
        
        assert isinstance(features, list)
        assert "node_embeddings" in features
        assert "similarity_matrix" in features
    
    def test_enhancement_consistency(self, core_isne: CoreISNE, sample_input: GraphEnhancementInput):
        """Test that enhancement produces consistent results."""
        result1 = core_isne.enhance(sample_input)
        result2 = core_isne.enhance(sample_input)
        
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
    
    def test_enhancement_score_calculation(self, core_isne: CoreISNE):
        """Test enhancement score calculation."""
        original = [1.0, 0.0, 0.0]
        enhanced = [0.8, 0.2, 0.1]
        
        score = core_isne._calculate_enhancement_score(original, enhanced)
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
    
    def test_similarity_enhancement_with_weights(self, core_isne: CoreISNE, sample_embeddings: List[List[float]]):
        """Test similarity-based enhancement with weighted neighbors."""
        core_isne.configure({
            "enhancement_method": "similarity",
            "use_similarity_weights": True,
            "neighbor_count": 3
        })
        
        enhanced = core_isne._enhance_similarity_based(sample_embeddings)
        
        assert len(enhanced) == len(sample_embeddings)
        assert all(len(emb) == len(sample_embeddings[0]) for emb in enhanced)
    
    def test_isne_enhancement_method(self, core_isne: CoreISNE, sample_embeddings: List[List[float]]):
        """Test ISNE enhancement method."""
        enhanced = core_isne._enhance_isne_method(sample_embeddings)
        
        assert len(enhanced) == len(sample_embeddings)
        assert all(len(emb) == len(sample_embeddings[0]) for emb in enhanced)
        
        # Enhanced embeddings should be different from originals (for most cases)
        different_count = 0
        for orig, enh in zip(sample_embeddings, enhanced):
            if not np.allclose(orig, enh, atol=1e-6):
                different_count += 1
        
        # At least some embeddings should be enhanced
        assert different_count > 0
    
    def test_error_handling_invalid_embedding(self, core_isne: CoreISNE):
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
        
        result = core_isne.enhance(invalid_input)
        
        # Should handle gracefully and return valid output
        assert isinstance(result, GraphEnhancementOutput)
        assert result.metadata.status in [ProcessingStatus.SUCCESS, ProcessingStatus.ERROR]
    
    def test_performance_metrics_update(self, core_isne: CoreISNE, sample_input: GraphEnhancementInput):
        """Test that performance metrics are updated after enhancement."""
        initial_metrics = core_isne.get_metrics()
        initial_count = initial_metrics["total_enhancements"]
        initial_time = initial_metrics["total_processing_time"]
        
        core_isne.enhance(sample_input)
        
        updated_metrics = core_isne.get_metrics()
        assert updated_metrics["total_enhancements"] > initial_count
        assert updated_metrics["total_processing_time"] >= initial_time


if __name__ == "__main__":
    pytest.main([__file__, "-v"])