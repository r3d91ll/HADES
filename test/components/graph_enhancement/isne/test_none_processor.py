"""
Unit tests for None/Passthrough ISNE Processor.

Tests the NoneISNE component implementation including passthrough functionality,
validation, and configuration management.
"""

import pytest
import tempfile
import json
from typing import Dict, Any, List
from datetime import datetime

from src.components.graph_enhancement.isne.none.processor import NoneISNE
from src.types.components.contracts import (
    ComponentType,
    GraphEnhancementInput,
    GraphEnhancementOutput,
    ChunkEmbedding,
    EnhancedEmbedding,
    ProcessingStatus
)


class TestNoneISNE:
    """Test suite for NoneISNE processor."""
    
    @pytest.fixture
    def sample_config(self) -> Dict[str, Any]:
        """Sample configuration for testing."""
        return {
            "preserve_original_embeddings": True,
            "add_metadata": True,
            "validate_inputs": True,
            "validate_outputs": True,
            "copy_tensors": False,
            "log_operations": False
        }
    
    @pytest.fixture
    def none_isne(self, sample_config: Dict[str, Any]) -> NoneISNE:
        """Create NoneISNE instance for testing."""
        return NoneISNE(config=sample_config)
    
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
            enhancement_options={"method": "none"}
        )
    
    def test_initialization(self, sample_config: Dict[str, Any]):
        """Test NoneISNE initialization."""
        none_isne = NoneISNE(config=sample_config)
        
        assert none_isne.name == "none"
        assert none_isne.version == "1.0.0"
        assert none_isne.component_type == ComponentType.GRAPH_ENHANCEMENT
        assert none_isne._preserve_original is True
        assert none_isne._add_metadata is True
        assert none_isne._validate_inputs is True
        assert none_isne._validate_outputs is True
        assert none_isne._copy_tensors is False
    
    def test_initialization_default_config(self):
        """Test NoneISNE initialization with default config."""
        none_isne = NoneISNE()
        
        assert none_isne.name == "none"
        assert none_isne._preserve_original is True  # Default
        assert none_isne._add_metadata is True  # Default
        assert none_isne._validate_inputs is True  # Default
        assert none_isne._validate_outputs is True  # Default
        assert none_isne._copy_tensors is False  # Default
    
    def test_configure(self, none_isne: NoneISNE):
        """Test configuration updates."""
        new_config = {
            "preserve_original_embeddings": False,
            "add_metadata": False,
            "validate_inputs": False,
            "validate_outputs": False,
            "copy_tensors": True
        }
        
        none_isne.configure(new_config)
        
        assert none_isne._preserve_original is False
        assert none_isne._add_metadata is False
        assert none_isne._validate_inputs is False
        assert none_isne._validate_outputs is False
        assert none_isne._copy_tensors is True
    
    def test_configure_invalid_config(self, none_isne: NoneISNE):
        """Test configuration with invalid parameters."""
        invalid_configs = [
            {"preserve_original_embeddings": "not_bool"},
            {"add_metadata": 123},
            {"validate_inputs": "yes"},
            {"validate_outputs": []},
            {"copy_tensors": "false"},
        ]
        
        for invalid_config in invalid_configs:
            with pytest.raises(ValueError, match="Invalid configuration"):
                none_isne.configure(invalid_config)
    
    def test_validate_config(self, none_isne: NoneISNE):
        """Test configuration validation."""
        valid_configs = [
            {"preserve_original_embeddings": True},
            {"add_metadata": False},
            {"validate_inputs": True, "validate_outputs": False},
            {},  # Empty config should be valid
        ]
        
        for config in valid_configs:
            assert none_isne.validate_config(config) is True
        
        invalid_configs = [
            {"preserve_original_embeddings": "not_bool"},
            {"add_metadata": 123},
            "not_a_dict",
            {"unknown_param": True}  # Should still be valid, just ignored
        ]
        
        for config in invalid_configs[:3]:  # Skip the unknown_param test
            assert none_isne.validate_config(config) is False
    
    def test_get_config_schema(self, none_isne: NoneISNE):
        """Test configuration schema retrieval."""
        schema = none_isne.get_config_schema()
        
        assert isinstance(schema, dict)
        assert "type" in schema
        assert schema["type"] == "object"
        assert "properties" in schema
        
        properties = schema["properties"]
        assert "preserve_original_embeddings" in properties
        assert "add_metadata" in properties
        assert "validate_inputs" in properties
        assert "validate_outputs" in properties
        assert "copy_tensors" in properties
        assert "log_operations" in properties
        
        # Check boolean types
        for prop in properties.values():
            assert prop["type"] == "boolean"
    
    def test_health_check(self, none_isne: NoneISNE):
        """Test health check functionality."""
        # Passthrough component should always be healthy
        assert none_isne.health_check() is True
    
    def test_get_metrics(self, none_isne: NoneISNE):
        """Test metrics retrieval."""
        metrics = none_isne.get_metrics()
        
        assert isinstance(metrics, dict)
        assert "component_name" in metrics
        assert "component_version" in metrics
        assert "enhancement_method" in metrics
        assert "total_processed" in metrics
        assert "avg_processing_time" in metrics
        assert "preserve_original" in metrics
        assert "add_metadata" in metrics
        
        assert metrics["component_name"] == "none"
        assert metrics["component_version"] == "1.0.0"
        assert metrics["enhancement_method"] == "passthrough"
        assert metrics["total_processed"] == 0  # Initially
    
    def test_enhance_passthrough(self, none_isne: NoneISNE, sample_input: GraphEnhancementInput):
        """Test passthrough enhancement functionality."""
        result = none_isne.enhance(sample_input)
        
        assert isinstance(result, GraphEnhancementOutput)
        assert len(result.enhanced_embeddings) == len(sample_input.embeddings)
        assert result.metadata.status == ProcessingStatus.SUCCESS
        assert len(result.errors) == 0
        
        # Check that embeddings are passed through unchanged
        for i, enhanced_emb in enumerate(result.enhanced_embeddings):
            assert isinstance(enhanced_emb, EnhancedEmbedding)
            assert enhanced_emb.chunk_id == sample_input.embeddings[i].chunk_id
            
            # Enhanced embedding should be the same as original
            assert enhanced_emb.enhanced_embedding == sample_input.embeddings[i].embedding
            
            # Original should be preserved (if configured)
            if none_isne._preserve_original:
                assert enhanced_emb.original_embedding == sample_input.embeddings[i].embedding
            
            # Enhancement score should be 1.0 (perfect preservation)
            assert enhanced_emb.enhancement_score == 1.0
            
            # Check graph features
            assert enhanced_emb.graph_features["enhancement_method"] == "passthrough"
            assert enhanced_emb.graph_features["preserve_original"] == none_isne._preserve_original
    
    def test_enhance_without_original_preservation(self, sample_input: GraphEnhancementInput):
        """Test enhancement without preserving original embeddings."""
        config = {"preserve_original_embeddings": False}
        none_isne = NoneISNE(config=config)
        
        result = none_isne.enhance(sample_input)
        
        assert isinstance(result, GraphEnhancementOutput)
        assert len(result.enhanced_embeddings) == len(sample_input.embeddings)
        
        for enhanced_emb in result.enhanced_embeddings:
            # Original should be empty when not preserved
            assert enhanced_emb.original_embedding == []
    
    def test_enhance_without_metadata(self, sample_input: GraphEnhancementInput):
        """Test enhancement without adding metadata."""
        config = {"add_metadata": False}
        none_isne = NoneISNE(config=config)
        
        result = none_isne.enhance(sample_input)
        
        assert isinstance(result, GraphEnhancementOutput)
        assert len(result.enhanced_embeddings) == len(sample_input.embeddings)
        
        # Should still work, just without additional metadata
        for enhanced_emb in result.enhanced_embeddings:
            assert enhanced_emb.enhancement_score == 1.0
    
    def test_enhance_with_copy_tensors(self, sample_input: GraphEnhancementInput):
        """Test enhancement with tensor copying enabled."""
        config = {"copy_tensors": True}
        none_isne = NoneISNE(config=config)
        
        result = none_isne.enhance(sample_input)
        
        assert isinstance(result, GraphEnhancementOutput)
        assert len(result.enhanced_embeddings) == len(sample_input.embeddings)
        assert result.metadata.status == ProcessingStatus.SUCCESS
    
    def test_enhance_without_validation(self, sample_input: GraphEnhancementInput):
        """Test enhancement without input/output validation."""
        config = {
            "validate_inputs": False,
            "validate_outputs": False
        }
        none_isne = NoneISNE(config=config)
        
        result = none_isne.enhance(sample_input)
        
        assert isinstance(result, GraphEnhancementOutput)
        assert len(result.enhanced_embeddings) == len(sample_input.embeddings)
        assert result.metadata.status == ProcessingStatus.SUCCESS
    
    def test_enhance_empty_input(self, none_isne: NoneISNE):
        """Test enhancement with empty input."""
        empty_input = GraphEnhancementInput(
            embeddings=[],
            metadata={},
            enhancement_options={}
        )
        
        result = none_isne.enhance(empty_input)
        
        assert isinstance(result, GraphEnhancementOutput)
        assert len(result.enhanced_embeddings) == 0
        assert result.metadata.status == ProcessingStatus.SUCCESS
    
    def test_enhance_invalid_input_with_validation(self):
        """Test enhancement with invalid input when validation is enabled."""
        config = {"validate_inputs": True}
        none_isne = NoneISNE(config=config)
        
        invalid_input = GraphEnhancementInput(
            embeddings=[ChunkEmbedding(
                chunk_id="",  # Empty chunk_id
                embedding=[],  # Empty embedding
                metadata={}
            )],
            metadata={},
            enhancement_options={}
        )
        
        result = none_isne.enhance(invalid_input)
        
        # Should handle validation errors gracefully
        assert isinstance(result, GraphEnhancementOutput)
        assert len(result.errors) > 0  # Should have validation errors
    
    def test_train_method(self, none_isne: NoneISNE, sample_input: GraphEnhancementInput):
        """Test training method (should be no-op)."""
        # Training should do nothing for passthrough component
        none_isne.train(sample_input)
        
        # Should still be "trained"
        assert none_isne.is_trained() is True
    
    def test_is_trained(self, none_isne: NoneISNE):
        """Test is_trained method."""
        # Passthrough component is always "trained"
        assert none_isne.is_trained() is True
    
    def test_save_model(self, none_isne: NoneISNE):
        """Test model saving functionality."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name
        
        try:
            none_isne.save_model(temp_path)
            
            # Check that file was created and contains expected data
            with open(temp_path, 'r') as f:
                saved_data = json.load(f)
            
            assert saved_data["component_type"] == "none_isne"
            assert saved_data["component_name"] == "none"
            assert saved_data["component_version"] == "1.0.0"
            assert "config" in saved_data
            assert "statistics" in saved_data
            assert "saved_at" in saved_data
            
        finally:
            import os
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_load_model(self, none_isne: NoneISNE):
        """Test model loading functionality."""
        # Create a test configuration file
        test_config = {
            "component_type": "none_isne",
            "component_name": "none",
            "component_version": "1.0.0",
            "config": {
                "preserve_original_embeddings": False,
                "add_metadata": False
            },
            "statistics": {
                "total_processed": 100,
                "total_processing_time": 5.0
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            json.dump(test_config, f)
            temp_path = f.name
        
        try:
            none_isne.load_model(temp_path)
            
            # Check that configuration was loaded
            assert none_isne._preserve_original is False
            assert none_isne._add_metadata is False
            assert none_isne._total_processed == 100
            assert none_isne._total_processing_time == 5.0
            
        finally:
            import os
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_get_model_info(self, none_isne: NoneISNE):
        """Test model info retrieval."""
        info = none_isne.get_model_info()
        
        assert isinstance(info, dict)
        assert info["model_type"] == "passthrough"
        assert info["component_name"] == "none"
        assert info["component_version"] == "1.0.0"
        assert info["enhancement_method"] == "none"
        assert "preserve_original" in info
        assert "add_metadata" in info
        assert "total_processed" in info
    
    def test_supports_incremental_training(self, none_isne: NoneISNE):
        """Test incremental training support (should be False)."""
        assert none_isne.supports_incremental_training() is False
    
    def test_passthrough_embedding_creation(self, none_isne: NoneISNE):
        """Test creation of passthrough enhanced embedding."""
        chunk_emb = ChunkEmbedding(
            chunk_id="test_chunk",
            embedding=[0.1, 0.2, 0.3],
            metadata={"test": True}
        )
        
        enhanced_emb = none_isne._passthrough_embedding(chunk_emb)
        
        assert isinstance(enhanced_emb, EnhancedEmbedding)
        assert enhanced_emb.chunk_id == "test_chunk"
        assert enhanced_emb.enhanced_embedding == [0.1, 0.2, 0.3]
        assert enhanced_emb.enhancement_score == 1.0
        assert enhanced_emb.graph_features["enhancement_method"] == "passthrough"
    
    def test_validate_input_data(self, none_isne: NoneISNE, sample_input: GraphEnhancementInput):
        """Test input data validation."""
        errors = none_isne._validate_input_data(sample_input)
        assert len(errors) == 0  # Valid input should have no errors
        
        # Test invalid input
        invalid_input = GraphEnhancementInput(
            embeddings=[],  # No embeddings
            metadata={},
            enhancement_options={}
        )
        
        errors = none_isne._validate_input_data(invalid_input)
        assert len(errors) > 0
    
    def test_validate_output_data(self, none_isne: NoneISNE, sample_input: GraphEnhancementInput):
        """Test output data validation."""
        result = none_isne.enhance(sample_input)
        errors = none_isne._validate_output_data(result.enhanced_embeddings)
        assert len(errors) == 0  # Valid output should have no errors
        
        # Test invalid output
        invalid_embeddings = [EnhancedEmbedding(
            chunk_id="",  # Empty chunk_id
            original_embedding=[],
            enhanced_embedding=[],  # Empty embedding
            graph_features={},
            enhancement_score=-1.0,  # Invalid score
            metadata={}
        )]
        
        errors = none_isne._validate_output_data(invalid_embeddings)
        assert len(errors) > 0
    
    def test_performance_metrics_update(self, none_isne: NoneISNE, sample_input: GraphEnhancementInput):
        """Test that performance metrics are updated after processing."""
        initial_metrics = none_isne.get_metrics()
        initial_count = initial_metrics["total_processed"]
        initial_time = initial_metrics["total_processing_time"]
        
        none_isne.enhance(sample_input)
        
        updated_metrics = none_isne.get_metrics()
        assert updated_metrics["total_processed"] > initial_count
        assert updated_metrics["total_processing_time"] >= initial_time
    
    def test_processing_statistics(self, none_isne: NoneISNE, sample_input: GraphEnhancementInput):
        """Test processing statistics tracking."""
        # Process multiple batches
        for _ in range(3):
            result = none_isne.enhance(sample_input)
            assert result.metadata.status == ProcessingStatus.SUCCESS
        
        metrics = none_isne.get_metrics()
        assert metrics["total_processed"] == 3 * len(sample_input.embeddings)
        assert metrics["avg_processing_time"] > 0.0
    
    def test_graph_stats_in_output(self, none_isne: NoneISNE, sample_input: GraphEnhancementInput):
        """Test that graph stats are properly included in output."""
        result = none_isne.enhance(sample_input)
        
        assert "graph_stats" in result.__dict__
        assert "processing_time" in result.graph_stats
        assert "embeddings_processed" in result.graph_stats
        assert "embeddings_failed" in result.graph_stats
        assert "enhancement_method" in result.graph_stats
        assert "throughput_embeddings_per_second" in result.graph_stats
        
        assert result.graph_stats["enhancement_method"] == "passthrough"
        assert result.graph_stats["embeddings_processed"] == len(sample_input.embeddings)
        assert result.graph_stats["embeddings_failed"] == 0
    
    def test_model_info_in_output(self, none_isne: NoneISNE, sample_input: GraphEnhancementInput):
        """Test that model info is properly included in output."""
        result = none_isne.enhance(sample_input)
        
        assert "model_info" in result.__dict__
        assert result.model_info["model_type"] == "passthrough"
        assert result.model_info["enhancement_method"] == "none"
        assert "preserve_original" in result.model_info
        assert "validation_enabled" in result.model_info


if __name__ == "__main__":
    pytest.main([__file__, "-v"])