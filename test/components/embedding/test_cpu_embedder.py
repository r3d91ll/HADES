"""
Unit tests for CPU Embedder Component
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from typing import List, Dict, Any

from src.components.embedding.cpu.processor import CPUEmbedder
from src.types.components.contracts import (
    ComponentType,
    EmbeddingInput,
    EmbeddingOutput,
    DocumentChunk,
    ProcessingStatus
)


class TestCPUEmbedder:
    """Test suite for CPU embedder component."""
    
    @pytest.fixture
    def cpu_embedder(self):
        """Create CPU embedder instance for testing."""
        config = {
            'model_name': 'test-model',
            'batch_size': 16,
            'max_length': 512,
            'normalize_embeddings': True
        }
        return CPUEmbedder(config)
    
    @pytest.fixture
    def sample_input(self):
        """Create sample embedding input."""
        chunks = [
            DocumentChunk(
                id="chunk_1",
                content="This is the first test chunk.",
                metadata={"source": "test"}
            ),
            DocumentChunk(
                id="chunk_2", 
                content="This is the second test chunk.",
                metadata={"source": "test"}
            )
        ]
        
        return EmbeddingInput(
            chunks=chunks,
            model_name="test-model",
            batch_size=16,
            metadata={"test": True}
        )
    
    def test_init(self, cpu_embedder):
        """Test CPU embedder initialization."""
        assert cpu_embedder.name == "cpu"
        assert cpu_embedder.version == "1.0.0"
        assert cpu_embedder.component_type == ComponentType.EMBEDDING
        assert cpu_embedder._model_name == "test-model"
        assert cpu_embedder._batch_size == 16
        assert cpu_embedder._normalize_embeddings is True
    
    def test_configure(self, cpu_embedder):
        """Test configuration update."""
        new_config = {
            'model_name': 'new-model',
            'batch_size': 32,
            'max_length': 1024
        }
        
        cpu_embedder.configure(new_config)
        
        assert cpu_embedder._model_name == "new-model"
        assert cpu_embedder._batch_size == 32
        assert cpu_embedder._max_length == 1024
    
    def test_validate_config(self, cpu_embedder):
        """Test configuration validation."""
        # Valid config
        valid_config = {
            'batch_size': 64,
            'max_length': 2048,
            'pooling_strategy': 'mean'
        }
        assert cpu_embedder.validate_config(valid_config) is True
        
        # Invalid batch size
        invalid_config = {
            'batch_size': -1
        }
        assert cpu_embedder.validate_config(invalid_config) is False
        
        # Invalid pooling strategy
        invalid_config = {
            'pooling_strategy': 'invalid'
        }
        assert cpu_embedder.validate_config(invalid_config) is False
    
    def test_get_config_schema(self, cpu_embedder):
        """Test config schema generation."""
        schema = cpu_embedder.get_config_schema()
        
        assert schema['type'] == 'object'
        assert 'properties' in schema
        assert 'model_name' in schema['properties']
        assert 'batch_size' in schema['properties']
        assert 'max_length' in schema['properties']
    
    @patch('src.components.embedding.cpu.processor.create_model_engine')
    def test_health_check(self, mock_factory, cpu_embedder):
        """Test health check."""
        # Mock successful model initialization
        mock_model = Mock()
        mock_factory.return_value = mock_model
        
        # Health check should succeed
        assert cpu_embedder.health_check() is True
        
        # Test failed health check
        cpu_embedder._model_loaded = False
        cpu_embedder._initialize_model = Mock(side_effect=Exception("Model error"))
        assert cpu_embedder.health_check() is False
    
    def test_get_metrics(self, cpu_embedder):
        """Test metrics collection."""
        # Set some test statistics
        cpu_embedder._total_embeddings_created = 100
        cpu_embedder._total_processing_time = 10.5
        
        metrics = cpu_embedder.get_metrics()
        
        assert metrics['component_name'] == "cpu"
        assert metrics['component_version'] == "1.0.0"
        assert metrics['total_embeddings_created'] == 100
        assert metrics['total_processing_time'] == 10.5
        assert metrics['avg_processing_time'] == 0.105
    
    @patch('src.components.embedding.cpu.processor.create_model_engine')
    def test_embed_success(self, mock_factory, cpu_embedder, sample_input):
        """Test successful embedding generation."""
        # Mock model engine
        mock_model = Mock()
        mock_model.embed_texts.return_value = [
            [0.1] * 384,  # Mock embedding for chunk 1
            [0.2] * 384   # Mock embedding for chunk 2
        ]
        mock_factory.return_value = mock_model
        
        # Force model initialization
        cpu_embedder._model = mock_model
        cpu_embedder._model_loaded = True
        
        # Generate embeddings
        output = cpu_embedder.embed(sample_input)
        
        # Verify output
        assert isinstance(output, EmbeddingOutput)
        assert len(output.embeddings) == 2
        assert output.metadata.status == ProcessingStatus.SUCCESS
        assert output.metadata.component_name == "cpu"
        assert len(output.errors) == 0
        
        # Verify embeddings
        assert output.embeddings[0].chunk_id == "chunk_1"
        assert output.embeddings[0].embedding_dimension == 384
        assert output.embeddings[1].chunk_id == "chunk_2"
    
    def test_embed_fallback(self, cpu_embedder, sample_input):
        """Test fallback embedding generation."""
        # Force fallback mode
        cpu_embedder._model = "basic_fallback"
        cpu_embedder._model_loaded = True
        
        # Generate embeddings
        output = cpu_embedder.embed(sample_input)
        
        # Verify output
        assert isinstance(output, EmbeddingOutput)
        assert len(output.embeddings) == 2
        assert output.metadata.status == ProcessingStatus.SUCCESS
        
        # Verify fallback embeddings have correct dimension
        assert output.embeddings[0].embedding_dimension == 384
        assert len(output.embeddings[0].embedding) == 384
    
    def test_embed_error(self, cpu_embedder, sample_input):
        """Test embedding error handling."""
        # Force error during embedding
        cpu_embedder._model_loaded = False
        cpu_embedder._initialize_model = Mock(side_effect=Exception("Init error"))
        
        # Generate embeddings
        output = cpu_embedder.embed(sample_input)
        
        # Verify error handling
        assert isinstance(output, EmbeddingOutput)
        assert len(output.embeddings) == 0
        assert output.metadata.status == ProcessingStatus.ERROR
        assert len(output.errors) > 0
        assert "CPU embedding failed" in output.errors[0]
    
    def test_estimate_tokens(self, cpu_embedder, sample_input):
        """Test token estimation."""
        estimated = cpu_embedder.estimate_tokens(sample_input)
        
        # Should estimate based on character count
        total_chars = sum(len(chunk.content) for chunk in sample_input.chunks)
        expected = max(1, total_chars // 4)
        
        assert estimated == expected
    
    def test_supports_model(self, cpu_embedder):
        """Test model support checking."""
        # Supported models
        assert cpu_embedder.supports_model("all-MiniLM-L6-v2") is True
        assert cpu_embedder.supports_model("all-mpnet-base-v2") is True
        assert cpu_embedder.supports_model("sentence-transformers/all-MiniLM-L6-v2") is True
        
        # Unsupported models
        assert cpu_embedder.supports_model("gpt-3") is False
        assert cpu_embedder.supports_model("unknown-model") is False
    
    def test_get_embedding_dimension(self, cpu_embedder):
        """Test embedding dimension retrieval."""
        assert cpu_embedder.get_embedding_dimension("all-MiniLM-L6-v2") == 384
        assert cpu_embedder.get_embedding_dimension("all-mpnet-base-v2") == 768
        assert cpu_embedder.get_embedding_dimension("unknown-model") == 384  # Default
    
    def test_embedding_properties(self, cpu_embedder):
        """Test embedder properties."""
        assert cpu_embedder.embedding_dimension == 384  # Based on test-model
        assert cpu_embedder.max_sequence_length == 512
        assert cpu_embedder.supports_batch_processing() is True
        assert cpu_embedder.get_optimal_batch_size() == 16
    
    def test_extract_text_features(self, cpu_embedder):
        """Test fallback text feature extraction."""
        text = "This is a test sentence with some features."
        features = cpu_embedder._extract_text_features(text)
        
        assert isinstance(features, list)
        assert len(features) == 384  # Target dimension
        assert all(isinstance(f, float) for f in features)
        
        # Test normalization
        if cpu_embedder._normalize_embeddings:
            import math
            norm = math.sqrt(sum(f * f for f in features))
            assert abs(norm - 1.0) < 0.01  # Should be normalized