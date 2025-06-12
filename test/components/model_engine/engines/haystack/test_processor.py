"""
Unit tests for Haystack Model Engine Processor

Tests the HaystackModelEngine component implementation including:
- Haystack pipeline configuration and management
- Document processing and question answering
- Retrieval and reader component integration
- Multi-modal processing capabilities
- Protocol compliance with ModelEngine interface
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from typing import Dict, Any, List

from src.components.model_engine.engines.haystack.processor import HaystackModelEngine
from src.types.components.contracts import (
    ComponentType,
    ModelEngineInput,
    ModelEngineOutput,
    ModelInferenceResult,
    ProcessingStatus
)


class TestHaystackModelEngine:
    """Test suite for HaystackModelEngine component."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.default_config = {
            "pipeline_config": "test_pipeline.yaml",
            "server_url": "http://localhost:8080",
            "retriever_config": {
                "top_k": 10,
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
            },
            "reader_config": {
                "model": "deepset/roberta-base-squad2",
                "max_seq_len": 384
            }
        }
        self.engine = HaystackModelEngine(config=self.default_config)
    
    def test_initialization(self):
        """Test Haystack component initialization."""
        # Test default initialization
        default_engine = HaystackModelEngine()
        assert default_engine.name == "haystack"
        assert default_engine.version == "1.0.0"
        assert default_engine.component_type == ComponentType.MODEL_ENGINE
        
        # Test with configuration
        assert self.engine._config == self.default_config
        assert self.engine._pipeline_config == "test_pipeline.yaml"
        assert self.engine._server_url == "http://localhost:8080"
    
    def test_configuration(self):
        """Test Haystack component configuration."""
        new_config = {
            "pipeline_config": "production_pipeline.yaml",
            "server_url": "http://prod-server:8080",
            "retriever_config": {
                "top_k": 20,
                "embedding_model": "sentence-transformers/all-mpnet-base-v2"
            },
            "reader_config": {
                "model": "deepset/deberta-v3-large-squad2",
                "max_seq_len": 512
            },
            "document_store_config": {
                "type": "elasticsearch",
                "host": "localhost",
                "port": 9200
            }
        }
        
        # Test valid configuration
        self.engine.configure(new_config)
        assert self.engine._pipeline_config == "production_pipeline.yaml"
        assert self.engine._server_url == "http://prod-server:8080"
        
        # Test invalid configuration
        invalid_config = {
            "pipeline_config": 123,  # Should be string
            "server_url": ["invalid"],  # Should be string
            "retriever_config": "not_dict"  # Should be dict
        }
        
        with pytest.raises(ValueError, match="Invalid configuration provided"):
            self.engine.configure(invalid_config)
    
    def test_config_validation(self):
        """Test Haystack configuration validation."""
        # Valid configurations
        valid_configs = [
            {"pipeline_config": "pipeline.yaml"},
            {"server_url": "http://localhost:8080"},
            {"retriever_config": {"top_k": 5}},
            {"reader_config": {"model": "test-model"}},
            {"document_store_config": {"type": "memory"}},
            {}  # Empty config should be valid
        ]
        
        for config in valid_configs:
            assert self.engine.validate_config(config), f"Config should be valid: {config}"
        
        # Invalid configurations
        invalid_configs = [
            "not_a_dict",
            {"pipeline_config": 123},
            {"server_url": ["not_string"]},
            {"retriever_config": "not_dict"},
            {"reader_config": ["not_dict"]},
            {"document_store_config": "not_dict"}
        ]
        
        for config in invalid_configs:
            assert not self.engine.validate_config(config), f"Config should be invalid: {config}"
    
    def test_config_schema(self):
        """Test Haystack configuration schema generation."""
        schema = self.engine.get_config_schema()
        
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "pipeline_config" in schema["properties"]
        assert "server_url" in schema["properties"]
        assert "retriever_config" in schema["properties"]
        assert "reader_config" in schema["properties"]
        assert "document_store_config" in schema["properties"]
    
    @patch('src.components.model_engine.engines.haystack.processor.Pipeline')
    def test_health_check(self, mock_pipeline):
        """Test Haystack health check functionality."""
        # Test without initialized pipeline
        assert not self.engine.health_check()
        
        # Test with healthy mock pipeline
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.get_nodes.return_value = ["retriever", "reader"]
        mock_pipeline_instance.run.return_value = {"answers": []}
        self.engine._pipeline = mock_pipeline_instance
        
        assert self.engine.health_check()
        
        # Test pipeline initialization failure
        mock_pipeline.side_effect = Exception("Pipeline init failed")
        self.engine._pipeline = None
        assert not self.engine.health_check()
        
        # Test pipeline run failure
        mock_pipeline_instance.run.side_effect = Exception("Pipeline run failed")
        self.engine._pipeline = mock_pipeline_instance
        assert not self.engine.health_check()
    
    @patch('src.components.model_engine.engines.haystack.processor.Pipeline')
    def test_metrics_collection(self, mock_pipeline):
        """Test Haystack metrics collection."""
        # Test basic metrics
        metrics = self.engine.get_metrics()
        
        assert metrics["component_name"] == "haystack"
        assert metrics["component_version"] == "1.0.0"
        assert metrics["pipeline_config"] == "test_pipeline.yaml"
        assert metrics["server_url"] == "http://localhost:8080"
        assert "last_health_check" in metrics
        
        # Test with mock pipeline that has stats
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.get_stats.return_value = {
            "queries_processed": 500,
            "avg_retrieval_time": 0.2,
            "avg_reading_time": 0.8,
            "document_count": 10000
        }
        self.engine._pipeline = mock_pipeline_instance
        
        metrics = self.engine.get_metrics()
        assert "pipeline_stats" in metrics
        assert metrics["pipeline_stats"]["queries_processed"] == 500
        
        # Test metrics collection failure
        mock_pipeline_instance.get_stats.side_effect = Exception("Stats failed")
        metrics = self.engine.get_metrics()
        assert "pipeline_stats" not in metrics  # Should handle gracefully
    
    @patch('src.components.model_engine.engines.haystack.processor.Pipeline')
    def test_process_question_answering(self, mock_pipeline):
        """Test Haystack question answering processing."""
        # Setup mock pipeline
        mock_pipeline_instance = Mock()
        mock_answers = [
            {
                "answer": "The capital of France is Paris.",
                "score": 0.95,
                "context": "France is a country in Europe. Its capital is Paris...",
                "document_id": "doc_1",
                "start": 45,
                "end": 50
            },
            {
                "answer": "Paris",
                "score": 0.87,
                "context": "Paris is the most populous city of France...",
                "document_id": "doc_2", 
                "start": 0,
                "end": 5
            }
        ]
        
        mock_pipeline_instance.run.return_value = {
            "answers": mock_answers,
            "documents": [
                {"content": "Document 1 content", "id": "doc_1"},
                {"content": "Document 2 content", "id": "doc_2"}
            ]
        }
        mock_pipeline.return_value = mock_pipeline_instance
        
        # Create test input for question answering
        test_input = ModelEngineInput(
            requests=[
                {
                    "request_id": "qa-test-1",
                    "request_type": "question_answering",
                    "input_text": "What is the capital of France?",
                    "parameters": {
                        "retriever": {"top_k": 10},
                        "reader": {"top_k": 3}
                    }
                }
            ],
            model_config={
                "retriever_model": "sentence-transformers/all-MiniLM-L6-v2",
                "reader_model": "deepset/roberta-base-squad2"
            },
            batch_config={},
            metadata={}
        )
        
        # Process
        result = self.engine.process(test_input)
        
        # Validate output
        assert isinstance(result, ModelEngineOutput)
        assert len(result.results) == 1
        assert result.results[0].request_id == "qa-test-1"
        
        response_data = result.results[0].response_data
        assert "answers" in response_data
        assert len(response_data["answers"]) == 2
        assert response_data["answers"][0]["answer"] == "The capital of France is Paris."
        assert response_data["answers"][0]["score"] == 0.95
        
        assert result.metadata.status == ProcessingStatus.SUCCESS
        assert result.engine_stats["engine_type"] == "haystack"
    
    @patch('src.components.model_engine.engines.haystack.processor.Pipeline')
    def test_process_document_retrieval(self, mock_pipeline):
        """Test Haystack document retrieval processing."""
        # Setup mock pipeline for retrieval
        mock_pipeline_instance = Mock()
        mock_documents = [
            {
                "content": "Relevant document about AI and machine learning...",
                "id": "doc_ai_1",
                "score": 0.92,
                "meta": {"title": "Introduction to AI", "source": "textbook"}
            },
            {
                "content": "Another document discussing neural networks...",
                "id": "doc_ai_2", 
                "score": 0.85,
                "meta": {"title": "Neural Networks", "source": "research_paper"}
            }
        ]
        
        mock_pipeline_instance.run.return_value = {
            "documents": mock_documents
        }
        mock_pipeline.return_value = mock_pipeline_instance
        
        # Create test input for document retrieval
        test_input = ModelEngineInput(
            requests=[
                {
                    "request_id": "retrieval-test-1",
                    "request_type": "retrieve",
                    "input_text": "artificial intelligence machine learning",
                    "parameters": {
                        "retriever": {"top_k": 5, "similarity_threshold": 0.7}
                    }
                }
            ],
            model_config={},
            batch_config={},
            metadata={}
        )
        
        # Process
        result = self.engine.process(test_input)
        
        # Validate retrieval output
        assert isinstance(result, ModelEngineOutput)
        assert len(result.results) == 1
        assert result.results[0].request_id == "retrieval-test-1"
        
        response_data = result.results[0].response_data
        assert "documents" in response_data
        assert len(response_data["documents"]) == 2
        assert response_data["documents"][0]["score"] == 0.92
        assert "AI and machine learning" in response_data["documents"][0]["content"]
    
    @patch('src.components.model_engine.engines.haystack.processor.Pipeline')
    def test_process_document_indexing(self, mock_pipeline):
        """Test Haystack document indexing processing."""
        # Setup mock pipeline for indexing
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.run.return_value = {
            "status": "success",
            "indexed_documents": 5,
            "failed_documents": 0,
            "processing_time": 2.5
        }
        mock_pipeline.return_value = mock_pipeline_instance
        
        # Create test input for document indexing
        documents_to_index = [
            {"content": "Document 1 content", "meta": {"title": "Doc 1"}},
            {"content": "Document 2 content", "meta": {"title": "Doc 2"}},
            {"content": "Document 3 content", "meta": {"title": "Doc 3"}},
            {"content": "Document 4 content", "meta": {"title": "Doc 4"}},
            {"content": "Document 5 content", "meta": {"title": "Doc 5"}}
        ]
        
        test_input = ModelEngineInput(
            requests=[
                {
                    "request_id": "index-test-1",
                    "request_type": "index",
                    "input_data": documents_to_index,
                    "parameters": {
                        "batch_size": 100,
                        "update_existing": True
                    }
                }
            ],
            model_config={},
            batch_config={},
            metadata={}
        )
        
        # Process
        result = self.engine.process(test_input)
        
        # Validate indexing output
        assert isinstance(result, ModelEngineOutput)
        assert len(result.results) == 1
        assert result.results[0].request_id == "index-test-1"
        
        response_data = result.results[0].response_data
        assert response_data["status"] == "success"
        assert response_data["indexed_documents"] == 5
        assert response_data["failed_documents"] == 0
    
    @patch('src.components.model_engine.engines.haystack.processor.Pipeline')
    def test_multi_modal_processing(self, mock_pipeline):
        """Test Haystack multi-modal processing capabilities."""
        # Setup mock pipeline for multi-modal
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.run.return_value = {
            "answers": [
                {
                    "answer": "The image shows a cat sitting on a table.",
                    "score": 0.88,
                    "modality": "image",
                    "document_id": "img_doc_1"
                }
            ],
            "documents": [
                {
                    "content": "Image description: A fluffy cat...",
                    "id": "img_doc_1",
                    "meta": {"type": "image", "format": "jpg"}
                }
            ]
        }
        mock_pipeline.return_value = mock_pipeline_instance
        
        # Create test input for multi-modal query
        test_input = ModelEngineInput(
            requests=[
                {
                    "request_id": "multimodal-test-1",
                    "request_type": "multimodal_qa",
                    "input_text": "What do you see in this image?",
                    "input_data": {
                        "image_path": "/path/to/image.jpg",
                        "modalities": ["text", "image"]
                    },
                    "parameters": {
                        "include_image_context": True,
                        "cross_modal_retrieval": True
                    }
                }
            ],
            model_config={},
            batch_config={},
            metadata={}
        )
        
        # Process
        result = self.engine.process(test_input)
        
        # Validate multi-modal output
        assert isinstance(result, ModelEngineOutput)
        assert len(result.results) == 1
        response_data = result.results[0].response_data
        assert "answers" in response_data
        assert response_data["answers"][0]["modality"] == "image"
    
    @patch('src.components.model_engine.engines.haystack.processor.Pipeline')
    def test_batch_processing(self, mock_pipeline):
        """Test Haystack batch processing capabilities."""
        # Setup mock pipeline for batch processing
        mock_pipeline_instance = Mock()
        
        def batch_run(queries):
            results = []
            for i, query in enumerate(queries):
                results.append({
                    "answers": [{"answer": f"Answer {i+1}", "score": 0.9}],
                    "query_id": i
                })
            return results
        
        mock_pipeline_instance.run_batch.return_value = batch_run
        mock_pipeline.return_value = mock_pipeline_instance
        
        # Create batch input
        batch_requests = [
            {
                "request_id": f"batch-{i}",
                "request_type": "question_answering",
                "input_text": f"Question {i}?",
                "parameters": {"retriever": {"top_k": 5}}
            }
            for i in range(3)
        ]
        
        test_input = ModelEngineInput(
            requests=batch_requests,
            model_config={},
            batch_config={"batch_size": 3, "enable_batching": True},
            metadata={}
        )
        
        # Process batch
        result = self.engine.process(test_input)
        
        # Validate batch processing
        assert len(result.results) == 3
        assert result.engine_stats["request_count"] == 3
        assert result.metadata.status == ProcessingStatus.SUCCESS
    
    def test_pipeline_configuration_loading(self):
        """Test Haystack pipeline configuration loading."""
        # Test different pipeline configurations
        configs = [
            {"pipeline_config": "qa_pipeline.yaml"},
            {"pipeline_config": "indexing_pipeline.yaml"},
            {"pipeline_config": "retrieval_pipeline.yaml"}
        ]
        
        for config in configs:
            engine = HaystackModelEngine(config=config)
            assert engine._pipeline_config == config["pipeline_config"]
    
    def test_document_store_integration(self):
        """Test integration with different document stores."""
        document_store_configs = [
            {
                "document_store_config": {
                    "type": "elasticsearch",
                    "host": "localhost",
                    "port": 9200,
                    "index": "haystack_test"
                }
            },
            {
                "document_store_config": {
                    "type": "faiss",
                    "faiss_index_factory_str": "Flat",
                    "embedding_dim": 768
                }
            },
            {
                "document_store_config": {
                    "type": "memory",
                    "embedding_dim": 384
                }
            }
        ]
        
        for config in document_store_configs:
            engine = HaystackModelEngine(config=config)
            assert engine.validate_config(config)
    
    def test_performance_estimation(self):
        """Test Haystack performance estimation."""
        # Test estimation for different request types
        qa_input = ModelEngineInput(
            requests=[{"request_id": "1", "request_type": "question_answering"}],
            model_config={},
            batch_config={},
            metadata={}
        )
        
        retrieval_input = ModelEngineInput(
            requests=[{"request_id": "1", "request_type": "retrieve"}],
            model_config={},
            batch_config={},
            metadata={}
        )
        
        qa_estimate = self.engine.estimate_processing_time(qa_input)
        retrieval_estimate = self.engine.estimate_processing_time(retrieval_input)
        
        assert qa_estimate > 0
        assert retrieval_estimate > 0
        # QA should typically take longer than pure retrieval
        assert qa_estimate >= retrieval_estimate
    
    def test_supported_models(self):
        """Test Haystack supported models listing."""
        models = self.engine.get_supported_models()
        
        assert isinstance(models, list)
        assert len(models) > 0
        
        # Should include common Haystack-compatible models
        expected_models = [
            "deepset/roberta-base-squad2",
            "sentence-transformers/all-MiniLM-L6-v2",
            "deepset/deberta-v3-large-squad2"
        ]
        
        # Check if any expected models are in the list
        found_models = [m for m in expected_models if m in models]
        assert len(found_models) > 0
    
    def test_error_handling(self):
        """Test Haystack error handling and recovery."""
        # Test pipeline initialization failure
        with patch('src.components.model_engine.engines.haystack.processor.Pipeline') as mock_pipeline:
            mock_pipeline.side_effect = Exception("Pipeline load failed")
            
            test_input = ModelEngineInput(
                requests=[{"request_id": "error-test", "input_text": "test"}],
                model_config={},
                batch_config={},
                metadata={}
            )
            
            result = self.engine.process(test_input)
            assert len(result.errors) > 0
            assert result.metadata.status == ProcessingStatus.ERROR
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test with minimal configuration
        minimal_engine = HaystackModelEngine(config={})
        assert minimal_engine._pipeline_config is None
        assert minimal_engine._server_url == "http://localhost:8080"
        
        # Test empty requests
        empty_input = ModelEngineInput(
            requests=[],
            model_config={},
            batch_config={},
            metadata={}
        )
        
        result = self.engine.process(empty_input)
        assert len(result.results) == 0
        
        # Test malformed requests
        malformed_input = ModelEngineInput(
            requests=[
                {"invalid": "request"},  # Missing required fields
                {"request_id": "valid", "input_text": "Good request"}
            ],
            model_config={},
            batch_config={},
            metadata={}
        )
        
        # Should handle gracefully
        result = self.engine.process(malformed_input)
        assert isinstance(result, ModelEngineOutput)


if __name__ == "__main__":
    pytest.main([__file__])