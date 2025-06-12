"""
Unit tests for Core Model Engine Processor

Tests the CoreModelEngine component implementation including:
- Component initialization and configuration
- Input validation and processing 
- Health checks and metrics
- Error handling and recovery
- Protocol compliance
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from typing import Dict, Any, List

from src.components.model_engine.core.processor import CoreModelEngine
from src.types.components.contracts import (
    ComponentType,
    ModelEngineInput,
    ModelEngineOutput,
    ModelInferenceResult,
    ProcessingStatus
)


class TestCoreModelEngine:
    """Test suite for CoreModelEngine component."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.default_config = {
            "engine_type": "vllm",
            "model_name": "BAAI/bge-large-en-v1.5",
            "server_config": {
                "host": "localhost",
                "port": 8000,
                "gpu_memory_utilization": 0.8
            }
        }
        self.engine = CoreModelEngine(config=self.default_config)
    
    def test_initialization(self):
        """Test component initialization."""
        # Test default initialization
        default_engine = CoreModelEngine()
        assert default_engine.name == "core"
        assert default_engine.version == "1.0.0"
        assert default_engine.component_type == ComponentType.MODEL_ENGINE
        
        # Test with configuration
        assert self.engine._config == self.default_config
        assert self.engine._engine_type == "vllm"
        assert self.engine._model_name == "BAAI/bge-large-en-v1.5"
    
    def test_configuration(self):
        """Test component configuration."""
        new_config = {
            "engine_type": "haystack",
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "server_config": {
                "host": "remote-server",
                "port": 8080
            }
        }
        
        # Test valid configuration
        self.engine.configure(new_config)
        assert self.engine._engine_type == "haystack"
        assert self.engine._model_name == "sentence-transformers/all-MiniLM-L6-v2"
        
        # Test invalid configuration
        invalid_config = {
            "engine_type": "invalid_engine",
            "model_name": 123  # Should be string
        }
        
        with pytest.raises(ValueError, match="Invalid configuration provided"):
            self.engine.configure(invalid_config)
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid configurations
        valid_configs = [
            {"engine_type": "vllm"},
            {"engine_type": "haystack"},
            {"model_name": "test-model"},
            {"server_config": {"host": "localhost", "port": 8000}},
            {}  # Empty config should be valid
        ]
        
        for config in valid_configs:
            assert self.engine.validate_config(config), f"Config should be valid: {config}"
        
        # Invalid configurations
        invalid_configs = [
            "not_a_dict",
            {"engine_type": "invalid_type"},
            {"model_name": 123},
            {"server_config": "not_a_dict"}
        ]
        
        for config in invalid_configs:
            assert not self.engine.validate_config(config), f"Config should be invalid: {config}"
    
    def test_config_schema(self):
        """Test configuration schema generation."""
        schema = self.engine.get_config_schema()
        
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "engine_type" in schema["properties"]
        assert "model_name" in schema["properties"]
        assert "server_config" in schema["properties"]
        assert "inference_config" in schema["properties"]
        
        # Check enum values for engine_type
        assert schema["properties"]["engine_type"]["enum"] == ["vllm", "haystack"]
    
    @patch('src.components.model_engine.core.processor.ModelEngineFactory')
    def test_health_check(self, mock_factory):
        """Test health check functionality."""
        # Test without initialized engine
        assert not self.engine.health_check()
        
        # Test with mock engine
        mock_engine = Mock()
        mock_engine.is_ready.return_value = True
        mock_factory.return_value.create_engine.return_value = mock_engine
        
        self.engine._engine = mock_engine
        assert self.engine.health_check()
        
        # Test engine without is_ready method
        mock_engine_no_ready = Mock(spec=[])  # No is_ready method
        self.engine._engine = mock_engine_no_ready
        assert self.engine.health_check()  # Should default to True
        
        # Test engine initialization failure
        mock_factory.return_value.create_engine.side_effect = Exception("Init failed")
        self.engine._engine = None
        assert not self.engine.health_check()
    
    @patch('src.components.model_engine.core.processor.ModelEngineFactory')
    def test_metrics(self, mock_factory):
        """Test metrics collection."""
        # Test basic metrics
        metrics = self.engine.get_metrics()
        
        assert metrics["component_name"] == "core"
        assert metrics["component_version"] == "1.0.0"
        assert metrics["engine_type"] == "vllm"
        assert metrics["model_name"] == "BAAI/bge-large-en-v1.5"
        assert "last_health_check" in metrics
        
        # Test with mock engine that has stats
        mock_engine = Mock()
        mock_engine.get_stats.return_value = {"requests": 100, "errors": 2}
        self.engine._engine = mock_engine
        
        metrics = self.engine.get_metrics()
        assert "engine_stats" in metrics
        assert metrics["engine_stats"]["requests"] == 100
        
        # Test with engine that raises exception on stats
        mock_engine.get_stats.side_effect = Exception("Stats failed")
        metrics = self.engine.get_metrics()
        assert "engine_stats" not in metrics  # Should not include stats on error
    
    @patch('src.components.model_engine.core.processor.ModelEngineFactory')
    @patch('src.components.model_engine.core.processor.ModelEngineConfig')
    @patch('src.components.model_engine.core.processor.InferenceRequest')
    def test_process_success(self, mock_inference_request, mock_config, mock_factory):
        """Test successful processing."""
        # Setup mocks
        mock_engine = Mock()
        mock_response = Mock()
        mock_response.output_text = "Generated text"
        mock_response.tokens = [1, 2, 3]
        mock_response.metadata = {"model": "test"}
        
        mock_engine.generate.return_value = mock_response
        mock_factory.return_value.create_engine.return_value = mock_engine
        
        # Create test input
        test_input = ModelEngineInput(
            requests=[
                {
                    "request_id": "test-1",
                    "input_text": "Test prompt",
                    "parameters": {"max_tokens": 100}
                }
            ],
            model_config={},
            batch_config={},
            metadata={}
        )
        
        # Process
        result = self.engine.process(test_input)
        
        # Validate output
        assert isinstance(result, ModelEngineOutput)
        assert len(result.results) == 1
        assert result.results[0].request_id == "test-1"
        assert result.results[0].response_data["output_text"] == "Generated text"
        assert result.metadata.status == ProcessingStatus.SUCCESS
        assert result.engine_stats["request_count"] == 1
        assert result.engine_stats["success_count"] == 1
        assert result.engine_stats["error_count"] == 0
    
    @patch('src.components.model_engine.core.processor.ModelEngineFactory')
    def test_process_failure(self, mock_factory):
        """Test processing with failures."""
        # Setup mock to raise exception
        mock_factory.return_value.create_engine.side_effect = Exception("Engine init failed")
        
        # Create test input
        test_input = ModelEngineInput(
            requests=[{"request_id": "test-1", "input_text": "Test prompt"}],
            model_config={},
            batch_config={},
            metadata={}
        )
        
        # Process
        result = self.engine.process(test_input)
        
        # Validate error handling
        assert isinstance(result, ModelEngineOutput)
        assert len(result.errors) > 0
        assert result.metadata.status == ProcessingStatus.ERROR
        assert "Engine init failed" in str(result.errors)
    
    @patch('src.components.model_engine.core.processor.ModelEngineFactory')
    def test_process_partial_failure(self, mock_factory):
        """Test processing with partial failures."""
        # Setup mock engine that fails on second request
        mock_engine = Mock()
        mock_factory.return_value.create_engine.return_value = mock_engine
        
        def generate_side_effect(request):
            if "fail" in request.input_text:
                raise Exception("Processing failed")
            mock_response = Mock()
            mock_response.output_text = "Success"
            mock_response.tokens = []
            mock_response.metadata = {}
            return mock_response
        
        mock_engine.generate.side_effect = generate_side_effect
        
        # Create test input with one success and one failure
        test_input = ModelEngineInput(
            requests=[
                {"request_id": "success", "input_text": "Good prompt"},
                {"request_id": "failure", "input_text": "This will fail"}
            ],
            model_config={},
            batch_config={},
            metadata={}
        )
        
        # Process
        result = self.engine.process(test_input)
        
        # Validate partial success
        assert isinstance(result, ModelEngineOutput)
        assert len(result.results) == 2
        assert result.results[0].request_id == "success"
        assert result.results[0].error is None
        assert result.results[1].request_id == "failure"
        assert result.results[1].error is not None
        assert len(result.errors) > 0
        assert result.metadata.status == ProcessingStatus.ERROR
    
    def test_supported_models(self):
        """Test supported models listing."""
        models = self.engine.get_supported_models()
        
        assert isinstance(models, list)
        assert len(models) > 0
        assert "BAAI/bge-large-en-v1.5" in models
        assert "sentence-transformers/all-MiniLM-L6-v2" in models
    
    def test_processing_time_estimation(self):
        """Test processing time estimation."""
        # Test with small input
        small_input = ModelEngineInput(
            requests=[{"request_id": "1", "input_text": "test"}],
            model_config={},
            batch_config={},
            metadata={}
        )
        
        estimate = self.engine.estimate_processing_time(small_input)
        assert estimate >= 0.1  # Should estimate at least 100ms for vLLM
        
        # Test with large input
        large_input = ModelEngineInput(
            requests=[{"request_id": str(i), "input_text": "test"} for i in range(100)],
            model_config={},
            batch_config={},
            metadata={}
        )
        
        large_estimate = self.engine.estimate_processing_time(large_input)
        assert large_estimate > estimate  # Should estimate more time for more requests
        
        # Test with haystack engine (should be slower)
        self.engine._engine_type = "haystack"
        haystack_estimate = self.engine.estimate_processing_time(small_input)
        assert haystack_estimate >= 0.5  # Should estimate 500ms+ for haystack
    
    @patch('src.components.model_engine.core.processor.ServerManager')
    def test_server_management(self, mock_server_manager_class):
        """Test server start/stop functionality."""
        mock_server_manager = Mock()
        mock_server_manager_class.return_value = mock_server_manager
        
        # Test successful server start
        mock_server_manager.start.return_value = None
        assert self.engine.start_server()
        mock_server_manager.start.assert_called_once()
        
        # Test successful server stop
        mock_server_manager.stop.return_value = None
        mock_server_manager.is_running.return_value = True
        self.engine._server_manager = mock_server_manager
        
        assert self.engine.stop_server()
        mock_server_manager.stop.assert_called_once()
        
        # Test server status check
        assert self.engine.is_server_running()
        mock_server_manager.is_running.assert_called()
        
        # Test server start failure
        mock_server_manager.start.side_effect = Exception("Start failed")
        assert not self.engine.start_server()
    
    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Test with None config
        engine = CoreModelEngine(config=None)
        assert engine._config == {}
        
        # Test empty requests processing
        empty_input = ModelEngineInput(
            requests=[],
            model_config={},
            batch_config={},
            metadata={}
        )
        
        result = self.engine.process(empty_input)
        assert len(result.results) == 0
        assert result.engine_stats["request_count"] == 0
        
        # Test server operations without server manager
        self.engine._server_manager = None
        assert not self.engine.stop_server()
        assert not self.engine.is_server_running()


if __name__ == "__main__":
    pytest.main([__file__])