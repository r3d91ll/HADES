"""
Integration tests for Model Engine Component

Tests the complete model engine system including:
- Component factory and registration
- Engine switching and A/B testing
- Configuration loading and validation
- Cross-engine compatibility
- Performance benchmarking
"""

import pytest
from unittest.mock import Mock, patch
import tempfile
import yaml
import os
from typing import Dict, Any

from src.components.model_engine import (
    MODEL_ENGINE_REGISTRY,
    register_model_engine,
    get_model_engine,
    list_model_engines
)
from src.components.model_engine.core.processor import CoreModelEngine
from src.components.model_engine.engines.vllm.processor import VLLMModelEngine  
from src.components.model_engine.engines.haystack.processor import HaystackModelEngine
from src.types.components.contracts import (
    ModelEngineInput,
    ModelEngineOutput,
    ComponentType
)


class TestModelEngineIntegration:
    """Integration test suite for model engine components."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create temporary config files
        self.temp_dir = tempfile.mkdtemp()
        
        # Core engine config
        self.core_config = {
            "engine_type": "vllm",
            "model_name": "test-model",
            "server_config": {
                "host": "localhost",
                "port": 8000
            }
        }
        
        # vLLM engine config
        self.vllm_config = {
            "server_url": "http://localhost:8000",
            "device": "cuda",
            "model_config": {
                "tensor_parallel_size": 1,
                "gpu_memory_utilization": 0.8
            }
        }
        
        # Haystack engine config
        self.haystack_config = {
            "pipeline_config": os.path.join(self.temp_dir, "test_pipeline.yaml"),
            "server_url": "http://localhost:8080",
            "retriever_config": {
                "top_k": 10
            }
        }
        
        # Create test pipeline config
        pipeline_config = {
            "version": "1.0",
            "components": [
                {
                    "name": "retriever",
                    "type": "BM25Retriever"
                },
                {
                    "name": "reader", 
                    "type": "FARMReader"
                }
            ],
            "pipelines": [
                {
                    "name": "query",
                    "nodes": [
                        {
                            "name": "retriever",
                            "inputs": ["Query"]
                        },
                        {
                            "name": "reader",
                            "inputs": ["retriever"]
                        }
                    ]
                }
            ]
        }
        
        with open(self.haystack_config["pipeline_config"], 'w') as f:
            yaml.dump(pipeline_config, f)
    
    def teardown_method(self):
        """Clean up after each test method."""
        # Clean up temporary files
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_component_registration(self):
        """Test model engine component registration."""
        # Test registry starts populated
        initial_count = len(MODEL_ENGINE_REGISTRY)
        assert initial_count >= 0
        
        # Test manual registration
        class TestEngine:
            pass
            
        register_model_engine("test_engine", TestEngine)
        assert "test_engine" in MODEL_ENGINE_REGISTRY
        assert MODEL_ENGINE_REGISTRY["test_engine"] == TestEngine
        
        # Test getting registered engine
        retrieved_engine = get_model_engine("test_engine")
        assert retrieved_engine == TestEngine
        
        # Test getting non-existent engine
        with pytest.raises(ValueError, match="Model engine 'nonexistent' not found"):
            get_model_engine("nonexistent")
        
        # Test listing engines
        engines = list_model_engines()
        assert "test_engine" in engines
    
    def test_component_initialization(self):
        """Test initialization of all model engine components."""
        # Test Core engine initialization
        core_engine = CoreModelEngine(config=self.core_config)
        assert core_engine.name == "core"
        assert core_engine.component_type == ComponentType.MODEL_ENGINE
        assert core_engine.validate_config(self.core_config)
        
        # Test vLLM engine initialization
        vllm_engine = VLLMModelEngine(config=self.vllm_config)
        assert vllm_engine.name == "vllm"
        assert vllm_engine.component_type == ComponentType.MODEL_ENGINE
        assert vllm_engine.validate_config(self.vllm_config)
        
        # Test Haystack engine initialization
        haystack_engine = HaystackModelEngine(config=self.haystack_config)
        assert haystack_engine.name == "haystack"
        assert haystack_engine.component_type == ComponentType.MODEL_ENGINE
        assert haystack_engine.validate_config(self.haystack_config)
    
    def test_configuration_compatibility(self):
        """Test configuration compatibility across engines."""
        # Test that each engine properly validates its own config
        core_engine = CoreModelEngine()
        vllm_engine = VLLMModelEngine()
        haystack_engine = HaystackModelEngine()
        
        # Core engine should accept its config
        assert core_engine.validate_config(self.core_config)
        
        # vLLM engine should accept its config
        assert vllm_engine.validate_config(self.vllm_config)
        
        # Haystack engine should accept its config
        assert haystack_engine.validate_config(self.haystack_config)
        
        # Cross-validation should fail gracefully
        assert not vllm_engine.validate_config(self.haystack_config)
        assert not haystack_engine.validate_config(self.vllm_config)
    
    def test_unified_interface_compliance(self):
        """Test that all engines comply with unified interface."""
        engines = [
            CoreModelEngine(config=self.core_config),
            VLLMModelEngine(config=self.vllm_config),
            HaystackModelEngine(config=self.haystack_config)
        ]
        
        for engine in engines:
            # Test required properties
            assert hasattr(engine, 'name')
            assert hasattr(engine, 'version')
            assert hasattr(engine, 'component_type')
            
            # Test required methods
            assert hasattr(engine, 'configure')
            assert hasattr(engine, 'validate_config')
            assert hasattr(engine, 'get_config_schema')
            assert hasattr(engine, 'health_check')
            assert hasattr(engine, 'get_metrics')
            assert hasattr(engine, 'process')
            assert hasattr(engine, 'estimate_processing_time')
            assert hasattr(engine, 'get_supported_models')
            
            # Test method signatures (basic check)
            assert callable(engine.configure)
            assert callable(engine.validate_config)
            assert callable(engine.health_check)
            assert callable(engine.process)
    
    @patch('src.components.model_engine.core.processor.ModelEngineFactory')
    @patch('src.components.model_engine.engines.vllm.processor.LegacyVLLMEngine')
    def test_engine_switching(self, mock_vllm, mock_factory):
        """Test switching between different engine implementations."""
        # Setup mocks
        mock_core_engine = Mock()
        mock_core_engine.generate.return_value = Mock(output_text="Core response")
        mock_factory.return_value.create_engine.return_value = mock_core_engine
        
        mock_vllm_engine = Mock()
        mock_vllm_engine.generate.return_value = Mock(output_text="vLLM response")
        mock_vllm.return_value = mock_vllm_engine
        
        # Create test input
        test_input = ModelEngineInput(
            requests=[
                {
                    "request_id": "switch-test",
                    "input_text": "Test prompt",
                    "parameters": {}
                }
            ],
            model_config={},
            batch_config={},
            metadata={}
        )
        
        # Test Core engine processing
        core_engine = CoreModelEngine(config=self.core_config)
        core_result = core_engine.process(test_input)
        assert isinstance(core_result, ModelEngineOutput)
        assert core_result.engine_stats["engine_type"] == "vllm"  # Core delegates to vLLM
        
        # Test vLLM engine processing
        vllm_engine = VLLMModelEngine(config=self.vllm_config)
        vllm_result = vllm_engine.process(test_input)
        assert isinstance(vllm_result, ModelEngineOutput)
        assert vllm_result.engine_stats["engine_type"] == "vllm"
        
        # Both should return valid outputs
        assert len(core_result.results) == 1
        assert len(vllm_result.results) == 1
    
    def test_ab_testing_capability(self):
        """Test A/B testing capability between engines."""
        # Create engines for A/B testing
        engine_a = CoreModelEngine(config=self.core_config)
        engine_b = VLLMModelEngine(config=self.vllm_config)
        
        # Test that both engines can handle the same input format
        test_input = ModelEngineInput(
            requests=[
                {
                    "request_id": "ab-test",
                    "input_text": "Compare engines",
                    "parameters": {"max_tokens": 100}
                }
            ],
            model_config={},
            batch_config={},
            metadata={"experiment": "engine_comparison"}
        )
        
        # Both engines should accept the same input
        assert engine_a.validate_config(engine_a._config)
        assert engine_b.validate_config(engine_b._config)
        
        # Both should be able to estimate processing time
        time_a = engine_a.estimate_processing_time(test_input)
        time_b = engine_b.estimate_processing_time(test_input)
        
        assert time_a > 0
        assert time_b > 0
    
    def test_config_schema_compatibility(self):
        """Test configuration schema compatibility and validation."""
        engines = [
            CoreModelEngine(),
            VLLMModelEngine(),
            HaystackModelEngine()
        ]
        
        for engine in engines:
            schema = engine.get_config_schema()
            
            # All schemas should be valid JSON schema format
            assert schema["type"] == "object"
            assert "properties" in schema
            
            # Schemas should be engine-specific but follow common patterns
            if engine.name == "core":
                assert "engine_type" in schema["properties"]
                assert "model_name" in schema["properties"]
            elif engine.name == "vllm":
                assert "server_url" in schema["properties"]
                assert "device" in schema["properties"]
            elif engine.name == "haystack":
                assert "pipeline_config" in schema["properties"]
                assert "retriever_config" in schema["properties"]
    
    def test_metrics_aggregation(self):
        """Test metrics collection and aggregation across engines."""
        engines = [
            CoreModelEngine(config=self.core_config),
            VLLMModelEngine(config=self.vllm_config),
            HaystackModelEngine(config=self.haystack_config)
        ]
        
        all_metrics = {}
        for engine in engines:
            metrics = engine.get_metrics()
            all_metrics[engine.name] = metrics
            
            # All engines should provide basic metrics
            assert "component_name" in metrics
            assert "component_version" in metrics
            assert "last_health_check" in metrics
            
            # Engine-specific metrics
            if engine.name == "core":
                assert "engine_type" in metrics
                assert "model_name" in metrics
            elif engine.name == "vllm":
                assert "server_url" in metrics
                assert "device" in metrics
            elif engine.name == "haystack":
                assert "pipeline_config" in metrics
        
        # Should have metrics from all engines
        assert len(all_metrics) == 3
        assert "core" in all_metrics
        assert "vllm" in all_metrics
        assert "haystack" in all_metrics
    
    def test_error_handling_consistency(self):
        """Test consistent error handling across engines."""
        engines = [
            CoreModelEngine(config=self.core_config),
            VLLMModelEngine(config=self.vllm_config),
            HaystackModelEngine(config=self.haystack_config)
        ]
        
        # Test invalid configuration handling
        invalid_config = {"invalid": "config", "bad_field": 123}
        
        for engine in engines:
            # All should reject invalid config
            assert not engine.validate_config(invalid_config)
            
            # All should raise ValueError on configure with invalid config
            with pytest.raises(ValueError):
                engine.configure(invalid_config)
    
    def test_supported_models_consistency(self):
        """Test supported models listing consistency."""
        engines = [
            CoreModelEngine(config=self.core_config),
            VLLMModelEngine(config=self.vllm_config),
            HaystackModelEngine(config=self.haystack_config)
        ]
        
        for engine in engines:
            models = engine.get_supported_models()
            
            # All engines should return a list
            assert isinstance(models, list)
            
            # List should not be empty
            assert len(models) > 0
            
            # All items should be strings
            assert all(isinstance(model, str) for model in models)
    
    @patch('src.components.model_engine.core.processor.ModelEngineFactory')
    def test_performance_comparison(self, mock_factory):
        """Test performance comparison between engines."""
        # Setup mock for consistent testing
        mock_engine = Mock()
        mock_engine.generate.return_value = Mock(output_text="Test response")
        mock_factory.return_value.create_engine.return_value = mock_engine
        
        engines = [
            CoreModelEngine(config=self.core_config),
            VLLMModelEngine(config=self.vllm_config),
            HaystackModelEngine(config=self.haystack_config)
        ]
        
        # Test processing time estimation
        test_input = ModelEngineInput(
            requests=[{"request_id": "perf-test", "input_text": "Performance test"}],
            model_config={},
            batch_config={},
            metadata={}
        )
        
        estimates = {}
        for engine in engines:
            estimate = engine.estimate_processing_time(test_input)
            estimates[engine.name] = estimate
            assert estimate > 0
        
        # All engines should provide reasonable estimates
        assert all(est > 0 for est in estimates.values())
    
    def test_health_check_consistency(self):
        """Test health check consistency across engines."""
        engines = [
            CoreModelEngine(config=self.core_config),
            VLLMModelEngine(config=self.vllm_config),
            HaystackModelEngine(config=self.haystack_config)
        ]
        
        for engine in engines:
            # Health check should return a boolean
            health = engine.health_check()
            assert isinstance(health, bool)
            
            # Health check should be callable multiple times
            health2 = engine.health_check()
            assert isinstance(health2, bool)
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow with engine selection."""
        # Simulate engine selection based on task type
        task_configs = {
            "text_generation": {
                "engine": "vllm",
                "config": self.vllm_config
            },
            "question_answering": {
                "engine": "haystack", 
                "config": self.haystack_config
            },
            "general_inference": {
                "engine": "core",
                "config": self.core_config
            }
        }
        
        engines = {}
        for task, task_config in task_configs.items():
            if task_config["engine"] == "core":
                engine = CoreModelEngine(config=task_config["config"])
            elif task_config["engine"] == "vllm":
                engine = VLLMModelEngine(config=task_config["config"])
            elif task_config["engine"] == "haystack":
                engine = HaystackModelEngine(config=task_config["config"])
            
            engines[task] = engine
            
            # Verify engine is properly configured
            assert engine.validate_config(task_config["config"])
            assert isinstance(engine.health_check(), bool)
        
        # Should have engines for all task types
        assert len(engines) == 3
        assert "text_generation" in engines
        assert "question_answering" in engines
        assert "general_inference" in engines


if __name__ == "__main__":
    pytest.main([__file__])