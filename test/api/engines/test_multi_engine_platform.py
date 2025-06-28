"""
Test suite for the multi-engine RAG platform.

This module tests the complete multi-engine platform including:
- Engine registration and management
- PathRAG engine functionality
- A/B testing framework
- Comparison metrics
- FastAPI endpoints
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List

from src.api.engines.base import (
    EngineType, RetrievalMode, EngineRetrievalRequest, EngineRetrievalResult,
    EngineRetrievalResponse, EngineRegistry, get_engine_registry
)
from src.api.engines.pathrag_engine import PathRAGEngine
from src.api.engines.comparison import (
    ExperimentManager, ComparisonMetric, get_experiment_manager
)
from src.types.components.contracts import RAGMode


class TestEngineRegistry:
    """Test engine registry functionality."""
    
    def test_engine_registry_singleton(self):
        """Test that engine registry is a singleton."""
        registry1 = get_engine_registry()
        registry2 = get_engine_registry()
        assert registry1 is registry2
    
    def test_engine_registration(self):
        """Test engine registration and retrieval."""
        registry = EngineRegistry()
        
        # Create mock engine
        mock_engine = Mock()
        mock_engine.engine_type = EngineType.PATHRAG
        mock_engine.supported_modes = [RetrievalMode.FULL_ENGINE]
        
        # Register engine
        registry.register_engine(mock_engine, is_default=True)
        
        # Test retrieval
        retrieved_engine = registry.get_engine(EngineType.PATHRAG)
        assert retrieved_engine is mock_engine
        
        # Test default engine
        default_engine = registry.get_default_engine()
        assert default_engine is mock_engine
        
        # Test listing
        engines = registry.list_engines()
        assert EngineType.PATHRAG in engines
    
    def test_supported_modes_listing(self):
        """Test listing supported modes for engines."""
        registry = EngineRegistry()
        
        mock_engine = Mock()
        mock_engine.engine_type = EngineType.PATHRAG
        mock_engine.supported_modes = [RetrievalMode.NAIVE, RetrievalMode.HYBRID]
        
        registry.register_engine(mock_engine)
        
        modes = registry.list_supported_modes(EngineType.PATHRAG)
        assert RetrievalMode.NAIVE in modes
        assert RetrievalMode.HYBRID in modes
    
    @pytest.mark.asyncio
    async def test_health_check_all(self):
        """Test health checking all engines."""
        registry = EngineRegistry()
        
        # Mock healthy engine
        healthy_engine = AsyncMock()
        healthy_engine.engine_type = EngineType.PATHRAG
        healthy_engine.health_check.return_value = True
        
        registry.register_engine(healthy_engine)
        
        health_results = await registry.health_check_all()
        assert health_results[EngineType.PATHRAG] is True


class TestPathRAGEngine:
    """Test PathRAG engine implementation."""
    
    @pytest.fixture
    def pathrag_config(self):
        """Create test PathRAG configuration."""
        return {
            "flow_decay_factor": 0.8,
            "pruning_threshold": 0.3,
            "max_path_length": 5,
            "storage": {"type": "memory"},
            "embedder": {"type": "cpu"}
        }
    
    def test_pathrag_engine_initialization(self, pathrag_config):
        """Test PathRAG engine initialization."""
        engine = PathRAGEngine(pathrag_config)
        
        assert engine.engine_type == EngineType.PATHRAG
        assert RetrievalMode.FULL_ENGINE in engine.supported_modes
        assert RetrievalMode.HYBRID in engine.supported_modes
    
    def test_supported_modes(self):
        """Test PathRAG supported modes."""
        engine = PathRAGEngine()
        
        expected_modes = [
            RetrievalMode.NAIVE,
            RetrievalMode.LOCAL,
            RetrievalMode.GLOBAL,
            RetrievalMode.HYBRID,
            RetrievalMode.FULL_ENGINE
        ]
        
        for mode in expected_modes:
            assert mode in engine.supported_modes
    
    @pytest.mark.asyncio
    async def test_config_update(self, pathrag_config):
        """Test configuration updates."""
        engine = PathRAGEngine(pathrag_config)
        
        new_config = {"flow_decay_factor": 0.9}
        success = await engine.configure(new_config)
        
        assert success is True
        
        current_config = await engine.get_config()
        assert current_config["flow_decay_factor"] == 0.9
    
    @pytest.mark.asyncio
    async def test_config_schema(self):
        """Test configuration schema retrieval."""
        engine = PathRAGEngine()
        
        schema = await engine.get_config_schema()
        
        assert "type" in schema
        assert "properties" in schema
        assert "flow_decay_factor" in schema["properties"]
        assert "pruning_threshold" in schema["properties"]
    
    @pytest.mark.asyncio
    @patch('src.api.engines.pathrag_engine.PathRAGProcessor')
    async def test_retrieval_execution(self, mock_processor_class, pathrag_config):
        """Test PathRAG retrieval execution."""
        # Mock processor
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        
        # Mock processor output
        from src.types.components.contracts import RAGStrategyOutput, RAGResult, ComponentMetadata, ComponentType
        
        mock_result = RAGResult(
            content="Test content",
            score=0.8,
            metadata={"source": "test"},
            chunk_metadata={},
            document_metadata={},
            path_info=None
        )
        
        mock_output = RAGStrategyOutput(
            query="test query",
            mode=RAGMode.PATHRAG,
            results=[mock_result],
            metadata=ComponentMetadata(
                component_type=ComponentType.RAG_STRATEGY,
                component_name="pathrag",
                component_version="1.0.0"
            ),
            retrieval_stats={},
            context_info={}
        )
        
        mock_processor.retrieve_and_generate.return_value = mock_output
        mock_processor.health_check.return_value = True
        
        # Test retrieval
        engine = PathRAGEngine(pathrag_config)
        
        request = EngineRetrievalRequest(
            query="test query",
            mode=RetrievalMode.FULL_ENGINE,
            top_k=5
        )
        
        response = await engine.retrieve(request)
        
        assert response.engine_type == EngineType.PATHRAG
        assert response.query == "test query"
        assert len(response.results) == 1
        assert response.results[0].content == "Test content"
        assert response.results[0].score == 0.8


class TestComparisonFramework:
    """Test A/B testing and comparison framework."""
    
    @pytest.fixture
    def mock_engines(self):
        """Create mock engines for testing."""
        # PathRAG engine mock
        pathrag_engine = AsyncMock()
        pathrag_engine.engine_type = EngineType.PATHRAG
        pathrag_engine.supported_modes = [RetrievalMode.FULL_ENGINE, RetrievalMode.HYBRID]
        
        pathrag_response = EngineRetrievalResponse(
            engine_type=EngineType.PATHRAG,
            mode=RetrievalMode.FULL_ENGINE,
            query="test query",
            results=[
                EngineRetrievalResult(
                    content="PathRAG result 1",
                    score=0.9,
                    source="pathrag_source_1",
                    metadata={}
                ),
                EngineRetrievalResult(
                    content="PathRAG result 2", 
                    score=0.8,
                    source="pathrag_source_2",
                    metadata={}
                )
            ],
            execution_time_ms=150.0,
            total_candidates=2
        )
        pathrag_engine.retrieve.return_value = pathrag_response
        
        # SageGraph engine mock (for future comparison)
        sagegraph_engine = AsyncMock()
        sagegraph_engine.engine_type = EngineType.SAGEGRAPH
        sagegraph_engine.supported_modes = [RetrievalMode.FULL_ENGINE]
        
        sagegraph_response = EngineRetrievalResponse(
            engine_type=EngineType.SAGEGRAPH,
            mode=RetrievalMode.FULL_ENGINE,
            query="test query",
            results=[
                EngineRetrievalResult(
                    content="SageGraph result 1",
                    score=0.85,
                    source="sage_source_1",
                    metadata={}
                )
            ],
            execution_time_ms=200.0,
            total_candidates=1
        )
        sagegraph_engine.retrieve.return_value = sagegraph_response
        
        return {
            EngineType.PATHRAG: pathrag_engine,
            EngineType.SAGEGRAPH: sagegraph_engine
        }
    
    @pytest.mark.asyncio
    async def test_single_comparison(self, mock_engines):
        """Test single query comparison between engines."""
        experiment_manager = ExperimentManager()
        
        # Mock registry
        with patch('src.api.engines.comparison.get_engine_registry') as mock_registry_func:
            mock_registry = Mock()
            mock_registry.get_engine.side_effect = lambda engine_type: mock_engines.get(engine_type)
            mock_registry_func.return_value = mock_registry
            
            # Run comparison
            result = await experiment_manager.run_comparison(
                query="test query",
                engines=[EngineType.PATHRAG, EngineType.SAGEGRAPH],
                metrics=[ComparisonMetric.RELEVANCE, ComparisonMetric.SPEED]
            )
            
            # Verify results
            assert result.query == "test query"
            assert len(result.engines) == 2
            assert EngineType.PATHRAG in result.responses
            assert EngineType.SAGEGRAPH in result.responses
            
            # Check metrics
            assert "relevance" in result.metrics
            assert "speed" in result.metrics
            
            # PathRAG should score higher on relevance (0.85 avg vs 0.85)
            # Speed scores are inverted, so faster execution gets higher score
            pathrag_relevance = result.metrics["relevance"][EngineType.PATHRAG]
            sagegraph_relevance = result.metrics["relevance"][EngineType.SAGEGRAPH]
            
            assert pathrag_relevance > 0  # Should have some relevance score
            assert sagegraph_relevance > 0  # Should have some relevance score
    
    def test_metric_calculation_relevance(self):
        """Test relevance metric calculation."""
        responses = {
            EngineType.PATHRAG: EngineRetrievalResponse(
                engine_type=EngineType.PATHRAG,
                mode=RetrievalMode.FULL_ENGINE,
                query="test",
                results=[
                    EngineRetrievalResult(content="result1", score=0.9, source="s1", metadata={}),
                    EngineRetrievalResult(content="result2", score=0.8, source="s2", metadata={})
                ],
                execution_time_ms=100.0
            )
        }
        
        experiment_manager = ExperimentManager()
        
        # Test relevance calculation (should be average of scores)
        metrics = asyncio.run(experiment_manager._calculate_metrics(
            "test", responses, [ComparisonMetric.RELEVANCE]
        ))
        
        expected_relevance = (0.9 + 0.8) / 2  # 0.85
        assert abs(metrics["relevance"][EngineType.PATHRAG] - expected_relevance) < 0.001
    
    def test_metric_calculation_speed(self):
        """Test speed metric calculation."""
        responses = {
            EngineType.PATHRAG: EngineRetrievalResponse(
                engine_type=EngineType.PATHRAG,
                mode=RetrievalMode.FULL_ENGINE,
                query="test",
                results=[],
                execution_time_ms=100.0
            ),
            EngineType.SAGEGRAPH: EngineRetrievalResponse(
                engine_type=EngineType.SAGEGRAPH,
                mode=RetrievalMode.FULL_ENGINE,
                query="test",
                results=[],
                execution_time_ms=200.0
            )
        }
        
        experiment_manager = ExperimentManager()
        
        # Test speed calculation (faster should get higher score)
        metrics = asyncio.run(experiment_manager._calculate_metrics(
            "test", responses, [ComparisonMetric.SPEED]
        ))
        
        # PathRAG is faster (100ms vs 200ms), so should get higher speed score
        pathrag_speed = metrics["speed"][EngineType.PATHRAG]
        sagegraph_speed = metrics["speed"][EngineType.SAGEGRAPH]
        
        assert pathrag_speed > sagegraph_speed
        assert pathrag_speed == 0.5  # 1.0 - (100/200)
        assert sagegraph_speed == 0.0  # 1.0 - (200/200)
    
    def test_metric_calculation_diversity(self):
        """Test diversity metric calculation."""
        responses = {
            EngineType.PATHRAG: EngineRetrievalResponse(
                engine_type=EngineType.PATHRAG,
                mode=RetrievalMode.FULL_ENGINE,
                query="test",
                results=[
                    EngineRetrievalResult(content="result1", score=0.9, source="source1", metadata={}),
                    EngineRetrievalResult(content="result2", score=0.8, source="source2", metadata={}),
                    EngineRetrievalResult(content="result3", score=0.7, source="source1", metadata={})  # Duplicate source
                ],
                execution_time_ms=100.0
            )
        }
        
        experiment_manager = ExperimentManager()
        
        metrics = asyncio.run(experiment_manager._calculate_metrics(
            "test", responses, [ComparisonMetric.DIVERSITY]
        ))
        
        # 2 unique sources out of 3 results = 2/3 ≈ 0.667
        expected_diversity = 2.0 / 3.0
        assert abs(metrics["diversity"][EngineType.PATHRAG] - expected_diversity) < 0.001
    
    def test_winner_determination(self):
        """Test winner determination logic."""
        experiment_manager = ExperimentManager()
        
        metrics = {
            "relevance": {
                EngineType.PATHRAG: 0.9,
                EngineType.SAGEGRAPH: 0.7
            },
            "speed": {
                EngineType.PATHRAG: 0.8,
                EngineType.SAGEGRAPH: 0.6
            }
        }
        
        winner = experiment_manager._determine_winner(metrics)
        
        # PathRAG should win (avg: 0.85 vs 0.65)
        assert winner == EngineType.PATHRAG
    
    @pytest.mark.asyncio
    async def test_experiment_lifecycle(self, mock_engines):
        """Test complete experiment lifecycle."""
        experiment_manager = ExperimentManager()
        
        # Mock registry
        with patch('src.api.engines.comparison.get_engine_registry') as mock_registry_func:
            mock_registry = Mock()
            mock_registry.get_engine.side_effect = lambda engine_type: mock_engines.get(engine_type)
            mock_registry_func.return_value = mock_registry
            
            # Create experiment request
            from src.api.engines.base import ExperimentRequest
            
            request = ExperimentRequest(
                name="Test Experiment",
                description="Testing PathRAG vs SageGraph",
                engines=[EngineType.PATHRAG, EngineType.SAGEGRAPH],
                test_queries=["query 1", "query 2"],
                metrics=["relevance", "speed"],
                iterations=1
            )
            
            # Start experiment
            experiment_id = await experiment_manager.start_experiment(request)
            assert experiment_id is not None
            
            # Wait a bit for experiment to start
            await asyncio.sleep(0.1)
            
            # Get experiment
            experiment = await experiment_manager.get_experiment(experiment_id)
            assert experiment is not None
            assert experiment.name == "Test Experiment"
            assert experiment.engines_tested == [EngineType.PATHRAG, EngineType.SAGEGRAPH]
            
            # List experiments
            experiments = await experiment_manager.list_experiments()
            assert len(experiments) >= 1
            assert any(exp.experiment_id == experiment_id for exp in experiments)


@pytest.mark.integration
class TestEngineIntegration:
    """Integration tests for the complete multi-engine platform."""
    
    @pytest.mark.asyncio
    async def test_pathrag_engine_integration(self):
        """Test PathRAG engine with real components (mocked dependencies)."""
        # This test would require actual component integration
        # For now, we'll test the structure is correct
        
        engine = PathRAGEngine({
            "flow_decay_factor": 0.8,
            "storage": {"type": "memory"}
        })
        
        # Test configuration
        assert engine.engine_type == EngineType.PATHRAG
        assert await engine.get_config_schema() is not None
        
        # Test supported modes
        modes = engine.supported_modes
        assert RetrievalMode.PATHRAG in modes or RetrievalMode.FULL_ENGINE in modes
    
    def test_engine_registry_integration(self):
        """Test engine registry integration."""
        registry = get_engine_registry()
        
        # Should be able to register PathRAG engine
        pathrag_engine = PathRAGEngine()
        registry.register_engine(pathrag_engine)
        
        # Should be able to retrieve it
        retrieved = registry.get_engine(EngineType.PATHRAG)
        assert retrieved is not None
        assert retrieved.engine_type == EngineType.PATHRAG
    
    def test_experiment_manager_integration(self):
        """Test experiment manager integration."""
        manager = get_experiment_manager()
        
        # Should be able to create comparisons
        assert manager is not None
        
        # Should be singleton
        manager2 = get_experiment_manager()
        assert manager is manager2


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])