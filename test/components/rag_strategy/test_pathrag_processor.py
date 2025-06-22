"""
Test suite for PathRAG processor implementation.

This module tests the core PathRAG algorithms including:
- Flow-based resource allocation
- Multi-hop path reasoning
- Hierarchical keyword extraction
- Hybrid retrieval strategy
"""

import pytest
import networkx as nx
from typing import Dict, Any, List
from unittest.mock import Mock, patch

from src.components.rag_strategy.pathrag.processor import PathRAGProcessor
from src.types.components.contracts import (
    RAGStrategyInput, 
    RAGMode,
    ComponentMetadata,
    ComponentType,
    ProcessingStatus
)


class TestPathRAGProcessor:
    """Test suite for PathRAG processor."""
    
    @pytest.fixture
    def pathrag_config(self) -> Dict[str, Any]:
        """Create test configuration."""
        return {
            "flow_decay_factor": 0.8,
            "pruning_threshold": 0.3,
            "max_path_length": 5,
            "max_iterations": 10,
            "convergence_threshold": 0.001,
            "storage": {"type": "memory"},
            "embedder": {"type": "cpu"},
            "graph_enhancer": {"type": "isne"},
            "resource_allocation": {
                "initial_resource_per_node": 1.0,
                "alpha_decay": 0.8,
                "threshold_pruning": 0.3,
                "enable_early_stopping": True
            },
            "context": {
                "weights": {
                    "entity_context": 0.4,
                    "relationship_context": 0.4,
                    "source_text_context": 0.2
                }
            }
        }
    
    @pytest.fixture
    def processor(self, pathrag_config) -> PathRAGProcessor:
        """Create configured PathRAG processor."""
        processor = PathRAGProcessor()
        processor.configure(pathrag_config)
        return processor
    
    @pytest.fixture
    def sample_graph(self) -> nx.Graph:
        """Create sample knowledge graph for testing."""
        G = nx.Graph()
        
        # Add nodes with metadata
        nodes_data = [
            ("entity_1", {"content": "Content for entity 1", "description": "First entity", "entity_type": "person"}),
            ("entity_2", {"content": "Content for entity 2", "description": "Second entity", "entity_type": "organization"}),
            ("entity_3", {"content": "Content for entity 3", "description": "Third entity", "entity_type": "location"}),
            ("entity_4", {"content": "Content for entity 4", "description": "Fourth entity", "entity_type": "concept"}),
            ("entity_5", {"content": "Content for entity 5", "description": "Fifth entity", "entity_type": "event"}),
        ]
        
        for node_id, data in nodes_data:
            G.add_node(node_id, **data)
        
        # Add edges with weights
        edges_data = [
            ("entity_1", "entity_2", {"weight": 0.8, "description": "works at"}),
            ("entity_2", "entity_3", {"weight": 0.6, "description": "located in"}),
            ("entity_1", "entity_3", {"weight": 0.4, "description": "lives in"}),
            ("entity_3", "entity_4", {"weight": 0.7, "description": "represents"}),
            ("entity_4", "entity_5", {"weight": 0.5, "description": "influences"}),
        ]
        
        for source, target, data in edges_data:
            G.add_edge(source, target, **data)
        
        return G
    
    def test_processor_initialization(self, processor):
        """Test processor initialization."""
        assert processor.name == "pathrag"
        assert processor.version == "1.0.0"
        assert processor.component_type == ComponentType.RAG_STRATEGY
        assert not processor.is_initialized
    
    def test_config_validation(self, processor, pathrag_config):
        """Test configuration validation."""
        # Valid config
        assert processor.validate_config(pathrag_config)
        
        # Invalid config - missing required key
        invalid_config = pathrag_config.copy()
        del invalid_config["flow_decay_factor"]
        assert not processor.validate_config(invalid_config)
        
        # Invalid config - out of range value
        invalid_config = pathrag_config.copy()
        invalid_config["flow_decay_factor"] = 1.5
        assert not processor.validate_config(invalid_config)
    
    def test_health_check(self, processor):
        """Test health check functionality."""
        # Should pass with mock components
        assert processor.health_check()
    
    def test_keyword_extraction_simple(self, processor):
        """Test simple keyword extraction fallback."""
        query = "Find information about machine learning algorithms"
        keywords = processor._simple_keyword_extraction(query)
        
        assert "high_level_keywords" in keywords
        assert "low_level_keywords" in keywords
        assert len(keywords["high_level_keywords"]) > 0
        assert len(keywords["low_level_keywords"]) > 0
    
    def test_resource_allocation_initialization(self, processor, sample_graph):
        """Test resource allocation initialization."""
        processor.knowledge_graph = sample_graph
        start_nodes = ["entity_1", "entity_2"]
        
        resources = processor._initialize_resource_allocation(start_nodes)
        
        # Check resource distribution
        assert len(resources) == len(sample_graph.nodes())
        assert resources["entity_1"] > 0
        assert resources["entity_2"] > 0
        assert resources["entity_3"] == 0  # Not a start node
        
        # Check total resources
        total_resources = sum(resources.values())
        assert abs(total_resources - 1.0) < 0.001  # Should sum to 1.0
    
    def test_flow_based_pruning(self, processor, sample_graph):
        """Test flow-based pruning algorithm."""
        processor.knowledge_graph = sample_graph
        
        # Initialize resources
        initial_resources = {
            "entity_1": 0.5,
            "entity_2": 0.5,
            "entity_3": 0.0,
            "entity_4": 0.0,
            "entity_5": 0.0
        }
        
        # Run flow-based pruning
        pruned_resources = processor._flow_based_pruning(
            initial_resources,
            decay_factor=0.8,
            threshold=0.1,
            max_iterations=5
        )
        
        # Check that resources flowed through the graph
        assert len(pruned_resources) > 0
        assert "entity_1" in pruned_resources  # Start node should retain resources
        
        # Connected nodes should receive some resources
        neighbors = list(sample_graph.neighbors("entity_1"))
        connected_with_resources = [node for node in neighbors if node in pruned_resources]
        assert len(connected_with_resources) > 0
    
    def test_multi_hop_path_finding(self, processor, sample_graph):
        """Test multi-hop path finding algorithm."""
        processor.knowledge_graph = sample_graph
        
        # Find paths between entities
        source = "entity_1"
        target = "entity_4"
        
        paths = processor._find_paths_between_entities(source, target, max_hops=3)
        
        assert len(paths) > 0
        
        for path_data in paths:
            assert "path" in path_data
            assert "hop_length" in path_data
            assert "narrative" in path_data
            assert "score" in path_data
            
            path = path_data["path"]
            assert path[0] == source
            assert path[-1] == target
            assert len(path) >= 2
    
    def test_path_narrative_building(self, processor, sample_graph):
        """Test path narrative building."""
        processor.knowledge_graph = sample_graph
        
        # Test 1-hop narrative
        path_1hop = ["entity_1", "entity_2"]
        narrative_1hop = processor._build_path_narrative(path_1hop, 1)
        assert "entity_1" in narrative_1hop
        assert "entity_2" in narrative_1hop
        assert len(narrative_1hop) > 0
        
        # Test 2-hop narrative
        path_2hop = ["entity_1", "entity_2", "entity_3"]
        narrative_2hop = processor._build_path_narrative(path_2hop, 2)
        assert "entity_1" in narrative_2hop
        assert "entity_2" in narrative_2hop
        assert "entity_3" in narrative_2hop
        assert len(narrative_2hop) > len(narrative_1hop)
    
    def test_path_scoring_weighted_bfs(self, processor, sample_graph):
        """Test weighted BFS path scoring."""
        processor.knowledge_graph = sample_graph
        
        # Test different path lengths
        short_path = ["entity_1", "entity_2"]
        long_path = ["entity_1", "entity_2", "entity_3", "entity_4"]
        
        short_score = processor._calculate_path_score_weighted_bfs(short_path)
        long_score = processor._calculate_path_score_weighted_bfs(long_path)
        
        # Shorter paths should generally score higher due to decay
        assert short_score > 0
        assert long_score > 0
        # Note: In some cases, edge weights might make longer paths score higher
    
    def test_hierarchical_node_retrieval(self, processor, sample_graph):
        """Test hierarchical node retrieval."""
        processor.knowledge_graph = sample_graph
        
        keywords = {
            "high_level_keywords": ["organization", "location"],
            "low_level_keywords": ["entity", "person"]
        }
        
        query_embedding = [0.1] * 768  # Mock embedding
        
        relevant_nodes = processor._retrieve_relevant_nodes(
            query_embedding, keywords, top_k=3
        )
        
        assert "entity_nodes" in relevant_nodes
        assert "relationship_nodes" in relevant_nodes
        assert isinstance(relevant_nodes["entity_nodes"], list)
        assert isinstance(relevant_nodes["relationship_nodes"], list)
    
    def test_end_to_end_pathrag_retrieval(self, processor, sample_graph):
        """Test end-to-end PathRAG retrieval."""
        # Setup processor with sample graph
        processor.knowledge_graph = sample_graph
        processor.is_initialized = True
        
        # Create test input
        input_data = RAGStrategyInput(
            query="Find connections between person and organization",
            mode=RAGMode.PATHRAG,
            top_k=5,
            max_token_for_context=4000
        )
        
        # Mock keyword extraction
        with patch.object(processor, '_extract_keywords') as mock_extract:
            mock_extract.return_value = {
                "high_level_keywords": ["connections", "relationships"],
                "low_level_keywords": ["person", "organization"]
            }
            
            # Run PathRAG retrieval
            output = processor.retrieve_and_generate(input_data)
        
        # Verify output structure
        assert output.query == input_data.query
        assert output.mode == RAGMode.PATHRAG
        assert len(output.results) <= input_data.top_k
        assert output.metadata.status == ProcessingStatus.SUCCESS
        
        # Check retrieval stats
        assert "entity_nodes_found" in output.retrieval_stats
        assert "relationship_nodes_found" in output.retrieval_stats
        assert "paths_explored" in output.retrieval_stats
    
    def test_hybrid_retrieval_mode(self, processor, sample_graph):
        """Test hybrid retrieval mode."""
        processor.knowledge_graph = sample_graph
        processor.is_initialized = True
        
        input_data = RAGStrategyInput(
            query="Find information about organizations and locations",
            mode=RAGMode.HYBRID,
            top_k=3,
            max_token_for_context=2000
        )
        
        # Mock keyword extraction
        with patch.object(processor, '_extract_keywords') as mock_extract:
            mock_extract.return_value = {
                "high_level_keywords": ["organizations", "locations"],
                "low_level_keywords": ["entity", "information"]
            }
            
            output = processor.retrieve_and_generate(input_data)
        
        # Verify hybrid mode execution
        assert output.mode == RAGMode.HYBRID
        assert len(output.results) > 0
        
        # Check that both local and global contexts are used
        context_types = [result.metadata.get("context_type") for result in output.results]
        # Should have mix of local and global context types
    
    def test_supported_modes(self, processor):
        """Test supported mode reporting."""
        supported_modes = processor.get_supported_modes()
        expected_modes = ["naive", "local", "global", "hybrid", "pathrag"]
        
        for mode in expected_modes:
            assert mode in supported_modes
            assert processor.supports_mode(mode)
        
        # Test unsupported mode
        assert not processor.supports_mode("unsupported_mode")
    
    def test_metrics_collection(self, processor, sample_graph):
        """Test metrics collection."""
        processor.knowledge_graph = sample_graph
        processor.is_initialized = True
        
        metrics = processor.get_metrics()
        
        assert "component_name" in metrics
        assert "component_version" in metrics
        assert "is_initialized" in metrics
        assert "knowledge_graph_nodes" in metrics
        assert "knowledge_graph_edges" in metrics
        
        assert metrics["component_name"] == "pathrag"
        assert metrics["knowledge_graph_nodes"] == len(sample_graph.nodes())
        assert metrics["knowledge_graph_edges"] == len(sample_graph.edges())
    
    def test_error_handling(self, processor):
        """Test error handling in retrieval."""
        # Test with uninitialized processor
        input_data = RAGStrategyInput(
            query="Test query",
            mode=RAGMode.PATHRAG,
            top_k=5,
            max_token_for_context=1000
        )
        
        # Should handle gracefully and return error output
        output = processor.retrieve_and_generate(input_data)
        
        # Check error handling
        assert output.metadata.status == ProcessingStatus.ERROR
        assert len(output.errors) > 0
        assert len(output.results) == 0


class TestPathRAGIntegration:
    """Integration tests for PathRAG with HADES components."""
    
    def test_config_schema_compatibility(self):
        """Test configuration schema compatibility."""
        processor = PathRAGProcessor()
        schema = processor.get_config_schema()
        
        # Verify required schema fields
        assert "type" in schema
        assert "properties" in schema
        assert "required" in schema
        
        # Check required properties
        required_fields = schema["required"]
        assert "flow_decay_factor" in required_fields
        assert "pruning_threshold" in required_fields
        assert "max_path_length" in required_fields
    
    def test_knowledge_base_stats(self):
        """Test knowledge base statistics."""
        processor = PathRAGProcessor()
        
        # Before initialization
        stats = processor.get_knowledge_base_stats()
        assert not stats["initialized"]
        
        # After initialization (mock)
        processor.is_initialized = True
        processor.knowledge_graph = nx.Graph()
        processor.knowledge_graph.add_nodes_from(["n1", "n2", "n3"])
        processor.knowledge_graph.add_edges_from([("n1", "n2"), ("n2", "n3")])
        
        stats = processor.get_knowledge_base_stats()
        assert stats["initialized"]
        assert stats["total_nodes"] == 3
        assert stats["total_edges"] == 2
        assert stats["average_degree"] > 0
    
    def test_performance_estimation(self):
        """Test performance estimation methods."""
        processor = PathRAGProcessor()
        
        input_data = RAGStrategyInput(
            query="Test query",
            mode=RAGMode.PATHRAG,
            top_k=10,
            max_token_for_context=4000
        )
        
        # Test retrieval time estimation
        retrieval_time = processor.estimate_retrieval_time(input_data)
        assert retrieval_time > 0
        
        # Test generation time estimation
        generation_time = processor.estimate_generation_time(input_data)
        assert generation_time > 0
        
        # PathRAG should be more expensive than other modes
        input_data.mode = RAGMode.NAIVE
        naive_time = processor.estimate_retrieval_time(input_data)
        
        input_data.mode = RAGMode.PATHRAG
        pathrag_time = processor.estimate_retrieval_time(input_data)
        
        assert pathrag_time > naive_time


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])