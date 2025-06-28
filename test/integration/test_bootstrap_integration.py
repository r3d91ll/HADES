"""
Integration test for bootstrap pipeline.
Tests the actual working pipeline without creating new code.
"""

import pytest
import sys
from pathlib import Path
from typing import Dict, Any, List

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.pipelines.bootstrap.supra_weight import SupraWeightBootstrapPipeline


@pytest.fixture
def test_config() -> Dict[str, Any]:
    """Test configuration."""
    return {
        "database": {
            "url": "http://localhost:8529",
            "username": "root", 
            "password": "",
            "database": "test_integration_db"
        },
        "bootstrap": {
            "batch_size": 10,
            "max_edges_per_node": 5,
            "min_edge_weight": 0.5
        },
        "supra_weight": {
            "aggregation_method": "adaptive",
            "semantic_threshold": 0.6
        },
        "density": {
            "target_density": 0.1
        }
    }


@pytest.fixture
def sample_nodes() -> List[Dict[str, Any]]:
    """Sample nodes for testing."""
    return [
        {
            "node_id": "test_1",
            "node_type": "document",
            "content": "Test document about PathRAG",
            "embedding": [0.1] * 384
        },
        {
            "node_id": "test_2", 
            "node_type": "document",
            "content": "Test document about ISNE",
            "embedding": [0.2] * 384
        },
        {
            "node_id": "test_3",
            "node_type": "code",
            "content": "def process(): pass",
            "embedding": [0.3] * 384
        }
    ]


@pytest.mark.integration
def test_bootstrap_pipeline_with_nodes(test_config, sample_nodes):
    """Test bootstrap pipeline with sample nodes."""
    pipeline = SupraWeightBootstrapPipeline(test_config)
    
    # Run pipeline
    results = pipeline.run(sample_nodes)
    
    # Verify results
    assert results['nodes_processed'] == len(sample_nodes)
    assert results['edges_created'] >= 0
    assert 'duration_seconds' in results
    assert results.get('errors', []) == []
    
    # Validate graph
    validation = pipeline.validate_graph()
    assert validation['is_valid'] is True


@pytest.mark.integration 
def test_bootstrap_pipeline_validation(test_config, sample_nodes):
    """Test graph validation after bootstrap."""
    pipeline = SupraWeightBootstrapPipeline(test_config)
    
    # Run pipeline
    pipeline.run(sample_nodes)
    
    # Validate
    validation = pipeline.validate_graph()
    
    assert 'is_valid' in validation
    assert 'warnings' in validation
    assert 'errors' in validation
    
    # Check specific validations
    if validation['warnings']:
        for warning in validation['warnings']:
            assert isinstance(warning, str)


@pytest.mark.integration
def test_relationship_detection(test_config):
    """Test that relationships are properly detected."""
    # Create nodes with known relationships
    nodes = [
        {
            "node_id": "file_1",
            "node_type": "file",
            "file_path": "/src/main.py",
            "content": "from utils import helper",
            "embedding": [0.5] * 384
        },
        {
            "node_id": "file_2",
            "node_type": "file", 
            "file_path": "/src/utils.py",
            "content": "def helper(): return 42",
            "embedding": [0.6] * 384
        }
    ]
    
    pipeline = SupraWeightBootstrapPipeline(test_config)
    results = pipeline.run(nodes)
    
    # Should detect at least semantic relationships
    assert 'relationships_detected' in results
    relationships = results['relationships_detected']
    assert len(relationships) > 0
    
    # Check that some relationships were found
    total_relationships = sum(relationships.values())
    assert total_relationships > 0


@pytest.mark.integration
@pytest.mark.parametrize("node_count", [1, 10, 50])
def test_scalability(test_config, node_count):
    """Test pipeline scales with different node counts."""
    # Generate nodes
    nodes = []
    for i in range(node_count):
        nodes.append({
            "node_id": f"node_{i}",
            "node_type": "test",
            "content": f"Test content {i}",
            "embedding": [0.1 * (i % 10)] * 384
        })
    
    pipeline = SupraWeightBootstrapPipeline(test_config)
    results = pipeline.run(nodes)
    
    assert results['nodes_processed'] == node_count
    assert results['errors'] == []
    
    # Density should be reasonable
    if 'density' in results and results['density'] is not None:
        assert 0 <= results['density'] <= 1


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "-m", "integration"])