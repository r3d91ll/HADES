#!/usr/bin/env python3
"""
Test script for supra-weight bootstrap pipeline.
"""

import sys
import json
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, Any, List

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.pipelines.bootstrap.supra_weight import SupraWeightBootstrapPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_config() -> Dict[str, Any]:
    """Create test configuration for bootstrap pipeline."""
    return {
        "database": {
            "url": "http://localhost:8529",
            "username": "root",
            "password": "",
            "database": "isne_bootstrap_test"
        },
        "bootstrap": {
            "batch_size": 1000,
            "max_edges_per_node": 50,
            "min_edge_weight": 0.3,
            "file_extensions": [".py", ".md", ".json"],
            "similarity_batch_size": 500
        },
        "supra_weight": {
            "aggregation_method": "adaptive",
            "semantic_threshold": 0.5,
            "importance_weights": {
                "co_location": 1.0,
                "import": 0.95,
                "structural": 0.9,
                "reference": 0.85,
                "semantic": 0.7,
                "sequential": 0.6,
                "temporal": 0.5
            }
        },
        "density": {
            "target_density": 0.1,
            "local_density_factor": 2.0
        }
    }


def test_with_sample_nodes() -> Dict[str, Any]:
    """Test with sample nodes."""
    # Create sample nodes
    nodes: List[Dict[str, Any]] = [
        {
            "node_id": "file_1",
            "node_type": "file",
            "file_path": "/project/src/main.py",
            "file_name": "main.py",
            "directory": "/project/src",
            "content": "import utils\\nfrom models import Model\\n\\ndef main():\\n    pass",
            "modified_at": datetime.utcnow().isoformat()
        },
        {
            "node_id": "file_2",
            "node_type": "file",
            "file_path": "/project/src/utils.py",
            "file_name": "utils.py",
            "directory": "/project/src",
            "content": "def helper():\\n    return 42",
            "modified_at": datetime.utcnow().isoformat()
        },
        {
            "node_id": "chunk_1_0",
            "node_type": "chunk",
            "chunk_index": 0,
            "source_file_id": "file_1",
            "file_path": "/project/src/main.py",
            "directory": "/project/src",
            "content": "import utils\\nfrom models import Model",
            "embedding": [0.1] * 384  # Mock embedding
        },
        {
            "node_id": "chunk_1_1",
            "node_type": "chunk", 
            "chunk_index": 1,
            "source_file_id": "file_1",
            "file_path": "/project/src/main.py",
            "directory": "/project/src",
            "content": "def main():\\n    pass",
            "embedding": [0.2] * 384  # Mock embedding
        }
    ]
    
    # Initialize pipeline
    config = create_test_config()
    pipeline = SupraWeightBootstrapPipeline(config)
    
    # Progress callback
    def progress_callback(info: Dict[str, Any]) -> None:
        logger.info(f"Progress: {info.get('progress', 0):.1%}")
    
    # Run pipeline
    logger.info("Starting bootstrap pipeline with sample nodes...")
    results = pipeline.run(nodes, progress_callback=progress_callback)
    
    # Print results
    logger.info("\\nPipeline Results:")
    logger.info(f"Nodes processed: {results['nodes_processed']}")
    logger.info(f"Edges created: {results['edges_created']}")
    logger.info(f"Graph density: {results.get('density', 0):.4f}")
    logger.info(f"Duration: {results.get('duration_seconds', 0):.2f} seconds")
    
    logger.info("\\nRelationships detected:")
    for rel_type, count in results.get('relationships_detected', {}).items():
        logger.info(f"  {rel_type}: {count}")
        
    logger.info("\\nDensity control statistics:")
    density_stats = results.get('density_control', {})
    logger.info(f"  Acceptance rate: {density_stats.get('acceptance_rate', 0):.2%}")
    logger.info(f"  Average node degree: {density_stats.get('avg_node_degree', 0):.2f}")
    
    # Validate graph
    validation = pipeline.validate_graph()
    logger.info(f"\\nGraph validation: {'PASSED' if validation['is_valid'] else 'FAILED'}")
    if validation['warnings']:
        logger.warning(f"Warnings: {validation['warnings']}")
    if validation['errors']:
        logger.error(f"Errors: {validation['errors']}")
        
    return results


def test_with_directory(directory_path: Path) -> Dict[str, Any]:
    """Test with a directory of files."""
    config = create_test_config()
    pipeline = SupraWeightBootstrapPipeline(config)
    
    logger.info(f"Processing directory: {directory_path}")
    results = pipeline.run(directory_path)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test supra-weight bootstrap pipeline")
    parser.add_argument("--dir", type=Path, help="Directory to process")
    parser.add_argument("--sample", action="store_true", help="Use sample nodes")
    
    args = parser.parse_args()
    
    try:
        if args.sample or not args.dir:
            results = test_with_sample_nodes()
        else:
            results = test_with_directory(args.dir)
            
        # Save results
        output_file = Path("bootstrap_results.json")
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"\\nResults saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise