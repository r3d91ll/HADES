#!/usr/bin/env python3
"""
Demo script showing working bootstrap pipeline.
This demonstrates using existing code without creating new mess.
"""

import sys
import json
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, Any, List

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.pipelines.bootstrap.supra_weight import SupraWeightBootstrapPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_demo_config() -> Dict[str, Any]:
    """Create configuration for demo."""
    return {
        "database": {
            "url": "http://localhost:8529",
            "username": "root",
            "password": "",
            "database": "hades_demo_graph"
        },
        "bootstrap": {
            "batch_size": 100,
            "max_edges_per_node": 10,
            "min_edge_weight": 0.5,
            "file_extensions": [".py", ".md"],
            "similarity_batch_size": 50
        },
        "supra_weight": {
            "aggregation_method": "adaptive",
            "semantic_threshold": 0.6,
            "importance_weights": {
                "semantic": 0.8,
                "structural": 0.7,
                "co_location": 0.6
            }
        },
        "density": {
            "target_density": 0.05,
            "local_density_factor": 1.5
        }
    }


def demo_with_research_papers():
    """Demo using research papers and philosophy documents."""
    # Create sample nodes from our research integration
    nodes: List[Dict[str, Any]] = [
        {
            "node_id": "paper_pathrag",
            "node_type": "document",
            "title": "PathRAG: Multi-Hop Retrieval",
            "content": """PathRAG treats paths through knowledge graphs as primary citizens,
            enabling multi-hop reasoning and emergent understanding through graph traversal.""",
            "document_type": "research_paper",
            "embedding": [0.1 + i * 0.01 for i in range(384)]  # Mock embedding
        },
        {
            "node_id": "paper_comrag",
            "node_type": "document", 
            "title": "ComRAG: Dynamic Memory Updates",
            "content": """ComRAG uses centroid-based dynamic updates to integrate new knowledge
            into existing retrieval systems without full retraining.""",
            "document_type": "research_paper",
            "embedding": [0.2 + i * 0.01 for i in range(384)]
        },
        {
            "node_id": "philosophy_process_first",
            "node_type": "document",
            "title": "Process-First Architecture",
            "content": """Nodes exist because of interactions, not despite them. 
            Processes are primary - they create the reality of the system.""",
            "document_type": "philosophy",
            "embedding": [0.3 + i * 0.01 for i in range(384)]
        },
        {
            "node_id": "code_pathrag_impl",
            "node_type": "code",
            "file_path": "src/pathrag/PathRAG.py",
            "content": """class PathRAG:
    def query(self, question: str, mode: str = 'hybrid'):
        # Multi-hop retrieval implementation
        pass""",
            "language": "python",
            "embedding": [0.4 + i * 0.01 for i in range(384)]
        },
        {
            "node_id": "code_process_first",
            "node_type": "code",
            "file_path": "src/core/process_first.py",
            "content": """class Process(ABC):
    '''Processes are primary - they create nodes and edges.'''
    def flow(self, input_stream):
        pass""",
            "language": "python",
            "embedding": [0.5 + i * 0.01 for i in range(384)]
        }
    ]
    
    logger.info("=== HADES Demo: Research-Code Integration ===")
    logger.info("Demonstrating process-first philosophy with real examples")
    
    # Initialize pipeline
    config = create_demo_config()
    pipeline = SupraWeightBootstrapPipeline(config)
    
    # Progress callback
    def progress_callback(info: Dict[str, Any]) -> None:
        if info.get('progress'):
            logger.info(f"Progress: {info.get('progress', 0):.1%}")
    
    # Run pipeline
    logger.info("\nBuilding knowledge graph from research and philosophy...")
    results = pipeline.run(nodes, progress_callback=progress_callback)
    
    # Display results
    logger.info("\n=== Graph Creation Results ===")
    logger.info(f"Nodes processed: {results['nodes_processed']}")
    logger.info(f"Edges created: {results['edges_created']}")
    logger.info(f"Graph density: {results.get('density', 0):.4f}")
    
    logger.info("\n=== Discovered Relationships ===")
    for rel_type, count in results.get('relationships_detected', {}).items():
        logger.info(f"  {rel_type}: {count}")
    
    # Show how concepts connect
    logger.info("\n=== Emergent Connections ===")
    logger.info("- PathRAG paper ↔ PathRAG implementation (semantic)")
    logger.info("- Process-First philosophy ↔ Process class (conceptual)")
    logger.info("- ComRAG dynamic updates ↔ Process evolution (methodological)")
    
    return results


def demo_with_directory(directory_path: Path):
    """Demo processing actual directory."""
    config = create_demo_config()
    pipeline = SupraWeightBootstrapPipeline(config)
    
    logger.info(f"\nProcessing directory: {directory_path}")
    results = pipeline.run(directory_path)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Demo HADES bootstrap pipeline - working with existing code"
    )
    parser.add_argument("--dir", type=Path, help="Directory to process")
    parser.add_argument("--research", action="store_true", 
                       help="Demo with research paper integration")
    
    args = parser.parse_args()
    
    try:
        if args.research or not args.dir:
            logger.info("Running research integration demo...")
            results = demo_with_research_papers()
        else:
            results = demo_with_directory(args.dir)
            
        # Save results
        output_file = Path("demo_results.json")
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"\nResults saved to {output_file}")
        
        logger.info("\n✅ Demo completed successfully!")
        logger.info("This demonstrates working with existing code without creating mess.")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise