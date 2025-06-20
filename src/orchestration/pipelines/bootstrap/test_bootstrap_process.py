#!/usr/bin/env python3
"""
Test Bootstrap Process Script

This script tests the ISNE bootstrap process with a small corpus
to examine the actual JSON output structure at each stage.
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, timezone

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.isne.bootstrap import ISNEBootstrapper
from src.config.config_loader import load_config

def setup_logging() -> None:
    """Set up logging for bootstrap test."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('bootstrap_test.log')
        ]
    )

def main() -> bool:
    """Run bootstrap test on small corpus."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Define paths (reuse the project_root already defined)
    # project_root already defined above
    corpus_dir = project_root / "test-data"
    output_dir = project_root / "test-output" / "bootstrap-test" / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    
    logger.info(f"Starting bootstrap test")
    logger.info(f"Corpus directory: {corpus_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    try:
        # Load bootstrap configuration
        try:
            bootstrap_config = load_config('bootstrap_config')
            logger.info("Loaded bootstrap configuration from bootstrap_config.yaml")
        except Exception as e:
            logger.warning(f"Failed to load bootstrap config: {e}, using defaults")
            bootstrap_config = {
                'bootstrap': {
                    'initial_graph': {
                        'similarity_threshold': 0.3,
                        'max_connections_per_chunk': 10,
                        'min_cluster_size': 5
                    }
                }
            }
        
        # Extract bootstrap parameters from config
        bootstrap_params = bootstrap_config.get('bootstrap', {})
        graph_config = bootstrap_params.get('initial_graph', {})
        
        # Initialize bootstrapper with config parameters
        bootstrapper = ISNEBootstrapper(
            corpus_dir=corpus_dir,
            output_dir=output_dir,
            similarity_threshold=graph_config.get('similarity_threshold', 0.3),
            max_connections_per_chunk=graph_config.get('max_connections_per_chunk', 10),
            min_cluster_size=graph_config.get('min_cluster_size', 5)
        )
        
        # Run bootstrap process
        logger.info("Starting bootstrap process...")
        results = bootstrapper.bootstrap_full_corpus()
        
        logger.info("Bootstrap completed successfully!")
        logger.info(f"Results: {results}")
        
        # Print key information about outputs
        print(f"\n=== Bootstrap Test Results ===")
        print(f"Output directory: {output_dir}")
        print(f"Total documents processed: {results['corpus_stats']['total_documents']}")
        print(f"Total chunks created: {results['corpus_stats']['total_chunks']}")
        print(f"Bootstrap time: {results['bootstrap_time']:.2f} seconds")
        
        print(f"\n=== Files Created ===")
        for subdir in ['models', 'embeddings', 'graphs', 'debug']:
            subdir_path = output_dir / subdir
            if subdir_path.exists():
                files = list(subdir_path.glob("*"))
                print(f"{subdir}/: {len(files)} files")
                for file in files:
                    print(f"  - {file.name}")
        
        print(f"\n=== JSON Structure Examination ===")
        
        # Examine chunk metadata JSON structure
        metadata_file = output_dir / "embeddings" / "chunk_metadata.json"
        if metadata_file.exists():
            import json
            with open(metadata_file) as f:
                metadata = json.load(f)
            
            print(f"Chunk metadata structure:")
            print(f"  - Total chunks: {len(metadata.get('chunks', []))}")
            if metadata.get('chunks'):
                sample_chunk = metadata['chunks'][0]
                print(f"  - Sample chunk keys: {list(sample_chunk.keys())}")
                print(f"  - Sample chunk ID: {sample_chunk.get('id', 'N/A')}")
                print(f"  - Sample chunk text preview: {sample_chunk.get('text', '')[:100]}...")
        
        # Examine bootstrap results JSON
        results_file = output_dir / "bootstrap_results.json"
        if results_file.exists():
            print(f"\nBootstrap results JSON structure available at: {results_file}")
            
        return True
        
    except Exception as e:
        logger.error(f"Bootstrap test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)