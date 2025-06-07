#!/usr/bin/env python3
"""
Run ISNE Bootstrap with Configuration

This script runs the ISNE bootstrap process using the bootstrap_config.yaml
file, providing a clean interface for cold start initialization.
"""

import sys
import logging
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.isne.bootstrap import ISNEBootstrapper
from src.config.config_loader import load_config


def setup_logging(log_level: str = "INFO", log_file: str = None):
    """Set up logging configuration."""
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def load_bootstrap_config(config_file: str = None):
    """Load bootstrap configuration from file or default."""
    try:
        if config_file:
            # Load custom config file
            logger = logging.getLogger(__name__)
            logger.info(f"Loading custom config from: {config_file}")
            with open(config_file) as f:
                custom_config = json.load(f)
            return custom_config
        else:
            # Load default bootstrap config
            bootstrap_config = load_config('bootstrap_config')
            logger = logging.getLogger(__name__)
            logger.info("Loaded default bootstrap_config.yaml")
            return bootstrap_config
            
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to load configuration: {e}")
        raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run HADES-PathRAG ISNE bootstrap with configuration"
    )
    
    # Input/Output arguments
    parser.add_argument(
        "--corpus-dir", "-i",
        required=True,
        help="Directory containing documents for bootstrap corpus"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        help="Output directory for bootstrap results"
    )
    
    # Configuration arguments
    parser.add_argument(
        "--config", "-c",
        help="Path to custom bootstrap configuration JSON file"
    )
    
    parser.add_argument(
        "--config-override",
        help="JSON string to override specific config values"
    )
    
    # Bootstrap parameters (override config)
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        help="Similarity threshold for graph connections"
    )
    
    parser.add_argument(
        "--max-connections",
        type=int,
        help="Maximum connections per chunk"
    )
    
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        help="Minimum cluster size"
    )
    
    parser.add_argument(
        "--max-files",
        type=int,
        help="Maximum number of files to process"
    )
    
    # File type filtering
    parser.add_argument(
        "--file-types",
        nargs="+",
        default=[".pdf"],
        help="File types to include in bootstrap (default: .pdf)"
    )
    
    # Logging
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    parser.add_argument(
        "--log-file",
        help="Log file path"
    )
    
    # Advanced options
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training phases (graph construction only)"
    )
    
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="Run validation checks only"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)
    
    try:
        # Validate corpus directory
        corpus_dir = Path(args.corpus_dir)
        if not corpus_dir.exists():
            logger.error(f"Corpus directory does not exist: {corpus_dir}")
            return 1
        
        # Check for files in corpus
        all_files = []
        for file_type in args.file_types:
            pattern = f"*{file_type}" if not file_type.startswith("*") else file_type
            all_files.extend(list(corpus_dir.glob(pattern)))
        
        if not all_files:
            logger.error(f"No files found matching types {args.file_types} in {corpus_dir}")
            return 1
        
        logger.info(f"Found {len(all_files)} files for bootstrap")
        
        # Set up output directory
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = Path(f"./bootstrap-output-{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Bootstrap output directory: {output_dir}")
        
        # Load configuration
        config = load_bootstrap_config(args.config)
        logger.info("Loaded bootstrap configuration")
        
        # Apply config overrides from command line JSON
        if args.config_override:
            try:
                override_config = json.loads(args.config_override)
                logger.info("Applying configuration overrides")
                
                # Deep merge override config
                def deep_merge(base, override):
                    for key, value in override.items():
                        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                            deep_merge(base[key], value)
                        else:
                            base[key] = value
                
                deep_merge(config, override_config)
                
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in config override: {e}")
                return 1
        
        # Extract configuration parameters
        bootstrap_params = config.get('bootstrap', {})
        graph_config = bootstrap_params.get('initial_graph', {})
        corpus_config = bootstrap_params.get('corpus', {})
        
        # Apply command line parameter overrides
        similarity_threshold = args.similarity_threshold or graph_config.get('similarity_threshold', 0.3)
        max_connections = args.max_connections or graph_config.get('max_connections_per_chunk', 10)
        min_cluster_size = args.min_cluster_size or graph_config.get('min_cluster_size', 5)
        max_files = args.max_files or corpus_config.get('max_files', None)
        
        logger.info(f"Bootstrap parameters:")
        logger.info(f"  - Similarity threshold: {similarity_threshold}")
        logger.info(f"  - Max connections per chunk: {max_connections}")
        logger.info(f"  - Min cluster size: {min_cluster_size}")
        logger.info(f"  - Max files: {max_files or 'unlimited'}")
        
        # Limit files if specified
        if max_files and len(all_files) > max_files:
            all_files = all_files[:max_files]
            logger.info(f"Limited to {len(all_files)} files")
        
        # Initialize bootstrapper
        bootstrapper = ISNEBootstrapper(
            corpus_dir=corpus_dir,
            output_dir=output_dir,
            similarity_threshold=similarity_threshold,
            max_connections_per_chunk=max_connections,
            min_cluster_size=min_cluster_size
        )
        
        if args.validation_only:
            logger.info("Running validation-only mode")
            # TODO: Implement validation-only mode
            logger.warning("Validation-only mode not yet implemented")
            return 0
        
        # Run bootstrap process
        logger.info("Starting ISNE bootstrap process...")
        start_time = datetime.now()
        
        if args.skip_training:
            logger.info("Skipping training phases (graph construction only)")
            # TODO: Implement graph-only mode
            logger.warning("Graph-only mode not yet implemented, running full bootstrap")
        
        results = bootstrapper.bootstrap_full_corpus()
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Report results
        print("\n" + "="*50)
        print("HADES-PathRAG ISNE Bootstrap Results")
        print("="*50)
        
        corpus_stats = results['corpus_stats']
        graph_stats = results['graph_stats']
        training_results = results['training_results']
        
        print(f"Success: ✓")
        print(f"Processing time: {processing_time:.2f} seconds")
        print(f"Total documents: {corpus_stats['total_documents']}")
        print(f"Total chunks: {corpus_stats['total_chunks']}")
        print(f"Average chunk length: {corpus_stats['avg_chunk_length']:.1f} characters")
        
        print(f"\nGraph Statistics:")
        print(f"Initial graph:")
        print(f"  - Nodes: {graph_stats['initial_graph']['nodes']}")
        print(f"  - Edges: {graph_stats['initial_graph']['edges']}")
        print(f"  - Density: {graph_stats['initial_graph']['density']:.6f}")
        print(f"Refined graph:")
        print(f"  - Nodes: {graph_stats['refined_graph']['nodes']}")
        print(f"  - Edges: {graph_stats['refined_graph']['edges']}")
        print(f"  - Density: {graph_stats['refined_graph']['density']:.6f}")
        
        print(f"\nTraining Results:")
        initial_training = training_results['initial_training']
        final_training = training_results['final_training']
        print(f"Initial training loss: {initial_training.get('final_loss', 'N/A')}")
        print(f"Final training loss: {final_training.get('final_loss', 'N/A')}")
        
        # Output file locations
        print(f"\nOutput Files:")
        print(f"Bootstrap directory: {output_dir}")
        print(f"Models: {output_dir}/models/")
        print(f"Embeddings: {output_dir}/embeddings/")
        print(f"Graphs: {output_dir}/graphs/")
        print(f"Results: {output_dir}/bootstrap_results.json")
        
        # Save enhanced results with config info
        enhanced_results = {
            **results,
            'config_used': config,
            'command_line_args': vars(args),
            'processing_time_seconds': processing_time,
            'files_processed': len(all_files)
        }
        
        enhanced_results_file = output_dir / "enhanced_bootstrap_results.json"
        with open(enhanced_results_file, 'w') as f:
            json.dump(enhanced_results, f, indent=2, default=str)
        
        print(f"Enhanced results: {enhanced_results_file}")
        
        # Performance recommendations
        files_per_second = len(all_files) / processing_time if processing_time > 0 else 0
        print(f"\nPerformance:")
        print(f"Processing rate: {files_per_second:.2f} files/second")
        
        if processing_time > 600:  # > 10 minutes
            print("For large-scale processing consider:")
            print("  - GPU acceleration for training")
            print("  - Parallel document processing")
            print("  - Increased batch sizes")
        
        return 0
        
    except Exception as e:
        logger.error(f"Bootstrap execution failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())