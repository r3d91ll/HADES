#!/usr/bin/env python3
"""
Run Modular Bootstrap Pipeline

This script runs the modular bootstrap pipeline that can reuse data ingestion
components through the component factory system.
"""

import sys
import logging
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.orchestration.pipelines.bootstrap.modular_pipeline import run_modular_bootstrap
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


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Run modular bootstrap pipeline for HADES"
    )
    
    # Required arguments
    parser.add_argument(
        "--corpus-dir",
        required=True,
        help="Directory containing corpus files"
    )
    
    # Optional arguments
    parser.add_argument(
        "--output-dir",
        help="Output directory for bootstrap results (default: ./bootstrap_output)"
    )
    
    parser.add_argument(
        "--config",
        help="Configuration file or name (default: modular_pipeline_config)"
    )
    
    parser.add_argument(
        "--file-pattern",
        default="*.pdf",
        help="File pattern to match (default: *.pdf)"
    )
    
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--log-file",
        help="Log file path (optional)"
    )
    
    # Component selection overrides
    parser.add_argument(
        "--document-processor",
        help="Override document processor implementation"
    )
    
    parser.add_argument(
        "--chunker",
        help="Override chunker implementation"
    )
    
    parser.add_argument(
        "--embedder",
        help="Override embedder implementation"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)
    
    try:
        # Validate input directory
        corpus_dir = Path(args.corpus_dir)
        if not corpus_dir.exists():
            raise ValueError(f"Corpus directory does not exist: {corpus_dir}")
        
        # Setup output directory
        output_dir = Path(args.output_dir) if args.output_dir else Path("./bootstrap_output")
        
        # Load configuration
        config = None
        if args.config:
            try:
                # Try to load as config name first
                config = load_config(args.config)
                logger.info(f"Loaded configuration: {args.config}")
            except:
                # Try to load as file path
                config_path = Path(args.config)
                if config_path.exists():
                    import yaml
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f)
                    logger.info(f"Loaded configuration from file: {config_path}")
                else:
                    logger.warning(f"Configuration not found: {args.config}, using default")
        
        # Apply command line component overrides
        if config and any([args.document_processor, args.chunker, args.embedder]):
            if 'components' not in config:
                config['components'] = {}
            
            if args.document_processor:
                config['components']['document_processor'] = {
                    'implementation': args.document_processor,
                    'enabled': True
                }
            
            if args.chunker:
                config['components']['chunker'] = {
                    'implementation': args.chunker,
                    'enabled': True
                }
            
            if args.embedder:
                config['components']['embedder'] = {
                    'implementation': args.embedder,
                    'enabled': True
                }
            
            logger.info("Applied command line component overrides")
        
        # Log configuration
        logger.info(f"Corpus directory: {corpus_dir}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"File pattern: {args.file_pattern}")
        
        # Run bootstrap
        logger.info("Starting modular bootstrap pipeline...")
        start_time = datetime.now()
        
        results = run_modular_bootstrap(
            corpus_dir=corpus_dir,
            output_dir=output_dir,
            config=config,
            file_pattern=args.file_pattern
        )
        
        end_time = datetime.now()
        total_time = end_time - start_time
        
        # Log results
        logger.info("=== Bootstrap Completed Successfully ===")
        logger.info(f"Total time: {total_time}")
        logger.info(f"Files processed: {results['corpus_stats']['total_files']}")
        logger.info(f"Chunks generated: {results['corpus_stats']['total_chunks']}")
        logger.info(f"Component configuration: {results['component_configuration']}")
        
        # Initial graph stats
        initial_graph = results['graph_stats']['initial_graph']
        logger.info(f"Initial graph: {initial_graph['nodes']} nodes, {initial_graph['edges']} edges")
        
        # Refined graph stats
        refined_graph = results['graph_stats']['refined_graph']
        logger.info(f"Refined graph: {refined_graph['nodes']} nodes, {refined_graph['edges']} edges")
        
        # Save summary to file
        summary_file = output_dir / "bootstrap_summary.json"
        summary_data = {
            'execution_info': {
                'command_line_args': vars(args),
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'total_time_seconds': total_time.total_seconds()
            },
            'results': results
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
        
        logger.info(f"Bootstrap summary saved to: {summary_file}")
        
    except Exception as e:
        logger.error(f"Bootstrap failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()