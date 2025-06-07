#!/usr/bin/env python3
"""
Run Data Ingestion Pipeline with Configuration

This script runs the complete data ingestion pipeline using the configuration
files we created, providing a clean interface for production use.
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

from src.orchestration.pipelines.data_ingestion_pipeline import run_data_ingestion_pipeline
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


def load_pipeline_config(config_file: str = None):
    """Load pipeline configuration from file or default."""
    try:
        if config_file:
            # Load custom config file
            logger = logging.getLogger(__name__)
            logger.info(f"Loading custom config from: {config_file}")
            with open(config_file) as f:
                custom_config = json.load(f)
            return custom_config
        else:
            # Load default data ingestion config
            data_ingestion_config = load_config('data_ingestion_config')
            logger = logging.getLogger(__name__)
            logger.info("Loaded default data_ingestion_config.yaml")
            
            # Extract stage-specific configs
            config = {
                'docproc': data_ingestion_config.get('docproc', {}),
                'chunking': data_ingestion_config.get('chunking', {}),
                'embedding': data_ingestion_config.get('embedding', {}),
                'isne': data_ingestion_config.get('isne', {}),
                'storage': data_ingestion_config.get('storage', {})
            }
            
            return config
            
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to load configuration: {e}")
        raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run HADES-PathRAG data ingestion pipeline with configuration"
    )
    
    # Input/Output arguments
    parser.add_argument(
        "--input-dir", "-i",
        required=True,
        help="Directory containing input files to process"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        help="Output directory for debug files (if debug enabled)"
    )
    
    # Configuration arguments
    parser.add_argument(
        "--config", "-c",
        help="Path to custom configuration JSON file"
    )
    
    parser.add_argument(
        "--config-override",
        help="JSON string to override specific config values"
    )
    
    # Processing options
    parser.add_argument(
        "--file-types",
        nargs="+",
        default=[".pdf"],
        help="File types to process (default: .pdf)"
    )
    
    parser.add_argument(
        "--max-files",
        type=int,
        help="Maximum number of files to process"
    )
    
    # Debug and logging
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with intermediate file saving"
    )
    
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
    
    # Storage options
    parser.add_argument(
        "--storage-mode",
        choices=["create", "append", "upsert"],
        help="Storage mode (overrides config)"
    )
    
    parser.add_argument(
        "--database-name",
        help="Database name (overrides config)"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)
    
    try:
        # Find input files
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            logger.error(f"Input directory does not exist: {input_dir}")
            return 1
        
        input_files = []
        for file_type in args.file_types:
            pattern = f"*{file_type}" if not file_type.startswith("*") else file_type
            input_files.extend(str(f) for f in input_dir.glob(pattern))
        
        if not input_files:
            logger.error(f"No files found matching types {args.file_types} in {input_dir}")
            return 1
        
        # Limit files if specified
        if args.max_files:
            input_files = input_files[:args.max_files]
            logger.info(f"Limited to {len(input_files)} files")
        
        logger.info(f"Found {len(input_files)} files to process")
        for file_path in input_files[:5]:  # Show first 5 files
            logger.info(f"  - {Path(file_path).name}")
        if len(input_files) > 5:
            logger.info(f"  ... and {len(input_files) - 5} more files")
        
        # Load configuration
        config = load_pipeline_config(args.config)
        logger.info("Loaded pipeline configuration")
        
        # Apply config overrides
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
        
        # Apply command line overrides
        if args.storage_mode:
            config.setdefault('storage', {})['mode'] = args.storage_mode
            logger.info(f"Override storage mode: {args.storage_mode}")
        
        if args.database_name:
            config.setdefault('storage', {})['database_name'] = args.database_name
            logger.info(f"Override database name: {args.database_name}")
        
        # Set up output directory for debug
        output_dir = None
        if args.debug:
            if args.output_dir:
                output_dir = args.output_dir
            else:
                output_dir = f"./pipeline-output-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            logger.info(f"Debug mode enabled, output directory: {output_dir}")
        
        # Run pipeline
        logger.info("Starting data ingestion pipeline...")
        start_time = datetime.now()
        
        results = run_data_ingestion_pipeline(
            input_files=input_files,
            config=config,
            enable_debug=args.debug,
            debug_output_dir=output_dir,
            filter_types=args.file_types
        )
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Report results
        print("\n" + "="*50)
        print("HADES-PathRAG Data Ingestion Pipeline Results")
        print("="*50)
        
        print(f"Success: {'✓' if results['success'] else '✗'}")
        print(f"Processing time: {processing_time:.2f} seconds")
        print(f"Files processed: {results['pipeline_stats']['total_files']}")
        print(f"Documents created: {results['pipeline_stats']['processed_documents']}")
        print(f"Chunks generated: {results['pipeline_stats']['generated_chunks']}")
        print(f"Chunks embedded: {results['pipeline_stats']['embedded_chunks']}")
        print(f"Chunks ISNE enhanced: {results['pipeline_stats']['isne_enhanced_chunks']}")
        print(f"Documents stored: {results['pipeline_stats']['stored_documents']}")
        print(f"Chunks stored: {results['pipeline_stats']['stored_chunks']}")
        print(f"Relationships stored: {results['pipeline_stats']['stored_relationships']}")
        
        # Stage performance
        print(f"\nStage Performance:")
        stage_times = results['stage_times']
        for stage, time_taken in stage_times.items():
            print(f"  {stage}: {time_taken:.2f}s")
        
        # Error summary
        errors = results['pipeline_stats']['errors']
        if errors:
            print(f"\nErrors ({len(errors)}):")
            for error in errors:
                print(f"  - {error['stage']}: {error['message']}")
        
        # Debug files
        if args.debug and results['debug_output_dir']:
            print(f"\nDebug files saved to: {results['debug_output_dir']}")
        
        # Save results to file
        results_file = f"pipeline_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Detailed results saved to: {results_file}")
        
        return 0 if results['success'] else 1
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())