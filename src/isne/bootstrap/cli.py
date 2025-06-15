"""
Command Line Interface for ISNE Bootstrap Pipeline

Provides command-line access to the ISNE bootstrap pipeline
for creating ISNE models from document collections.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from .config import BootstrapConfig
from .pipeline import ISNEBootstrapPipeline
from .monitoring import BootstrapMonitor


def setup_logging(log_level: str = "INFO", log_file: Optional[Path] = None):
    """Set up logging configuration."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def collect_input_files(paths: List[str], recursive: bool = True) -> List[Path]:
    """
    Collect input files from given paths.
    
    Args:
        paths: List of file or directory paths
        recursive: Whether to search directories recursively
        
    Returns:
        List of input file paths
    """
    input_files = []
    
    for path_str in paths:
        path = Path(path_str)
        
        if path.is_file():
            input_files.append(path)
        elif path.is_dir():
            if recursive:
                # Recursively find all files
                for file_path in path.rglob("*"):
                    if file_path.is_file():
                        input_files.append(file_path)
            else:
                # Only direct children
                for file_path in path.iterdir():
                    if file_path.is_file():
                        input_files.append(file_path)
        else:
            print(f"Warning: Path does not exist: {path}")
    
    return input_files


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="ISNE Bootstrap Pipeline - Create ISNE models from document collections",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default config
  python -m src.isne.bootstrap.cli --input-dir /path/to/docs --output-dir /path/to/output
  
  # Custom configuration
  python -m src.isne.bootstrap.cli \\
    --config config/custom_bootstrap.yaml \\
    --input-dir /path/to/docs \\
    --output-dir /path/to/output \\
    --model-name "my_isne_model"
  
  # Multiple input sources
  python -m src.isne.bootstrap.cli \\
    --input-files /path/to/file1.pdf /path/to/file2.py \\
    --input-dir /path/to/docs \\
    --output-dir /path/to/output
  
  # Debug mode with verbose logging
  python -m src.isne.bootstrap.cli \\
    --input-dir /path/to/docs \\
    --output-dir /path/to/output \\
    --log-level DEBUG \\
    --save-intermediate
        """
    )
    
    # Input sources
    input_group = parser.add_argument_group('Input Sources')
    input_group.add_argument(
        '--input-dir', '-i',
        type=str,
        action='append',
        help='Input directory containing documents (can specify multiple)'
    )
    input_group.add_argument(
        '--input-files', '-f',
        type=str,
        nargs='+',
        help='Specific input files to process'
    )
    input_group.add_argument(
        '--input-files-list',
        type=str,
        help='Path to file containing list of input files (one per line)'
    )
    input_group.add_argument(
        '--recursive', '-r',
        action='store_true',
        default=True,
        help='Recursively search input directories (default: True)'
    )
    input_group.add_argument(
        '--no-recursive',
        action='store_false',
        dest='recursive',
        help='Do not recursively search input directories'
    )
    
    # Output configuration
    output_group = parser.add_argument_group('Output Configuration')
    output_group.add_argument(
        '--output-dir', '-o',
        type=str,
        required=True,
        help='Output directory for model and results'
    )
    output_group.add_argument(
        '--model-name', '-n',
        type=str,
        default='isne_bootstrap_model',
        help='Name for the trained ISNE model (default: isne_bootstrap_model)'
    )
    output_group.add_argument(
        '--save-intermediate',
        action='store_true',
        help='Save intermediate stage results for debugging'
    )
    
    # Configuration
    config_group = parser.add_argument_group('Configuration')
    config_group.add_argument(
        '--config', '-c',
        type=str,
        help='Path to bootstrap configuration YAML file'
    )
    config_group.add_argument(
        '--override',
        type=str,
        action='append',
        help='Override configuration values (format: section.key=value)'
    )
    
    # Processing options
    process_group = parser.add_argument_group('Processing Options')
    process_group.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be processed without actually running'
    )
    process_group.add_argument(
        '--estimate-resources',
        action='store_true',
        help='Estimate resource requirements and exit'
    )
    process_group.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate configuration and inputs, do not run pipeline'
    )
    
    # Logging and monitoring
    logging_group = parser.add_argument_group('Logging and Monitoring')
    logging_group.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    logging_group.add_argument(
        '--log-file',
        type=str,
        help='Path to log file (logs to stdout by default)'
    )
    logging_group.add_argument(
        '--monitor',
        action='store_true',
        help='Enable detailed monitoring and progress tracking'
    )
    
    args = parser.parse_args()
    
    # Set up logging
    log_file = Path(args.log_file) if args.log_file else None
    setup_logging(args.log_level, log_file)
    logger = logging.getLogger(__name__)
    
    try:
        # Collect input files
        input_files = []
        
        if args.input_files:
            input_files.extend([Path(f) for f in args.input_files])
        
        if args.input_files_list:
            list_path = Path(args.input_files_list)
            if list_path.exists():
                with open(list_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):  # Skip empty lines and comments
                            input_files.append(Path(line))
                logger.info(f"Loaded {len(input_files)} files from {args.input_files_list}")
            else:
                logger.error(f"Input files list not found: {args.input_files_list}")
                sys.exit(1)
        
        if args.input_dir:
            for dir_path in args.input_dir:
                dir_files = collect_input_files([dir_path], args.recursive)
                input_files.extend(dir_files)
        
        if not input_files:
            print("Error: No input files specified. Use --input-dir or --input-files.")
            sys.exit(1)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_files = []
        for file_path in input_files:
            if file_path not in seen:
                seen.add(file_path)
                unique_files.append(file_path)
        input_files = unique_files
        
        logger.info(f"Collected {len(input_files)} input files")
        
        # Dry run mode
        if args.dry_run:
            print(f"DRY RUN MODE - Would process {len(input_files)} files:")
            for i, file_path in enumerate(input_files[:10]):  # Show first 10
                print(f"  {i+1:3d}: {file_path}")
            if len(input_files) > 10:
                print(f"  ... and {len(input_files) - 10} more files")
            print(f"Output directory: {args.output_dir}")
            print(f"Model name: {args.model_name}")
            sys.exit(0)
        
        # Load configuration
        if args.config:
            config = BootstrapConfig.from_yaml(args.config)
            logger.info(f"Loaded configuration from {args.config}")
        else:
            config = BootstrapConfig.get_default()
            logger.info("Using default configuration")
        
        # Apply configuration overrides
        if args.override:
            for override in args.override:
                try:
                    key_path, value = override.split('=', 1)
                    sections = key_path.split('.')
                    
                    # Navigate to the correct config section
                    current_config = config
                    for section in sections[:-1]:
                        current_config = getattr(current_config, section)
                    
                    # Set the value (with type conversion)
                    final_key = sections[-1]
                    current_value = getattr(current_config, final_key)
                    
                    if isinstance(current_value, bool):
                        value = value.lower() in ('true', '1', 'yes', 'on')
                    elif isinstance(current_value, int):
                        value = int(value)
                    elif isinstance(current_value, float):
                        value = float(value)
                    
                    setattr(current_config, final_key, value)
                    logger.info(f"Override applied: {key_path} = {value}")
                    
                except Exception as e:
                    logger.warning(f"Failed to apply override '{override}': {e}")
        
        # Apply CLI flags to config
        if args.save_intermediate:
            config.save_intermediate_results = True
        
        # Estimate resources if requested
        if args.estimate_resources:
            logger.info("Estimating resource requirements...")
            
            # Create pipeline instance for estimation
            pipeline = ISNEBootstrapPipeline(config)
            
            duration_estimate = pipeline.estimate_total_duration(len(input_files))
            resource_reqs = pipeline.get_resource_requirements(len(input_files))
            
            print("\nResource Estimation:")
            print("=" * 40)
            print(f"Input files: {len(input_files)}")
            print(f"Estimated duration: {duration_estimate:.0f} seconds "
                  f"({duration_estimate/3600:.1f} hours)")
            print(f"Required RAM: {resource_reqs['recommended_specs']['ram']}")
            print(f"Required storage: {resource_reqs['recommended_specs']['storage']}")
            print(f"GPU memory: {resource_reqs['recommended_specs']['gpu_memory']}")
            print(f"CPU cores: {resource_reqs['cpu_cores']}")
            print(f"Network required: {resource_reqs['network_required']}")
            
            sys.exit(0)
        
        # Validation only mode
        if args.validate_only:
            logger.info("Validation mode - checking configuration and inputs...")
            
            # Create pipeline for validation
            pipeline = ISNEBootstrapPipeline(config)
            
            # Validate each stage configuration
            validation_passed = True
            for stage_name in pipeline.stage_order:
                stage = pipeline.stages[stage_name]
                stage_config = getattr(config, stage_name)
                
                if stage_name == 'document_processing':
                    errors = stage.validate_inputs(input_files, stage_config)
                elif stage_name == 'isne_training':
                    errors = stage.validate_inputs({}, stage_config, Path(args.output_dir))
                else:
                    errors = stage.validate_inputs([], stage_config)
                
                if errors:
                    print(f"❌ {stage_name}: {len(errors)} validation errors:")
                    for error in errors:
                        print(f"   - {error}")
                    validation_passed = False
                else:
                    print(f"✅ {stage_name}: validation passed")
            
            if validation_passed:
                print("\n✅ All validations passed. Configuration is valid.")
                sys.exit(0)
            else:
                print("\n❌ Validation failed. Please fix the errors above.")
                sys.exit(1)
        
        # Initialize monitoring - will be done in pipeline
        monitor = None
        
        # Create and run pipeline
        pipeline = ISNEBootstrapPipeline(config, monitor)
        
        logger.info("Starting ISNE Bootstrap Pipeline...")
        result = pipeline.run(
            input_files=input_files,
            output_dir=Path(args.output_dir),
            model_name=args.model_name
        )
        
        # Report results
        if result.success:
            print("\n🎉 Bootstrap pipeline completed successfully!")
            print(f"📁 Output directory: {result.output_directory}")
            print(f"🤖 Model saved to: {result.model_path}")
            print(f"⏱️  Total time: {result.total_time_seconds:.2f} seconds")
            
            # Show key statistics
            stats = result.final_stats
            if 'data_flow' in stats:
                print("\n📊 Key Statistics:")
                flow = stats['data_flow']
                print(f"   Documents processed: {flow.get('documents_generated', 'N/A')}")
                print(f"   Chunks created: {flow.get('chunks_generated', 'N/A')}")
                print(f"   Embeddings generated: {flow.get('embeddings_generated', 'N/A')}")
                print(f"   Graph nodes: {flow.get('graph_nodes', 'N/A')}")
                print(f"   Graph edges: {flow.get('graph_edges', 'N/A')}")
                print(f"   Model parameters: {flow.get('model_parameters', 'N/A'):,}")
            
            sys.exit(0)
        else:
            print(f"\n❌ Bootstrap pipeline failed: {result.error_message}")
            if result.error_stage:
                print(f"   Failed at stage: {result.error_stage}")
            print(f"   Check logs for details")
            sys.exit(1)
    
    except KeyboardInterrupt:
        logger.info("Bootstrap pipeline interrupted by user")
        print("\n⚠️  Pipeline interrupted by user")
        sys.exit(130)
    
    except Exception as e:
        logger.error(f"Bootstrap pipeline failed with error: {e}", exc_info=True)
        print(f"\n💥 Pipeline failed with error: {e}")
        print("   Check logs for full traceback")
        sys.exit(1)


if __name__ == '__main__':
    main()