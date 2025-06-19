#!/usr/bin/env python3
"""
Bootstrap Pipeline CLI Script

Command-line interface for running the Sequential-ISNE Bootstrap Pipeline.
"""

import sys
import argparse
import json
import yaml
from pathlib import Path
from typing import Dict, Any

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.pipelines.bootstrap import BootstrapPipeline, BootstrapConfig
from src.pipelines.bootstrap.config import FileTypeFilter


def load_config_file(config_path: Path) -> Dict[str, Any]:
    """Load configuration from YAML or JSON file."""
    try:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            return yaml.safe_load(config_path.read_text())
        elif config_path.suffix.lower() == '.json':
            return json.loads(config_path.read_text())
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")
    except Exception as e:
        print(f"Error loading config file {config_path}: {e}")
        sys.exit(1)


def create_sample_config(output_path: Path) -> None:
    """Create a sample configuration file."""
    sample_config = {
        "input_directory": "/path/to/your/project",
        "output_database": "sequential_isne_bootstrap",
        "file_type_filter": "all",
        "max_file_size": 10485760,  # 10MB
        "exclude_patterns": [
            "*.pyc", "*.pyo", "*.pyd", "__pycache__", ".git", ".svn",
            "node_modules", ".venv", "venv", "*.log", "*.tmp"
        ],
        "enable_directory_bootstrap": True,
        "enable_import_analysis": True,
        "enable_semantic_similarity": True,
        "enable_cross_modal_discovery": True,
        "semantic_similarity_threshold": 0.7,
        "co_location_weight": 0.8,
        "import_weight": 0.9,
        "directory_hierarchy_weight": 0.6,
        "batch_size": 100,
        "max_workers": 4,
        "chunk_size": 512,
        "chunk_overlap": 50,
        "storage_component": "arangodb_v2",
        "enable_parallel_processing": True,
        "log_level": "INFO"
    }
    
    try:
        if output_path.suffix.lower() in ['.yaml', '.yml']:
            output_path.write_text(yaml.dump(sample_config, default_flow_style=False, indent=2))
        else:
            output_path.write_text(json.dumps(sample_config, indent=2))
        
        print(f"Sample configuration written to: {output_path}")
    except Exception as e:
        print(f"Error writing sample config: {e}")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Sequential-ISNE Bootstrap Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic bootstrap from directory
  python run_bootstrap_pipeline.py /path/to/project

  # Bootstrap with custom database name
  python run_bootstrap_pipeline.py /path/to/project --database my_graph

  # Bootstrap with configuration file
  python run_bootstrap_pipeline.py --config bootstrap_config.yaml

  # Perform dry run analysis
  python run_bootstrap_pipeline.py /path/to/project --dry-run

  # Generate sample configuration
  python run_bootstrap_pipeline.py --sample-config bootstrap_config.yaml

  # Bootstrap only code files
  python run_bootstrap_pipeline.py /path/to/project --file-filter code_only

  # Bootstrap with parallel processing disabled
  python run_bootstrap_pipeline.py /path/to/project --no-parallel
        """
    )
    
    # Main arguments
    parser.add_argument(
        "input_directory",
        nargs="?",
        type=Path,
        help="Directory to bootstrap from"
    )
    
    # Configuration options
    parser.add_argument(
        "--config", "-c",
        type=Path,
        help="Configuration file (YAML or JSON)"
    )
    
    parser.add_argument(
        "--database", "-d",
        type=str,
        default="sequential_isne_bootstrap",
        help="Output database name (default: sequential_isne_bootstrap)"
    )
    
    parser.add_argument(
        "--file-filter",
        type=str,
        choices=["all", "code_only", "docs_only", "config_only", "code_and_docs"],
        default="all",
        help="Filter files by type (default: all)"
    )
    
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum worker threads (default: 4)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for processing (default: 100)"
    )
    
    # Feature toggles
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel processing"
    )
    
    parser.add_argument(
        "--no-imports",
        action="store_true",
        help="Disable import analysis"
    )
    
    parser.add_argument(
        "--no-cross-modal",
        action="store_true",
        help="Disable cross-modal discovery"
    )
    
    # Operation modes
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform analysis without building graph"
    )
    
    parser.add_argument(
        "--sample-config",
        type=Path,
        help="Generate sample configuration file and exit"
    )
    
    # Logging
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Handle sample config generation
    if args.sample_config:
        create_sample_config(args.sample_config)
        return
    
    # Validate required arguments
    if not args.input_directory and not args.config:
        parser.error("Must provide either input_directory or --config")
    
    # Setup logging
    import logging
    log_level = logging.DEBUG if args.verbose else getattr(logging, args.log_level)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Load configuration
        if args.config:
            print(f"Loading configuration from: {args.config}")
            config_dict = load_config_file(args.config)
            
            # Override with command line arguments
            if args.input_directory:
                config_dict["input_directory"] = args.input_directory
            if not config_dict.get("input_directory"):
                parser.error("input_directory must be specified in config file or command line")
        else:
            # Create config from command line arguments
            config_dict = {
                "input_directory": args.input_directory,
                "output_database": args.database,
                "file_type_filter": args.file_filter,
                "max_workers": args.max_workers,
                "batch_size": args.batch_size,
                "enable_parallel_processing": not args.no_parallel,
                "enable_import_analysis": not args.no_imports,
                "enable_cross_modal_discovery": not args.no_cross_modal,
                "log_level": args.log_level
            }
        
        # Create configuration object
        config = BootstrapConfig(**config_dict)
        
        # Validate input directory
        if not config.input_directory.exists():
            print(f"Error: Input directory does not exist: {config.input_directory}")
            sys.exit(1)
        
        # Create and run pipeline
        pipeline = BootstrapPipeline(config)
        
        if args.dry_run:
            print("Performing dry run analysis...")
            result = pipeline.dry_run()
            
            print("\n" + "="*60)
            print("DRY RUN ANALYSIS RESULTS")
            print("="*60)
            
            if "error" in result:
                print(f"Error: {result['error']}")
                sys.exit(1)
            
            analysis = result["analysis"]
            print(f"Total files: {analysis['total_files']}")
            print(f"File types: {analysis['file_types']}")
            print(f"Total size: {analysis['total_size_bytes']:,} bytes")
            print(f"Total relationships: {analysis['total_relationships']}")
            print(f"Edge types: {analysis['edge_types']}")
            print(f"Cross-modal edges: {analysis['cross_modal_edges']}")
            print(f"Estimated processing time: {result['estimated_processing_time']:.1f} seconds")
            
            if result["recommendations"]:
                print("\nRecommendations:")
                for rec in result["recommendations"]:
                    print(f"  - {rec}")
        
        else:
            print(f"Starting Bootstrap Pipeline for: {config.input_directory}")
            print(f"Target database: {config.output_database}")
            print(f"File filter: {config.file_type_filter}")
            print(f"Parallel processing: {'enabled' if config.enable_parallel_processing else 'disabled'}")
            print("")
            
            # Execute pipeline
            result = pipeline.execute()
            
            print("\n" + "="*60)
            print("BOOTSTRAP PIPELINE RESULTS")
            print("="*60)
            
            if result.success:
                print("✅ Pipeline completed successfully!")
                print(f"Database: {result.database_name}")
                print(f"Nodes created: {result.node_count}")
                print(f"Edges created: {result.edge_count}")
                print(f"Files processed: {result.metrics.total_files_processed}")
                print(f"Processing time: {result.metrics.total_processing_time:.2f}s")
                
                if result.metrics.files_by_type:
                    print(f"Files by type: {result.metrics.files_by_type}")
                
                if result.metrics.edges_by_type:
                    print(f"Edges by type: {result.metrics.edges_by_type}")
                
                if result.metrics.cross_modal_edges > 0:
                    print(f"Cross-modal edges: {result.metrics.cross_modal_edges}")
                
                print(f"Graph density: {result.metrics.graph_density:.4f}")
                
                if result.warnings:
                    print("\nWarnings:")
                    for warning in result.warnings:
                        print(f"  ⚠️  {warning}")
            else:
                print("❌ Pipeline failed!")
                if result.errors:
                    print("Errors:")
                    for error in result.errors:
                        print(f"  - {error}")
                sys.exit(1)
    
    except Exception as e:
        print(f"Fatal error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()