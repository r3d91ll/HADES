#!/usr/bin/env python
"""
Example script demonstrating the use of the ISNE pipeline integration
for repository ingestion and code similarity search.
"""
import sys
import os
from pathlib import Path
import logging
from datetime import datetime
import argparse

# Add project root to path to allow importing from src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingest.ingestor import RepositoryIngestor
from src.config.settings import load_config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process a code repository using the ISNE pipeline"
    )
    parser.add_argument(
        "--repo_path", 
        type=str, 
        required=True,
        help="Path to the repository to process"
    )
    parser.add_argument(
        "--repo_name", 
        type=str, 
        help="Name for the repository (defaults to directory name)"
    )
    parser.add_argument(
        "--config_path", 
        type=str, 
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--db_host", 
        type=str, 
        default="localhost",
        help="ArangoDB host"
    )
    parser.add_argument(
        "--db_port", 
        type=int, 
        default=8529,
        help="ArangoDB port"
    )
    parser.add_argument(
        "--db_name", 
        type=str, 
        default="pathrag",
        help="ArangoDB database name"
    )
    parser.add_argument(
        "--db_user", 
        type=str, 
        default="root",
        help="ArangoDB username"
    )
    parser.add_argument(
        "--db_password", 
        type=str, 
        default="",
        help="ArangoDB password"
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the script."""
    start_time = datetime.now()
    args = parse_args()
    
    logger.info(f"Starting ISNE repository processing at {start_time}")
    
    # Load configuration
    try:
        config = load_config(args.config_path)
        logger.info(f"Loaded configuration from {args.config_path}")
    except Exception as e:
        logger.warning(f"Failed to load configuration: {e}. Using defaults.")
        config = {}
    
    # Initialize the repository ingestor
    ingestor = RepositoryIngestor(
        database=args.db_name,
        host=args.db_host,
        port=args.db_port,
        username=args.db_user,
        password=args.db_password
    )
    
    logger.info("Setting up database collections...")
    ingestor.setup_collections()
    
    # Process the repository with ISNE
    repo_path = Path(args.repo_path)
    if not repo_path.exists():
        logger.error(f"Repository path {repo_path} does not exist")
        return 1
    
    logger.info(f"Processing repository at {repo_path}")
    try:
        stats = ingestor.process_repository_with_isne(
            repo_path=repo_path,
            repo_name=args.repo_name
        )
        
        logger.info("Repository processing completed successfully")
        logger.info(f"Stats: {stats}")
        
        # Example for finding similar code
        if stats.get("document_count", 0) > 0:
            logger.info("Finding similar code example:")
            # Get a sample code file
            code_files = list(repo_path.glob("**/*.py"))
            if code_files:
                sample_file = code_files[0]
                logger.info(f"Using {sample_file} as sample code")
                with open(sample_file, "r") as f:
                    sample_code = f.read()
                
                similar_results = ingestor.find_similar_code(
                    code_content=sample_code,
                    limit=5
                )
                
                logger.info(f"Found {len(similar_results)} similar code snippets")
                for i, result in enumerate(similar_results):
                    logger.info(f"Result {i+1}:")
                    logger.info(f"  ID: {result['id']}")
                    logger.info(f"  Title: {result['title']}")
                    logger.info(f"  Score: {result['score']}")
    
    except Exception as e:
        logger.error(f"Error processing repository: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    end_time = datetime.now()
    duration = end_time - start_time
    logger.info(f"Completed in {duration}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
