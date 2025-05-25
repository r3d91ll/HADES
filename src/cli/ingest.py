"""
Documentation ingestion CLI for HADES-PathRAG.

This module provides command-line interfaces for ingesting documentation
and other content into HADES-PathRAG knowledge graph.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, cast
import json
from datetime import datetime

from src.pipelines.ingest.orchestrator.ingestor import RepositoryIngestor
from src.storage.arango.connection import ArangoConnection
from src.types.common import StorageConfig, PreProcessorConfig
from src.storage.arango.repository import ArangoRepository

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_documentation_config(db_name: str = "pathrag_docs") -> Dict[str, Any]:
    """
    Create a configuration optimized for documentation ingestion.
    
    Args:
        db_name: Name of the ArangoDB database to use
        
    Returns:
        Configuration dictionary
    """
    # Storage configuration
    storage_config = {
        "host": "localhost",
        "port": 8529,
        "username": "root",
        "password": "",
        "database": db_name,
        "collection_prefix": "docs_",
        "use_vector_index": True,
        "vector_dimensions": 768,
        "embedding": {
            "model_name": "mirth/chonky_modernbert_large_1",
            "model_dimension": 768,
            "batch_size": 8,
            "use_gpu": True,
            "normalize_embeddings": True,
            "use_vllm": True,
            "vllm_server_url": "http://localhost:8000"
        }
    }
    
    # Preprocessor configuration
    preprocessor_config = {
        "exclude_patterns": [
            r"\.git/.*",
            r"__pycache__/.*", 
            r".*\.pyc$",
            r"node_modules/.*",
            r".*\.swp$",
            r".*\.swo$",
            r"\.DS_Store$"
        ],
        "max_workers": 4,
        "batch_size": 10,
        "file_type_map": {
            "markdown": [".md", ".mdx", ".markdown"],
            "text": [".txt", ".text"],
            "pdf": [".pdf"],
            "html": [".html", ".htm"],
            "xml": [".xml"],
            "json": [".json"],
            "yaml": [".yml", ".yaml"],
            "rst": [".rst"]
        }
    }
    
    return {
        "storage": storage_config,
        "preprocessor": preprocessor_config
    }


def ingest_documentation(docs_dir: Path, dataset_name: Optional[str] = None, 
                    db_name: str = "pathrag_docs", db_mode: str = "append",
                    force: bool = False) -> Dict[str, Any]:
    """
    Ingest documentation from a directory.
    
    Args:
        docs_dir: Directory containing documentation
        dataset_name: Optional name for the dataset
        db_name: Name of ArangoDB database to use
        db_mode: Database mode ('create' or 'append')
        force: Force recreation of collections even if they exist
        
    Returns:
        Ingestion results and statistics
    """
    # Get configuration
    config = create_documentation_config(db_name=db_name)
    
    # Convert configurations to expected format for RepositoryIngestor
    from src.pipelines.ingest.orchestrator.config import IngestionConfig
    
    # Create IngestionConfig object from our configuration dictionary
    ingest_config = IngestionConfig()
    # Apply storage config
    ingest_config.storage.update(config["storage"])
    # Apply preprocessor config
    ingest_config.preprocessor.update(config["preprocessor"])
    # Set initialize_db based on db_mode
    ingest_config.initialize_db = (db_mode == "create")
    
    logger.info(f"Initializing ingestor with db_mode: {db_mode} (initialize_db={ingest_config.initialize_db})")
    
    # Create repository ingestor with the proper config
    ingestor = RepositoryIngestor(
        config=ingest_config,
    )
    
    # Handle database mode
    repository = ingestor.repository
    if db_mode == "create" and repository:
        # Cast to ArangoRepository to access its specific attributes
        arango_repo = cast(ArangoRepository, repository)
        
        # If force is True, first drop collections if they exist
        if force:
            try:
                conn = arango_repo.connection
                graph_name = arango_repo.graph_name
                node_col = arango_repo.node_collection_name
                edge_col = arango_repo.edge_collection_name
                
                # Delete graph first (important)
                if conn.graph_exists(graph_name):
                    logger.info(f"Dropping existing graph {graph_name}")
                    conn.delete_graph(graph_name, drop_collections=True)
                
                # Delete collections as fallback
                if conn.collection_exists(node_col):
                    logger.info(f"Dropping existing node collection {node_col}")
                    conn.delete_collection(node_col)
                    
                if conn.collection_exists(edge_col):
                    logger.info(f"Dropping existing edge collection {edge_col}")
                    conn.delete_collection(edge_col)
            except Exception as e:
                logger.error(f"Error during forced collection cleanup: {e}")
        
        # Explicitly setup collections
        logger.info("Setting up collections explicitly in 'create' mode")
        arango_repo.setup_collections()
    
    # Ingest documentation
    logger.info(f"Starting documentation ingestion from {docs_dir}")
    results = ingestor.ingest(docs_dir, dataset_name)
    
    # Log results
    logger.info(f"Documentation ingestion completed. Stats: {results}")
    
    # Return results with repository reference for querying
    return {
        "results": results,
        "repository": ingestor.get_repository()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest documentation into HADES-PathRAG")
    parser.add_argument("--docs-dir", type=str, required=True, help="Directory containing documentation files")
    parser.add_argument("--dataset-name", type=str, help="Optional name for the dataset")
    parser.add_argument("--db-name", type=str, default="pathrag_docs", help="Name of the ArangoDB database to use")
    parser.add_argument("--db-mode", type=str, choices=["create", "append"], default="append", 
                      help="Database mode: 'create' initializes collections, 'append' adds to existing ones")
    parser.add_argument("--force", action="store_true", help="Force recreation of collections even if they exist")
    parser.add_argument("--output-file", type=str, help="Optional file to save ingestion statistics")
    args = parser.parse_args()
    
    # Convert to Path
    docs_dir = Path(args.docs_dir)
    
    # Verify documentation directory exists
    if not docs_dir.exists() or not docs_dir.is_dir():
        logger.error(f"Documentation directory does not exist: {docs_dir}")
        return 1
    
    # Check for supported files
    supported_extensions = [
        ".md", ".mdx", ".markdown", ".txt", ".text", ".pdf", 
        ".html", ".htm", ".xml", ".json", ".yml", ".yaml", ".rst"
    ]
    
    has_supported_files = any(
        any(f.suffix.lower() in supported_extensions for f in docs_dir.glob(f"**/*{ext}"))
        for ext in supported_extensions
    )
    
    if not has_supported_files:
        logger.warning(f"No supported documentation files found in {docs_dir}")
        return 1
    
    # Ingest documentation with specified database settings
    result = ingest_documentation(
        docs_dir, 
        dataset_name=args.dataset_name,
        db_name=args.db_name,
        db_mode=args.db_mode,
        force=args.force
    )
    
    # Save statistics if output file is specified
    if args.output_file:
        # Convert to serializable format
        serializable_results = {
            "stats": result.get("results", {}),
            "timestamp": datetime.now().isoformat(),
            "config": {
                "database": args.db_name,
                "mode": args.db_mode,
                "force": args.force,
                "docs_dir": str(docs_dir),
                "dataset_name": args.dataset_name
            }
        }
        
        with open(args.output_file, "w") as f:
            json.dump(serializable_results, f, indent=2)
            logger.info(f"Saved ingestion statistics to {args.output_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
