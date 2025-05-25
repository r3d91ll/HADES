"""
Query interface CLI for HADES-PathRAG.

This module provides command-line tools for querying the knowledge graph.
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, cast
import json
from datetime import datetime

from src.storage.arango.connection import ArangoConnection
from src.ingest.repository.arango_repository import ArangoRepository
from src.isne.integrations.pathrag_connector import PathRAGConnector
from src.isne.pipeline import ISNEPipeline, PipelineConfig
from src.isne.types.models import EmbeddingConfig
from src.types.common import StorageConfig, PathRankingConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def query_documentation(query: str, db_name: str = "pathrag_docs", 
                       top_k: int = 5, collection_name: Optional[str] = None,
                       embedding_model: str = "mirth/chonky_modernbert_large_1") -> Any:
    """
    Query documentation in the knowledge graph.
    
    Args:
        query: Natural language query
        db_name: Name of the ArangoDB database
        top_k: Number of results to return
        collection_name: Optional specific collection to query
        embedding_model: Model name for generating query embeddings
        
    Returns:
        Query results with document matches and paths
    """
    # Create storage config
    storage_config = StorageConfig({
        "host": "http://localhost:8529",
        "username": "root",
        "password": "",
        "database": db_name,
        "embedding": {
            "model_name": embedding_model,
            "model_dimension": 768,
            "use_gpu": True,
            "normalize_embeddings": True
        }
    })
    
    # Connect to database
    connection = ArangoConnection.bootstrap(config=storage_config)
    
    # Use collection name with prefix if specified
    collection_prefix = "" # Change if needed
    node_collection = f"{collection_prefix}nodes" if collection_prefix else "nodes"
    edge_collection = f"{collection_prefix}edges" if collection_prefix else "edges"
    graph_name = f"{collection_prefix}pathrag" if collection_prefix else "pathrag"
    
    # Override if explicit collection provided
    if collection_name:
        node_collection = collection_name
    
    # Create repository and connector
    repository = ArangoRepository(
        connection=connection,
        node_collection=node_collection,
        edge_collection=edge_collection,
        graph_name=graph_name
    )
    
    # Initialize PathRAG connector
    connector = PathRAGConnector(repository=repository)
    
    # Create embedding config
    embedding_cfg = EmbeddingConfig(
        model_name=embedding_model,
        model_dimension=768,
        use_gpu=True,
        normalize_embeddings=True
    )
    
    # Create pipeline config
    pipeline_config = PipelineConfig(
        embedding_config=embedding_cfg,
        use_gpu=True
    )
    
    # Set up path ranking config
    path_ranking = PathRankingConfig({
        "semantic_weight": 0.7,
        "path_length_weight": 0.2,
        "edge_strength_weight": 0.1,
        "max_path_length": 3,
        "max_paths": top_k
    })
    
    # We'll use the connector directly since pipeline doesn't have a search method
    logger.info(f"Querying with: '{query}'")
    results = connector.rank_paths(
        query=query,
        document_limit=top_k,
        path_limit=top_k,
        filters=None
    )
    
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Query the HADES-PathRAG knowledge graph")
    parser.add_argument("--query", type=str, required=True, help="Natural language query")
    parser.add_argument("--db-name", type=str, default="pathrag_docs", help="Name of the ArangoDB database")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    parser.add_argument("--collection", type=str, help="Optional specific collection to query")
    parser.add_argument("--model", type=str, default="mirth/chonky_modernbert_large_1", 
                        help="Model for embedding generation")
    parser.add_argument("--output-file", type=str, help="Optional file to save query results")
    args = parser.parse_args()
    
    try:
        # Query documentation
        results = query_documentation(
            args.query,
            db_name=args.db_name,
            top_k=args.top_k,
            collection_name=args.collection,
            embedding_model=args.model
        )
        
        # Process results
        print("\n=== Query Results ===")
        print(f"Query: '{args.query}'")
        print(f"Database: {args.db_name}")
        print(f"Found {len(results.documents)} result(s)")
        
        # Display results
        for i, doc in enumerate(results.documents):
            print(f"\n--- Result {i+1} ---")
            print(f"Title: {doc.title if hasattr(doc, 'title') and doc.title else 'Untitled'}")
            print(f"ID: {doc.document_id}")
            print(f"Score: {doc.score:.4f}")
            
            # Show content snippet (truncated if too long)
            content = doc.content if hasattr(doc, 'content') else "No content"
            if len(content) > 300:
                content = content[:300] + "..."
            print(f"Content: {content}")
            
            # Show paths if available
            if hasattr(results, 'paths') and results.paths:
                paths_for_doc = [p for p in results.paths if p.target_id == doc.document_id]
                if paths_for_doc:
                    print("\nPaths:")
                    for j, path in enumerate(paths_for_doc[:3]):  # Limit to 3 paths per doc
                        print(f"  Path {j+1}: {' -> '.join([n.title for n in path.nodes if hasattr(n, 'title')])}")
        
        # Save to file if requested
        if args.output_file:
            # Convert to serializable format
            serializable = {
                "query": args.query,
                "timestamp": datetime.now().isoformat(),
                "results": [
                    {
                        "id": doc.document_id,
                        "title": doc.title if hasattr(doc, 'title') else "Untitled",
                        "score": doc.score,
                        "content": doc.content if hasattr(doc, 'content') else "No content"
                    }
                    for doc in results.documents
                ]
            }
            
            # Add paths if available
            if hasattr(results, 'paths') and results.paths:
                serializable["paths"] = [
                    {
                        "target_id": path.target_id,
                        "score": path.score,
                        "nodes": [
                            {"id": n.id, "title": n.title if hasattr(n, 'title') else "Untitled"}
                            for n in path.nodes
                        ]
                    }
                    for path in results.paths
                ]
            
            with open(args.output_file, 'w') as f:
                json.dump(serializable, f, indent=2)
                print(f"\nSaved results to {args.output_file}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during query: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
