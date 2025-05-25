"""
Administrative CLI tools for HADES-PathRAG.

This module provides utilities for managing and administrating
the ArangoDB database and collections.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, cast
import json
from datetime import datetime
from pprint import pprint

from src.storage.arango.connection import ArangoConnection, get_client
from src.types.common import StorageConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def setup_graph(db_name: str = "hades", 
                node_collection: str = "nodes", 
                edge_collection: str = "edges",
                graph_name: str = "pathrag",
                force: bool = False,
                username: str = "root",
                password: str = "") -> Dict[str, Any]:
    """
    Set up standard ArangoDB graph structure for HADES-PathRAG.
    
    Args:
        db_name: Name of the database
        node_collection: Name for node collection
        edge_collection: Name for edge collection
        graph_name: Name for the graph
        force: Force recreation of collections even if they exist
        username: ArangoDB username
        password: ArangoDB password
        
    Returns:
        Information about created structure
    """
    # Create storage config
    config = StorageConfig({
        "database": db_name,
        "host": "http://localhost:8529",
        "username": username,
        "password": password,
    })
    
    # Bootstrap connection with collection setup
    connection = ArangoConnection.bootstrap(
        config=config,
        node_collection=node_collection,
        edge_collection=edge_collection,
        graph_name=graph_name,
        force=force
    )
    
    return {
        "database": db_name,
        "graph": graph_name,
        "collections": {
            "node": node_collection,
            "edge": edge_collection
        },
        "status": "created" if force else "verified"
    }


def reset_database(db_name: str, username: str = "root", password: str = "") -> bool:
    """
    Reset (delete) a database completely.
    
    Args:
        db_name: Database name to reset
        username: ArangoDB username
        password: ArangoDB password
        
    Returns:
        Success status
    """
    logger.warning(f"Attempting to delete database: {db_name}")
    
    try:
        # Connect to _system database
        client = get_client(hosts="http://localhost:8529")
        sys_db = client.db("_system", username=username, password=password)
        
        # Check if database exists
        if sys_db.has_database(db_name):
            # Delete the database
            sys_db.delete_database(db_name)
            logger.info(f"Successfully deleted database: {db_name}")
            return True
        else:
            logger.warning(f"Database does not exist: {db_name}")
            return False
            
    except Exception as e:
        logger.error(f"Error deleting database: {e}")
        return False


def verify_database(db_name: str, username: str = "root", password: str = "") -> Dict[str, Any]:
    """
    Verify and display ArangoDB database contents.
    
    Args:
        db_name: Database name to verify
        username: ArangoDB username
        password: ArangoDB password
        
    Returns:
        Database info dictionary
    """
    # Create storage config
    config = StorageConfig({
        "database": db_name,
        "host": "http://localhost:8529",
        "username": username,
        "password": password,
    })
    
    # Connect with bootstrap (creates DB and collections if needed)
    conn = ArangoConnection.bootstrap(config=config)
    
    # Gather database info
    db_info: Dict[str, Any] = {}
    
    # List collections
    collections = cast(List[Dict[str, Any]], conn.raw_db.collections())
    
    # Filter out system collections
    regular_collections = [c for c in collections if not c['name'].startswith('_')]
    
    # Get collection info
    db_info["collections"] = {}
    for collection in regular_collections:
        coll_name = collection["name"]
        coll = conn.raw_db.collection(coll_name)
        is_edge = hasattr(coll, 'is_edge') and coll.is_edge()
        
        db_info["collections"][coll_name] = {
            "type": "edge" if is_edge else "document",
            "count": coll.count(),
            "properties": collection.get("properties", {})
        }
    
    # List graphs
    graphs = cast(List[Dict[str, Any]], conn.raw_db.graphs())
    db_info["graphs"] = {}
    
    for graph in graphs:
        graph_name = graph["name"]
        
        # Get edge definitions
        edge_defs = graph.get("edge_definitions", [])
        
        db_info["graphs"][graph_name] = {
            "edge_collections": [e["edge_collection"] for e in edge_defs],
            "vertex_collections": list(set(
                v for e in edge_defs 
                for v_list in [e.get("from_vertex_collections", []), e.get("to_vertex_collections", [])]
                for v in v_list
            ))
        }
    
    return db_info


def main_setup_graph() -> int:
    parser = argparse.ArgumentParser(description="Set up ArangoDB graph structure for HADES-PathRAG")
    parser.add_argument("--db-name", type=str, default="hades", help="Database name")
    parser.add_argument("--node-collection", type=str, default="nodes", help="Node collection name")
    parser.add_argument("--edge-collection", type=str, default="edges", help="Edge collection name")
    parser.add_argument("--graph-name", type=str, default="pathrag", help="Graph name")
    parser.add_argument("--force", action="store_true", help="Force recreation even if exists")
    parser.add_argument("--username", type=str, default="root", help="ArangoDB username")
    parser.add_argument("--password", type=str, default="", help="ArangoDB password")
    args = parser.parse_args()
    
    result = setup_graph(
        db_name=args.db_name,
        node_collection=args.node_collection,
        edge_collection=args.edge_collection,
        graph_name=args.graph_name,
        force=args.force,
        username=args.username,
        password=args.password
    )
    
    print("\n=== Graph Setup Results ===")
    pprint(result)
    return 0


def main_reset_db() -> int:
    parser = argparse.ArgumentParser(description="Reset (delete) an ArangoDB database")
    parser.add_argument("--db-name", type=str, required=True, help="Database name to reset")
    parser.add_argument("--username", type=str, default="root", help="ArangoDB username")
    parser.add_argument("--password", type=str, default="", help="ArangoDB password")
    parser.add_argument("--confirm", action="store_true", 
                      help="Confirm deletion (required to prevent accidental deletion)")
    args = parser.parse_args()
    
    if not args.confirm:
        print(f"ERROR: Please use --confirm to confirm deletion of database '{args.db_name}'")
        return 1
    
    success = reset_database(
        db_name=args.db_name,
        username=args.username,
        password=args.password
    )
    
    if success:
        print(f"Successfully reset database: {args.db_name}")
        return 0
    else:
        print(f"Failed to reset database: {args.db_name}")
        return 1


def main_verify_db() -> int:
    parser = argparse.ArgumentParser(description="Verify ArangoDB database contents")
    parser.add_argument("--db-name", type=str, default="hades", help="Database name to verify")
    parser.add_argument("--username", type=str, default="root", help="ArangoDB username")
    parser.add_argument("--password", type=str, default="", help="ArangoDB password")
    parser.add_argument("--output-file", type=str, help="Optional file to save database info")
    args = parser.parse_args()
    
    db_info = verify_database(
        db_name=args.db_name,
        username=args.username,
        password=args.password
    )
    
    print(f"\n=== Database '{args.db_name}' Info ===")
    
    # Print collection info
    print("\n== Collections ==")
    for name, info in db_info["collections"].items():
        print(f" - {name} ({info['type']})")
        print(f"   Documents: {info['count']}")
    
    # Print graph info
    print("\n== Graphs ==")
    for name, info in db_info["graphs"].items():
        print(f" - {name}")
        print(f"   Edge Collections: {', '.join(info['edge_collections']) if info['edge_collections'] else 'None'}")
        print(f"   Vertex Collections: {', '.join(info['vertex_collections']) if info['vertex_collections'] else 'None'}")
    
    # Save to file if requested
    if args.output_file:
        with open(args.output_file, "w") as f:
            json.dump({
                "database": args.db_name,
                "timestamp": datetime.now().isoformat(),
                "info": db_info
            }, f, indent=2)
        print(f"\nSaved database info to {args.output_file}")
    
    return 0


# Entry points for each tool
if __name__ == "__main__":
    print("This module provides multiple administrative tools.")
    print("Please use one of the following command-line scripts:")
    print("  - pathrag-setup-graph")
    print("  - pathrag-reset-db")
    print("  - pathrag-verify-db")
    sys.exit(1)
