#!/usr/bin/env python3
"""
Script to inspect ArangoDB databases for ISNE training data structures.

This script examines the available databases and their collections to understand
what data is available for ISNE model training.
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from database.arango_client import ArangoClient


def inspect_database(client: ArangoClient, db_name: str) -> dict:
    """Inspect a database and return its structure."""
    print(f"\n{'='*60}")
    print(f"INSPECTING DATABASE: {db_name}")
    print(f"{'='*60}")
    
    # Connect to the database
    if not client.connect_to_database(db_name):
        return {"error": f"Failed to connect to database {db_name}"}
    
    # Get database info
    db_info = client.get_database_info()
    if 'error' in db_info:
        return db_info
    
    print(f"Database: {db_info['database_name']}")
    print(f"Total documents: {db_info['total_documents']}")
    print(f"Total collections: {len(db_info['collections'])}")
    
    # Examine each collection
    collection_details = {}
    for collection_info in db_info['collections']:
        collection_name = collection_info['name']
        collection_type = collection_info['type']
        collection_count = collection_info['count']
        
        print(f"\n  Collection: {collection_name}")
        print(f"    Type: {'Edge' if collection_type == 3 else 'Document'}")
        print(f"    Count: {collection_count}")
        
        # Get a sample document to understand structure
        sample_doc = client.get_sample_document(collection_name)
        if sample_doc:
            print(f"    Sample document structure:")
            # Show only the keys and types, not full values
            for key, value in sample_doc.items():
                value_type = type(value).__name__
                if isinstance(value, list) and value:
                    value_type = f"list[{type(value[0]).__name__}]"
                elif isinstance(value, dict):
                    value_type = f"dict with keys: {list(value.keys())[:5]}"
                print(f"      {key}: {value_type}")
        else:
            print(f"    No sample document available")
        
        collection_details[collection_name] = {
            "type": "edge" if collection_type == 3 else "document",
            "count": collection_count,
            "sample_structure": sample_doc
        }
    
    return {
        "database_name": db_name,
        "total_documents": db_info['total_documents'],
        "total_collections": len(db_info['collections']),
        "collections": collection_details
    }


def analyze_isne_readiness(db_structures: dict) -> dict:
    """Analyze if the databases have sufficient structure for ISNE training."""
    print(f"\n{'='*60}")
    print("ISNE TRAINING READINESS ANALYSIS")
    print(f"{'='*60}")
    
    analysis = {
        "has_nodes": False,
        "has_edges": False,
        "has_embeddings": False,
        "has_features": False,
        "recommendations": []
    }
    
    # Look for common ISNE training data patterns
    for db_name, db_info in db_structures.items():
        if "error" in db_info:
            continue
            
        print(f"\nDatabase: {db_name}")
        collections = db_info.get("collections", {})
        
        # Check for node-like collections
        node_collections = []
        edge_collections = []
        embedding_collections = []
        
        for coll_name, coll_info in collections.items():
            if coll_info["type"] == "edge":
                edge_collections.append(coll_name)
                print(f"  ✓ Found edge collection: {coll_name} ({coll_info['count']} edges)")
            else:
                # Check if this looks like a node collection
                sample = coll_info.get("sample_structure", {})
                if sample:
                    has_id = any(key in sample for key in ['_id', '_key', 'id', 'node_id'])
                    has_content = any(key in sample for key in ['content', 'text', 'chunk', 'document'])
                    has_embedding = any(key in sample for key in ['embedding', 'vector', 'embeddings'])
                    
                    if has_id and (has_content or has_embedding):
                        node_collections.append(coll_name)
                        print(f"  ✓ Found node-like collection: {coll_name} ({coll_info['count']} documents)")
                        
                        if has_embedding:
                            embedding_collections.append(coll_name)
                            print(f"    - Contains embeddings")
        
        # Update analysis
        if node_collections:
            analysis["has_nodes"] = True
        if edge_collections:
            analysis["has_edges"] = True
        if embedding_collections:
            analysis["has_embeddings"] = True
    
    # Generate recommendations
    if not analysis["has_nodes"]:
        analysis["recommendations"].append("No node collections found. Need document/chunk collections with IDs.")
    
    if not analysis["has_edges"]:
        analysis["recommendations"].append("No edge collections found. Need relationship data between nodes.")
    
    if not analysis["has_embeddings"]:
        analysis["recommendations"].append("No embedding data found. Need pre-computed embeddings for nodes.")
    else:
        analysis["recommendations"].append("Found existing embeddings - can be used as initial features for ISNE training.")
    
    # Overall assessment
    if analysis["has_nodes"] and analysis["has_edges"]:
        print(f"\n✓ READY FOR ISNE TRAINING")
        print("  - Have node and edge data")
        if analysis["has_embeddings"]:
            print("  - Have existing embeddings for enhanced training")
    else:
        print(f"\n⚠ NOT READY FOR ISNE TRAINING")
        for rec in analysis["recommendations"]:
            print(f"  - {rec}")
    
    return analysis


def main():
    """Main inspection function."""
    # Database names to inspect
    databases = [
        "sequential_isne_performance_test",
        "sequential_isne_testdata", 
        "test_embedding_pipeline"
    ]
    
    # Initialize client
    client = ArangoClient(
        host="127.0.0.1",
        port=8529,
        username="root",
        password="",
        database="hades"  # Default, will be changed
    )
    
    # First, list all available databases
    print("Available databases:")
    all_dbs = client.list_databases()
    for db in all_dbs:
        marker = "✓" if db in databases else " "
        print(f"  {marker} {db}")
    
    # Inspect each target database
    db_structures = {}
    for db_name in databases:
        if db_name in all_dbs:
            db_structures[db_name] = inspect_database(client, db_name)
        else:
            print(f"\n❌ Database '{db_name}' not found!")
            db_structures[db_name] = {"error": "Database not found"}
    
    # Analyze ISNE readiness
    analysis = analyze_isne_readiness(db_structures)
    
    # Save detailed results
    output_file = Path(__file__).parent / "arango_inspection_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            "inspection_results": db_structures,
            "isne_analysis": analysis
        }, f, indent=2, default=str)
    
    print(f"\n📄 Detailed results saved to: {output_file}")


if __name__ == "__main__":
    main()