#!/usr/bin/env python3
"""
Query Bootstrap Results

Script to check what data was created by the Bootstrap Pipeline.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.database.arango_client import ArangoClient


def main():
    """Check bootstrap results in ArangoDB."""
    
    print("🔍 Checking Bootstrap Pipeline Results")
    print("=" * 50)
    
    # Connect to ArangoDB
    client = ArangoClient()
    
    # Check databases
    print("\n📊 Available Databases:")
    databases = client.list_databases()
    for db in databases:
        print(f"  - {db}")
    
    # Check sequential_isne_testdata database
    target_db = "sequential_isne_testdata"
    if target_db in databases:
        print(f"\n🗂️  Collections in '{target_db}':")
        
        # Connect to target database
        if client.connect_to_database(target_db):
            collections = client.list_collections()
            
            for collection in collections:
                count = client.get_collection_count(collection)
                print(f"  - {collection}: {count:,} documents")
                
                # Show sample document if collection has data
                if count > 0:
                    sample = client.get_sample_document(collection)
                    if sample:
                        print(f"    Sample fields: {list(sample.keys())}")
        else:
            print(f"❌ Failed to connect to database: {target_db}")
    else:
        print(f"❌ Database '{target_db}' not found")
    
    # Check the other database too
    other_db = "hades_sequential_isne"  
    if other_db in databases:
        print(f"\n🗂️  Collections in '{other_db}':")
        
        if client.connect_to_database(other_db):
            collections = client.list_collections()
            
            for collection in collections:
                count = client.get_collection_count(collection)
                print(f"  - {collection}: {count:,} documents")
        else:
            print(f"❌ Failed to connect to database: {other_db}")


if __name__ == "__main__":
    main()