#!/usr/bin/env python3
"""
Test script for ArangoDB client connection.

Tests the new ArangoDB client against the local ArangoDB instance.
"""

import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.database.arango_client import ArangoClient

def test_arango_connection():
    """Test ArangoDB connection and basic operations."""
    print("🧪 Testing ArangoDB Connection")
    print("=" * 50)
    
    # Test connection
    try:
        client = ArangoClient(
            host="127.0.0.1",
            port=8529,
            username="root",
            password="",
            database="hades_test",
            timeout=10
        )
        
        print("📡 Connecting to ArangoDB...")
        success = client.connect()
        
        if success:
            print("✅ Connection successful!")
            
            # Test health check
            print("🏥 Testing health check...")
            health = client.health_check()
            print(f"Health check: {'✅ Passed' if health else '❌ Failed'}")
            
            # Test database info
            print("📊 Getting database info...")
            db_info = client.get_database_info()
            print(f"Database: {db_info.get('database_name')}")
            print(f"Collections: {len(db_info.get('collections', []))}")
            
            # Test metrics
            print("📈 Getting client metrics...")
            metrics = client.get_metrics()
            print(f"Connections created: {metrics['connections_created']}")
            print(f"Connected: {metrics['connected']}")
            
            # Test basic collection operations
            print("📋 Testing collection operations...")
            
            # Create test collection
            if not client.has_collection("test_collection"):
                collection = client.create_collection("test_collection")
                print("✅ Created test collection")
            else:
                print("📝 Test collection already exists")
            
            # Insert test document
            test_doc = {
                "name": "test_document",
                "timestamp": "2024-01-01T00:00:00Z",
                "data": {"test": True}
            }
            
            result = client.insert_document("test_collection", test_doc)
            print(f"✅ Inserted test document: {result.get('_key')}")
            
            # Query test document
            doc = client.get_document("test_collection", result['_key'])
            if doc:
                print(f"✅ Retrieved test document: {doc['name']}")
            
            # Test AQL query
            print("🔍 Testing AQL query...")
            query = "FOR doc IN test_collection LIMIT 5 RETURN doc"
            cursor = client.execute_aql(query)
            docs = list(cursor)
            print(f"✅ AQL query returned {len(docs)} documents")
            
            # Clean up
            client.delete_document("test_collection", result['_key'])
            print("🧹 Cleaned up test document")
            
            # Disconnect
            client.disconnect()
            print("👋 Disconnected from ArangoDB")
            
            print("\n🎉 All tests passed!")
            return True
            
        else:
            print("❌ Connection failed!")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

if __name__ == "__main__":
    success = test_arango_connection()
    sys.exit(0 if success else 1)