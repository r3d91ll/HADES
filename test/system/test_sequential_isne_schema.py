#!/usr/bin/env python3
"""
Test script for Sequential-ISNE schema.

Tests the Sequential-ISNE modality-specific schema with real ArangoDB connection.
"""

import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.database.arango_client import ArangoClient
from src.storage.incremental.sequential_isne_schema import SequentialISNESchemaManager
from src.storage.incremental.sequential_isne_types import (
    CodeFile, DocumentationFile, ConfigFile,
    FileType, CodeFileType, DocumentationType, ConfigFileType,
    ProcessingStatus, classify_file_type
)
from datetime import datetime, timezone

def test_sequential_isne_schema():
    """Test Sequential-ISNE schema creation and validation."""
    print("🧪 Testing Sequential-ISNE Schema")
    print("=" * 60)
    
    # Create ArangoDB client
    try:
        client = ArangoClient(
            host="127.0.0.1",
            port=8529,
            username="root",
            password="",
            database="sequential_isne_test",
            timeout=10
        )
        
        print("📡 Connecting to ArangoDB...")
        success = client.connect()
        
        if not success:
            print("❌ Failed to connect to ArangoDB")
            return False
        
        print("✅ Connection successful!")
        
        # Create schema manager
        print("🏗️  Creating Sequential-ISNE schema manager...")
        schema_manager = SequentialISNESchemaManager(
            client._client, 
            "sequential_isne_test"
        )
        
        # Initialize database with Sequential-ISNE schema
        print("📋 Initializing Sequential-ISNE database schema...")
        init_success = schema_manager.initialize_database()
        
        if not init_success:
            print("❌ Failed to initialize schema")
            return False
        
        print("✅ Schema initialized successfully!")
        
        # Validate schema
        print("🔍 Validating schema structure...")
        validation_success = schema_manager.validate_schema()
        
        if not validation_success:
            print("❌ Schema validation failed")
            return False
        
        print("✅ Schema validation passed!")
        
        # Test modality collections
        print("📁 Testing modality-specific collections...")
        modality_collections = schema_manager.get_modality_collections()
        
        for modality, collection_name in modality_collections.items():
            if client.has_collection(collection_name):
                count = client.get_collection_count(collection_name)
                print(f"  ✅ {modality}: {collection_name} ({count} documents)")
            else:
                print(f"  ❌ Missing collection: {collection_name}")
                return False
        
        # Test cross-modal edge types
        print("🌉 Testing cross-modal edge types...")
        cross_modal_types = schema_manager.get_cross_modal_edge_types()
        print(f"  Cross-modal edge types: {len(cross_modal_types)}")
        for edge_type in cross_modal_types[:3]:  # Show first 3
            print(f"    - {edge_type}")
        
        # Test data insertion with Pydantic models
        print("💾 Testing data insertion with Pydantic models...")
        
        # Test code file
        code_file = CodeFile(
            file_path="/test/example.py",
            file_name="example.py",
            directory="/test",
            extension=".py",
            file_type=CodeFileType.PYTHON,
            content="def hello_world():\n    print('Hello, world!')",
            content_hash="abc123",
            size=45,
            modified_time=datetime.fromisoformat("2024-01-01T00:00:00"),
            directory_depth=1,
            lines_of_code=2
        )
        
        # Insert code file
        code_result = client.insert_document("code_files", code_file.model_dump(mode='json'))
        print(f"  ✅ Inserted code file: {code_result['_key']}")
        
        # Test documentation file
        doc_file = DocumentationFile(
            file_path="/test/README.md",
            file_name="README.md",
            directory="/test",
            extension=".md",
            file_type=DocumentationType.MARKDOWN,
            content="# Test Project\n\nThis is a test project.",
            content_hash="def456",
            size=35,
            modified_time=datetime.fromisoformat("2024-01-01T00:00:00"),
            directory_depth=1,
            word_count=7
        )
        
        # Insert documentation file
        doc_result = client.insert_document("documentation_files", doc_file.model_dump(mode='json'))
        print(f"  ✅ Inserted documentation file: {doc_result['_key']}")
        
        # Test config file
        config_file = ConfigFile(
            file_path="/test/config.yaml",
            file_name="config.yaml",
            directory="/test",
            extension=".yaml",
            file_type=ConfigFileType.YAML,
            content="database:\n  host: localhost\n  port: 5432",
            content_hash="ghi789",
            size=42,
            modified_time=datetime.fromisoformat("2024-01-01T00:00:00"),
            directory_depth=1
        )
        
        # Insert config file
        config_result = client.insert_document("config_files", config_file.model_dump(mode='json'))
        print(f"  ✅ Inserted config file: {config_result['_key']}")
        
        # Test cross-modal edge insertion
        print("🔗 Testing cross-modal edge insertion...")
        
        cross_modal_edge = {
            "_from": f"code_files/{code_result['_key']}",
            "_to": f"documentation_files/{doc_result['_key']}",
            "_from_modality": "code",
            "_to_modality": "documentation",
            "edge_type": "code_to_doc",
            "weight": 0.85,
            "confidence": 0.9,
            "similarity_score": 0.75,
            "source": "isne_training",
            "discovery_method": "sequential_isne",
            "created_at": "2024-01-01T00:00:00Z",
            "metadata": {"test": True}
        }
        
        edge_result = client.insert_document("cross_modal_edges", cross_modal_edge)
        print(f"  ✅ Inserted cross-modal edge: {edge_result['_key']}")
        
        # Test utility functions
        print("🔧 Testing utility functions...")
        
        test_files = [
            "/test/script.py",
            "/test/README.md", 
            "/test/config.json",
            "/test/data.csv"
        ]
        
        for file_path in test_files:
            file_type = classify_file_type(file_path)
            print(f"  {file_path} -> {file_type}")
        
        # Get database statistics
        print("📊 Getting database statistics...")
        stats = schema_manager.get_database_statistics()
        
        print(f"  Schema version: {stats['schema_version']}")
        print(f"  Total files: {stats['total_files']}")
        print(f"  Cross-modal edges: {stats['cross_modal_edges']}")
        print(f"  Modality distribution:")
        for modality, count in stats['modality_distribution'].items():
            print(f"    - {modality}: {count}")
        
        # Clean up test data
        print("🧹 Cleaning up test data...")
        client.delete_document("code_files", code_result['_key'])
        client.delete_document("documentation_files", doc_result['_key'])
        client.delete_document("config_files", config_result['_key'])
        client.delete_document("cross_modal_edges", edge_result['_key'])
        print("  ✅ Test data cleaned up")
        
        # Disconnect
        client.disconnect()
        print("👋 Disconnected from ArangoDB")
        
        print("\n🎉 All Sequential-ISNE schema tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_sequential_isne_schema()
    sys.exit(0 if success else 1)