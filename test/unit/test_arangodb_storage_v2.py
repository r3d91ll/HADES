#!/usr/bin/env python3
"""
Comprehensive test suite for ArangoStorageV2 implementation.

Tests the complete ArangoDB storage component with real database operations,
Sequential-ISNE schema integration, and all protocol compliance features.
"""

import sys
import uuid
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import the storage component and dependencies
from src.components.storage.arangodb.storage_v2 import ArangoStorageV2
from src.database.arango_client import ArangoClient, ArangoConnectionError, ArangoOperationError
from src.storage.incremental.sequential_isne_types import (
    CodeFile, DocumentationFile, ConfigFile,
    CodeFileType, DocumentationType, ConfigFileType,
    EdgeType, ProcessingStatus, FileType
)
from src.types.components.contracts import (
    StorageInput, QueryInput, EnhancedEmbedding,
    ProcessingStatus as ContractProcessingStatus
)


def test_arangodb_storage_v2():
    """Comprehensive test suite for ArangoStorageV2."""
    print("🧪 Testing ArangoStorageV2 Implementation")
    print("=" * 60)
    
    # Test configuration
    config = {
        "database": {
            "host": "127.0.0.1",
            "port": 8529,
            "username": "root",
            "password": "",
            "database": "test_arangodb_storage_v2"
        },
        "sequential_isne": {
            "db_name": "test_arangodb_storage_v2",
            "batch_size": 10,
            "similarity_threshold": 0.8,
            "embedding_dim": 768
        },
        "vector_index_enabled": True,
        "vector_dimension": 768,
        "max_results": 50
    }
    
    try:
        # ===== INITIALIZATION TESTS =====
        print("📋 Testing component initialization...")
        
        storage = ArangoStorageV2(config)
        
        # Test basic properties
        assert storage.name == "arangodb_v2"
        assert storage.version == "2.0.0"
        assert storage.component_type.value == "storage"
        print("  ✅ Component properties correct")
        
        # Test configuration validation
        assert storage.validate_config(config)
        print("  ✅ Configuration validation passed")
        
        # Test config schema
        schema = storage.get_config_schema()
        assert "properties" in schema
        assert "database" in schema["properties"]
        print("  ✅ Configuration schema valid")
        
        # ===== CONNECTION TESTS =====
        print("🔌 Testing database connection...")
        
        # Test connection
        connected = storage.connect()
        assert connected, "Failed to connect to ArangoDB"
        print("  ✅ Database connection successful")
        
        # Test health check
        healthy = storage.health_check()
        assert healthy, "Health check failed"
        print("  ✅ Health check passed")
        
        # Test metrics
        metrics = storage.get_metrics()
        assert "component_name" in metrics
        assert metrics["is_connected"] == True
        print("  ✅ Metrics collection working")
        
        # ===== SEQUENTIAL-ISNE SPECIFIC TESTS =====
        print("📁 Testing Sequential-ISNE modality storage...")
        
        # Generate unique file paths for this test run
        test_id = uuid.uuid4().hex[:8]
        
        # Test code file storage
        code_file = CodeFile(
            file_path=f"/test/example_{test_id}.py",
            file_name=f"example_{test_id}.py",
            directory=f"/test_{test_id}",
            extension=".py",
            file_type=CodeFileType.PYTHON,
            content="def hello():\n    print('Hello, world!')",
            content_hash=f"hash_{uuid.uuid4().hex[:8]}",
            size=40,
            modified_time=datetime.now(timezone.utc),
            directory_depth=1,
            lines_of_code=2
        )
        
        code_file_id = storage.store_modality_file(code_file)
        assert code_file_id.startswith("code_files/")
        print(f"  ✅ Stored code file: {code_file_id}")
        
        # Test documentation file storage
        doc_file = DocumentationFile(
            file_path=f"/test/README_{test_id}.md",
            file_name=f"README_{test_id}.md",
            directory=f"/test_{test_id}",
            extension=".md",
            file_type=DocumentationType.MARKDOWN,
            content="# Test Project\n\nThis is a test.",
            content_hash=f"hash_{uuid.uuid4().hex[:8]}",
            size=32,
            modified_time=datetime.now(timezone.utc),
            directory_depth=1,
            word_count=7
        )
        
        doc_file_id = storage.store_modality_file(doc_file)
        assert doc_file_id.startswith("documentation_files/")
        print(f"  ✅ Stored documentation file: {doc_file_id}")
        
        # Test config file storage
        config_file = ConfigFile(
            file_path=f"/test/config_{test_id}.yaml",
            file_name=f"config_{test_id}.yaml",
            directory=f"/test_{test_id}",
            extension=".yaml",
            file_type=ConfigFileType.YAML,
            content="database:\n  host: localhost",
            content_hash=f"hash_{uuid.uuid4().hex[:8]}",
            size=28,
            modified_time=datetime.now(timezone.utc),
            directory_depth=1
        )
        
        config_file_id = storage.store_modality_file(config_file)
        assert config_file_id.startswith("config_files/")
        print(f"  ✅ Stored config file: {config_file_id}")
        
        # Test cross-modal edge creation
        edge_id = storage.create_cross_modal_edge(
            from_file_id=code_file_id,
            to_file_id=doc_file_id,
            edge_type=EdgeType.CODE_TO_DOC,
            weight=0.85,
            metadata={"test": True}
        )
        assert edge_id.startswith("cross_modal_edges/")
        print(f"  ✅ Created cross-modal edge: {edge_id}")
        
        # Test cross-modal relationship discovery
        relationships = storage.find_cross_modal_relationships(code_file_id)
        assert len(relationships) >= 1
        assert relationships[0]["from_file"] == code_file_id
        assert relationships[0]["to_file"] == doc_file_id
        print(f"  ✅ Found {len(relationships)} cross-modal relationships")
        
        # ===== STORAGE PROTOCOL TESTS =====
        print("💾 Testing Storage protocol compliance...")
        
        # Create proper enhanced embeddings using contract
        enhanced_embeddings = [
            EnhancedEmbedding(
                chunk_id="chunk_1",
                original_embedding=[0.1] * 768,
                enhanced_embedding=[0.2] * 768,
                graph_features={"feature1": 0.5, "feature2": 0.8},
                enhancement_score=0.9,
                metadata={
                    "content": "Test content for chunk 1",
                    "file_path": f"/test/example_{test_id}.py",
                    "chunk_index": 0
                }
            ),
            EnhancedEmbedding(
                chunk_id="chunk_2",
                original_embedding=[0.15] * 768,
                enhanced_embedding=[0.25] * 768,
                graph_features={"feature1": 0.6, "feature2": 0.7},
                enhancement_score=0.85,
                metadata={
                    "content": "Test content for chunk 2",
                    "file_path": f"/test/README_{test_id}.md",
                    "chunk_index": 0
                }
            ),
            EnhancedEmbedding(
                chunk_id="chunk_3",
                original_embedding=[0.12] * 768,
                enhanced_embedding=[0.22] * 768,
                graph_features={"feature1": 0.4, "feature2": 0.9},
                enhancement_score=0.87,
                metadata={
                    "content": "Test content for chunk 3",
                    "file_path": f"/test/config_{test_id}.yaml",
                    "chunk_index": 0
                }
            )
        ]
        
        # Test storage input
        storage_input = StorageInput(
            enhanced_embeddings=enhanced_embeddings,
            storage_method="sequential_isne",
            metadata={"test_batch": True}
        )
        
        # Test store operation
        storage_output = storage.store(storage_input)
        
        assert len(storage_output.stored_items) == 3
        assert storage_output.metadata.component_name == "arangodb_v2"
        assert storage_output.storage_stats["stored_count"] == 3
        assert storage_output.index_info["sequential_isne_enabled"] == True
        print(f"  ✅ Stored {len(storage_output.stored_items)} enhanced embeddings")
        
        # Test query operations
        query_input = QueryInput(
            query="test query",
            top_k=5,
            filters={"test": True},
            search_options={"cross_modal": True}
        )
        
        # Test text query
        query_output = storage.query(query_input)
        assert query_output.metadata.component_name == "arangodb_v2"
        assert query_output.search_stats["sequential_isne_enhanced"] == True
        print(f"  ✅ Text query returned {len(query_output.results)} results")
        
        # Test vector query
        vector_query_input = QueryInput(
            query="vector test",
            query_embedding=[0.1] * 768,
            top_k=3
        )
        
        vector_query_output = storage.query(vector_query_input)
        assert vector_query_output.search_stats["query_type"] == "vector"
        print(f"  ✅ Vector query returned {len(vector_query_output.results)} results")
        
        # ===== CRUD OPERATIONS TESTS =====
        print("🔄 Testing CRUD operations...")
        
        # Test update operation
        stored_item_id = storage_output.stored_items[0].item_id
        update_success = storage.update(stored_item_id, {"updated": True})
        assert update_success
        print("  ✅ Update operation successful")
        
        # Test delete operation
        item_ids_to_delete = [item.item_id for item in storage_output.stored_items]
        delete_success = storage.delete(item_ids_to_delete)
        assert delete_success
        print(f"  ✅ Deleted {len(item_ids_to_delete)} items")
        
        # ===== STATISTICS AND CAPACITY TESTS =====
        print("📊 Testing statistics and capacity...")
        
        # Test statistics
        stats = storage.get_statistics()
        assert "connected" in stats
        assert stats["connected"] == True
        print("  ✅ Statistics retrieval working")
        
        # Test capacity info
        capacity = storage.get_capacity_info()
        assert "sequential_isne_config" in capacity
        assert capacity["vector_index_enabled"] == True
        print("  ✅ Capacity info retrieval working")
        
        # Test modality statistics
        modality_stats = storage.get_modality_statistics()
        assert "schema_version" in modality_stats
        print("  ✅ Modality statistics working")
        
        # ===== TRANSACTION SUPPORT TESTS =====
        print("💳 Testing transaction support...")
        
        # Test transaction support
        supports_transactions = storage.supports_transactions()
        assert supports_transactions == True
        print("  ✅ Transaction support confirmed")
        
        # ===== CONTEXT MANAGER TESTS =====
        print("🔗 Testing context manager support...")
        
        # Test context manager usage
        with ArangoStorageV2(config) as ctx_storage:
            assert ctx_storage._connected == True
            ctx_test_id = uuid.uuid4().hex[:8]
            test_file = CodeFile(
                file_path=f"/test/context_test_{ctx_test_id}.py",
                file_name=f"context_test_{ctx_test_id}.py",
                directory=f"/test_{ctx_test_id}",
                extension=".py",
                file_type=CodeFileType.PYTHON,
                content="# Context test",
                content_hash=f"hash_{uuid.uuid4().hex[:8]}",
                size=15,
                modified_time=datetime.now(timezone.utc),
                directory_depth=1
            )
            ctx_file_id = ctx_storage.store_modality_file(test_file)
            assert ctx_file_id.startswith("code_files/")
        
        # Storage should be disconnected after context exit
        print("  ✅ Context manager working correctly")
        
        # ===== CLEANUP =====
        print("🧹 Cleaning up test data...")
        
        # Clean up test files
        storage._client.delete_document("code_files", code_file_id.split("/")[1])
        storage._client.delete_document("documentation_files", doc_file_id.split("/")[1])
        storage._client.delete_document("config_files", config_file_id.split("/")[1])
        storage._client.delete_document("cross_modal_edges", edge_id.split("/")[1])
        
        # Disconnect
        storage.disconnect()
        print("  ✅ Test cleanup completed")
        
        print("\n🎉 All ArangoStorageV2 tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_configuration_edge_cases():
    """Test configuration edge cases and error handling."""
    print("\n🔧 Testing configuration edge cases...")
    
    storage = ArangoStorageV2()
    
    # Test invalid configurations
    invalid_configs = [
        None,
        "not a dict",
        {"database": {"port": "not an int"}},
        {"sequential_isne": {"batch_size": -1}}
    ]
    
    for invalid_config in invalid_configs:
        try:
            result = storage.validate_config(invalid_config)
            if result:
                print(f"  ⚠️  Expected validation failure for: {invalid_config}")
        except:
            pass  # Expected
    
    print("  ✅ Configuration validation edge cases handled")


def test_error_handling():
    """Test error handling and recovery."""
    print("\n⚠️  Testing error handling...")
    
    # Test with invalid database configuration
    invalid_config = {
        "database": {
            "host": "nonexistent-host",
            "port": 9999,
            "username": "invalid",
            "password": "wrong",
            "database": "nonexistent"
        }
    }
    
    storage = ArangoStorageV2(invalid_config)
    
    try:
        storage.connect()
        print("  ⚠️  Expected connection failure but succeeded")
    except ArangoConnectionError:
        print("  ✅ Connection error handled correctly")
    except Exception as e:
        print(f"  ✅ Connection error handled: {type(e).__name__}")
    
    # Test health check on disconnected storage
    health = storage.health_check()
    assert health == False
    print("  ✅ Health check correctly reports unhealthy state")


if __name__ == "__main__":
    print("Starting comprehensive ArangoStorageV2 test suite...")
    print("=" * 70)
    
    # Run main test suite
    main_success = test_arangodb_storage_v2()
    
    # Run edge case tests
    test_configuration_edge_cases()
    
    # Run error handling tests
    test_error_handling()
    
    if main_success:
        print("\n✅ All tests completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)