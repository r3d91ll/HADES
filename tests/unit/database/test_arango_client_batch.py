"""
Unit tests for ArangoDB client batch operations and type safety features.

These tests verify the functionality of the improved ArangoDB client with
a focus on batch document operations, edge handling, and type safety.
"""

import unittest
from unittest.mock import patch, MagicMock
import os
import tempfile
from typing import List, Dict, Any, Optional

from src.database.arango_client import ArangoClient


class ArangoClientBatchOperationsTests(unittest.TestCase):
    """Tests for the ArangoDB client batch operations and type safety features."""
    
    @patch("src.database.arango_client.BaseArangoClient")
    def setUp(self, mock_base_client):
        """Set up test environment with mocked ArangoDB client."""
        # Mock the base client
        self.mock_base_client = mock_base_client
        
        # Mock database
        self.mock_db = MagicMock()
        self.mock_base_client.return_value.db.return_value = self.mock_db
        
        # Create client with test configuration
        self.client = ArangoClient(
            host="test-host",
            port=8888,
            username="test-user",
            password="test-password",
            database="test-db"
        )
    
    def test_batch_document_insert(self):
        """Test batch insertion of documents with proper type handling."""
        # Mock database and collection
        mock_db = MagicMock()
        mock_collection = MagicMock()
        mock_db.collection.return_value = mock_collection
        self.client.get_database = MagicMock(return_value=mock_db)
        
        # Define test documents
        docs = [
            {"_key": "doc1", "content": "Test Document 1", "tags": ["test", "document"]},
            {"_key": "doc2", "content": "Test Document 2", "tags": ["test", "batch"]},
            {"_key": "doc3", "content": "Test Document 3", "tags": ["test", "insert"]}
        ]
        
        # Mock the collection.insert_many method
        mock_collection.insert_many.return_value = [
            {"_id": "test_collection/doc1", "_key": "doc1", "_rev": "1"},
            {"_id": "test_collection/doc2", "_key": "doc2", "_rev": "1"},
            {"_id": "test_collection/doc3", "_key": "doc3", "_rev": "1"}
        ]
        
        # Execute batch insert
        result = self.client.insert_documents("test-db", "test_collection", docs)
        
        # Verify results
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]["_key"], "doc1")
        self.assertEqual(result[1]["_key"], "doc2")
        self.assertEqual(result[2]["_key"], "doc3")
        
        # Verify collection was called correctly
        mock_db.collection.assert_called_with("test_collection")
        mock_collection.insert_many.assert_called_once()
    
    def test_batch_insert_with_error_handling(self):
        """Test batch insertion with error handling and proper type conversions."""
        # Mock database and collection
        mock_db = MagicMock()
        mock_collection = MagicMock()
        mock_db.collection.return_value = mock_collection
        self.client.get_database = MagicMock(return_value=mock_db)
        
        # Define test documents
        docs = [
            {"_key": "doc1", "content": "Test Document 1"},
            {"_key": "doc2", "content": "Test Document 2"},
            {"_key": "doc3", "content": "Test Document 3"}
        ]
        
        # Mock the collection.insert_many method to raise an exception
        mock_collection.insert_many.side_effect = Exception("Database error")
        
        # Execute batch insert with error
        result = self.client.insert_documents("test-db", "test_collection", docs)
        
        # Verify results are None for each document due to error
        self.assertEqual(len(result), 3)
        self.assertIsNone(result[0])
        self.assertIsNone(result[1])
        self.assertIsNone(result[2])
    
    def test_empty_batch_operation(self):
        """Test handling of empty document lists in batch operations."""
        # Call with empty document list
        result = self.client.insert_documents("test-db", "test_collection", [])
        
        # Should return empty list
        self.assertEqual(result, [])
    
    def test_edge_type_safety(self):
        """Test edge operations with proper type safety."""
        # Mock database and collection
        mock_db = MagicMock()
        mock_edge_collection = MagicMock()
        mock_db.collection.return_value = mock_edge_collection
        self.client.get_database = MagicMock(return_value=mock_db)
        
        # Mock the insert method
        mock_edge_collection.insert.return_value = {
            "_id": "edges/test_edge",
            "_key": "test_edge",
            "_rev": "1"
        }
        
        # Test edge insertion with separate from_vertex, to_vertex and data
        edge_data = {"weight": 0.75, "label": "CONTAINS"}
        result = self.client.insert_edge(
            "test-db",
            "edges",
            from_vertex="nodes/source",
            to_vertex="nodes/target",
            data=edge_data
        )
        
        # Verify the result
        self.assertEqual(result["_id"], "edges/test_edge")
        self.assertEqual(result["_key"], "test_edge")
        
        # Verify the collection.insert was called with properly formatted edge
        expected_doc = {
            "_from": "nodes/source",
            "_to": "nodes/target",
            "weight": 0.75,
            "label": "CONTAINS"
        }
        mock_edge_collection.insert.assert_called_with(expected_doc)
    
    def test_query_with_type_safety(self):
        """Test query execution with proper type handling."""
        # Mock database
        mock_db = MagicMock()
        mock_cursor = MagicMock()
        # Mix of different result types to test handling
        mock_cursor.__iter__.return_value = iter([
            {"_key": "doc1", "name": "Test 1"},
            {"_key": "doc2", "name": "Test 2"},
            None  # Intentionally add None to test handling
        ])
        mock_db.aql.execute.return_value = mock_cursor
        
        self.client.get_database = MagicMock(return_value=mock_db)
        
        # Execute query
        query = "FOR d IN test_collection FILTER d.type == @type RETURN d"
        bind_vars = {"type": "test"}
        
        results = self.client.execute_query("test-db", query, bind_vars)
        
        # Verify results - should exclude None values
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["_key"], "doc1")
        self.assertEqual(results[1]["_key"], "doc2")
        
        # Verify query execution
        mock_db.aql.execute.assert_called_with(query, bind_vars=bind_vars)


if __name__ == "__main__":
    unittest.main()
