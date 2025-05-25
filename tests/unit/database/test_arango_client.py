"""
Unit tests for ArangoDB client.

These tests verify the functionality of the ArangoDB client configuration
and integration with the database configuration system.
"""

import unittest
from unittest.mock import patch, MagicMock
import os
import tempfile
import yaml

from src.database.arango_client import ArangoClient
from src.config.database_config import (
    ArangoDBConfig,
    load_database_config,
    get_database_config,
    get_connection_params
)


class DatabaseConfigTests(unittest.TestCase):
    """Tests for the database configuration module."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary config file
        self.temp_config = tempfile.NamedTemporaryFile(delete=False, suffix=".yaml")
        self.config_data = {
            "host": "test-host",
            "port": 8888,
            "username": "test-user",
            "password": "test-password",
            "database_name": "test-db",
            "documents_collection": "test-docs",
            "chunks_collection": "test-chunks",
            "relationships_collection": "test-relations"
        }
        
        # Write config data to file
        with open(self.temp_config.name, "w") as f:
            yaml.dump(self.config_data, f)
    
    def tearDown(self):
        """Clean up test environment."""
        # Remove temporary config file
        os.unlink(self.temp_config.name)
        
        # Clear environment variables
        for var in ["ARANGO_HOST", "ARANGO_PORT", "ARANGO_USER", "ARANGO_PASSWORD", "ARANGO_DB"]:
            if var in os.environ:
                del os.environ[var]
    
    def test_load_config_from_file(self):
        """Test loading configuration from YAML file."""
        config = load_database_config(self.temp_config.name)
        
        self.assertEqual(config.host, self.config_data["host"])
        self.assertEqual(config.port, self.config_data["port"])
        self.assertEqual(config.username, self.config_data["username"])
        self.assertEqual(config.password, self.config_data["password"])
        self.assertEqual(config.database_name, self.config_data["database_name"])
        self.assertEqual(config.documents_collection, self.config_data["documents_collection"])
    
    def test_get_config_as_dict(self):
        """Test getting configuration as dictionary."""
        config_dict = get_database_config(self.temp_config.name)
        
        self.assertEqual(config_dict["host"], self.config_data["host"])
        self.assertEqual(config_dict["port"], self.config_data["port"])
        self.assertEqual(config_dict["username"], self.config_data["username"])
        self.assertEqual(config_dict["password"], self.config_data["password"])
        self.assertEqual(config_dict["database_name"], self.config_data["database_name"])
    
    def test_environment_variable_override(self):
        """Test environment variable overrides."""
        # Set environment variables
        os.environ["ARANGO_HOST"] = "env-host"
        os.environ["ARANGO_PORT"] = "9999"
        os.environ["ARANGO_USER"] = "env-user"
        os.environ["ARANGO_PASSWORD"] = "env-password"
        os.environ["ARANGO_DB"] = "env-db"
        
        # Get connection parameters
        params = get_connection_params()
        
        # Check that environment variables take precedence
        self.assertEqual(params["host"], "env-host")
        self.assertEqual(params["port"], 9999)
        self.assertEqual(params["username"], "env-user")
        self.assertEqual(params["password"], "env-password")
        self.assertEqual(params["database"], "env-db")
    
    def test_default_config_values(self):
        """Test default configuration values."""
        # Create empty config file
        with open(self.temp_config.name, "w") as f:
            f.write("{}\n")
            
        config = load_database_config(self.temp_config.name)
        
        # Check default values
        self.assertEqual(config.host, "localhost")
        self.assertEqual(config.port, 8529)
        self.assertEqual(config.username, "root")
        self.assertEqual(config.database_name, "hades")


class ArangoClientTests(unittest.TestCase):
    """Tests for the ArangoDB client."""
    
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
            database_name="test-db"
        )
    
    def test_client_initialization(self):
        """Test client initialization with explicit parameters."""
        # We can't directly check private attributes since the implementation doesn't store them
        # But we can verify the client was initialized and database was accessed
        
        # Verify username is stored (this is explicitly stored)
        self.assertEqual(self.client.username, "test-user")
        
        # Check that base client was initialized correctly
        self.mock_base_client.assert_called_once()
        
        # Verify the URL was constructed correctly
        self.mock_base_client.assert_called_with(
            hosts="http://test-host:8888",
            http_client=None,
            serializer=None,
            deserializer=None
        )
        
        # Verify database connection was established
        self.mock_base_client.return_value.db.assert_called_once()
    
    @patch("src.database.arango_client.get_connection_params")
    @patch("src.database.arango_client.BaseArangoClient")
    def test_client_with_config(self, mock_base_client, mock_get_params):
        """Test client initialization with configuration."""
        # Mock connection parameters
        mock_get_params.return_value = {
            "host": "config-host",
            "port": 7777,
            "username": "config-user",
            "password": "config-password",
            "database": "config-db",  # Note: The actual implementation expects 'database', not 'database_name'
            "use_ssl": True,
            "timeout": 30
        }
        
        # Mock database
        mock_base_client.return_value.db.return_value = MagicMock()
        
        # Create client without explicit parameters
        # Directly set username and password to match our mocked values
        client = ArangoClient(username="config-user", password="config-password")
        
        # Verify username is stored
        self.assertEqual(client.username, "config-user")
        
        # Verify the URL was constructed correctly with SSL
        mock_base_client.assert_called_with(
            hosts="https://config-host:7777", # should use https with use_ssl=True
            http_client=None,
            serializer=None,
            deserializer=None
        )
        
        # Check that get_connection_params was called
        mock_get_params.assert_called_once()
    
    def test_database_operations(self):
        """Test database operations."""
        # Mock database existence check
        self.mock_db.has_database.return_value = False
        
        # Check database existence
        result = self.client.database_exists("test-db")
        self.assertFalse(result)
        self.mock_db.has_database.assert_called_with("test-db")
        
        # Create database
        self.mock_db.create_database.return_value = True
        result = self.client.create_database("test-db")
        self.assertTrue(result)
        self.mock_db.create_database.assert_called_with("test-db")
        
        # Delete database
        self.mock_db.has_database.return_value = True
        self.mock_db.delete_database.return_value = True
        result = self.client.delete_database("test-db")
        self.assertTrue(result)
        self.mock_db.delete_database.assert_called_with("test-db")
    
    def test_collection_operations(self):
        """Test collection operations."""
        # Mock database
        mock_db = MagicMock()
        self.client.get_database = MagicMock(return_value=mock_db)
        
        # Mock collection existence check
        mock_db.has_collection.return_value = False
        
        # Check collection existence
        result = self.client.collection_exists("test-db", "test-collection")
        self.assertFalse(result)
        mock_db.has_collection.assert_called_with("test-collection")
        
        # Create collection - note that our implementation now passes edge=False by default
        mock_db.create_collection.return_value = MagicMock()
        self.client.create_collection("test-db", "test-collection")
        mock_db.create_collection.assert_called_with("test-collection", edge=False)
        
        # Create edge collection
        self.client.create_collection("test-db", "test-edges", edge=True)
        mock_db.create_collection.assert_called_with("test-edges", edge=True)
        
        # Delete collection
        mock_db.has_collection.return_value = True
        mock_db.delete_collection.return_value = True
        result = self.client.delete_collection("test-db", "test-collection")
        self.assertTrue(result)
        mock_db.delete_collection.assert_called_with("test-collection")
    
    def test_document_operations(self):
        """Test document operations."""
        # Mock database and collection
        mock_db = MagicMock()
        mock_collection = MagicMock()
        mock_db.collection.return_value = mock_collection
        mock_db.has_collection.return_value = True
        
        self.client.get_database = MagicMock(return_value=mock_db)
        
        # Test document insert
        doc = {"_key": "test-doc", "name": "Test Document"}
        # For insert_document, we need to mock the internal _insert_document method
        # since the actual implementation gets the document after insertion
        mock_result = {"_id": "collection/test-doc", "_key": "test-doc", "name": "Test Document"}
        self.client._insert_document = MagicMock(return_value=mock_result)
        
        result = self.client.insert_document("test-db", "test-collection", doc)
        self.assertEqual(result["_id"], "collection/test-doc")
        self.client._insert_document.assert_called_with("test-db", "test-collection", doc)
        
        # Test document update
        update_doc = {"name": "Updated Document"}
        
        # Mock the _update_document method with correct result
        expected_updated_doc = {
            "_id": "collection/test-doc", 
            "_key": "test-doc", 
            "name": "Updated Document"
        }
        self.client._update_document = MagicMock(return_value=expected_updated_doc)
        
        result = self.client.update_document("test-db", "test-collection", "test-doc", update_doc)
        self.assertEqual(result["name"], "Updated Document")
        
        # Verify internal method was called correctly
        self.client._update_document.assert_called_with("test-db", "test-collection", "test-doc", update_doc)
        
        # Test document delete
        # In the actual implementation, delete_document returns a boolean
        # We need to mock the internal _delete_document method to return True
        self.client._delete_document = MagicMock(return_value=True)
        
        # Call delete_document
        result = self.client.delete_document("test-db", "test-collection", "test-doc")
        
        # Verify the result is True
        self.assertTrue(result)
        
        # Verify the internal method was called correctly
        self.client._delete_document.assert_called_with("test-db", "test-collection", "test-doc")
        
        # Test document existence check
        mock_collection.has.return_value = True
        
        result = self.client.document_exists("test-db", "test-collection", "test-doc")
        self.assertTrue(result)
        mock_collection.has.assert_called_with("test-doc")
    
    def test_edge_operations(self):
        """Test edge operations."""
        # Mock database and collection
        mock_db = MagicMock()
        mock_collection = MagicMock()
        mock_db.collection.return_value = mock_collection
        mock_db.has_collection.return_value = True
        
        self.client.get_database = MagicMock(return_value=mock_db)
        
        # Test edge insert with from_vertex, to_vertex, and data parameters
        edge_data = {"weight": 0.5}
        # In the actual implementation, we need to mock the collection.insert and collection.get
        mock_collection.insert.return_value = {"_id": "edges/test-edge", "_key": "test-edge"}
        mock_collection.get.return_value = {
            "_id": "edges/test-edge", 
            "_key": "test-edge",
            "_from": "vertices/source", 
            "_to": "vertices/target", 
            "weight": 0.5
        }
        
        result = self.client.insert_edge(
            "test-db", 
            "test-edges", 
            from_vertex="vertices/source",
            to_vertex="vertices/target",
            data=edge_data
        )
        
        # Check the result - should be the document returned by collection.get
        self.assertIsNotNone(result)
        self.assertEqual(result["_id"], "edges/test-edge")
        self.assertEqual(result["_from"], "vertices/source")
        self.assertEqual(result["_to"], "vertices/target")
        
        # Check that the collection.insert was called with properly constructed edge
        expected_edge = {
            "_from": "vertices/source",
            "_to": "vertices/target",
            "weight": 0.5
        }
        mock_collection.insert.assert_called_with(expected_edge)
        
        # Test edge update
        update_edge = {"weight": 0.8}
        
        # Mock the _update_edge method to return expected result
        expected_updated_edge = {
            "_id": "edges/test-edge", 
            "_key": "test-edge",
            "_from": "vertices/source", 
            "_to": "vertices/target", 
            "weight": 0.8
        }
        self.client._update_edge = MagicMock(return_value=expected_updated_edge)
        
        result = self.client.update_edge("test-db", "test-edges", "edges/test-edge", update_edge)
        self.assertEqual(result["weight"], 0.8)
        
        # Verify the internal method was called correctly
        self.client._update_edge.assert_called_with("test-db", "test-edges", "edges/test-edge", update_edge)
        
        # Test get edge - reset mock first
        mock_db.aql.execute.reset_mock()
        mock_cursor = MagicMock()
        mock_cursor.__iter__.return_value = iter([{"_id": "edges/test-edge"}])
        mock_db.aql.execute.return_value = mock_cursor
        
        result = self.client.get_edge(
            "test-db", "test-edges", "vertices/source", "vertices/target"
        )
        self.assertEqual(result["_id"], "edges/test-edge")
        self.assertEqual(mock_db.aql.execute.call_count, 1)
        
        # Test delete edges - reset mock first
        mock_db.aql.execute.reset_mock()
        mock_cursor = MagicMock()
        mock_cursor.__iter__.return_value = iter([1, 1, 1])  # 3 edges deleted
        mock_db.aql.execute.return_value = mock_cursor
        
        result = self.client.delete_edges("test-db", "test-edges", "vertices/source")
        self.assertEqual(result, 3)
        self.assertEqual(mock_db.aql.execute.call_count, 1)
    
    def test_query_execution(self):
        """Test AQL query execution."""
        # Mock database
        mock_db = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.__iter__.return_value = iter([{"name": "doc1"}, {"name": "doc2"}])
        mock_db.aql.execute.return_value = mock_cursor
        
        self.client.get_database = MagicMock(return_value=mock_db)
        
        # Execute query
        query = "FOR d IN test-collection RETURN d"
        bind_vars = {"param": "value"}
        
        result = self.client.execute_query("test-db", query, bind_vars)
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["name"], "doc1")
        self.assertEqual(result[1]["name"], "doc2")
        mock_db.aql.execute.assert_called_with(query, bind_vars=bind_vars)
    
    def test_retry_mechanism(self):
        """Test retry mechanism for failed operations."""
        # Mock operation that fails twice then succeeds
        mock_operation = MagicMock(side_effect=[Exception("First failure"), Exception("Second failure"), "success"])
        
        # Call with retry
        result = self.client._retry_operation(mock_operation, "test_operation")
        
        # Check result
        self.assertEqual(result, "success")
        self.assertEqual(mock_operation.call_count, 3)
    
    def test_batch_document_operations(self):
        """Test batch operations for documents."""
        # Mock database and collection
        mock_db = MagicMock()
        mock_collection = MagicMock()
        mock_db.collection.return_value = mock_collection
        mock_db.has_collection.return_value = True
        
        self.client.get_database = MagicMock(return_value=mock_db)
        
        # Test batch document insert
        docs = [
            {"_key": "doc1", "name": "Document 1"},
            {"_key": "doc2", "name": "Document 2"},
            {"_key": "doc3", "name": "Document 3"}
        ]
        
        # Mock the internal _insert_documents method
        self.client._insert_documents = MagicMock(return_value=[
            {"_id": "collection/doc1", "_key": "doc1", "_rev": "1"},
            {"_id": "collection/doc2", "_key": "doc2", "_rev": "1"},
            {"_id": "collection/doc3", "_key": "doc3", "_rev": "1"}
        ])
        
        # Call the batch insert method
        result = self.client.insert_documents("test-db", "test-collection", docs)
        
        # Verify results
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]["_key"], "doc1")
        self.assertEqual(result[1]["_key"], "doc2")
        self.assertEqual(result[2]["_key"], "doc3")
        
        # Verify internal method was called correctly
        self.client._insert_documents.assert_called_once_with("test-db", "test-collection", docs)
        
    def test_batch_document_operation_with_failures(self):
        """Test batch operations with some failures."""
        # Mock database and collection
        mock_db = MagicMock()
        mock_collection = MagicMock()
        mock_db.collection.return_value = mock_collection
        mock_db.has_collection.return_value = True
        
        self.client.get_database = MagicMock(return_value=mock_db)
        
        # Test documents
        docs = [
            {"_key": "doc1", "name": "Document 1"},
            {"_key": "doc2", "name": "Document 2"},
            {"_key": "doc3", "name": "Document 3"}
        ]
        
        # Mock an exception for the _insert_documents method
        self.client._insert_documents = MagicMock(side_effect=Exception("Batch insert failed"))
        
        # Call the batch insert method
        result = self.client.insert_documents("test-db", "test-collection", docs)
        
        # Verify we get None values for each document when an error occurs
        self.assertEqual(len(result), 3)
        self.assertIsNone(result[0])
        self.assertIsNone(result[1])
        self.assertIsNone(result[2])
        
    def test_edge_operations_with_type_safety(self):
        """Test edge operations with type safety improvements."""
        # Mock database and collection
        mock_db = MagicMock()
        mock_collection = MagicMock()
        mock_db.collection.return_value = mock_collection
        mock_db.has_collection.return_value = True
        
        self.client.get_database = MagicMock(return_value=mock_db)
        
        # Test edge insert with from_vertex and to_vertex parameters
        # Make sure to provide the correct mock return value structure
        mock_collection.insert.return_value = {
            "_id": "edges/test-edge", 
            "_key": "test-edge", 
            "_rev": "1"
        }
        
        # Mock the get method as it's called after insertion
        mock_collection.get.return_value = {
            "_id": "edges/test-edge", 
            "_key": "test-edge", 
            "_rev": "1",
            "_from": "vertices/source",
            "_to": "vertices/target",
            "weight": 0.5
        }
        
        # Call insert_edge with the new parameter format
        result = self.client.insert_edge(
            "test-db", 
            "test-edges", 
            from_vertex="vertices/source", 
            to_vertex="vertices/target", 
            data={"weight": 0.5}
        )
        
        # Verify the result has the expected structure
        self.assertIsNotNone(result)
        self.assertEqual(result["_id"], "edges/test-edge")
        self.assertEqual(result["_key"], "test-edge")
        self.assertEqual(result["_from"], "vertices/source")
        self.assertEqual(result["_to"], "vertices/target")
        
        # Verify the collection.insert was called with properly formatted edge document
        expected_edge = {
            "_from": "vertices/source",
            "_to": "vertices/target",
            "weight": 0.5
        }
        mock_collection.insert.assert_called_with(expected_edge)
        
    def test_empty_batch_operations(self):
        """Test batch operations with empty document list."""
        # Call with empty document list
        result = self.client.insert_documents("test-db", "test-collection", [])
        
        # Should return empty list
        self.assertEqual(result, [])
        
    def test_retry_mechanism_with_failure(self):
        """Test retry mechanism with a consistently failing operation."""
        # Mock operation that always fails
        mock_operation = MagicMock(side_effect=Exception("Always fails"))
        
        # Set fewer retries for the test
        self.client.max_retries = 2
        
        # Call with retry and check exception
        with self.assertRaises(Exception) as context:
            self.client._retry_operation(mock_operation, "failing_operation")
            
        self.assertEqual(str(context.exception), "Always fails")
        self.assertEqual(mock_operation.call_count, 3)  # Initial + 2 retries


if __name__ == "__main__":
    unittest.main()
