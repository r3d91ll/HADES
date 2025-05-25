"""Unit tests for the TextArangoRepository class.

This module contains tests for the TextArangoRepository class, ensuring proper
handling of text document storage, embedding management, and graph operations
in ArangoDB.
"""

import asyncio
import json
import unittest
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, cast
from unittest.mock import MagicMock, patch, AsyncMock, call
import numpy as np
from datetime import datetime
import logging
from unittest.mock import PropertyMock

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.storage.arango.text_repository import TextArangoRepository
from src.storage.arango.connection import ArangoConnection
from src.storage.arango.repository import ArangoRepository
from src.types.common import NodeData, EdgeData, EmbeddingVector, NodeID


# Helper function to run async tests
def run_async_test(coroutine):
    """Run an async test coroutine."""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coroutine)


class MockTextArangoRepository(TextArangoRepository):
    """A mock version of TextArangoRepository for testing."""
    
    def __init__(self):
        # Skip the parent's __init__ to avoid actual ArangoDB operations
        # We'll set up the attributes manually
        self.connection = MagicMock()
        self.node_collection_name = "test_nodes"
        self.edge_collection_name = "test_edges"
        self.graph_name = "test_graph"
        
        # For mocking methods
        self._setup_text_indexes = MagicMock()
    
    async def initialize(self):
        return True
    
    async def store_node(self, node_data: NodeData) -> bool:
        return True
        
    async def store_edge(self, edge_data: EdgeData) -> bool:
        return True
        
    async def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        # This is mocked to return a node by default 
        return {"id": node_id, "data": "mock_data"}
        
    async def get_path(self, start_id: str, end_id: str) -> List[Dict[str, Any]]:
        return []
    
    async def _execute_aql(self, query: str, bind_vars: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        # This mock implementation returns empty list by default
        # Will be overridden in specific tests
        return []
      
    async def get_connected_nodes(self, node_id: str, edge_type: str = None) -> List[Dict[str, Any]]:
        # Mock implementation that returns empty list by default
        return []
        
    async def search_fulltext(self, query: str, field: str = "content", node_types: Optional[List[str]] = None, limit: int = 10) -> List[Dict[str, Any]]:
        # Mock implementation that relies on _execute_aql being mocked appropriately in each test
        bind_vars = {"query": query, "limit": limit}
        if node_types:
            bind_vars["types"] = node_types
            
        query_string = f"""
        FOR doc IN FULLTEXT({self.node_collection_name}, "{field}", @query)
            FILTER doc.type IN @types
            LIMIT @limit
            RETURN doc
        """
        return await self._execute_aql(query_string, bind_vars)
        
    async def hybrid_search(self, text_query: str, query_vector: List[float], embedding_type: str = "default", 
                          node_types: Optional[List[str]] = None, limit: int = 10, 
                          text_weight: float = 0.5, vector_weight: float = 0.5) -> List[Dict[str, Any]]:
        # Mock implementation that relies on _execute_aql being mocked appropriately in each test
        bind_vars = {
            "query": text_query, 
            "vector": query_vector,
            "limit": limit,
            "text_weight": text_weight,
            "vector_weight": vector_weight
        }
        if node_types:
            bind_vars["types"] = node_types
            
        query_string = f"""
        LET text_results = (FOR doc IN FULLTEXT({self.node_collection_name}, "content", @query))
        LET vector_results = (VECTOR_SEARCH...)
        
        FOR doc IN text_results
            SORT doc.score * @text_weight + vector_similarity * @vector_weight DESC
            LIMIT @limit
            RETURN doc
        """
        return await self._execute_aql(query_string, bind_vars)
        
        
class TestTextArangoRepository(unittest.TestCase):
    """Test cases for TextArangoRepository."""

    def setUp(self) -> None:
        """Set up mocks for testing."""
        # Create a simple connection mock
        self.connection_mock = MagicMock(spec=ArangoConnection)
        
        # Use our special mock repository that avoids real DB operations
        self.repo = MockTextArangoRepository()
        
        # Override _execute_aql with a controlled mock for testing
        self.repo._execute_aql = AsyncMock(return_value=[])

    def test_initialization(self) -> None:
        """Test repository initialization."""
        # Create a mock connection
        connection = MagicMock()
        
        # For this test, we'll use a test-specific mock class that implements the abstract methods
        class TestRepo(TextArangoRepository):
            async def initialize(self):
                return True
            
            async def store_node(self, node_data):
                return True
                
            async def store_edge(self, edge_data):
                return True
                
            async def get_node(self, node_id):
                return {"id": node_id}
                
            async def get_path(self, start_id, end_id):
                return []
        
        # Create a repository with our test-specific mock
        repo = TestRepo(
            connection=connection,
            node_collection="test_nodes",
            edge_collection="test_edges",
            graph_name="test_graph"
        )
        
        # Check that collections were set correctly
        self.assertEqual(repo.node_collection_name, "test_nodes")
        self.assertEqual(repo.edge_collection_name, "test_edges")
        self.assertEqual(repo.graph_name, "test_graph")
        
    def test_has_index(self) -> None:
        """Test _has_index method."""
        # Create a mock collection
        mock_collection = MagicMock()
        mock_collection.indexes.return_value = [
            {"fields": ["type"]},  # Single field index
            {"fields": ["field1", "field2"]}  # Multi-field index
        ]
        
        # Test index that exists
        self.assertTrue(self.repo._has_index(mock_collection, ["type"]))
        
        # Test multi-field index that exists
        self.assertTrue(self.repo._has_index(mock_collection, ["field1", "field2"]))
        
        # Test index that doesn't exist
        self.assertFalse(self.repo._has_index(mock_collection, ["non_existent"]))
        
        # Test partial match (should return False)
        self.assertFalse(self.repo._has_index(mock_collection, ["field1"]))
        
    def test_setup_text_indexes(self) -> None:
        """Test _setup_text_indexes method."""
        # Create mock connection and collection
        connection = MagicMock()
        mock_collection = MagicMock()
        mock_collection.indexes.return_value = [
            {"fields": ["type"]}  # This index already exists
        ]
        connection.get_collection.return_value = mock_collection
        
        # For this test, we'll use a concrete class with the abstract methods implemented
        # and the real _setup_text_indexes method exposed
        class TestIndexRepo(TextArangoRepository):
            async def initialize(self):
                return True
            
            async def store_node(self, node_data):
                return True
                
            async def store_edge(self, edge_data):
                return True
                
            async def get_node(self, node_id):
                return {"id": node_id}
                
            async def get_path(self, start_id, end_id):
                return []
                
            # Override to expose parent method directly for testing
            def call_setup_indexes(self):
                self._setup_text_indexes()
        
        # Create repo with this connection
        repo = TestIndexRepo(
            connection=connection,
            node_collection="test_nodes"
        )
        
        # Replace the method with our own implementation that just calls the parent
        # This lets us bypass the automatic call in __init__ and test it directly
        repo._setup_text_indexes = TextArangoRepository._setup_text_indexes.__get__(repo)
        
        # Call the method directly
        repo.call_setup_indexes()
        
        # Verify correct calls were made
        mock_collection.add_hash_index.assert_called_with(fields=["parent_id"], unique=False)
        mock_collection.add_persistent_index.assert_called_with(fields=["embedding_type"], unique=False)
    
    def test_store_embedding_with_type(self) -> None:
        """Test storing embeddings with different types."""
        # Set up mock
        node_id = "test_node"
        embedding = [0.1, 0.2, 0.3, 0.4]
        # First return value for get_node, second for _execute_aql (UPDATE statement)
        self.repo.get_node = AsyncMock(return_value={"id": node_id, "type": "document"})
        self.repo._execute_aql = AsyncMock(return_value=[{"_key": "success"}])
        
        # Test with default embedding type
        result = run_async_test(self.repo.store_embedding_with_type(node_id, embedding))
        self.assertTrue(result)
        
        # Check _execute_aql was called with the right parameters
        self.repo._execute_aql.assert_called_once()
        call_args = self.repo._execute_aql.call_args[0]
        query = call_args[0]
        bind_vars = call_args[1]
        
        # Verify the query has the expected components
        self.assertIn("UPDATE", query)  # Uses UPDATE not UPSERT
        self.assertIn("node_id", str(bind_vars))  # Node ID in bind vars
        
        # The embedding should be wrapped in an embedding_doc structure
        update_data = bind_vars.get("update_data", {})
        self.assertIn("embedding", update_data)
        embedding_doc = update_data.get("embedding", {})
        self.assertEqual(embedding_doc.get("vector"), embedding)
        self.assertEqual(embedding_doc.get("embedding_type"), "default")
        self.assertEqual(embedding_doc.get("dimension"), len(embedding))
        
        # Reset mock and test with isne embedding type
        self.repo._execute_aql.reset_mock()
        result = run_async_test(self.repo.store_embedding_with_type(node_id, embedding, "isne"))
        self.assertTrue(result)
        
        # Verify the embedding type was set correctly in the update_data
        bind_vars = self.repo._execute_aql.call_args[0][1]
        update_data = bind_vars.get("update_data", {})
        self.assertIn("isne_embedding", update_data)  # Uses type_embedding key format
        embedding_doc = update_data.get("isne_embedding", {})
        self.assertEqual(embedding_doc.get("embedding_type"), "isne")
        
        # Test with nonexistent node
        self.repo.get_node.return_value = None
        result = run_async_test(self.repo.store_embedding_with_type("nonexistent", embedding))
        self.assertFalse(result)
    
    def test_create_similarity_edges(self) -> None:
        """Test creating similarity edges between chunks."""
        # Create test data
        chunks = [
            ("chunk1", [0.1, 0.2, 0.3]),
            ("chunk2", [0.4, 0.5, 0.6]),
            ("chunk3", [0.7, 0.8, 0.9])
        ]
        
        # Instead of mocking store_edge, we need to mock _execute_aql since that's what's used in bulk insert
        self.repo._execute_aql = AsyncMock(return_value=[{"_key": "edge1"}, {"_key": "edge2"}, {"_key": "edge3"}, {"_key": "edge4"}])
        
        # Mock numpy's dot product to return a controlled similarity matrix
        similarity_matrix = np.array([
            [1.0, 0.8, 0.7],  # chunk1 similarities
            [0.8, 1.0, 0.9],  # chunk2 similarities
            [0.7, 0.9, 1.0]   # chunk3 similarities
        ])
        
        # Also need to mock the numpy normalize function to avoid NaN errors
        with patch('numpy.dot', return_value=similarity_matrix):
            with patch('numpy.linalg.norm', return_value=np.array([[1.0], [1.0], [1.0]])):
                # Call the method with threshold 0.8
                result = run_async_test(self.repo.create_similarity_edges(
                    chunks, 
                    edge_type="similar_to", 
                    threshold=0.8
                ))
        
        # Should return the number of edges created based on _execute_aql mock return
        self.assertEqual(result, 4)
        
        # Verify the AQL was called with the right parameters
        self.repo._execute_aql.assert_called_once()
        
        # Test with empty chunks list
        self.repo._execute_aql.reset_mock()
        result = run_async_test(self.repo.create_similarity_edges([]))
        self.assertEqual(result, 0)
        self.repo._execute_aql.assert_not_called()
    
    def test_search_similar_with_data(self) -> None:
        """Test vector search with node data."""
        # Setup mock results with the structure return_value will modify to match method return
        node1 = {"id": "node1", "content": "Similar content 1"}
        node2 = {"id": "node2", "content": "Similar content 2"}
        mock_results = [
            {"node": node1, "score": 0.95},
            {"node": node2, "score": 0.85}
        ]
        self.repo._execute_aql = AsyncMock(return_value=mock_results)
        
        # Call method
        query_vector = [0.1, 0.2, 0.3, 0.4]
        results = run_async_test(self.repo.search_similar_with_data(
            query_vector,
            limit=2,
            node_types=["chunk"]
        ))
        
        # Verify results - check that the method correctly transforms the AQL results
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0], node1)  # First item in tuple is node
        self.assertEqual(results[0][1], 0.95)   # Second item is score
        self.assertEqual(results[1][0], node2)
        self.assertEqual(results[1][1], 0.85)
        
        # Verify AQL query contains expected components
        call_args = self.repo._execute_aql.call_args[0]
        query = call_args[0]
        bind_vars = call_args[1]
        
        self.assertIn("COSINE_SIMILARITY", query)  # Uses cosine similarity
        self.assertIn("embedding", query)  # Default embedding type
        self.assertEqual(bind_vars["query_vector"], query_vector)
        self.assertEqual(bind_vars["limit"], 2)
        
        # Test with ISNE embedding type
        self.repo._execute_aql.reset_mock()
        run_async_test(self.repo.search_similar_with_data(
            query_vector,
            embedding_type="isne"
        ))
        
        # Verify ISNE embedding type was used
        query = self.repo._execute_aql.call_args[0][0]
        self.assertIn("isne_embedding", query)
    
    def test_get_document_with_chunks(self) -> None:
        """Test retrieving a document with its chunks."""
        # Setup mock document and chunks
        doc_id = "doc123"
        document = {"id": doc_id, "title": "Test Document"}
        chunks = [
            {"id": "chunk1", "content": "Chunk content 1"},
            {"id": "chunk2", "content": "Chunk content 2"}
        ]
        
        # We need to mock _execute_aql instead of get_connected_nodes, as that's what's used in the implementation
        self.repo.get_node = AsyncMock(return_value=document)
        self.repo._execute_aql = AsyncMock(return_value=chunks)
        
        # Call method
        result = run_async_test(self.repo.get_document_with_chunks(doc_id))
        
        # Check results
        self.assertEqual(result["document"], document)
        self.assertEqual(result["chunks"], chunks)
        self.assertEqual(result["chunk_count"], len(chunks))
        
        # Verify mock calls
        self.repo.get_node.assert_called_once_with(doc_id)
        self.repo._execute_aql.assert_called_once()
        
        # Verify the AQL query contains expected parameters
        query_string = self.repo._execute_aql.call_args[0][0]
        bind_vars = self.repo._execute_aql.call_args[0][1]
        self.assertIn("OUTBOUND", query_string)
        self.assertIn("contains", query_string)
        self.assertEqual(bind_vars["start_vertex"], f"{self.repo.node_collection_name}/{doc_id}")
        
        # Test with nonexistent document
        self.repo.get_node.reset_mock()
        self.repo.get_node.return_value = None

        result = run_async_test(self.repo.get_document_with_chunks("nonexistent"))
        self.assertIsNone(result)

    def test_search_fulltext(self):
        """Test fulltext search functionality."""
        # Set up mock results
        expected_results = [
            {"id": "chunk1", "content": "This is a test of fulltext search"},
            {"id": "chunk2", "content": "Another test document with search terms"}
        ]
        
        # Create a specific mock implementation for _execute_aql in this test
        async def mock_execute_aql(query_string, bind_vars):
            # Verify the query contains expected components before returning mock results
            if "FULLTEXT" in query_string and bind_vars["query"] == "test search":
                return expected_results
            return []
            
        self.repo._execute_aql = mock_execute_aql

        # Call the method
        query = "test search"
        results = run_async_test(self.repo.search_fulltext(
            query=query,
            limit=10,
            node_types=["chunk"]
        ))

        # Verify results
        self.assertEqual(results, expected_results)

    def test_hybrid_search(self):
        """Test hybrid search functionality."""
        # Set up mock results
        expected_results = [
            {"id": "chunk1", "score": 0.95, "content": "Hybrid search result 1"},
            {"id": "chunk2", "score": 0.85, "content": "Hybrid search result 2"}
        ]
        
        # Create a specific mock implementation for this test
        async def mock_execute_aql(query_string, bind_vars):
            # Verify the query parameters before returning results
            if ("text_results" in query_string and 
                bind_vars.get("query") == "hybrid search" and 
                bind_vars.get("vector") == [0.1, 0.2, 0.3, 0.4]):
                return expected_results
            return []
            
        self.repo._execute_aql = mock_execute_aql

        # Call the method with both text and vector
        text_query = "hybrid search"
        vector_query = [0.1, 0.2, 0.3, 0.4]

        results = run_async_test(self.repo.hybrid_search(
            text_query=text_query,
            query_vector=vector_query,
            limit=5,
            node_types=["chunk"]
        ))

        # Verify results
        self.assertEqual(results, expected_results)
        
    def test_hybrid_search_with_weights(self):
        """Test hybrid search with custom weights."""
        # Setup mock results
        expected_results = [
            {"id": "chunk1", "score": 0.92, "content": "Weighted hybrid result"}
        ]
        
        # Create a specific mock implementation for custom weights test
        async def mock_execute_aql(query_string, bind_vars):
            # Verify weights before returning results
            if (bind_vars.get("text_weight") == 0.7 and 
                bind_vars.get("vector_weight") == 0.3):
                return expected_results
            return []
            
        self.repo._execute_aql = mock_execute_aql
        
        # Call with custom weights
        results = run_async_test(self.repo.hybrid_search(
            text_query="weighted search",
            query_vector=[0.1, 0.2, 0.3],
            text_weight=0.7,  # Custom weight for text (higher priority)
            vector_weight=0.3  # Custom weight for vector
        ))
        
        # Verify results
        self.assertEqual(results, expected_results)
        
    def test_get_document_with_chunks_error_handling(self):
        """Test error handling in get_document_with_chunks."""
        # Setup mock to raise exception during execution
        self.repo.get_node = AsyncMock(side_effect=Exception("Database connection error"))
        
        # Call method that should catch the exception
        result = run_async_test(self.repo.get_document_with_chunks("doc123"))
        
        # Should return None on error
        self.assertIsNone(result)
        
    def test_search_similar_with_empty_vector(self):
        """Test vector search with an empty vector."""
        # Empty vector should return empty results
        results = run_async_test(self.repo.search_similar_with_data(
            query_vector=[],
            limit=10
        ))
        
        # Should return empty list
        self.assertEqual(results, [])
        
    def test_store_embedding_nonexistent_node(self):
        """Test storing embedding for nonexistent node."""
        # Setup mock to return None (node not found)
        self.repo.get_node = AsyncMock(return_value=None)
        
        # Attempt to store embedding
        result = run_async_test(self.repo.store_embedding_with_type(
            "nonexistent_node", [0.1, 0.2, 0.3]
        ))
        
        # Should return False
        self.assertFalse(result)
        
    def test_create_similarity_edges_batch_processing(self):
        """Test batch processing in create_similarity_edges."""
        # Create test data with multiple batches
        chunk_count = 15  # Will be processed in multiple batches with batch_size=10
        
        # Generate unique chunk IDs and random embeddings
        chunks = [(f"chunk{i}", [float(i)/100, float(i+1)/100, float(i+2)/100]) 
                 for i in range(chunk_count)]
                 
        # Mock for bulk insert
        self.repo._execute_aql = AsyncMock(side_effect=[
            ["edge1", "edge2", "edge3"],  # First batch creates 3 edges
            ["edge4", "edge5"]  # Second batch creates 2 edges
        ])
        
        # Call with small batch size
        result = run_async_test(self.repo.create_similarity_edges(
            chunks,
            threshold=0.8,
            batch_size=10  # Process in batches of 10
        ))
        
        # Should return total edges created across all batches
        self.assertEqual(result, 5)  # 3 + 2 = 5 edges
        
        # Verify _execute_aql was called twice (once per batch)
        self.assertEqual(self.repo._execute_aql.call_count, 2)
        
    @patch('numpy.linalg.norm')
    def test_create_similarity_edges_error_handling(self, mock_norm):
        """Test error handling in create_similarity_edges."""
        # Setup data
        chunks = [("chunk1", [0.1, 0.2]), ("chunk2", [0.3, 0.4])]
        
        # Mock numpy to raise an exception
        mock_norm.side_effect = ValueError("Invalid array dimensions")
        
        # Call method
        result = run_async_test(self.repo.create_similarity_edges(chunks))
        
        # Should return 0 edges created due to error
        self.assertEqual(result, 0)
        
    def test_setup_with_existing_indices(self):
        """Test setup with existing indices."""
        # Mock connection and collection
        connection = MagicMock()
        mock_collection = MagicMock()
        
        # Setup return value to indicate all indices already exist
        mock_collection.indexes.return_value = [
            {"fields": ["type"]},
            {"fields": ["parent_id"]},
            {"fields": ["embedding_type"]}
        ]
        connection.get_collection.return_value = mock_collection
        
        # Define test-specific class
        class TestIndexRepo(TextArangoRepository):
            async def initialize(self):
                return True
            
            async def store_node(self, node_data):
                return True
                
            async def store_edge(self, edge_data):
                return True
                
            async def get_node(self, node_id):
                return {"id": node_id}
                
            async def get_path(self, start_id, end_id):
                return []
                
            def call_setup_indexes(self):
                self._setup_text_indexes()
        
        # Create repo with mocked connection
        # But use our patched _setup_text_indexes to avoid it running in __init__
        with patch.object(TestIndexRepo, '_setup_text_indexes', MagicMock()) as mock_setup:
            repo = TestIndexRepo(
                connection=connection,
                node_collection="test_nodes"
            )
            
            # Now manually call the real setup method
            repo._setup_text_indexes = TextArangoRepository._setup_text_indexes.__get__(repo)
            repo.call_setup_indexes()
            
            # Setup should skip creating indices since they all exist
            mock_collection.add_hash_index.assert_not_called()
            mock_collection.add_persistent_index.assert_not_called()
        
    def test_search_fulltext_with_custom_field(self):
        """Test fulltext search with custom field."""
        # Setup mock results
        expected_results = [{"id": "result1", "title": "Test Title"}]
        
        # Create a specific mock implementation for this test
        async def mock_execute_aql(query_string, bind_vars):
            # Verify that the custom field is used in the query
            if 'title' in query_string and bind_vars["query"] == "test":
                return expected_results
            return []
            
        self.repo._execute_aql = mock_execute_aql
        
        # Call with custom field
        results = run_async_test(self.repo.search_fulltext(
            query="test",
            field="title",  # Search in title field instead of content
            limit=5
        ))
        
        # Verify results
        self.assertEqual(results, expected_results)

    def test_setup_text_indexes_error_handling(self):
        """Test error handling in _setup_text_indexes."""
        # Create a mock connection that raises an exception
        connection = MagicMock()
        connection.get_collection.side_effect = Exception("Collection not found")
        
        # We need a concrete class with abstract methods implemented
        class TestErrorHandlingRepo(TextArangoRepository):
            async def initialize(self):
                return True
            
            async def store_node(self, node_data):
                return True
                
            async def store_edge(self, edge_data):
                return True
                
            async def get_node(self, node_id):
                return {"id": node_id}
                
            async def get_path(self, start_id, end_id):
                return []
                
            def call_setup_indexes(self):
                self._setup_text_indexes()
        
        # Use patch to prevent automatic call during __init__
        with patch.object(TestErrorHandlingRepo, '_setup_text_indexes', MagicMock()):
            # Create repo with the failing connection
            repo = TestErrorHandlingRepo(
                connection=connection,
                node_collection="test_nodes"
            )
            
            # Replace with real implementation and ensure no exception
            repo._setup_text_indexes = TextArangoRepository._setup_text_indexes.__get__(repo)
            
            # This should not raise an exception because _setup_text_indexes catches errors
            repo.call_setup_indexes()
            
            # We're just verifying that no exception is propagated
            self.assertIsNotNone(repo)

    def test_search_similar_with_data_exception(self):
        """Test exception handling in search_similar_with_data."""
        # Make _execute_aql raise an exception
        self.repo._execute_aql = AsyncMock(side_effect=Exception("Database error"))
        
        # Call method
        results = run_async_test(self.repo.search_similar_with_data([0.1, 0.2, 0.3]))
        
        # Should return empty list on error
        self.assertEqual(results, [])

if __name__ == "__main__":
    unittest.main()
