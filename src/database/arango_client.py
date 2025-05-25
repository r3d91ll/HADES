"""
ArangoDB client for HADES-PathRAG.

This module provides a client for interacting with ArangoDB,
handling connection, database operations, and error recovery.
"""

import logging
import time
import json
from typing import Dict, Any, List, Optional, Union, Tuple, Callable, TypeVar, cast, Set

# Import ArangoDB Python driver
try:
    from arango import ArangoClient as BaseArangoClient
    from arango.database import Database
    from arango.collection import Collection
    from arango.exceptions import (
        ArangoError, 
        DocumentInsertError,
        DocumentUpdateError,
        DocumentReplaceError,
        DocumentDeleteError,
        DocumentRevisionError
    )
except ImportError:
    raise ImportError(
        "The 'python-arango' package is required. "
        "Install it with: pip install python-arango"
    )

# Import database configuration
from src.config.database_config import get_connection_params

# Type variable for retry operation
T = TypeVar('T')

class ArangoClient:
    """ArangoDB client for HADES-PathRAG.
    
    This class provides methods for interacting with ArangoDB,
    handling connection, database operations, and error recovery.
    """
    
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
        use_ssl: Optional[bool] = None,
        timeout: Optional[int] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **kwargs: Any
    ):
        """Initialize ArangoDB client.
        
        If connection parameters are not provided, they will be loaded
        from the configuration.
        
        Args:
            host: Database host
            port: Database port
            username: Database username
            password: Database password
            database: Default database name
            use_ssl: Whether to use SSL
            timeout: Connection timeout in seconds
            max_retries: Maximum number of retries for operations
            retry_delay: Initial delay between retries in seconds
            **kwargs: Additional connection parameters
        """
        # Setup logger
        self.logger = logging.getLogger(__name__)
        
        # Store credentials for later use
        self.username = username
        self.password = password
        
        # Get connection parameters from config if not provided
        if not all([host, port, username, password]):
            config = get_connection_params()
            host = host or config.get('host')
            port = port or config.get('port')
            username = username or config.get('username')
            password = password or config.get('password')
            database = database or config.get('database')
            use_ssl = use_ssl if use_ssl is not None else config.get('use_ssl', False)
            timeout = timeout or config.get('timeout', 30)
        
        # Store retry configuration
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Initialize client with proper URL format
        protocol = "https" if use_ssl else "http"
        url = f"{protocol}://{host}:{port}"
        self.logger.info(f"Connecting to ArangoDB at {url}")
        
        self.client = BaseArangoClient(
            hosts=url,
            http_client=kwargs.get('http_client'),
            serializer=kwargs.get('serializer'),
            deserializer=kwargs.get('deserializer')
        )
        
        # Get system database for administrative operations
        self.sys_db = self.client.db(
            name='_system',
            username=username,
            password=password,
            timeout=timeout
        )
        
        try:
            self.sys_db.properties()
            self.logger.info(f"Connected to ArangoDB at {url}")
        except Exception as e:
            self.logger.error(f"Failed to connect to ArangoDB: {str(e)}")
            raise

    def database_exists(self, database_name: str) -> bool:
        """Check if a database exists.
        
        Args:
            database_name: Name of the database to check
            
        Returns:
            True if database exists, False otherwise
        """
        try:
            result = self.sys_db.has_database(database_name)
            return bool(result)  # Ensure boolean return type
        except Exception as e:
            self.logger.error(f"Failed to check if database {database_name} exists: {str(e)}")
            return False

    def create_database(self, database_name: str) -> bool:
        """Create a database if it doesn't exist.
        
        Args:
            database_name: Name of the database to create
            
        Returns:
            True if database was created or already exists, False otherwise
        """
        try:
            if self.database_exists(database_name):
                self.logger.info(f"Database {database_name} already exists")
                return True
            
            result = self.sys_db.create_database(database_name)
            self.logger.info(f"Created database {database_name}")
            return bool(result)  # Ensure boolean return type
        except Exception as e:
            self.logger.error(f"Failed to create database {database_name}: {str(e)}")
            return False

    def delete_database(self, database_name: str) -> bool:
        """Delete a database.
        
        Args:
            database_name: Name of the database to delete
            
        Returns:
            True if database was deleted, False otherwise
        """
        try:
            if not self.database_exists(database_name):
                self.logger.info(f"Database {database_name} does not exist")
                return True
            
            self.sys_db.delete_database(database_name)
            self.logger.info(f"Deleted database {database_name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete database {database_name}: {str(e)}")
            return False

    def get_database(self, database_name: str) -> Database:
        """Get a database object.
        
        Args:
            database_name: Name of the database
            
        Returns:
            Database object
            
        Raises:
            Exception: If database doesn't exist
        """
        try:
            if not self.database_exists(database_name):
                self.create_database(database_name)
            
            db = self.client.db(
                name=database_name,
                username=self.username,
                password=self.password
            )
            
            return db
        except Exception as e:
            self.logger.error(f"Failed to get database {database_name}: {str(e)}")
            raise

    def collection_exists(self, database_name: str, collection_name: str) -> bool:
        """Check if a collection exists in a database.
        
        Args:
            database_name: Name of the database
            collection_name: Name of the collection
            
        Returns:
            True if collection exists, False otherwise
        """
        try:
            db = self.get_database(database_name)
            result = db.has_collection(collection_name)
            return bool(result)  # Ensure boolean return type
        except Exception as e:
            self.logger.error(
                f"Failed to check if collection {collection_name} exists in database {database_name}: {str(e)}"
            )
            return False
            
    def create_collection(self, database_name: str, collection_name: str, edge: bool = False, **kwargs: Any) -> bool:
        """Create a collection in a database if it doesn't exist.
        
        Args:
            database_name: Name of the database
            collection_name: Name of the collection
            edge: Whether to create an edge collection
            **kwargs: Additional collection properties
            
        Returns:
            True if collection was created or already exists, False otherwise
        """
        try:
            db = self.get_database(database_name)
            
            if db.has_collection(collection_name):
                self.logger.info(f"Collection {collection_name} already exists in database {database_name}")
                return True
            
            result = db.create_collection(collection_name, edge=edge, **kwargs)
            self.logger.info(f"Created {'edge ' if edge else ''}collection {collection_name} in database {database_name}")
            return bool(result)  # Ensure boolean return type
        except Exception as e:
            self.logger.error(
                f"Failed to create collection {collection_name} in database {database_name}: {str(e)}"
            )
            return False
    
    def delete_collection(self, database_name: str, collection_name: str) -> bool:
        """Delete a collection from a database.
        
        Args:
            database_name: Name of the database
            collection_name: Name of the collection
            
        Returns:
            True if collection was deleted or doesn't exist, False otherwise
        """
        try:
            db = self.get_database(database_name)
            
            if not db.has_collection(collection_name):
                self.logger.info(f"Collection {collection_name} does not exist in database {database_name}")
                return True
            
            db.delete_collection(collection_name)
            self.logger.info(f"Deleted collection {collection_name} from database {database_name}")
            return True
        except Exception as e:
            self.logger.error(
                f"Failed to delete collection {collection_name} from database {database_name}: {str(e)}"
            )
            return False
    
    def get_collections(self, database_name: str) -> List[Dict[str, Any]]:
        """Get a list of collections in a database.
        
        Args:
            database_name: Name of the database
            
        Returns:
            List of collection information
        """
        try:
            db = self.get_database(database_name)
            collections = []
            
            for collection_name in db.collections():
                if not collection_name.startswith('_'):
                    try:
                        collection = db.collection(collection_name)
                        collections.append({
                            "name": collection_name,
                            "type": "edge" if collection.properties()["type"] == 3 else "document",
                            "count": collection.count()
                        })
                    except Exception as e:
                        self.logger.warning(f"Error getting details for collection {collection_name}: {str(e)}")
            
            return collections
        except Exception as e:
            self.logger.error(f"Failed to get collections in database {database_name}: {str(e)}")
            return []
    
    def list_documents(self, database_name: str, collection_name: str, limit: int = 100) -> List[Dict[str, Any]]:
        """List documents in a collection.
        
        Args:
            database_name: Name of the database
            collection_name: Name of the collection
            limit: Maximum number of documents to return
            
        Returns:
            List of documents
        """
        try:
            db = self.get_database(database_name)
            
            if not db.has_collection(collection_name):
                self.logger.warning(f"Collection {collection_name} does not exist in database {database_name}")
                return []
            
            collection = db.collection(collection_name)
            cursor = collection.all(limit=limit)
            return list(cursor)
        except Exception as e:
            self.logger.error(
                f"Failed to list documents in collection {collection_name}: {str(e)}"
            )
            return []
    
    def count_documents(self, database_name: str, collection_name: str) -> int:
        """Count documents in a collection.
        
        Args:
            database_name: Name of the database
            collection_name: Name of the collection
            
        Returns:
            Number of documents in the collection
        """
        try:
            db = self.get_database(database_name)
            
            if not db.has_collection(collection_name):
                return 0
            
            collection = db.collection(collection_name)
            result = collection.count()
            return int(result)  # Ensure integer return type
        except Exception as e:
            self.logger.error(
                f"Failed to count documents in collection {collection_name}: {str(e)}"
            )
            return 0
    
    def get_collection_count(self, database_name: str, collection_name: str) -> int:
        """Get the number of documents in a collection (alias for count_documents).
        
        Args:
            database_name: Name of the database
            collection_name: Name of the collection
            
        Returns:
            Number of documents in the collection
        """
        return self.count_documents(database_name, collection_name)
    
    def get_document(self, database_name: str, collection_name: str, document_key: str) -> Optional[Dict[str, Any]]:
        """Get a document from a collection.
        
        Args:
            database_name: Name of the database
            collection_name: Name of the collection
            document_key: Key of the document to get
            
        Returns:
            Document or None if not found
        """
        try:
            db = self.get_database(database_name)
            
            if not db.has_collection(collection_name):
                self.logger.warning(f"Collection {collection_name} does not exist in database {database_name}")
                return None
            
            collection = db.collection(collection_name)
            
            if not collection.has(document_key):
                return None
            
            result = collection.get(document_key)
            # Ensure we return a Dict or None
            return dict(result) if result else None
        except Exception as e:
            self.logger.error(
                f"Failed to get document {document_key} from collection {collection_name}: {str(e)}"
            )
            return None
    
    def document_exists(self, database_name: str, collection_name: str, document_key: str) -> bool:
        """Check if a document exists in a collection.
        
        Args:
            database_name: Name of the database
            collection_name: Name of the collection
            document_key: Key of the document to check
            
        Returns:
            True if document exists, False otherwise
        """
        try:
            db = self.get_database(database_name)
            
            if not db.has_collection(collection_name):
                return False
                
            collection = db.collection(collection_name)
            result = collection.has(document_key)
            return bool(result)  # Ensure boolean return type
        except Exception as e:
            self.logger.error(
                f"Failed to check if document {document_key} exists in {database_name}.{collection_name}: {str(e)}"
            )
            return False
    
    def insert_document(self, database_name: str, collection_name: str, document: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Insert a document into a collection.
        
        Args:
            database_name: Name of the database
            collection_name: Name of the collection
            document: Document to insert
            
        Returns:
            Inserted document with _id, _key, and _rev or None if insertion failed
        """
        try:
            return self._retry_operation(
                lambda: self._insert_document(database_name, collection_name, document),
                f"insert_document({collection_name})"
            )
        except Exception as e:
            self.logger.error(f"Failed to insert document into {collection_name}: {str(e)}")
            return None
    
    def _insert_document(self, database_name: str, collection_name: str, document: Dict[str, Any]) -> Dict[str, Any]:
        """Internal method to insert a document.
        
        Args:
            database_name: Name of the database
            collection_name: Name of the collection
            document: Document to insert
            
        Returns:
            Inserted document with _id, _key, and _rev
            
        Raises:
            Exception: If document insertion fails
        """
        db = self.get_database(database_name)
        
        # Create collection if it doesn't exist
        if not db.has_collection(collection_name):
            db.create_collection(collection_name)
            self.logger.info(f"Created collection {collection_name} in database {database_name}")
        
        collection = db.collection(collection_name)
        
        # Insert document
        result = collection.insert(document)
        self.logger.debug(f"Inserted document into {database_name}.{collection_name}: {result['_key']}")
        
        # Get the inserted document
        doc = collection.get(result['_key'])
        return dict(doc) if doc else {}
    
    def insert_documents(self, database_name: str, collection_name: str, documents: List[Dict[str, Any]]) -> List[Optional[Dict[str, Any]]]:
        """Insert multiple documents into a collection.
        
        Args:
            database_name: Name of the database
            collection_name: Name of the collection
            documents: List of document dictionaries to insert
            
        Returns:
            List of metadata for inserted documents (same length as documents)
        """
        if not documents:
            return []
        
        try:
            # Convert return type to match method signature
            result: List[Optional[Dict[str, Any]]] = self._retry_operation(
                lambda: self._insert_documents(database_name, collection_name, documents),
                f"insert_documents({collection_name}, {len(documents)} docs)"
            )
            return result
        except Exception as e:
            self.logger.error(f"Failed to insert documents into {collection_name}: {str(e)}")
            return [None] * len(documents)
    
    def _insert_documents(self, database_name: str, collection_name: str, documents: List[Dict[str, Any]]) -> List[Optional[Dict[str, Any]]]:
        """Internal method to insert multiple documents.
        
        Args:
            database_name: Name of the database
            collection_name: Name of the collection
            documents: List of documents to insert
            
        Returns:
            List of inserted documents with _id, _key, and _rev (or None for failed inserts)
            
        Raises:
            Exception: If document insertion fails
        """
        db = self.get_database(database_name)
        
        # Create collection if it doesn't exist
        if not db.has_collection(collection_name):
            db.create_collection(collection_name)
            self.logger.info(f"Created collection {collection_name} in database {database_name}")
        
        collection = db.collection(collection_name)
        
        # Insert documents in bulk
        results = collection.insert_many(documents)
        self.logger.info(f"Inserted {len(results)} documents into {database_name}.{collection_name}")
        
        # Get the inserted documents
        inserted_docs = []
        for result in results:
            try:
                inserted_docs.append(collection.get(result['_key']))
            except Exception as e:
                self.logger.warning(f"Failed to get inserted document {result['_key']}: {str(e)}")
                inserted_docs.append(None)
        
        return inserted_docs
    
    def update_document(self, database_name: str, collection_name: str, document_key: str, document: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update a document in a collection.
        
        Args:
            database_name: Name of the database
            collection_name: Name of the collection
            document_key: Key of the document to update
            document: New document data
            
        Returns:
            Updated document with _id, _key, and _rev or None if update failed
        """
        try:
            return self._retry_operation(
                lambda: self._update_document(database_name, collection_name, document_key, document),
                f"update_document({collection_name}, {document_key})"
            )
        except Exception as e:
            self.logger.error(f"Failed to update document {document_key} in {collection_name}: {str(e)}")
            return None
    
    def _update_document(self, database_name: str, collection_name: str, document_key: str, document: Dict[str, Any]) -> Dict[str, Any]:
        """Internal method to update a document.
        
        Args:
            database_name: Name of the database
            collection_name: Name of the collection
            document_key: Key of the document to update
            document: New document data
            
        Returns:
            Updated document with _id, _key, and _rev
            
        Raises:
            Exception: If document update fails
        """
        db = self.get_database(database_name)
        
        if not db.has_collection(collection_name):
            raise ValueError(f"Collection {collection_name} does not exist in database {database_name}")
        
        collection = db.collection(collection_name)
        
        if not collection.has(document_key):
            raise ValueError(f"Document {document_key} does not exist in collection {collection_name}")
        
        # Make sure document has the correct key
        document["_key"] = document_key
        
        # Update document
        collection.update(document)
        self.logger.debug(f"Updated document {document_key} in {database_name}.{collection_name}")
        
        # Get the updated document
        doc = collection.get(document_key)
        return dict(doc) if doc else {}
    
    def replace_document(self, database_name: str, collection_name: str, document_key: str, document: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Replace a document in a collection.
        
        Args:
            database_name: Name of the database
            collection_name: Name of the collection
            document_key: Key of the document to replace
            document: New document
            
        Returns:
            Replaced document with _id, _key, and _rev or None if replacement failed
        """
        try:
            return self._retry_operation(
                lambda: self._replace_document(database_name, collection_name, document_key, document),
                f"replace_document({collection_name}, {document_key})"
            )
        except Exception as e:
            self.logger.error(f"Failed to replace document {document_key} in {collection_name}: {str(e)}")
            return None
    
    def _replace_document(self, database_name: str, collection_name: str, document_key: str, document: Dict[str, Any]) -> Dict[str, Any]:
        """Internal method to replace a document.
        
        Args:
            database_name: Name of the database
            collection_name: Name of the collection
            document_key: Key of the document to replace
            document: New document
            
        Returns:
            Replaced document with _id, _key, and _rev
            
        Raises:
            Exception: If document replacement fails
        """
        db = self.get_database(database_name)
        
        if not db.has_collection(collection_name):
            raise ValueError(f"Collection {collection_name} does not exist in database {database_name}")
        
        collection = db.collection(collection_name)
        
        if not collection.has(document_key):
            raise ValueError(f"Document {document_key} does not exist in collection {collection_name}")
        
        # Make sure document has the correct key
        document["_key"] = document_key
        
        # Replace document
        collection.replace(document)
        self.logger.debug(f"Replaced document {document_key} in {database_name}.{collection_name}")
        
        # Get the replaced document
        doc = collection.get(document_key)
        return dict(doc) if doc else {}
    
    def delete_document(self, database_name: str, collection_name: str, document_key: str) -> bool:
        """Delete a document from a collection.
        
        Args:
            database_name: Name of the database
            collection_name: Name of the collection
            document_key: Key of the document to delete
            
        Returns:
            True if document was deleted, False otherwise
        """
        try:
            return self._retry_operation(
                lambda: self._delete_document(database_name, collection_name, document_key),
                f"delete_document({collection_name}, {document_key})"
            )
        except Exception as e:
            self.logger.error(f"Failed to delete document {document_key} from {collection_name}: {str(e)}")
            return False
    
    def _delete_document(self, database_name: str, collection_name: str, document_key: str) -> bool:
        """Internal method to delete a document.
        
        Args:
            database_name: Name of the database
            collection_name: Name of the collection
            document_key: Key of the document to delete
            
        Returns:
            True if document was deleted, False otherwise
            
        Raises:
            Exception: If document deletion fails
        """
        db = self.get_database(database_name)
        
        if not db.has_collection(collection_name):
            self.logger.info(f"Collection {collection_name} does not exist in database {database_name}")
            return True
        
        collection = db.collection(collection_name)
        
        if not collection.has(document_key):
            self.logger.info(f"Document {document_key} does not exist in collection {collection_name}")
            return True
        
        # Delete document
        collection.delete(document_key)
        self.logger.debug(f"Deleted document {document_key} from {database_name}.{collection_name}")
        
        return True
    
    def has_edge(self, database_name: str, collection_name: str, from_vertex: str, to_vertex: str) -> bool:
        """Check if an edge exists between two vertices.
        
        Args:
            database_name: Name of the database
            collection_name: Name of the edge collection
            from_vertex: ID of the source vertex
            to_vertex: ID of the target vertex
            
        Returns:
            True if edge exists, False otherwise
        """
        try:
            edge = self.get_edge(database_name, collection_name, from_vertex, to_vertex)
            return edge is not None
        except Exception as e:
            self.logger.error(
                f"Failed to check if edge exists from {from_vertex} to {to_vertex} in {collection_name}: {str(e)}"
            )
            return False
    
    def get_edge(self, database_name: str, collection_name: str, from_vertex: str, to_vertex: str) -> Optional[Dict[str, Any]]:
        """Get an edge between two vertices.
        
        Args:
            database_name: Name of the database
            collection_name: Name of the edge collection
            from_vertex: ID of the source vertex
            to_vertex: ID of the target vertex
            
        Returns:
            Edge document or None if not found
        """
        try:
            db = self.get_database(database_name)
            
            if not db.has_collection(collection_name):
                self.logger.warning(f"Collection {collection_name} does not exist in database {database_name}")
                return None
            
            # Query to find edge between vertices
            query = f"""
                FOR e IN {collection_name}
                FILTER e._from == @from_vertex AND e._to == @to_vertex
                LIMIT 1
                RETURN e
            """
            
            cursor = db.aql.execute(
                query,
                bind_vars={
                    "from_vertex": from_vertex,
                    "to_vertex": to_vertex
                }
            )
            
            # Get the first result (if any)
            results = list(cursor)
            if results and len(results) > 0:
                return dict(results[0]) if results[0] else None
            
            return None
        except Exception as e:
            self.logger.error(
                f"Failed to get edge from {from_vertex} to {to_vertex} in collection {collection_name}: {str(e)}"
            )
            return None
    
    def create_edge(self, database_name: str, collection_name: str, from_vertex: str, to_vertex: str, data: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Create an edge between two vertices.
        
        Args:
            database_name: Name of the database
            collection_name: Name of the edge collection
            from_vertex: ID of the source vertex
            to_vertex: ID of the target vertex
            data: Additional edge data
            
        Returns:
            Created edge document or None if creation failed
        """
        try:
            return self._retry_operation(
                lambda: self._create_edge(database_name, collection_name, from_vertex, to_vertex, data),
                f"create_edge({collection_name}, {from_vertex} -> {to_vertex})"
            )
        except Exception as e:
            self.logger.error(f"Failed to create edge from {from_vertex} to {to_vertex} in {collection_name}: {str(e)}")
            return None
    
    def _create_edge(self, database_name: str, collection_name: str, from_vertex: str, to_vertex: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Internal method to create an edge between two vertices.
        
        Args:
            database_name: Name of the database
            collection_name: Name of the edge collection
            from_vertex: ID of the source vertex
            to_vertex: ID of the target vertex
            data: Additional edge data
            
        Returns:
            Created edge document
            
        Raises:
            Exception: If edge creation fails
        """
        db = self.get_database(database_name)
        
        # Create collection if it doesn't exist
        if not db.has_collection(collection_name):
            db.create_collection(collection_name, edge=True)
            self.logger.info(f"Created edge collection {collection_name} in database {database_name}")
        
        collection = db.collection(collection_name)
        
        # Check if edge already exists
        existing_edge = self.get_edge(database_name, collection_name, from_vertex, to_vertex)
        if existing_edge:
            return dict(existing_edge)
        
        # Create edge document
        edge = {
            "_from": from_vertex,
            "_to": to_vertex
        }
        
        # Add additional data if provided
        if data:
            edge.update(data)
        
        # Insert edge
        result = collection.insert(edge)
        self.logger.info(f"Created edge from {from_vertex} to {to_vertex} in {database_name}.{collection_name}")
        
        # Get the created edge
        edge_key = result["_key"]
        doc = collection.get(edge_key)
        return dict(doc) if doc else {}
    
    def insert_edge(self, database_name: str, collection_name: str, from_vertex: str, to_vertex: str, data: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Insert an edge between two vertices (alias for create_edge).
        
        Args:
            database_name: Name of the database
            collection_name: Name of the edge collection
            from_vertex: ID of the source vertex
            to_vertex: ID of the target vertex
            data: Additional edge data
            
        Returns:
            Created edge document or None if creation failed
        """
        return self.create_edge(database_name, collection_name, from_vertex, to_vertex, data)
    
    def update_edge(self, database_name: str, collection_name: str, edge_key: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update an edge in a collection.
        
        Args:
            database_name: Name of the database
            collection_name: Name of the edge collection
            edge_key: Key of the edge to update
            data: New edge data
            
        Returns:
            Updated edge document or None if update failed
        """
        try:
            return self._retry_operation(
                lambda: self._update_edge(database_name, collection_name, edge_key, data),
                f"update_edge({collection_name}, {edge_key})"
            )
        except Exception as e:
            self.logger.error(f"Failed to update edge {edge_key} in {collection_name}: {str(e)}")
            return None
    
    def _update_edge(self, database_name: str, collection_name: str, edge_key: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Internal method to update an edge.
        
        Args:
            database_name: Name of the database
            collection_name: Name of the edge collection
            edge_key: Key of the edge to update
            data: New edge data
            
        Returns:
            Updated edge document
            
        Raises:
            Exception: If edge update fails
        """
        db = self.get_database(database_name)
        
        if not db.has_collection(collection_name):
            raise ValueError(f"Collection {collection_name} does not exist in database {database_name}")
        
        collection = db.collection(collection_name)
        
        if not collection.has(edge_key):
            raise ValueError(f"Edge {edge_key} does not exist in collection {collection_name}")
        
        # Make sure edge has the correct key
        data["_key"] = edge_key
        
        # Update edge
        collection.update(data)
        self.logger.info(f"Updated edge {edge_key} in {database_name}.{collection_name}")
        
        # Get the updated edge
        doc = collection.get(edge_key)
        return dict(doc) if doc else {}
    
    def delete_edges(self, database_name: str, collection_name: str, vertex_id: str, direction: Optional[str] = None) -> int:
        """Delete edges connected to a vertex.
        
        Args:
            database_name: Name of the database
            collection_name: Name of the edge collection
            vertex_id: ID of the vertex
            direction: Direction of edges to delete ('outbound', 'inbound', or None for both)
            
        Returns:
            Number of edges deleted
        """
        try:
            return self._retry_operation(
                lambda: self._delete_edges(database_name, collection_name, vertex_id, direction),
                f"delete_edges({collection_name}, {vertex_id}, {direction})"
            )
        except Exception as e:
            self.logger.error(f"Failed to delete edges for vertex {vertex_id} in {collection_name}: {str(e)}")
            return 0
    
    def _delete_edges(self, database_name: str, collection_name: str, vertex_id: str, direction: Optional[str] = None) -> int:
        """Internal method to delete edges connected to a vertex.
        
        Args:
            database_name: Name of the database
            collection_name: Name of the edge collection
            vertex_id: ID of the vertex
            direction: Direction of edges to delete ('outbound', 'inbound', or None for both)
            
        Returns:
            Number of edges deleted
            
        Raises:
            Exception: If edge deletion fails
        """
        db = self.get_database(database_name)
        
        if not db.has_collection(collection_name):
            return 0
        
        # Build query based on direction
        if direction == "outbound":
            query = f"""
                FOR e IN {collection_name}
                FILTER e._from == @vertex_id
                REMOVE e IN {collection_name}
                RETURN 1
            """
        elif direction == "inbound":
            query = f"""
                FOR e IN {collection_name}
                FILTER e._to == @vertex_id
                REMOVE e IN {collection_name}
                RETURN 1
            """
        else:  # both directions
            query = f"""
                FOR e IN {collection_name}
                FILTER e._from == @vertex_id OR e._to == @vertex_id
                REMOVE e IN {collection_name}
                RETURN 1
            """
        
        cursor = db.aql.execute(
            query,
            bind_vars={
                "vertex_id": vertex_id
            }
        )
        
        # Count number of deleted edges
        count = len(list(cursor))
        direction_str = f"{direction} " if direction else ""
        self.logger.info(f"Deleted {count} {direction_str}edges connected to {vertex_id} in {database_name}.{collection_name}")
        
        return count
    
    def execute_query(self, database_name: str, query: str, bind_vars: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute an AQL query.
        
        Args:
            database_name: Name of the database
            query: AQL query string
            bind_vars: Query bind variables
            
        Returns:
            Query results as a list of dictionaries
        """
        try:
            # Use explicit function for type safety
            def execute() -> List[Dict[str, Any]]:
                return self._execute_query(database_name, query, bind_vars)
                
            results = self._retry_operation(
                execute,
                f"execute_query({database_name})"
            )
            return results
        except Exception as e:
            self.logger.error(f"Failed to execute query: {str(e)}")
            return []
    
    def _execute_query(self, database_name: str, query: str, bind_vars: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Internal method to execute an AQL query.
        
        Args:
            database_name: Name of the database
            query: AQL query string
            bind_vars: Query bind variables
            
        Returns:
            Query results as a list of dictionaries
            
        Raises:
            Exception: If query execution fails
        """
        db = self.get_database(database_name)
        
        # Execute query
        cursor = db.aql.execute(query, bind_vars=bind_vars or {})
        
        # Convert cursor results to list of dictionaries
        result_list: List[Dict[str, Any]] = []
        for item in cursor:
            if item is not None:
                result_list.append(dict(item))
        
        return result_list
    
    def create_index(self, database_name: str, collection_name: str, index_type: str, fields: List[str], unique: bool = False) -> Dict[str, Any]:
        """Create an index on a collection.
        
        Args:
            database_name: Name of the database
            collection_name: Name of the collection
            index_type: Type of index (e.g., "hash", "skiplist", "fulltext", "geo")
            fields: Fields to index
            unique: Whether the index should enforce uniqueness
            
        Returns:
            Index information
        """
        try:
            db = self.get_database(database_name)
            
            if not db.has_collection(collection_name):
                raise ValueError(f"Collection {collection_name} does not exist in database {database_name}")
            
            collection = db.collection(collection_name)
            
            # Create index based on type
            if index_type == "hash":
                result = collection.add_hash_index(fields=fields, unique=unique)
            elif index_type == "skiplist":
                result = collection.add_skiplist_index(fields=fields, unique=unique)
            elif index_type == "fulltext":
                result = collection.add_fulltext_index(fields=fields)
            elif index_type == "geo":
                result = collection.add_geo_index(fields=fields)
            else:
                raise ValueError(f"Unsupported index type: {index_type}")
            
            self.logger.info(f"Created {index_type} index on {database_name}.{collection_name}: {fields}")
            return dict(result) if result else {}
        except Exception as e:
            self.logger.error(
                f"Failed to create {index_type} index on {database_name}.{collection_name}: {str(e)}"
            )
            raise
    
    def _retry_operation(self, operation: Callable[[], T], operation_name: str) -> T:
        """Retry an operation with exponential backoff.
        
        Args:
            operation: Function to retry
            operation_name: Name of the operation for logging
            
        Returns:
            Result of the operation
            
        Raises:
            Exception: If all retries fail
        """
        retries = 0
        last_error = None
        
        while retries <= self.max_retries:
            try:
                result = operation()
                # Explicitly return the typed result
                return result
            except Exception as e:
                last_error = e
                retries += 1
                
                if retries <= self.max_retries:
                    # Calculate delay with exponential backoff
                    delay = self.retry_delay * (2 ** (retries - 1))
                    
                    self.logger.warning(
                        f"Operation '{operation_name}' failed (attempt {retries}/{self.max_retries}), "
                        f"retrying in {delay:.2f}s: {str(e)}"
                    )
                    time.sleep(delay)
                else:
                    self.logger.error(
                        f"Operation '{operation_name}' failed after {retries} attempts: {str(e)}"
                    )
        
        # If we get here, all retries failed
        if last_error:
            raise last_error
        
        # This should never happen, but needed for type safety
        raise RuntimeError(f"Failed to execute {operation_name} after {self.max_retries} retries")
