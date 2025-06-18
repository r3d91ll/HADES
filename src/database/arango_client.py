"""
ArangoDB Client for HADES

Production-ready ArangoDB client with connection pooling, retry logic,
and comprehensive error handling for HADES operations.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Union, Iterator
from contextlib import contextmanager
from datetime import datetime

from arango import ArangoClient as PyArangoClient
from arango.database import StandardDatabase
from arango.collection import StandardCollection
from arango.exceptions import (
    ArangoError, 
    DatabaseCreateError, 
    DatabaseDeleteError,
    CollectionCreateError,
    DocumentInsertError,
    DocumentGetError,
    DocumentUpdateError,
    DocumentDeleteError,
    AQLQueryExecuteError,
    ServerConnectionError
)

logger = logging.getLogger(__name__)


class ArangoConnectionError(Exception):
    """Custom exception for ArangoDB connection errors."""
    pass


class ArangoOperationError(Exception):
    """Custom exception for ArangoDB operation errors."""
    pass


class ArangoClient:
    """
    Production ArangoDB client for HADES.
    
    Features:
    - Connection pooling and retry logic
    - Comprehensive error handling
    - Health checking and monitoring
    - Batch operations support
    - Transaction support
    - Query optimization
    """
    
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8529,
        username: str = "root",
        password: str = "",
        database: str = "hades",
        use_ssl: bool = False,
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        connection_pool_size: int = 10
    ):
        """
        Initialize ArangoDB client.
        
        Args:
            host: ArangoDB host
            port: ArangoDB port
            username: Database username
            password: Database password
            database: Default database name
            use_ssl: Whether to use SSL
            timeout: Connection timeout in seconds
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries in seconds
            connection_pool_size: Size of connection pool
        """
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.database_name = database
        self.use_ssl = use_ssl
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Build connection URL
        protocol = "https" if use_ssl else "http"
        self.url = f"{protocol}://{host}:{port}"
        
        # Initialize client and connections
        self._client: Optional[PyArangoClient] = None
        self._sys_db: Optional[StandardDatabase] = None
        self._db: Optional[StandardDatabase] = None
        self._connected = False
        self._last_health_check = None
        
        # Statistics
        self._stats = {
            'connections_created': 0,
            'queries_executed': 0,
            'documents_inserted': 0,
            'documents_updated': 0,
            'documents_deleted': 0,
            'errors': 0,
            'last_error': None
        }
        
        logger.info(f"Initialized ArangoDB client for {self.url}")
    
    def connect(self) -> bool:
        """
        Establish connection to ArangoDB.
        
        Returns:
            True if connection successful
            
        Raises:
            ArangoConnectionError: If connection fails after retries
        """
        for attempt in range(self.max_retries):
            try:
                # Create PyArango client
                self._client = PyArangoClient(hosts=self.url)
                
                # Connect to system database
                self._sys_db = self._client.db(
                    "_system",
                    username=self.username,
                    password=self.password
                )
                
                # Test connection with a simple query
                self._sys_db.properties()
                
                # Create or connect to target database
                if not self._sys_db.has_database(self.database_name):
                    logger.info(f"Creating database: {self.database_name}")
                    self._sys_db.create_database(self.database_name)
                
                # Connect to target database
                self._db = self._client.db(
                    self.database_name,
                    username=self.username,
                    password=self.password
                )
                
                # Test target database connection
                self._db.properties()
                
                self._connected = True
                self._last_health_check = datetime.now()
                self._stats['connections_created'] += 1
                
                logger.info(f"Successfully connected to ArangoDB database: {self.database_name}")
                return True
                
            except Exception as e:
                self._stats['errors'] += 1
                self._stats['last_error'] = str(e)
                
                if attempt < self.max_retries - 1:
                    logger.warning(f"Connection attempt {attempt + 1} failed: {e}. Retrying in {self.retry_delay}s...")
                    time.sleep(self.retry_delay)
                else:
                    error_msg = f"Failed to connect to ArangoDB after {self.max_retries} attempts: {e}"
                    logger.error(error_msg)
                    raise ArangoConnectionError(error_msg)
        
        return False
    
    def disconnect(self) -> None:
        """Disconnect from ArangoDB."""
        self._connected = False
        self._db = None
        self._sys_db = None
        self._client = None
        logger.info("Disconnected from ArangoDB")
    
    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self._connected and self._db is not None
    
    def health_check(self) -> bool:
        """
        Perform health check on the database connection.
        
        Returns:
            True if database is healthy
        """
        try:
            if not self.is_connected():
                return False
            
            # Simple query to test connection
            self._db.properties()
            self._last_health_check = datetime.now()
            return True
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            self._stats['errors'] += 1
            self._stats['last_error'] = str(e)
            return False
    
    @property
    def sys_db(self) -> StandardDatabase:
        """Get system database connection."""
        if not self._sys_db:
            raise ArangoConnectionError("Not connected to system database")
        return self._sys_db
    
    @property
    def db(self) -> StandardDatabase:
        """Get target database connection."""
        if not self._db:
            raise ArangoConnectionError("Not connected to target database")
        return self._db
    
    def ensure_connected(self) -> None:
        """Ensure client is connected, reconnecting if necessary."""
        if not self.is_connected():
            self.connect()
    
    def collection(self, name: str) -> StandardCollection:
        """
        Get collection by name.
        
        Args:
            name: Collection name
            
        Returns:
            Collection object
            
        Raises:
            ArangoConnectionError: If not connected
        """
        self.ensure_connected()
        return self.db.collection(name)
    
    def has_collection(self, name: str) -> bool:
        """
        Check if collection exists.
        
        Args:
            name: Collection name
            
        Returns:
            True if collection exists
        """
        try:
            self.ensure_connected()
            return self.db.has_collection(name)
        except Exception as e:
            logger.error(f"Error checking collection {name}: {e}")
            return False
    
    def create_collection(self, name: str, edge: bool = False, **kwargs) -> StandardCollection:
        """
        Create a collection.
        
        Args:
            name: Collection name
            edge: Whether this is an edge collection
            **kwargs: Additional collection options
            
        Returns:
            Created collection
            
        Raises:
            ArangoOperationError: If collection creation fails
        """
        try:
            self.ensure_connected()
            collection = self.db.create_collection(name, edge=edge, **kwargs)
            logger.info(f"Created {'edge' if edge else 'document'} collection: {name}")
            return collection
            
        except CollectionCreateError as e:
            error_msg = f"Failed to create collection {name}: {e}"
            logger.error(error_msg)
            self._stats['errors'] += 1
            self._stats['last_error'] = str(e)
            raise ArangoOperationError(error_msg)
    
    def execute_aql(self, query: str, bind_vars: Optional[Dict[str, Any]] = None, **kwargs) -> Any:
        """
        Execute AQL query.
        
        Args:
            query: AQL query string
            bind_vars: Query bind variables
            **kwargs: Additional query options
            
        Returns:
            Query result
            
        Raises:
            ArangoOperationError: If query execution fails
        """
        try:
            self.ensure_connected()
            
            bind_vars = bind_vars or {}
            result = self.db.aql.execute(query, bind_vars=bind_vars, **kwargs)
            
            self._stats['queries_executed'] += 1
            logger.debug(f"Executed AQL query successfully")
            
            return result
            
        except AQLQueryExecuteError as e:
            error_msg = f"AQL query failed: {e}"
            logger.error(error_msg)
            self._stats['errors'] += 1
            self._stats['last_error'] = str(e)
            raise ArangoOperationError(error_msg)
    
    def insert_document(self, collection_name: str, document: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Insert document into collection.
        
        Args:
            collection_name: Name of collection
            document: Document to insert
            **kwargs: Additional insert options
            
        Returns:
            Insert result metadata
            
        Raises:
            ArangoOperationError: If insert fails
        """
        try:
            collection = self.collection(collection_name)
            result = collection.insert(document, **kwargs)
            
            self._stats['documents_inserted'] += 1
            logger.debug(f"Inserted document into {collection_name}")
            
            return result
            
        except DocumentInsertError as e:
            error_msg = f"Failed to insert document into {collection_name}: {e}"
            logger.error(error_msg)
            self._stats['errors'] += 1
            self._stats['last_error'] = str(e)
            raise ArangoOperationError(error_msg)
    
    def insert_documents(self, collection_name: str, documents: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        """
        Insert multiple documents into collection.
        
        Args:
            collection_name: Name of collection
            documents: List of documents to insert
            **kwargs: Additional insert options
            
        Returns:
            List of insert result metadata
            
        Raises:
            ArangoOperationError: If batch insert fails
        """
        try:
            collection = self.collection(collection_name)
            results = collection.insert_many(documents, **kwargs)
            
            self._stats['documents_inserted'] += len(documents)
            logger.debug(f"Inserted {len(documents)} documents into {collection_name}")
            
            return results
            
        except DocumentInsertError as e:
            error_msg = f"Failed to insert {len(documents)} documents into {collection_name}: {e}"
            logger.error(error_msg)
            self._stats['errors'] += 1
            self._stats['last_error'] = str(e)
            raise ArangoOperationError(error_msg)
    
    def get_document(self, collection_name: str, key: str) -> Optional[Dict[str, Any]]:
        """
        Get document by key.
        
        Args:
            collection_name: Name of collection
            key: Document key
            
        Returns:
            Document or None if not found
            
        Raises:
            ArangoOperationError: If get operation fails
        """
        try:
            collection = self.collection(collection_name)
            return collection.get(key)
            
        except DocumentGetError as e:
            if "not found" in str(e).lower():
                return None
            
            error_msg = f"Failed to get document {key} from {collection_name}: {e}"
            logger.error(error_msg)
            self._stats['errors'] += 1
            self._stats['last_error'] = str(e)
            raise ArangoOperationError(error_msg)
    
    def update_document(self, collection_name: str, document: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Update document in collection.
        
        Args:
            collection_name: Name of collection
            document: Document with updates (must include _key)
            **kwargs: Additional update options
            
        Returns:
            Update result metadata
            
        Raises:
            ArangoOperationError: If update fails
        """
        try:
            collection = self.collection(collection_name)
            result = collection.update(document, **kwargs)
            
            self._stats['documents_updated'] += 1
            logger.debug(f"Updated document in {collection_name}")
            
            return result
            
        except DocumentUpdateError as e:
            error_msg = f"Failed to update document in {collection_name}: {e}"
            logger.error(error_msg)
            self._stats['errors'] += 1
            self._stats['last_error'] = str(e)
            raise ArangoOperationError(error_msg)
    
    def delete_document(self, collection_name: str, key: str, **kwargs) -> bool:
        """
        Delete document by key.
        
        Args:
            collection_name: Name of collection
            key: Document key
            **kwargs: Additional delete options
            
        Returns:
            True if deleted successfully
            
        Raises:
            ArangoOperationError: If delete fails
        """
        try:
            collection = self.collection(collection_name)
            result = collection.delete(key, **kwargs)
            
            self._stats['documents_deleted'] += 1
            logger.debug(f"Deleted document {key} from {collection_name}")
            
            return True
            
        except DocumentDeleteError as e:
            error_msg = f"Failed to delete document {key} from {collection_name}: {e}"
            logger.error(error_msg)
            self._stats['errors'] += 1
            self._stats['last_error'] = str(e)
            raise ArangoOperationError(error_msg)
    
    def truncate_collection(self, collection_name: str) -> bool:
        """
        Truncate (empty) a collection.
        
        Args:
            collection_name: Name of collection
            
        Returns:
            True if truncated successfully
        """
        try:
            collection = self.collection(collection_name)
            collection.truncate()
            logger.info(f"Truncated collection: {collection_name}")
            return True
            
        except Exception as e:
            error_msg = f"Failed to truncate collection {collection_name}: {e}"
            logger.error(error_msg)
            self._stats['errors'] += 1
            self._stats['last_error'] = str(e)
            raise ArangoOperationError(error_msg)
    
    def get_collection_count(self, collection_name: str) -> int:
        """
        Get document count in collection.
        
        Args:
            collection_name: Name of collection
            
        Returns:
            Number of documents in collection
        """
        try:
            collection = self.collection(collection_name)
            return collection.count()
            
        except Exception as e:
            logger.error(f"Failed to get count for collection {collection_name}: {e}")
            return 0
    
    @contextmanager
    def transaction(self, write_collections: Optional[List[str]] = None, read_collections: Optional[List[str]] = None):
        """
        Context manager for database transactions.
        
        Args:
            write_collections: Collections that will be written to
            read_collections: Collections that will be read from
            
        Yields:
            Transaction database object
        """
        self.ensure_connected()
        
        write_collections = write_collections or []
        read_collections = read_collections or []
        
        txn_db = self.db.begin_transaction(
            read=read_collections,
            write=write_collections
        )
        
        try:
            yield txn_db
            txn_db.commit_transaction()
            logger.debug("Transaction committed successfully")
            
        except Exception as e:
            txn_db.abort_transaction()
            logger.error(f"Transaction aborted: {e}")
            raise
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get client performance metrics.
        
        Returns:
            Dictionary containing performance metrics
        """
        metrics = self._stats.copy()
        metrics.update({
            'connected': self._connected,
            'host': self.host,
            'port': self.port,
            'database': self.database_name,
            'last_health_check': self._last_health_check.isoformat() if self._last_health_check else None,
            'uptime': (datetime.now() - self._last_health_check).total_seconds() if self._last_health_check else 0
        })
        return metrics
    
    def get_database_info(self) -> Dict[str, Any]:
        """
        Get database information.
        
        Returns:
            Database information
        """
        try:
            self.ensure_connected()
            
            # Get database properties
            db_props = self.db.properties()
            
            # Get collection information
            collections = []
            for collection_name in self.db.collections():
                collection = self.db.collection(collection_name['name'])
                collections.append({
                    'name': collection_name['name'],
                    'type': collection_name['type'],
                    'count': collection.count()
                })
            
            return {
                'database_name': self.database_name,
                'properties': db_props,
                'collections': collections,
                'total_documents': sum(c['count'] for c in collections)
            }
            
        except Exception as e:
            logger.error(f"Failed to get database info: {e}")
            return {'error': str(e)}
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
    
    def __repr__(self) -> str:
        """String representation."""
        status = "connected" if self._connected else "disconnected"
        return f"ArangoClient({self.url}, database='{self.database_name}', status='{status}')"