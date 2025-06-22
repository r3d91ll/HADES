"""
ArangoDB Client for HADES

Production-ready ArangoDB client with connection pooling, retry logic,
and comprehensive error handling for HADES operations.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Union, Iterator
from contextlib import contextmanager
from datetime import datetime, timezone

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

from src.config.manager import get_config_manager

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
        host: Optional[str] = None,
        port: Optional[int] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
        use_ssl: Optional[bool] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
        retry_delay: Optional[float] = None,
        connection_pool_size: Optional[int] = None
    ):
        """
        Initialize ArangoDB client.
        
        Args:
            host: ArangoDB host (optional, uses config if None)
            port: ArangoDB port (optional, uses config if None)
            username: Database username (optional, uses config if None)
            password: Database password (optional, uses config if None)
            database: Default database name (optional, uses config if None)
            use_ssl: Whether to use SSL (optional, uses config if None)
            timeout: Connection timeout in seconds (optional, uses config if None)
            max_retries: Maximum retry attempts (optional, uses config if None)
            retry_delay: Delay between retries in seconds (optional, uses config if None)
            connection_pool_size: Size of connection pool (optional, uses config if None)
        """
        # Get configuration manager
        config_manager = get_config_manager()
        
        # Use provided values or fallback to configuration
        self.host = host or config_manager.get('database.arango.host', '127.0.0.1')
        self.port = port or config_manager.get('database.arango.port', 8529)
        self.username = username or config_manager.get('database.arango.username', 'root')
        self.password = password or config_manager.get('database.arango.password', '')
        self.database_name = database or config_manager.get('database.arango.default_database', 'hades')
        self.use_ssl = use_ssl if use_ssl is not None else config_manager.get('database.arango.use_ssl', False)
        self.timeout = timeout or config_manager.get('database.arango.timeout', 30)
        self.max_retries = max_retries or config_manager.get('database.arango.max_retries', 3)
        self.retry_delay = retry_delay or config_manager.get('database.arango.retry_delay', 1.0)
        self.connection_pool_size = connection_pool_size or config_manager.get('database.arango.connection_pool_size', 10)
        
        # Build connection URL
        protocol = "https" if self.use_ssl else "http"
        self.url = f"{protocol}://{self.host}:{self.port}"
        
        # Initialize client and connections
        self._client: Optional[PyArangoClient] = None
        self._sys_db: Optional[StandardDatabase] = None
        self._db: Optional[StandardDatabase] = None
        self._connected = False
        self._last_health_check: Optional[datetime] = None
        
        # Statistics
        self._stats: Dict[str, Union[int, str, None]] = {
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
                self._last_health_check = datetime.now(timezone.utc)
                self._increment_stat('connections_created')
                
                logger.info(f"Successfully connected to ArangoDB database: {self.database_name}")
                return True
                
            except Exception as e:
                self._set_error(str(e))
                
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
            if self._db is not None:
                self._db.properties()
                self._last_health_check = datetime.now(timezone.utc)
                return True
            else:
                return False
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            self._set_error(str(e))
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
            result = self.db.has_collection(name)
            return bool(result) if result is not None else False
        except Exception as e:
            logger.error(f"Error checking collection {name}: {e}")
            return False
    
    def create_collection(self, name: str, edge: bool = False, **kwargs: Any) -> StandardCollection:
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
            collection_result = self.db.create_collection(name, edge=edge, **kwargs)
            # Handle union type from ArangoDB
            if hasattr(collection_result, '_key'):
                # It's a StandardCollection
                collection = collection_result
            else:
                # It might be an AsyncJob or BatchJob, convert to StandardCollection
                collection = self.db.collection(name)
            
            if collection is None:
                raise ArangoOperationError(f"Failed to create collection {name}: returned None")
            logger.info(f"Created {'edge' if edge else 'document'} collection: {name}")
            return collection  # type: ignore[return-value]
            
        except CollectionCreateError as e:
            error_msg = f"Failed to create collection {name}: {e}"
            logger.error(error_msg)
            self._set_error(str(e))
            raise ArangoOperationError(error_msg)
    
    def create_database(self, name: str) -> bool:
        """
        Create a database.
        
        Args:
            name: Database name
            
        Returns:
            True if database was created successfully
            
        Raises:
            ArangoOperationError: If database creation fails
        """
        try:
            if not self._sys_db:
                # Connect to system database if not already connected
                self.connect()
                
            # Check if database already exists
            if self._sys_db is not None and self._sys_db.has_database(name):
                logger.info(f"Database {name} already exists")
                return True
                
            # Create the database
            if self._sys_db is not None:
                self._sys_db.create_database(name)
                logger.info(f"Created database: {name}")
                self._increment_stat('databases_created')
                return True
            else:
                raise ArangoOperationError("System database not available")
            
        except DatabaseCreateError as e:
            error_msg = f"Failed to create database {name}: {e}"
            logger.error(error_msg)
            self._set_error(str(e))
            raise ArangoOperationError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error creating database {name}: {e}"
            logger.error(error_msg)
            self._set_error(str(e))
            raise ArangoOperationError(error_msg)
    
    def connect_to_database(self, database_name: str) -> bool:
        """
        Connect to a specific database.
        
        Args:
            database_name: Name of database to connect to
            
        Returns:
            True if connection successful
        """
        try:
            # First ensure we're connected to the system
            if not self._client:
                self.connect()
                
            # Connect to the specific database
            if self._client is not None:
                self._db = self._client.db(database_name, username=self.username, password=self.password)
                self.database_name = database_name
                
                # Test the connection
                if self._db is not None:
                    self._db.properties()
                    logger.info(f"Successfully connected to database: {database_name}")
                    return True
                else:
                    return False
            else:
                return False
            
        except Exception as e:
            logger.error(f"Failed to connect to database {database_name}: {e}")
            return False
    
    def execute_aql(self, query: str, bind_vars: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
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
            
            self._increment_stat('queries_executed')
            logger.debug(f"Executed AQL query successfully")
            
            return result
            
        except AQLQueryExecuteError as e:
            error_msg = f"AQL query failed: {e}"
            logger.error(error_msg)
            self._set_error(str(e))
            raise ArangoOperationError(error_msg)
    
    def insert_document(self, collection_name: str, document: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
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
            insert_result = collection.insert(document, **kwargs)
            
            # Handle union type from ArangoDB
            if isinstance(insert_result, dict):
                result = insert_result
            elif insert_result is not None and hasattr(insert_result, 'result'):
                # Handle AsyncJob/BatchJob
                result_data = insert_result.result() if callable(getattr(insert_result, 'result', None)) else {}
                result = result_data if isinstance(result_data, dict) else {}
            else:
                result = {}
            
            if not result:
                raise ArangoOperationError(f"Insert operation returned empty result for collection {collection_name}")
            
            self._increment_stat('documents_inserted')
            logger.debug(f"Inserted document into {collection_name}")
            
            return result
            
        except DocumentInsertError as e:
            error_msg = f"Failed to insert document into {collection_name}: {e}"
            logger.error(error_msg)
            self._set_error(str(e))
            raise ArangoOperationError(error_msg)
    
    def insert_documents(self, collection_name: str, documents: List[Dict[str, Any]], **kwargs: Any) -> List[Dict[str, Any]]:
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
            insert_results = collection.insert_many(documents, **kwargs)
            
            # Handle union type from ArangoDB
            if isinstance(insert_results, list):
                results = insert_results
            elif insert_results is not None and hasattr(insert_results, 'result'):
                # Handle AsyncJob/BatchJob
                result_data = insert_results.result() if callable(getattr(insert_results, 'result', None)) else []
                # Filter out any error objects and keep only dict results
                results = [r for r in result_data if isinstance(r, dict) and not hasattr(r, 'error')] if isinstance(result_data, list) else []
            else:
                results = []
            
            self._increment_stat('documents_inserted', len(documents))
            logger.debug(f"Inserted {len(documents)} documents into {collection_name}")
            
            return results  # type: ignore[return-value]
            
        except DocumentInsertError as e:
            error_msg = f"Failed to insert {len(documents)} documents into {collection_name}: {e}"
            logger.error(error_msg)
            self._set_error(str(e))
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
            get_result = collection.get(key)
            
            # Handle union type from ArangoDB
            if isinstance(get_result, dict):
                return get_result
            elif get_result is not None and hasattr(get_result, 'result'):
                # Handle AsyncJob/BatchJob
                result_data = get_result.result() if callable(getattr(get_result, 'result', None)) else None
                return result_data if isinstance(result_data, dict) else None
            else:
                return None
            
        except DocumentGetError as e:
            if "not found" in str(e).lower():
                return None
            
            error_msg = f"Failed to get document {key} from {collection_name}: {e}"
            logger.error(error_msg)
            self._set_error(str(e))
            raise ArangoOperationError(error_msg)
    
    def update_document(self, collection_name: str, document: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
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
            update_result = collection.update(document, **kwargs)
            
            # Handle union type from ArangoDB
            if isinstance(update_result, dict):
                result = update_result
            elif update_result is not None and hasattr(update_result, 'result'):
                # Handle AsyncJob/BatchJob
                result_data = update_result.result() if callable(getattr(update_result, 'result', None)) else {}
                result = result_data if isinstance(result_data, dict) else {}
            else:
                result = {}
            
            self._increment_stat('documents_updated')
            logger.debug(f"Updated document in {collection_name}")
            
            return result
            
        except DocumentUpdateError as e:
            error_msg = f"Failed to update document in {collection_name}: {e}"
            logger.error(error_msg)
            self._set_error(str(e))
            raise ArangoOperationError(error_msg)
    
    def delete_document(self, collection_name: str, key: str, **kwargs: Any) -> bool:
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
            
            self._increment_stat('documents_deleted')
            logger.debug(f"Deleted document {key} from {collection_name}")
            
            return True
            
        except DocumentDeleteError as e:
            error_msg = f"Failed to delete document {key} from {collection_name}: {e}"
            logger.error(error_msg)
            self._set_error(str(e))
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
            self._set_error(str(e))
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
            count_result = collection.count()
            # Handle union type from ArangoDB
            if isinstance(count_result, int):
                return count_result
            elif count_result is not None and hasattr(count_result, 'result'):
                # Handle AsyncJob/BatchJob
                result_data = count_result.result() if callable(getattr(count_result, 'result', None)) else 0
                return int(result_data) if isinstance(result_data, (int, str)) else 0
            else:
                return 0
            
        except Exception as e:
            logger.error(f"Failed to get count for collection {collection_name}: {e}")
            return 0
    
    @contextmanager
    def transaction(self, write_collections: Optional[List[str]] = None, read_collections: Optional[List[str]] = None) -> Iterator[StandardDatabase]:
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
            # Handle both TransactionDatabase and StandardDatabase
            yield txn_db  # type: ignore[misc]
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
            'uptime': int((datetime.now(timezone.utc) - self._last_health_check).total_seconds()) if self._last_health_check else 0
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
            collections_result = self.db.collections()
            # Handle union type from ArangoDB
            if isinstance(collections_result, list):
                collections_list = collections_result
            elif collections_result is not None and hasattr(collections_result, 'result'):
                # Handle AsyncJob/BatchJob
                result_data = collections_result.result() if callable(getattr(collections_result, 'result', None)) else []
                collections_list = result_data if isinstance(result_data, list) else []
            else:
                collections_list = []
            
            for collection_info in collections_list:
                if isinstance(collection_info, dict) and 'name' in collection_info:
                    collection = self.db.collection(collection_info['name'])
                    count_result = collection.count()
                    # Handle count union type
                    if isinstance(count_result, int):
                        count = count_result
                    elif count_result is not None and hasattr(count_result, 'result'):
                        count_data = count_result.result() if callable(getattr(count_result, 'result', None)) else 0
                        count = int(count_data) if isinstance(count_data, (int, str)) else 0
                    else:
                        count = 0
                    
                    collections.append({
                        'name': collection_info['name'],
                        'type': collection_info.get('type', 'unknown'),
                        'count': count
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
    
    def __enter__(self) -> 'ArangoClient':
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.disconnect()
    
    def list_databases(self) -> List[str]:
        """List all databases."""
        try:
            from arango import ArangoClient as PyArangoClient
            
            # Create a new client instance
            client = PyArangoClient(hosts=self.url)
            
            # Get system database
            sys_db = client.db('_system', username=self.username, password=self.password)
            
            # List all databases
            databases_result = sys_db.databases()
            # Handle union type from ArangoDB
            if isinstance(databases_result, list):
                return databases_result
            elif databases_result is not None and hasattr(databases_result, 'result'):
                # Handle AsyncJob/BatchJob
                result_data = databases_result.result() if callable(getattr(databases_result, 'result', None)) else []
                return result_data if isinstance(result_data, list) else []
            else:
                return []
            
        except Exception as e:
            logger.error(f"Failed to list databases: {e}")
            return []
    
    def switch_database(self, database_name: str) -> bool:
        """Switch to a different database."""
        try:
            # Update the database name
            self.database_name = database_name
            
            # Reconnect with the new database
            if self._client is None:
                self._client = PyArangoClient(hosts=self.url)
                
            self._db = self._client.db(
                database_name,
                username=self.username,
                password=self.password
            )
            
            # Test connection
            if self._db is not None:
                self._db.properties()
                self._connected = True
                logger.info(f"Switched to database: {database_name}")
                return True
            else:
                return False
            
        except Exception as e:
            logger.error(f"Failed to switch to database {database_name}: {e}")
            self._connected = False
            return False
    
    def list_collections(self) -> List[str]:
        """List collections in current database."""
        try:
            if not self._db:
                return []
            
            collections_result = self._db.collections()
            # Handle union type from ArangoDB
            if isinstance(collections_result, list):
                collections_list = collections_result
            elif collections_result is not None and hasattr(collections_result, 'result'):
                # Handle AsyncJob/BatchJob
                result_data = collections_result.result() if callable(getattr(collections_result, 'result', None)) else []
                collections_list = result_data if isinstance(result_data, list) else []
            else:
                collections_list = []
            
            return [c['name'] for c in collections_list if isinstance(c, dict) and 'name' in c and not c['name'].startswith('_')]
            
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return []
    
    def get_collection_document_count(self, collection_name: str) -> int:
        """Get document count for a collection."""
        try:
            if not self._db:
                return 0
            
            collection = self._db.collection(collection_name)
            count_result = collection.count()
            # Handle union type from ArangoDB
            if isinstance(count_result, int):
                return count_result
            elif count_result is not None and hasattr(count_result, 'result'):
                # Handle AsyncJob/BatchJob
                result_data = count_result.result() if callable(getattr(count_result, 'result', None)) else 0
                return int(result_data) if isinstance(result_data, (int, str)) else 0
            else:
                return 0
            
        except Exception as e:
            logger.error(f"Failed to get count for {collection_name}: {e}")
            return 0
    
    def get_sample_document(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """Get a sample document from collection."""
        try:
            if not self._db:
                return None
            
            collection = self._db.collection(collection_name)
            cursor_result = collection.all(limit=1)
            # Handle union type from ArangoDB
            if hasattr(cursor_result, '__iter__'):
                cursor = cursor_result
            elif cursor_result is not None and hasattr(cursor_result, 'result'):
                # Handle AsyncJob/BatchJob
                cursor = cursor_result.result() if callable(getattr(cursor_result, 'result', None)) else None
            else:
                cursor = None
            
            if cursor is not None:
                docs = list(cursor)
                return docs[0] if docs else None
            else:
                return None
            
        except Exception as e:
            logger.error(f"Failed to get sample from {collection_name}: {e}")
            return None

    def _increment_stat(self, stat_name: str, increment: int = 1) -> None:
        """Safely increment a statistic."""
        current_value = self._stats.get(stat_name, 0)
        if isinstance(current_value, int):
            self._stats[stat_name] = current_value + increment
        else:
            self._stats[stat_name] = increment
    
    def _set_error(self, error_message: str) -> None:
        """Set error statistics."""
        self._increment_stat('errors')
        self._stats['last_error'] = error_message
    
    def __repr__(self) -> str:
        """String representation."""
        status = "connected" if self._connected else "disconnected"
        return f"ArangoClient({self.url}, database='{self.database_name}', status='{status}')"