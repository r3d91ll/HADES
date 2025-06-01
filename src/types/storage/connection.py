"""Database connection type definitions.

This module defines types for database connections, connection pools,
and transaction management in storage systems.
"""

from typing import Dict, List, Any, Optional, Union, Protocol, Tuple, TypeVar, runtime_checkable, Callable
from enum import Enum
from datetime import datetime


class ConnectionState(str, Enum):
    """Enum for connection states."""
    
    DISCONNECTED = "disconnected"
    """Connection is not established."""
    
    CONNECTING = "connecting"
    """Connection is in progress."""
    
    CONNECTED = "connected"
    """Connection is established and active."""
    
    ERROR = "error"
    """Connection is in error state."""
    
    CLOSING = "closing"
    """Connection is in the process of closing."""


class DatabaseType(str, Enum):
    """Enum for supported database types."""
    
    ARANGO = "arango"
    """ArangoDB database."""
    
    POSTGRES = "postgres"
    """PostgreSQL database."""
    
    SQLITE = "sqlite"
    """SQLite database."""
    
    MEMORY = "memory"
    """In-memory database (for testing)."""


class ConnectionConfig(Dict[str, Any]):
    """Configuration for database connections."""
    pass


class ConnectionCredentials(Dict[str, str]):
    """Credentials for database connections."""
    pass


class ConnectionError(Exception):
    """Exception raised for connection errors."""
    pass


class ConnectionPoolConfig(Dict[str, Any]):
    """Configuration for connection pools."""
    pass


class TransactionOptions(Dict[str, Any]):
    """Options for database transactions."""
    pass


@runtime_checkable
class TransactionContext(Protocol):
    """Protocol for transaction contexts."""
    
    def commit(self) -> bool:
        """
        Commit the transaction.
        
        Returns:
            True if commit was successful, False otherwise
        """
        ...
    
    def rollback(self) -> bool:
        """
        Rollback the transaction.
        
        Returns:
            True if rollback was successful, False otherwise
        """
        ...
    
    def is_active(self) -> bool:
        """
        Check if the transaction is active.
        
        Returns:
            True if the transaction is active, False otherwise
        """
        ...


@runtime_checkable
class ConnectionInterface(Protocol):
    """Interface for database connections."""
    
    def connect(self) -> bool:
        """
        Establish connection to the database.
        
        Returns:
            True if connection was successful, False otherwise
        """
        ...
    
    def disconnect(self) -> bool:
        """
        Close connection to the database.
        
        Returns:
            True if disconnection was successful, False otherwise
        """
        ...
    
    def is_connected(self) -> bool:
        """
        Check if connection to the database is active.
        
        Returns:
            True if connected, False otherwise
        """
        ...
    
    def get_database(self) -> Any:
        """
        Get the database instance.
        
        Returns:
            The database instance
        """
        ...
    
    def begin_transaction(self) -> TransactionContext:
        """
        Begin a new transaction.
        
        Returns:
            Transaction context
        """
        ...
    
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """
        Execute a query on the database.
        
        Args:
            query: Query string
            params: Query parameters
            
        Returns:
            Query result
        """
        ...
    
    def get_connection_info(self) -> Dict[str, Any]:
        """
        Get information about the current connection.
        
        Returns:
            Dictionary with connection information
        """
        ...
    
    def get_state(self) -> ConnectionState:
        """
        Get the current state of the connection.
        
        Returns:
            Current connection state
        """
        ...


@runtime_checkable
class ConnectionPoolInterface(Protocol):
    """Interface for connection pools."""
    
    def initialize(self, config: ConnectionPoolConfig) -> bool:
        """
        Initialize the connection pool.
        
        Args:
            config: Pool configuration
            
        Returns:
            True if initialization was successful, False otherwise
        """
        ...
    
    def get_connection(self) -> ConnectionInterface:
        """
        Get a connection from the pool.
        
        Returns:
            Database connection
        """
        ...
    
    def release_connection(self, connection: ConnectionInterface) -> bool:
        """
        Release a connection back to the pool.
        
        Args:
            connection: Connection to release
            
        Returns:
            True if release was successful, False otherwise
        """
        ...
    
    def close_all(self) -> bool:
        """
        Close all connections in the pool.
        
        Returns:
            True if all connections were closed successfully, False otherwise
        """
        ...
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the connection pool.
        
        Returns:
            Dictionary with pool statistics
        """
        ...


# Connection factory type
ConnectionFactory = Callable[[], ConnectionInterface]

# Transaction callback type with proper generic type
T = TypeVar('T')
TransactionCallback = Callable[[ConnectionInterface], T]


class ConnectionResult(Dict[str, Any]):
    """Result of connection operations."""
    
    # Define __new__ to add default values
    def __new__(cls, success: bool = False, error: Optional[str] = None, 
                connection_id: Optional[str] = None, elapsed_time: float = 0.0) -> 'ConnectionResult':
        """Create a new ConnectionResult instance."""
        instance = super().__new__(cls)
        instance.update({
            "success": success,
            "error": error,
            "connection_id": connection_id,
            "elapsed_time": elapsed_time,
            "timestamp": datetime.now().isoformat()
        })
        return instance
