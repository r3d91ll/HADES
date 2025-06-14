"""
ArangoDB Database Component

This module provides an ArangoDB database connector component that implements
the DatabaseConnector protocol. It wraps the existing HADES ArangoDB client
with the new component contract system.
"""

import logging
from typing import Dict, Any, Optional, Union, List
from datetime import datetime

# Import the existing ArangoDB client
from src.database.arango_client import ArangoClient as LegacyArangoClient

# Import component contracts and protocols
from src.types.components.contracts import ComponentType, ComponentMetadata
from src.types.components.protocols import DatabaseConnector
from src.config.database_config import get_connection_params


class ArangoConnector(DatabaseConnector):
    """
    ArangoDB connector component implementing DatabaseConnector protocol.
    
    This component wraps the existing HADES ArangoDB client with the new
    component contract system, providing standardized database connectivity.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize ArangoDB connector.
        
        Args:
            config: Optional configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self._config = config or {}
        self._connection: Optional[LegacyArangoClient] = None
        self._is_connected = False
        
        # Component metadata
        self._metadata = ComponentMetadata(
            component_type=ComponentType.DATABASE,
            component_name="arango",
            component_version="1.0.0",
            config=self._config
        )
        
        self.logger.info(f"Initialized ArangoDB connector with config: {self._config}")
    
    @property
    def name(self) -> str:
        """Component name for identification."""
        return "arango"
    
    @property
    def version(self) -> str:
        """Component version string."""
        return "1.0.0"
    
    @property
    def component_type(self) -> ComponentType:
        """Type of component."""
        return ComponentType.DATABASE
    
    def configure(self, config: Dict[str, Any]) -> None:
        """
        Configure component with parameters.
        
        Args:
            config: Configuration dictionary containing component parameters
            
        Raises:
            ValueError: If configuration is invalid
        """
        if not self.validate_config(config):
            raise ValueError("Invalid configuration provided")
        
        self._config.update(config)
        self._metadata.config = self._config
        self._metadata.processed_at = datetime.utcnow()
        
        self.logger.info(f"Updated ArangoDB connector configuration")
    
    def validate_config(self, config: Any) -> bool:
        """
        Validate configuration parameters.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            True if configuration is valid, False otherwise
        """
        if not isinstance(config, dict):
            return False
        
        # Validate required connection parameters if provided
        connection_params = ['host', 'port', 'username', 'password', 'database']
        for param in connection_params:
            if param in config:
                if not isinstance(config[param], (str, int)):
                    return False
        
        # Validate boolean parameters
        boolean_params = ['use_ssl']
        for param in boolean_params:
            if param in config:
                if not isinstance(config[param], bool):
                    return False
        
        # Validate numeric parameters
        numeric_params = ['timeout', 'max_retries']
        for param in numeric_params:
            if param in config:
                if not isinstance(config[param], (int, float)) or config[param] <= 0:
                    return False
        
        return True
    
    def get_config_schema(self) -> Dict[str, Any]:
        """
        Get the JSON schema for component configuration.
        
        Returns:
            JSON schema dictionary describing valid configuration
        """
        return {
            "type": "object",
            "properties": {
                "host": {
                    "type": "string",
                    "description": "ArangoDB host address",
                    "default": "localhost"
                },
                "port": {
                    "type": "integer",
                    "description": "ArangoDB port",
                    "minimum": 1,
                    "maximum": 65535,
                    "default": 8529
                },
                "username": {
                    "type": "string", 
                    "description": "Database username"
                },
                "password": {
                    "type": "string",
                    "description": "Database password"
                },
                "database": {
                    "type": "string",
                    "description": "Default database name",
                    "default": "hades"
                },
                "use_ssl": {
                    "type": "boolean",
                    "description": "Whether to use SSL connection",
                    "default": False
                },
                "timeout": {
                    "type": "number",
                    "description": "Connection timeout in seconds",
                    "minimum": 1,
                    "default": 30
                },
                "max_retries": {
                    "type": "integer",
                    "description": "Maximum connection retries",
                    "minimum": 0,
                    "default": 3
                },
                "retry_delay": {
                    "type": "number",
                    "description": "Delay between retries in seconds",
                    "minimum": 0,
                    "default": 1.0
                }
            }
        }
    
    def health_check(self) -> bool:
        """
        Check if component is healthy and ready to process data.
        
        Returns:
            True if component is healthy, False otherwise
        """
        try:
            if not self._is_connected:
                return False
            
            if self._connection is None:
                return False
            
            # Try a simple operation to verify connection
            # This uses the existing health check method if available
            if hasattr(self._connection, 'health_check'):
                return bool(self._connection.health_check())
            
            # Fallback: try to get database properties
            if hasattr(self._connection, 'sys_db'):
                self._connection.sys_db.properties()
                return True
            
            return True
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get component performance metrics.
        
        Returns:
            Dictionary containing performance metrics
        """
        metrics = {
            "component_name": self.name,
            "component_version": self.version,
            "is_connected": self._is_connected,
            "connection_status": "connected" if self._is_connected else "disconnected",
            "last_health_check": datetime.utcnow().isoformat()
        }
        
        if self._connection and hasattr(self._connection, 'get_metrics'):
            # Include metrics from the underlying connection if available
            metrics.update(self._connection.get_metrics())
        
        return metrics
    
    def connect(self) -> bool:
        """
        Establish database connection.
        
        Returns:
            True if connection successful
        """
        try:
            # Get connection parameters
            connection_config = self._config.copy()
            
            # If no config provided, use defaults
            if not connection_config:
                connection_config = get_connection_params()
            
            # Create connection using the existing ArangoDB client
            self._connection = LegacyArangoClient(
                host=connection_config.get('host'),
                port=connection_config.get('port'), 
                username=connection_config.get('username'),
                password=connection_config.get('password'),
                database=connection_config.get('database'),
                use_ssl=connection_config.get('use_ssl', False),
                timeout=connection_config.get('timeout', 30),
                max_retries=connection_config.get('max_retries', 3),
                retry_delay=connection_config.get('retry_delay', 1.0)
            )
            
            self._is_connected = True
            self.logger.info("Successfully connected to ArangoDB")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to ArangoDB: {e}")
            self._is_connected = False
            return False
    
    def disconnect(self) -> bool:
        """
        Close database connection.
        
        Returns:
            True if disconnection successful
        """
        try:
            if self._connection:
                # The existing client doesn't have an explicit disconnect method
                # Just clear the connection reference
                self._connection = None
            
            self._is_connected = False
            self.logger.info("Disconnected from ArangoDB")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during disconnection: {e}")
            return False
    
    def is_connected(self) -> bool:
        """Check if database is connected."""
        return self._is_connected and self._connection is not None
    
    def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> Any:
        """
        Execute a database query.
        
        Args:
            query: Query to execute (AQL query for ArangoDB)
            parameters: Query parameters
            
        Returns:
            Query result
            
        Raises:
            RuntimeError: If not connected
            Exception: If query execution fails
        """
        if not self.is_connected():
            raise RuntimeError("Not connected to database")
        
        try:
            # Execute AQL query using the existing client
            if self._connection is None:
                raise RuntimeError("No connection available")
                
            if hasattr(self._connection, 'execute_aql'):
                result = self._connection.execute_aql(query, parameters or {})
            else:
                # Fallback: try to execute through sys_db
                result = self._connection.sys_db.aql.execute(query, bind_vars=parameters or {})
            
            self.logger.debug(f"Executed query successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Query execution failed: {e}")
            raise
    
    def get_connection_info(self) -> Dict[str, Any]:
        """
        Get connection information.
        
        Returns:
            Dictionary containing connection details
        """
        info: Dict[str, Any] = {
            "component_name": self.name,
            "component_type": self.component_type.value,
            "is_connected": self._is_connected,
            "connection_time": self._metadata.processed_at.isoformat() if self._metadata.processed_at else None
        }
        
        # Add config info (excluding sensitive data)
        config_info = self._config.copy()
        if 'password' in config_info:
            config_info['password'] = '***'
        info['config'] = config_info
        
        return info
    
    def get_underlying_client(self) -> Optional[LegacyArangoClient]:
        """
        Get the underlying ArangoDB client for advanced operations.
        
        Returns:
            The underlying ArangoClient instance
        """
        return self._connection