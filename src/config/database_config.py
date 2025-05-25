"""
Database configuration for HADES-PathRAG.

This module defines the configuration for connecting to ArangoDB
and managing database operations.
"""

import os
from typing import Dict, Any, Optional
import yaml
from pydantic import BaseModel, Field

# Default configuration file path
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "database_config.yaml")


class ArangoDBConfig(BaseModel):
    """Configuration for ArangoDB connection and operations."""
    
    # Connection settings
    host: str = Field(
        default="localhost", 
        description="ArangoDB server hostname or IP address"
    )
    port: int = Field(
        default=8529, 
        description="ArangoDB server port"
    )
    username: str = Field(
        default="root", 
        description="ArangoDB username"
    )
    password: str = Field(
        default="", 
        description="ArangoDB password"
    )
    use_ssl: bool = Field(
        default=False, 
        description="Whether to use SSL for connection"
    )
    
    # Database settings
    database_name: str = Field(
        default="hades", 
        description="Name of the database to use"
    )
    
    # Collection names
    documents_collection: str = Field(
        default="documents", 
        description="Name of the documents collection"
    )
    chunks_collection: str = Field(
        default="chunks", 
        description="Name of the chunks collection"
    )
    relationships_collection: str = Field(
        default="relationships", 
        description="Name of the relationships (edges) collection"
    )
    
    # Operation settings
    timeout: int = Field(
        default=60, 
        description="Timeout for database operations in seconds"
    )
    retry_attempts: int = Field(
        default=3, 
        description="Number of retry attempts for failed operations"
    )
    retry_delay: float = Field(
        default=1.0, 
        description="Delay between retry attempts in seconds"
    )
    batch_size: int = Field(
        default=100, 
        description="Batch size for bulk operations"
    )


def load_database_config(config_path: Optional[str] = None) -> ArangoDBConfig:
    """Load database configuration from YAML file.
    
    Args:
        config_path: Path to configuration file (default: database_config.yaml)
        
    Returns:
        Database configuration object
    """
    config_path = config_path or DEFAULT_CONFIG_PATH
    
    # Check if config file exists
    if not os.path.exists(config_path):
        # Return default configuration
        return ArangoDBConfig()
    
    # Load config from file
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)
    
    # Create config object
    return ArangoDBConfig(**config_data)


def get_database_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Get database configuration as dictionary.
    
    This is a convenience function for getting the configuration
    in a format suitable for passing to the ArangoClient.
    
    Args:
        config_path: Path to configuration file (default: database_config.yaml)
        
    Returns:
        Database configuration dictionary
    """
    config = load_database_config(config_path)
    # Explicit typing to ensure correct return type
    result: Dict[str, Any] = config.model_dump()
    return result


# Check for environment variable overrides
def _get_env_override(env_var: str, default: Any) -> Any:
    """Get environment variable value or default."""
    return os.environ.get(env_var, default)


# Allow environment variable overrides for sensitive information
def get_connection_params() -> Dict[str, Any]:
    """Get database connection parameters with environment variable overrides.
    
    This function prioritizes environment variables over configuration file
    values for sensitive information like username and password.
    
    Environment variables:
    - ARANGO_HOST: Database host
    - ARANGO_PORT: Database port
    - ARANGO_USER: Database username
    - ARANGO_PASSWORD: Database password
    - ARANGO_DB: Database name
    
    Returns:
        Dictionary of connection parameters
    """
    config = load_database_config()
    
    # Override with environment variables if available
    connection_params = {
        "host": _get_env_override("ARANGO_HOST", config.host),
        "port": int(_get_env_override("ARANGO_PORT", config.port)),
        "username": _get_env_override("ARANGO_USER", config.username),
        "password": _get_env_override("ARANGO_PASSWORD", config.password),
        "database": _get_env_override("ARANGO_DB", config.database_name),
        "use_ssl": config.use_ssl,
        "timeout": config.timeout
    }
    
    return connection_params
