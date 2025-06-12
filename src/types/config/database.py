"""
Database configuration types for HADES.

This module contains Pydantic models for database configuration structures.
"""

from pydantic import BaseModel, Field


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