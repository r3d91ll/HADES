"""
Configuration type definitions for HADES.

This module provides Pydantic schemas for validating configuration files.
Each configuration file should have a corresponding schema defined here.
"""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, validator
from pathlib import Path


class BaseConfig(BaseModel):
    """Base configuration schema with common settings."""
    
    class Config:
        extra = 'forbid'  # Fail on unknown fields
        validate_assignment = True
        
        
class LoggingConfig(BaseModel):
    """Logging configuration schema."""
    level: str = Field("INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    format: str = Field("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file: Optional[Path] = None
    

class CommonConfig(BaseConfig):
    """Common configuration shared across modules."""
    logging: LoggingConfig
    metrics: Dict[str, Any] = Field(default_factory=dict)
    cache: Dict[str, Any] = Field(default_factory=dict)
    gpu: Dict[str, Any] = Field(default_factory=dict)
    paths: Dict[str, Path] = Field(default_factory=dict)
    

class ServerConfig(BaseConfig):
    """API server configuration schema."""
    server: Dict[str, Any]
    cors: Dict[str, Any] = Field(default_factory=dict)
    rate_limiting: Dict[str, Any] = Field(default_factory=dict)
    timeouts: Dict[str, Any] = Field(default_factory=dict)
    middleware: Dict[str, Any] = Field(default_factory=dict)
    docs: Dict[str, Any] = Field(default_factory=dict)