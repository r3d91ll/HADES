"""
Configuration models for pipeline processing in HADES-PathRAG.

This module defines the schema models for configuring processing pipelines
including stage configurations, input/output specifications, and database connections.
"""

from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import re
from enum import Enum

from pydantic import Field, field_validator, model_validator

from src.schemas.common.base import BaseSchema


class InputSourceType(str, Enum):
    """Types of input sources for pipeline processing."""
    FILE = "file"
    DIRECTORY = "directory"
    DATABASE = "database"
    API = "api"
    STREAM = "stream"
    PIPELINE = "pipeline"


class OutputDestinationType(str, Enum):
    """Types of output destinations for pipeline processing."""
    FILE = "file"
    DIRECTORY = "directory"
    DATABASE = "database"
    API = "api"
    STREAM = "stream"
    PIPELINE = "pipeline"


class InputSourceConfig(BaseSchema):
    """Configuration for pipeline input sources."""
    
    type: str
    path: Optional[Union[str, Path]] = None
    format: str = "auto"
    pattern: Optional[str] = None
    recursive: bool = False
    connection: Optional[str] = None
    collection: Optional[str] = None
    query: Optional[Dict[str, Any]] = None
    options: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator('type')
    def validate_type(cls, v):
        """Validate input source type."""
        if v not in [t.value for t in InputSourceType]:
            raise ValueError(f"Invalid input source type: {v}")
        return v
    
    @model_validator(mode='after')
    def validate_source_config(self):
        """Validate configuration based on source type."""
        if self.type == InputSourceType.FILE or self.type == InputSourceType.DIRECTORY:
            if not self.path:
                raise ValueError(f"Path is required for source type {self.type}")
        
        if self.type == InputSourceType.DATABASE:
            if not self.connection:
                raise ValueError("Connection string is required for database source")
        
        return self


class OutputDestinationConfig(BaseSchema):
    """Configuration for pipeline output destinations."""
    
    type: str
    path: Optional[Union[str, Path]] = None
    format: str = "json"
    overwrite: bool = False
    connection: Optional[str] = None
    collection: Optional[str] = None
    update_fields: Optional[List[str]] = None
    options: Dict[str, Any] = Field(default_factory=dict)
    stage: Optional[str] = None
    
    @field_validator('type')
    def validate_type(cls, v):
        """Validate output destination type."""
        if v not in [t.value for t in OutputDestinationType]:
            raise ValueError(f"Invalid output destination type: {v}")
        return v
    
    @model_validator(mode='after')
    def validate_destination_config(self):
        """Validate configuration based on destination type."""
        if self.type in [OutputDestinationType.FILE, OutputDestinationType.DIRECTORY]:
            if not self.path and self.type != OutputDestinationType.PIPELINE:
                raise ValueError(f"Path is required for destination type {self.type}")
        
        if self.type == OutputDestinationType.DATABASE:
            if not self.connection:
                raise ValueError("Connection string is required for database destination")
        
        if self.type == OutputDestinationType.PIPELINE:
            if not self.stage:
                raise ValueError("Stage name is required for pipeline destination")
        
        return self


class DatabaseType(str, Enum):
    """Types of databases supported by the pipeline."""
    SQLITE = "sqlite"
    POSTGRES = "postgres"
    MYSQL = "mysql"
    MONGODB = "mongodb"
    ARANGO = "arango"
    REDIS = "redis"
    ELASTICSEARCH = "elasticsearch"


class DatabaseConfig(BaseSchema):
    """Database configuration for pipeline processing."""
    
    type: str
    connection_string: str
    username: Optional[str] = None
    password: Optional[str] = None
    host: Optional[str] = None
    port: Optional[int] = None
    database: Optional[str] = None
    collection: Optional[str] = None
    graph_name: Optional[str] = None
    options: Dict[str, Any] = Field(default_factory=dict)
    connection_timeout: Optional[int] = None
    connection_retries: Optional[int] = None
    
    @field_validator('type')
    def validate_type(cls, v):
        """Validate database type."""
        if v not in [t.value for t in DatabaseType]:
            raise ValueError(f"Invalid database type: {v}")
        return v


class ProcessingStageConfig(BaseSchema):
    """Configuration for a processing stage in the pipeline."""
    
    name: str
    module: str
    function: str
    input: Union[InputSourceConfig, OutputDestinationConfig]
    output: OutputDestinationConfig
    params: Dict[str, Any] = Field(default_factory=dict)
    enabled: bool = False
    timeout: int = 3600
    retry_attempts: Optional[int] = None
    description: str = ""
    dependencies: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator('name')
    def validate_name(cls, v):
        """Validate stage name."""
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError("Stage name must contain only alphanumeric characters, hyphens, and underscores")
        return v
    
    @field_validator('timeout', 'retry_attempts')
    def validate_positive_int(cls, v, info):
        """Validate positive integer values."""
        if v is not None and v <= 0:
            raise ValueError(f"{info.field_name} must be a positive integer")
        return v


class PipelineConfigSchema(BaseSchema):
    """Main configuration schema for processing pipelines."""
    
    name: str
    version: str
    description: str = ""
    stages: List[ProcessingStageConfig] = Field(default_factory=list)
    max_workers: int = 4
    enable_logging: bool = True
    log_level: str = "INFO"
    database: Optional[DatabaseConfig] = None
    timeout: Optional[int] = None
    retry_attempts: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @model_validator(mode='after')
    def validate_stage_dependencies(self):
        """Validate that stage dependencies exist in the pipeline."""
        stage_names = set(stage.name for stage in self.stages)
        
        for stage in self.stages:
            # Check dependencies
            for dep in stage.dependencies:
                if dep not in stage_names:
                    raise ValueError(f"Stage '{stage.name}' depends on non-existent stage '{dep}'")
            
            # Check pipeline input references
            if hasattr(stage.input, 'type') and (stage.input.type == 'pipeline' or stage.input.type == InputSourceType.PIPELINE):
                source_stage = stage.input.stage
                if source_stage and source_stage not in stage_names:
                    raise ValueError(f"Stage '{stage.name}' depends on output from non-existent stage '{source_stage}'")
        
        return self
