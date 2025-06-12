"""
Pydantic Schema Validation Component

This module provides a Pydantic-based schema validator component that implements
the SchemaValidator protocol. It wraps the existing HADES schema validation
logic with the new component contract system.
"""

import logging
from typing import Dict, Any, List, Optional, Type, Union
from datetime import datetime

# Import existing schema validation functionality
from src.schemas.common.validation import (
    ValidationResult as LegacyValidationResult,
    ValidationStage,
    validate_document,
    validate_dataset,
    validate_or_raise,
    ValidationCheckpoint,
    upgrade_schema_version
)

# Import schema classes
from src.schemas.documents.base import DocumentSchema
from src.schemas.documents.dataset import DatasetSchema
from src.schemas.documents.relations import DocumentRelationSchema
from src.schemas.embedding.models import EmbeddingConfig, EmbeddingResult, BatchEmbeddingResult
from src.schemas.pipeline.base import PipelineConfigSchema, PipelineStatsSchema

# Import component contracts and protocols
from src.types.components.contracts import ComponentType, ComponentMetadata
from src.types.components.protocols import SchemaValidator
from src.types.common import SchemaVersion

# Import Pydantic for validation
from pydantic import BaseModel, ValidationError


class PydanticValidator(SchemaValidator):
    """
    Pydantic-based schema validator implementing SchemaValidator protocol.
    
    This component provides comprehensive schema validation using Pydantic models
    and integrates with the existing HADES schema validation system.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Pydantic schema validator.
        
        Args:
            config: Optional configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self._config = config or {}
        
        # Component metadata
        self._metadata = ComponentMetadata(
            component_type=ComponentType.SCHEMAS,
            component_name="pydantic",
            component_version="1.0.0",
            config=self._config
        )
        
        # Schema registry - maps schema names to Pydantic model classes
        self._schema_registry: Dict[str, Type[BaseModel]] = {
            "document": DocumentSchema,
            "dataset": DatasetSchema,
            "document_relation": DocumentRelationSchema,
            "embedding_config": EmbeddingConfig,
            "embedding_result": EmbeddingResult,
            "batch_embedding_result": BatchEmbeddingResult,
            "pipeline_config": PipelineConfigSchema,
            "pipeline_stats": PipelineStatsSchema
        }
        
        # Default schema version
        self._default_schema_version = self._config.get(
            'default_schema_version', 
            SchemaVersion.V2
        )
        
        # Validation settings
        self._strict_mode = self._config.get('strict_mode', False)
        self._auto_upgrade = self._config.get('auto_upgrade_schemas', True)
        
        self.logger.info(f"Initialized Pydantic validator with {len(self._schema_registry)} schemas")
    
    @property
    def name(self) -> str:
        """Component name for identification."""
        return "pydantic"
    
    @property
    def version(self) -> str:
        """Component version string."""
        return "1.0.0"
    
    @property
    def component_type(self) -> ComponentType:
        """Type of component."""
        return ComponentType.SCHEMAS
    
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
        
        # Update settings based on new config
        self._default_schema_version = self._config.get(
            'default_schema_version', 
            self._default_schema_version
        )
        self._strict_mode = self._config.get('strict_mode', self._strict_mode)
        self._auto_upgrade = self._config.get('auto_upgrade_schemas', self._auto_upgrade)
        
        self.logger.info(f"Updated Pydantic validator configuration")
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate configuration parameters.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            True if configuration is valid, False otherwise
        """
        if not isinstance(config, dict):
            return False
        
        # Validate schema version if provided
        if 'default_schema_version' in config:
            version = config['default_schema_version']
            if not isinstance(version, SchemaVersion):
                try:
                    SchemaVersion(version)
                except ValueError:
                    return False
        
        # Validate boolean parameters
        boolean_params = ['strict_mode', 'auto_upgrade_schemas']
        for param in boolean_params:
            if param in config:
                if not isinstance(config[param], bool):
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
                "default_schema_version": {
                    "type": "string",
                    "enum": [v.value for v in SchemaVersion],
                    "description": "Default schema version to use",
                    "default": SchemaVersion.V2.value
                },
                "strict_mode": {
                    "type": "boolean",
                    "description": "Whether to raise exceptions on validation failures",
                    "default": False
                },
                "auto_upgrade_schemas": {
                    "type": "boolean",
                    "description": "Whether to automatically upgrade old schema versions",
                    "default": True
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
            # Test validation with a minimal valid document
            test_data = {
                "id": "health_check",
                "content": "test content",
                "source": "health_check",
                "document_type": "text",
                "schema_version": self._default_schema_version.value
            }
            
            result = self.validate(test_data, "document")
            return result
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get component performance metrics.
        
        Returns:
            Dictionary containing performance metrics
        """
        return {
            "component_name": self.name,
            "component_version": self.version,
            "available_schemas": list(self._schema_registry.keys()),
            "schema_count": len(self._schema_registry),
            "default_schema_version": self._default_schema_version.value,
            "strict_mode": self._strict_mode,
            "auto_upgrade": self._auto_upgrade,
            "last_health_check": datetime.utcnow().isoformat()
        }
    
    def validate(self, data: Any, schema_name: str) -> bool:
        """
        Validate data against a schema.
        
        Args:
            data: Data to validate
            schema_name: Name of schema to validate against
            
        Returns:
            True if data is valid
        """
        try:
            errors = self.get_validation_errors(data, schema_name)
            return len(errors) == 0
        except Exception as e:
            self.logger.error(f"Validation failed for schema '{schema_name}': {e}")
            return False
    
    def get_validation_errors(self, data: Any, schema_name: str) -> List[str]:
        """
        Get validation errors for data.
        
        Args:
            data: Data to validate
            schema_name: Name of schema to validate against
            
        Returns:
            List of validation error messages
        """
        if schema_name not in self._schema_registry:
            return [f"Unknown schema: {schema_name}"]
        
        schema_class = self._schema_registry[schema_name]
        
        try:
            # Auto-upgrade schema if needed and enabled
            if (self._auto_upgrade and 
                isinstance(data, dict) and 
                'schema_version' in data):
                data = upgrade_schema_version(data, self._default_schema_version)
            
            # Validate using Pydantic model
            schema_class(**data)
            return []
            
        except ValidationError as e:
            error_messages = []
            for error in e.errors():
                loc = " -> ".join(str(x) for x in error['loc'])
                msg = f"Field '{loc}': {error['msg']}"
                error_messages.append(msg)
            
            return error_messages
        
        except Exception as e:
            return [f"Validation error: {str(e)}"]
    
    def get_available_schemas(self) -> List[str]:
        """
        Get list of available schemas.
        
        Returns:
            List of schema names
        """
        return list(self._schema_registry.keys())
    
    def register_schema(self, schema_name: str, schema_class: Type[BaseModel]) -> None:
        """
        Register a new schema for validation.
        
        Args:
            schema_name: Name to register the schema under
            schema_class: Pydantic model class for the schema
        """
        if not issubclass(schema_class, BaseModel):
            raise ValueError("Schema class must be a Pydantic BaseModel")
        
        self._schema_registry[schema_name] = schema_class
        self.logger.info(f"Registered schema '{schema_name}'")
    
    def unregister_schema(self, schema_name: str) -> bool:
        """
        Unregister a schema.
        
        Args:
            schema_name: Name of schema to unregister
            
        Returns:
            True if schema was unregistered, False if not found
        """
        if schema_name in self._schema_registry:
            del self._schema_registry[schema_name]
            self.logger.info(f"Unregistered schema '{schema_name}'")
            return True
        return False
    
    def get_schema_info(self, schema_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific schema.
        
        Args:
            schema_name: Name of schema to get info for
            
        Returns:
            Dictionary containing schema information, or None if not found
        """
        if schema_name not in self._schema_registry:
            return None
        
        schema_class = self._schema_registry[schema_name]
        
        return {
            "name": schema_name,
            "class_name": schema_class.__name__,
            "module": schema_class.__module__,
            "schema": schema_class.schema(),
            "fields": list(schema_class.__fields__.keys())
        }
    
    def validate_with_legacy_system(
        self, 
        data: Dict[str, Any], 
        validation_stage: str = "processing"
    ) -> LegacyValidationResult:
        """
        Validate using the legacy HADES validation system.
        
        Args:
            data: Data to validate
            validation_stage: Pipeline stage for validation
            
        Returns:
            LegacyValidationResult: Validation result from legacy system
        """
        try:
            stage = ValidationStage(validation_stage)
        except ValueError:
            stage = ValidationStage.CUSTOM
        
        # Try to determine the data type and validate appropriately
        if 'chunks' in data or 'documents' in data:
            # Looks like a dataset
            return validate_dataset(data, stage, self._default_schema_version)
        else:
            # Assume it's a document
            return validate_document(data, stage, self._default_schema_version)
    
    def create_validation_checkpoint(
        self, 
        schema_name: str,
        stage: str = "processing",
        strict: Optional[bool] = None
    ) -> ValidationCheckpoint:
        """
        Create a validation checkpoint for use as a decorator.
        
        Args:
            schema_name: Name of schema to validate against
            stage: Pipeline stage name
            strict: Whether to use strict validation (overrides config)
            
        Returns:
            ValidationCheckpoint that can be used as a decorator
        """
        if schema_name not in self._schema_registry:
            raise ValueError(f"Unknown schema: {schema_name}")
        
        schema_class = self._schema_registry[schema_name]
        
        try:
            validation_stage = ValidationStage(stage)
        except ValueError:
            validation_stage = ValidationStage.CUSTOM
        
        use_strict = strict if strict is not None else self._strict_mode
        
        return ValidationCheckpoint(
            stage=validation_stage,
            input_schema=schema_class,
            output_schema=schema_class,
            strict=use_strict
        )