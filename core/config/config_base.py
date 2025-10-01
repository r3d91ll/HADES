"""
Base Configuration Classes
==========================

Theory Connection - Information Reconstructionism:
The configuration system embodies the WHERE dimension through hierarchical positioning
of configuration sources (environment > processor > base). Configuration validation
ensures shared-context components (L=local coherence, I=instruction fit, A=actionability,
G=grounding) maintain consistent semantic relationships across processing phases.

The Conveyance Framework `C_pair = Hmean(C_out, C_in)·C_ext^α·P_ij` applies directly:
- WHERE (R): Configuration hierarchy and source positioning
- WHAT (W): Schema validation ensuring semantic content quality
- WHO (H): Access patterns and permission structures
- TIME (T): Configuration loading and parsing efficiency
- Shared context (C_ext): Exponential amplification of configuration coherence

From Actor-Network Theory: Configuration acts as an "obligatory passage point"
through which all system components must translate their requirements.
"""

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar('T', bound='BaseConfig')


class ConfigError(Exception):
    """
    Configuration-related errors.

    Theory Connection: Represents breakdown in Context coherence,
    triggering zero-propagation gate where C = 0.
    """
    pass


class ConfigValidationError(ConfigError):
    """Configuration validation errors."""

    def __init__(self, message: str, errors: List[str]):
        self.errors = errors
        super().__init__(f"{message}: {'; '.join(errors)}")


class BaseConfig(BaseModel, ABC):
    """
    Abstract base for all configuration models.

    Theory Connection - Conveyance Framework:
    Implements the shared-context (`C_ext`) component through validated configuration
    that ensures L=local coherence, I=instruction fit, A=actionability, G=grounding.
    Schema validation prevents configuration drift that would degrade `C_ext^α` amplification.

    From Information Reconstructionism: Configuration represents the WHERE dimension
    through hierarchical positioning and semantic relationships between components.
    """

    # Metadata fields
    config_version: str = Field(default="1.0", description="Configuration schema version")
    created_at: Optional[datetime] = Field(default=None, description="Configuration creation timestamp")
    source: Optional[str] = Field(default=None, description="Configuration source identifier")

    class Config:
        """Pydantic configuration."""
        extra = "forbid"  # Strict validation
        validate_assignment = True
        use_enum_values = True
        arbitrary_types_allowed = False

    def __init__(self, **data):
        # Set creation timestamp if not provided
        if 'created_at' not in data:
            data['created_at'] = datetime.utcnow()
        super().__init__(**data)

    @abstractmethod
    def validate_semantics(self) -> List[str]:
        """
        Validate semantic consistency beyond schema validation.

        Theory Connection: Ensures shared-context components maintain coherent
        relationships that support exponential amplification (`C_ext^α`).

        Returns:
            List of validation error messages (empty if valid)
        """
        pass

    def validate_full(self) -> None:
        """
        Perform full validation including semantic checks.

        Theory Connection: Prevents zero propagation by ensuring
        all shared-context dimensions maintain positive values.

        Raises:
            ConfigValidationError: If validation fails
        """
        try:
            # Pydantic validation is automatic
            pass
        except ValidationError as e:
            raise ConfigValidationError("Schema validation failed",
                                      [f"{err['loc']}: {err['msg']}" for err in e.errors])

        # Semantic validation
        semantic_errors = self.validate_semantics()
        if semantic_errors:
            raise ConfigValidationError("Semantic validation failed", semantic_errors)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary format.

        Returns:
            Configuration as dictionary
        """
        return self.dict(exclude_none=True)

    def to_json(self, indent: int = 2) -> str:
        """
        Convert to JSON format.

        Args:
            indent: JSON indentation

        Returns:
            Configuration as JSON string
        """
        return self.json(exclude_none=True, indent=indent)

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """
        Create configuration from dictionary.

        Args:
            data: Configuration data

        Returns:
            Configuration instance

        Raises:
            ConfigValidationError: If validation fails
        """
        try:
            instance = cls(**data)
            instance.validate_full()
            return instance
        except ValidationError as e:
            raise ConfigValidationError("Failed to create from dict",
                                      [f"{err['loc']}: {err['msg']}" for err in e.errors])

    @classmethod
    def from_json(cls: Type[T], json_str: str) -> T:
        """
        Create configuration from JSON string.

        Args:
            json_str: JSON configuration string

        Returns:
            Configuration instance

        Raises:
            ConfigValidationError: If validation fails
        """
        try:
            data = json.loads(json_str)
            return cls.from_dict(data)
        except json.JSONDecodeError as e:
            raise ConfigValidationError("Invalid JSON format", [str(e)])

    @classmethod
    def from_file(cls: Type[T], file_path: Union[str, Path]) -> T:
        """
        Load configuration from file.

        Args:
            file_path: Path to configuration file

        Returns:
            Configuration instance

        Raises:
            ConfigValidationError: If file cannot be loaded or validated
        """
        path = Path(file_path)
        if not path.exists():
            raise ConfigValidationError(f"Configuration file not found: {path}", [])

        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()

            instance = cls.from_json(content)
            instance.source = str(path)
            return instance

        except (IOError, OSError) as e:
            raise ConfigValidationError(f"Failed to read configuration file: {path}", [str(e)])

    def save_to_file(self, file_path: Union[str, Path]) -> None:
        """
        Save configuration to file.

        Args:
            file_path: Path to save configuration

        Raises:
            ConfigError: If file cannot be saved
        """
        path = Path(file_path)
        try:
            # Ensure directory exists
            path.parent.mkdir(parents=True, exist_ok=True)

            with open(path, 'w', encoding='utf-8') as f:
                f.write(self.to_json())

            self.source = str(path)
            logger.info(f"Configuration saved to {path}")

        except (IOError, OSError) as e:
            raise ConfigError(f"Failed to save configuration to {path}: {e}")

    def merge(self: T, other: T) -> T:
        """
        Merge with another configuration of the same type.

        Theory Connection: Implements hierarchical Context composition
        where higher-priority configurations override lower-priority ones
        while maintaining semantic coherence.

        Args:
            other: Configuration to merge (higher priority)

        Returns:
            New merged configuration instance
        """
        # Get base data
        base_data = self.to_dict()
        other_data = other.to_dict()

        # Deep merge logic
        merged_data = self._deep_merge_dicts(base_data, other_data)

        # Create new instance
        return self.__class__.from_dict(merged_data)

    @staticmethod
    def _deep_merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two dictionaries.

        Args:
            base: Base dictionary
            override: Override dictionary (higher priority)

        Returns:
            Merged dictionary
        """
        result = base.copy()

        for key, value in override.items():
            if (key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)):
                result[key] = BaseConfig._deep_merge_dicts(result[key], value)
            else:
                result[key] = value

        return result

    def get_context_score(self) -> float:
        """
        Calculate shared-context (C_ext) component score for the Conveyance Framework.

        Theory Connection: Quantifies `C_ext = wL·L + wI·I + wA·A + wG·G`
        based on configuration completeness and semantic coherence.

        Returns:
            C_ext score between 0.0 and 1.0
        """
        # Base implementation - subclasses should override
        semantic_errors = self.validate_semantics()

        # L = Local coherence (no validation errors)
        local_coherence = 1.0 if not semantic_errors else max(0.0, 1.0 - len(semantic_errors) / 10)

        # I = Instruction fit (all required fields present)
        # Handle both Pydantic v1 and v2 compatibility
        required_fields = []
        for field, field_info in self.__fields__.items():
            # Check if field is required (compatible with both Pydantic versions)
            is_required = (
                getattr(field_info, 'required', False) or  # Pydantic v1
                (hasattr(field_info, 'is_required') and field_info.is_required()) or  # Pydantic v2 method
                (field_info.default is None and field_info.default_factory is None)  # Fallback check
            )
            if is_required:
                required_fields.append(field)

        # Calculate instruction fit, handling case where there are no required fields
        if required_fields:
            instruction_fit = sum(1.0 for field in required_fields
                                if getattr(self, field, None) is not None) / len(required_fields)
        else:
            instruction_fit = 1.0  # Perfect fit if no fields are required

        # A = Actionability (configuration is complete and usable)
        field_values = [getattr(self, field, None) for field in self.__fields__]
        actionability = sum(1.0 for value in field_values if value is not None) / len(field_values)

        # G = Grounding (source and versioning information present)
        grounding = 0.5 * (1.0 if self.source else 0.0) + 0.5 * (1.0 if self.config_version else 0.0)

        # Equal weights (0.25 each) as specified in framework
        context_score = 0.25 * (local_coherence + instruction_fit + actionability + grounding)

        return min(1.0, max(0.0, context_score))


class ProcessingConfig(BaseConfig):
    """
    Configuration for document processing operations.

    Theory Connection: Implements processing pipeline shared-context (`C_ext`) with
    specific focus on WHERE (file paths), WHAT (content processing),
    WHO (worker allocation), and TIME (timeout/batch settings).
    """

    # Worker configuration
    workers: int = Field(
        default=4,
        ge=1,
        le=128,
        description="Number of worker processes"
    )

    # Batch processing
    batch_size: int = Field(
        default=10,
        ge=1,
        le=1000,
        description="Batch size for processing"
    )

    # Resource limits
    memory_limit_gb: Optional[float] = Field(
        default=None,
        ge=0.1,
        description="Memory limit in GB"
    )

    timeout_seconds: int = Field(
        default=300,
        ge=1,
        description="Processing timeout in seconds"
    )

    # GPU configuration
    use_gpu: bool = Field(
        default=False,
        description="Enable GPU acceleration"
    )

    gpu_devices: Optional[List[int]] = Field(
        default=None,
        description="GPU device IDs to use"
    )

    def validate_semantics(self) -> List[str]:
        """
        Validate processing configuration semantics.

        Returns:
            List of validation errors
        """
        errors = []

        # GPU validation
        if self.use_gpu and self.gpu_devices is not None:
            if not self.gpu_devices:
                errors.append("GPU devices list cannot be empty when GPU is enabled")
            elif any(device < 0 for device in self.gpu_devices):
                errors.append("GPU device IDs must be non-negative")

        # Resource validation
        if self.memory_limit_gb is not None and self.memory_limit_gb < 1.0:
            if self.workers > 2:
                errors.append(f"Memory limit {self.memory_limit_gb}GB too low for {self.workers} workers")

        # Worker/batch relationship
        if self.batch_size > self.workers * 100:
            errors.append(f"Batch size {self.batch_size} may be too large for {self.workers} workers")

        return errors


class StorageConfig(BaseConfig):
    """
    Configuration for storage operations.

    Theory Connection: Implements WHERE dimension through hierarchical
    storage paths and access patterns. Ensures grounding (G) through
    explicit path validation and connection parameters.
    """

    # Database connection
    host: str = Field(
        default="localhost",
        description="Database host"
    )

    port: int = Field(
        default=8529,
        ge=1,
        le=65535,
        description="Database port"
    )

    username: str = Field(
        default="root",
        description="Database username"
    )

    password: str = Field(
        default="",
        description="Database password"
    )

    database: str = Field(
        default="academy_store",
        min_length=1,
        description="Database name"
    )

    # File storage paths
    pdf_directory: Optional[Path] = Field(
        default=None,
        description="Directory containing PDF files"
    )

    staging_directory: Optional[Path] = Field(
        default=None,
        description="Temporary staging directory"
    )

    # Connection settings
    connection_timeout: int = Field(
        default=30,
        ge=1,
        description="Connection timeout in seconds"
    )

    max_retries: int = Field(
        default=3,
        ge=0,
        description="Maximum connection retry attempts"
    )

    def validate_semantics(self) -> List[str]:
        """
        Validate storage configuration semantics.

        Returns:
            List of validation errors
        """
        errors = []

        # Path validation
        if self.pdf_directory is not None:
            if not self.pdf_directory.exists():
                errors.append(f"PDF directory does not exist: {self.pdf_directory}")
            elif not self.pdf_directory.is_dir():
                errors.append(f"PDF directory path is not a directory: {self.pdf_directory}")

        if self.staging_directory is not None:
            # Staging directory can be created if it doesn't exist
            try:
                self.staging_directory.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                errors.append(f"Cannot create staging directory {self.staging_directory}: {e}")

        # Database validation
        if not self.database.strip():
            errors.append("Database name cannot be empty or whitespace")

        if self.port <= 0 or self.port > 65535:
            errors.append(f"Invalid port number: {self.port}")

        return errors
