# Configuration Management System

The configuration system provides hierarchical, validated configuration management for all HADES-Lab components, implementing the WHERE dimension of the Conveyance Framework through structured configuration positioning.

## Overview

This module embodies configuration as an "obligatory passage point" (Actor-Network Theory) through which all system components translate their requirements. Configuration validation ensures shared-context (`C_ext`) coherence across all processing phases.

## Architecture

```
config/
├── config_base.py      # Abstract base classes and validation
├── config_loader.py    # YAML/JSON configuration loading
├── config_manager.py   # Centralized configuration management
└── __init__.py        # Public API exports
```

## Theoretical Foundation

### Conveyance Framework Mapping

```python
C_pair = Hmean(C_out, C_in)·C_ext^α·P_ij
```

- **WHERE (R)**: Configuration hierarchy (environment > processor > base)
- **WHAT (W)**: Schema validation ensuring semantic content quality
- **WHO (H)**: Access patterns and permission structures
- **TIME (T)**: Configuration loading and parsing efficiency
- **Shared context (C_ext)**: Exponential amplification through configuration coherence

### Zero-Propagation Gate

Invalid configuration triggers `C_pair = 0`, preventing system operation with malformed parameters by collapsing shared context or the protocol gate.

## Core Components

### BaseConfig

Abstract base class for all configuration objects with Pydantic validation:

```python
from core.config import BaseConfig
from pydantic import Field

class PipelineConfig(BaseConfig):
    """Pipeline configuration with validation."""

    batch_size: int = Field(default=32, ge=1, le=1024)
    num_workers: int = Field(default=8, ge=1)
    timeout_seconds: float = Field(default=300.0, gt=0)

    def validate_coherence(self) -> bool:
        """Ensure configuration maintains Context coherence."""
        return self.batch_size * self.num_workers <= 256
```

### ConfigManager

Centralized configuration management with hierarchical loading:

```python
from core.config import ConfigManager

# Load configuration from YAML
config = ConfigManager.load_config("pipeline_config.yaml")

# Access nested configurations
extraction_config = config.get("extraction")
embedding_config = config.get("embedding")

# Validate all configurations
ConfigManager.validate_all(config)
```

### ConfigLoader

Low-level configuration loading with environment variable substitution:

```python
from core.config import ConfigLoader

# Load with environment variable expansion
loader = ConfigLoader()
config = loader.load_yaml("config.yaml", expand_env=True)

# Supports ${VAR_NAME} syntax in YAML
# database:
#   password: ${ARANGO_PASSWORD}
#   host: ${ARANGO_HOST:-localhost}
```

## Configuration Hierarchy

### 1. Environment Variables (Highest Priority)

```bash
export BATCH_SIZE=64
export NUM_WORKERS=16
export USE_GPU=true
```

### 2. Processor-Specific Configuration

```yaml
# extraction_config.yaml
extraction:
  batch_size: 32
  timeout_seconds: 300
  max_retries: 3
```

### 3. Base Configuration (Lowest Priority)

```yaml
# base_config.yaml
defaults:
  batch_size: 24
  num_workers: 8
  device: cuda
```

## Usage Patterns

### Basic Configuration

```python
from core.config import BaseConfig, ConfigManager

# Define configuration schema
class ExtractionConfig(BaseConfig):
    extractor_type: str = "docling"
    batch_size: int = 32
    timeout: float = 300.0
    max_pages: Optional[int] = None

# Load from file
config = ConfigManager.load_config("extraction.yaml")
extraction = ExtractionConfig(**config["extraction"])

# Validate
extraction.validate()
```

### Hierarchical Configuration

```python
# pipeline_config.yaml structure
pipeline:
  name: "ACID Pipeline"
  phases:
    extraction:
      workers: 32
      batch_size: 24
    embedding:
      workers: 8
      use_fp16: true
    storage:
      database: "academy_store"
      collection: "arxiv_metadata"

# Load hierarchical config
config = ConfigManager.load_config("pipeline_config.yaml")

# Access nested values
extraction_workers = config.get("pipeline.phases.extraction.workers")
# Returns: 32
```

### Dynamic Configuration

```python
from core.config import ConfigManager

# Runtime configuration updates
config = ConfigManager.load_config("base.yaml")

# Override with runtime values
if gpu_available():
    config.update("embedding.device", "cuda")
    config.update("embedding.batch_size", 64)
else:
    config.update("embedding.device", "cpu")
    config.update("embedding.batch_size", 8)

# Validate after updates
ConfigManager.validate_all(config)
```

## Validation System

### Schema Validation

```python
from pydantic import BaseModel, Field, validator

class DatabaseConfig(BaseConfig):
    host: str = Field(..., min_length=1)
    port: int = Field(default=8529, ge=1024, le=65535)
    database: str = Field(..., min_length=1)
    password: str = Field(..., min_length=8)

    @validator("host")
    def validate_host(cls, v):
        """Ensure host is reachable."""
        if not is_host_reachable(v):
            raise ValueError(f"Host {v} is not reachable")
        return v
```

### Coherence Validation

```python
class PipelineConfig(BaseConfig):
    extraction: ExtractionConfig
    embedding: EmbeddingConfig
    storage: StorageConfig

    def validate_coherence(self) -> bool:
        """
        Validate Context coherence across components.
        Ensures L (local), I (instruction), A (actionability), G (grounding).
        """
        # Local coherence: batch sizes must align
        if self.extraction.batch_size != self.embedding.batch_size:
            return False

        # Instruction fit: GPU configs must match
        if self.embedding.use_gpu and not gpu_available():
            return False

        # Actionability: storage must be configured
        if not self.storage.is_configured():
            return False

        return True
```

## Error Handling

### ConfigError

Base exception for configuration issues:

```python
from core.config import ConfigError, ConfigValidationError

try:
    config = ConfigManager.load_config("invalid.yaml")
except ConfigValidationError as e:
    print(f"Validation failed: {e.errors}")
    # ['batch_size must be positive', 'workers exceeds maximum']
except ConfigError as e:
    print(f"Configuration error: {e}")
```

## Best Practices

### 1. Always Use Validation

```python
# Good
config = ExtractionConfig(**raw_config)
config.validate()  # Explicit validation

# Better
config = ExtractionConfig.parse_obj(raw_config)  # Auto-validates
```

### 2. Provide Defaults

```python
class ProcessorConfig(BaseConfig):
    # Always provide sensible defaults
    batch_size: int = Field(default=32)
    timeout: float = Field(default=300.0)
    retries: int = Field(default=3)
```

### 3. Document Configuration

```python
class EmbeddingConfig(BaseConfig):
    """
    Embedding configuration.

    Attributes:
        model_name: Name of embedding model (e.g., "jinaai/jina-embeddings-v4")
        device: Compute device ("cuda" or "cpu")
        batch_size: Number of texts to process simultaneously
        use_fp16: Use half-precision for memory efficiency
    """
    model_name: str = Field(..., description="Embedding model identifier")
    device: str = Field(default="cuda", description="Compute device")
    batch_size: int = Field(default=32, ge=1, le=256, description="Batch size")
    use_fp16: bool = Field(default=True, description="Use FP16 precision")
```

### 4. Environment Variable Templates

Create `.env.example` files:

```bash
# Database Configuration
ARANGO_HOST=localhost
ARANGO_PORT=8529
ARANGO_PASSWORD=changeme
ARANGO_DATABASE=academy_store

# Processing Configuration
BATCH_SIZE=32
NUM_WORKERS=8
USE_GPU=true
CUDA_VISIBLE_DEVICES=0,1
```

## Performance Considerations

- **Lazy Loading**: Configurations are loaded on-demand
- **Caching**: Parsed configurations are cached for reuse
- **Validation Cost**: ~1-5ms per configuration object
- **Memory**: Typical configuration tree < 1MB

## Testing Configurations

```python
import pytest
from core.config import BaseConfig

def test_config_validation():
    """Test configuration validation."""
    config = MyConfig(
        batch_size=-1,  # Invalid
        timeout=300.0
    )

    with pytest.raises(ValidationError) as exc:
        config.validate()

    assert "batch_size" in str(exc.value)

def test_config_coherence():
    """Test Context coherence validation."""
    config = PipelineConfig.parse_obj({
        "extraction": {"batch_size": 32},
        "embedding": {"batch_size": 64}  # Mismatch
    })

    assert not config.validate_coherence()
```

## Migration Guide

### From Dict-based Config

```python
# Old approach
config = {
    "batch_size": 32,
    "workers": 8
}
batch_size = config.get("batch_size", 24)  # Manual defaults

# New approach
from core.config import BaseConfig

class Config(BaseConfig):
    batch_size: int = 32
    workers: int = 8

config = Config()  # Type-safe with validation
batch_size = config.batch_size
```

### From Environment Variables Only

```python
# Old approach
import os
batch_size = int(os.getenv("BATCH_SIZE", "32"))
workers = int(os.getenv("NUM_WORKERS", "8"))

# New approach
from core.config import ConfigLoader

loader = ConfigLoader()
config = loader.load_yaml("config.yaml", expand_env=True)
# YAML: batch_size: ${BATCH_SIZE:-32}
```

## Related Components

- [Workflows](../workflows/README.md) - Uses configurations for pipeline orchestration
- [Embedders](../embedders/README.md) - Embedding-specific configurations
- [Extractors](../extractors/README.md) - Extraction configurations
- [Database](../database/README.md) - Database connection configurations
