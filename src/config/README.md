# HADES Configuration System

This directory contains the configuration files for all HADES components, organized by component type and implementation.

## Directory Structure

The configuration structure mirrors the component architecture:

```text
src/config/
├── components/            # Component-specific configurations
│   ├── chunking/          # Chunking component configurations
│   │   ├── core/          # Core chunking implementation
│   │   ├── text/          # Text-specific chunking
│   │   ├── code/          # Code-specific chunking  
│   │   ├── chonky/        # GPU-accelerated chunking
│   │   └── cpu/           # CPU-optimized chunking
│   ├── database/          # Database component configurations
│   │   └── arango/        # ArangoDB configuration
│   ├── docproc/           # Document processing configurations
│   │   ├── core/          # Core document processor
│   │   └── docling/       # Docling-based processor
│   ├── embedding/         # Embedding component configurations
│   │   ├── core/          # Core embedding implementation
│   │   ├── cpu/           # CPU embedding implementation
│   │   ├── encoder/       # Encoder-based embeddings
│   │   └── vllm/          # vLLM-based embeddings
│   ├── isne/              # ISNE component configurations
│   │   ├── core/          # Core ISNE implementation
│   │   ├── training/      # ISNE training configuration
│   │   ├── inference/     # ISNE inference configuration
│   │   ├── inductive/     # Inductive ISNE configuration
│   │   └── none/          # Passthrough/no enhancement
│   ├── model_engine/      # Model engine configurations
│   │   ├── core/          # Core model engine
│   │   ├── vllm/          # vLLM engine configuration
│   │   └── haystack/      # Haystack engine configuration
│   ├── schemas/           # Schema validation configurations
│   │   └── pydantic/      # Pydantic validation configs
│   └── storage/           # Storage component configurations
│       ├── arangodb/      # ArangoDB storage
│       ├── memory/        # In-memory storage
│       └── networkx/      # NetworkX graph storage
├── pipelines/             # Pipeline-level configurations
│   ├── bootstrap/         # Bootstrap pipeline configs
│   ├── data_ingestion/    # Data ingestion configs
│   └── training/          # Training pipeline configs
├── api/                   # API-specific configurations
├── alerts/                # Alert system configurations
└── orchestration/         # Orchestration configurations
```

## Configuration Loading

The configuration system supports both legacy and component-based loading:

### Component-Based Loading (Recommended)

```python
from src.config.config_loader import load_config

# Load specific component configuration
chunking_config = load_config("components/chunking/core/config")
embedding_config = load_config("components/embedding/vllm/config")
model_engine_config = load_config("components/model_engine/vllm/config")
```

### Legacy Support

The system maintains backward compatibility:

```python
# These legacy names are automatically mapped to new paths
chunker_config = load_config("chunker_config")  # -> components/chunking/core/config
embedding_config = load_config("embedding_config")  # -> components/embedding/core/config
vllm_config = load_config("vllm_config")  # -> components/model_engine/vllm/config
```

## Configuration Files

Each component configuration file follows the pattern:

```yaml
# Component-specific configuration
version: 1

# Component settings
component:
  enabled: true
  implementation: "core"  # or "vllm", "haystack", etc.
  
# Implementation-specific configuration
# ... (varies by component)

# Performance settings
performance:
  use_gpu: true
  batch_size: 32
  
# Monitoring settings
monitoring:
  enabled: true
  metrics_interval: 10
```

## Adding New Components

To add a new component configuration:

1. Create the component directory: `src/config/new_component/`
2. Create implementation subdirectories: `src/config/new_component/impl1/`, `src/config/new_component/impl2/`
3. Add configuration files: `src/config/new_component/impl1/config.yaml`
4. Update the component factory to recognize the new component
5. Add type definitions in `src/types/new_component/`

## Environment-Specific Configurations

Configurations can be overridden for different environments:

- Development: Use default configurations
- Production: Override with environment variables or separate config files
- Testing: Use minimal/mock configurations

## Configuration Validation

All configurations are validated using Pydantic models defined in `src/types/components/contracts.py`.

The validation ensures:

- Required fields are present
- Data types are correct
- Value ranges are valid
- Component contracts are satisfied
