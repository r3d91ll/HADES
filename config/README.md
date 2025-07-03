# Configuration

## Overview
This directory contains all configuration files for HADES, organized by component. All configurations use YAML format for readability and support environment variable substitution.

## Structure
- `jina_v4/` - Jina v4 processor configuration
- `isne/` - ISNE model and training configurations
- `storage/` - Database and storage configurations
- `pathrag_config.yaml` - PathRAG-specific settings

## Configuration Hierarchy
1. Default values in code
2. Configuration files (this directory)
3. Environment variables (override)
4. Runtime parameters (highest priority)

## Key Configurations

### Jina v4 (`jina_v4/config.yaml`)
- Model selection and parameters
- vLLM engine settings
- Late chunking configuration
- Keyword extraction settings
- Multimodal processing options

### ISNE (`isne/`)
- Model architecture parameters
- Training hyperparameters
- Directory-aware settings
- Enhancement strength

### Storage
- ArangoDB connection settings
- Collection names
- Index configurations
- Performance tuning

## Environment Variables
```bash
# Database
ARANGO_URL=http://localhost:8529
ARANGO_USER=root
ARANGO_PASS=password

# Model paths
MODEL_CACHE_DIR=/path/to/models
JINA_MODEL_PATH=jinaai/jina-embeddings-v4

# GPU settings
CUDA_VISIBLE_DEVICES=0,1
VLLM_GPU_MEMORY_UTILIZATION=0.9
```

## Usage
```python
# Load configuration
from src.config import load_config
config = load_config('jina_v4/config.yaml')

# Override with environment
config = load_config('jina_v4/config.yaml', use_env=True)
```

## Best Practices
1. Keep sensitive data in environment variables
2. Use descriptive comments in YAML files
3. Maintain consistent structure across configs
4. Version control configuration changes
5. Document non-obvious settings

## Related Resources
- Implementation: `/src/` uses these configurations
- Documentation: `/docs/configuration/` for detailed guides
- Examples: `/scripts/examples/` for usage patterns