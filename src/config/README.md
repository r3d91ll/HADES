# Configuration

## Overview
This directory mirrors the src/ structure exactly, providing configuration files for each module. This pattern ensures consistency for both human maintainability and ISNE pattern learning.

## Structure (Mirrors src/)
- `common.yaml` - Common settings shared across modules
- `api/` - API server and endpoint configurations
- `concepts/` - Concept processing configurations
- `isne/` - ISNE model, training, and bootstrap configurations
- `jina_v4/` - Jina v4 processor configuration
- `pathrag/` - PathRAG strategy and retrieval configurations
- `storage/` - Storage backend configurations
- `utils/` - Utility configurations
  - `filesystem/` - File and directory processing settings
- `validation/` - Validation configurations

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

## Design Principles
1. **Mirror Structure**: Each src/ module has corresponding config/ directory
2. **YAML Format**: Human-readable configuration files
3. **Environment Overrides**: Support for environment variable substitution
4. **Module Ownership**: Each module owns its configuration
5. **Pattern Learning**: Consistent structure aids ISNE directory understanding

## Usage
```python
# Load configuration
from src.config import load_config
config = load_config('src/config/jina_v4/config.yaml')

# Override with environment
config = load_config('src/config/jina_v4/config.yaml', use_env=True)
```

## Best Practices
1. Keep sensitive data in environment variables
2. Use descriptive comments in YAML files
3. Maintain consistent structure across configs
4. Version control configuration changes
5. Document non-obvious settings

## Directory Mirroring Benefits
1. **3AM Debugging**: Easy to find configs for any module
2. **Pattern Recognition**: ISNE can learn from consistent structure
3. **Maintainability**: Clear 1:1 mapping between code and configs
4. **Discoverability**: Config location is predictable from module path

## Related Resources
- Type Definitions: `/src/types/` - mirrors same structure for types
- Implementation: `/src/` - uses these configurations
- Documentation: `/docs/configuration/` - detailed guides
- Examples: `/scripts/examples/` - usage patterns