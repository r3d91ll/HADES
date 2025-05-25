# Configuration Module

This module provides utilities for loading and managing configuration files for different pipelines in the HADES-PathRAG system.

## Pipeline Configuration Files

The system now supports multiple pipeline configurations:

- `training_pipeline_config.yaml`: Configuration for the training pipeline
- `ingestion_pipeline_config.yaml`: Configuration for the data ingestion pipeline (planned)

Each pipeline configuration file contains settings specific to that pipeline's operation, including device configurations, batch sizes, and component-specific parameters.

## Configuration API

### Loading Pipeline Configuration

```python
from src.config.config_loader import load_pipeline_config

# Load the training pipeline configuration (default)
training_config = load_pipeline_config()

# Load a specific pipeline configuration
ingestion_config = load_pipeline_config(pipeline_type='ingestion')

# Load from a custom path
custom_config = load_pipeline_config(config_path='/path/to/custom_config.yaml')
```

### Getting Device Configuration

```python
from src.config.config_loader import get_device_config, get_component_device

# Get device configuration for the training pipeline
device_config = get_device_config()
print(f"Mode: {device_config['mode']}")  # 'gpu', 'cpu', or 'auto'

# Get device for a specific component
docproc_device = get_component_device('docproc')
chunking_device = get_component_device('chunking')
embedding_device = get_component_device('embedding')

# For a different pipeline
ingestion_docproc_device = get_component_device('docproc', pipeline_type='ingestion')
```

## Configuration Structure

### Training Pipeline Configuration

The training pipeline configuration (`training_pipeline_config.yaml`) has the following structure:

```yaml
# Pipeline settings
pipeline:
  mode: "training"
  save_intermediate_results: true
  output_dir: "./test-output"
  device_config:
    CUDA_VISIBLE_DEVICES: "0,1"  # Controls which physical GPUs are visible

# Device execution configurations
gpu_execution:
  enabled: true
  docproc:
    device: "cuda:0"
    batch_size: 4
  chunking:
    device: "cuda:0"
    batch_size: 8
  embedding:
    device: "cuda:0"
    batch_size: 8
    model_precision: "float16"

cpu_execution:
  enabled: false
  docproc:
    device: "cpu"
    num_threads: 8
  # Additional CPU settings...
```

## Component Integration

### Using Device Configuration in Components

Components should use the provided configuration utilities to determine their device settings:

```python
from src.config.config_loader import get_component_device

# Get the configured device for this component
device = get_component_device('chunking')
if device:
    # Use the configured device
    print(f"Using configured device: {device}")
else:
    # Fall back to default device
    device = "cpu"
    print(f"No specific device configured, using default: {device}")
```

## Testing

Tests for the configuration module are located in `tests/unit/utils/test_device_config.py`.

Run the tests with:

```bash
pytest tests/unit/utils/test_device_config.py
```

## Future Additions

- Support for configuration templates and inheritance
- Environment-specific configuration overrides
- Configuration validation and schema enforcement
