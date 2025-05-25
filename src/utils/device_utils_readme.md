# Device Utilities

This module provides a standardized way to manage CPU/GPU device configuration in the HADES-PathRAG pipeline.

## Overview

The device utilities implement a flexible approach to controlling whether the pipeline uses CPU or GPU resources through the standard `CUDA_VISIBLE_DEVICES` environment variable.

## Key Features

- **Simple Configuration**: Control CPU/GPU usage through a single setting in the pipeline configuration
- **Industry Standard Approach**: Uses `CUDA_VISIBLE_DEVICES` which is universally supported by PyTorch, TensorFlow, and other ML libraries
- **Graceful Fallbacks**: Components automatically detect available devices and adapt accordingly
- **Runtime Diagnostics**: Detailed logging of device availability and configuration

## Usage

### In `pipeline_config.yaml`

```yaml
pipeline:
  mode: "training"
  save_intermediate_results: true
  output_dir: "./test-output"
  device_config:
    cuda_visible_devices: ""  # Empty string to force CPU, comma-separated device IDs for GPU (e.g., "0,1"), or null for system default
```

### Valid values for `cuda_visible_devices`:

- `""` (empty string): Force CPU-only mode
- `"0"`: Use only the first GPU
- `"1"`: Use only the second GPU
- `"0,1"`: Use the first and second GPUs
- `"1,2"`: Use the second and third GPUs
- `null`: Use system default (all available GPUs)

> **IMPORTANT:** When specifying a specific GPU (e.g., "1"), the system will still use it as "cuda:0" internally. This is because CUDA_VISIBLE_DEVICES creates a mapping where the first visible device always appears as device 0 to the application. For example, with `CUDA_VISIBLE_DEVICES="1"`, physical GPU 1 will be accessed as "cuda:0" in your code.

### In Code

```python
from src.utils.device_utils import is_gpu_available, get_device_info

# Check if GPU is available for use
if is_gpu_available():
    print("Running with GPU acceleration")
else:
    print("Running in CPU-only mode")

# Get detailed device information for logging or debugging
device_info = get_device_info()
print(f"Device type: {device_info['device_type']}")
print(f"Device name: {device_info['device_name']}")
```

## How It Works

1. The `DocumentProcessorManager` reads the `cuda_visible_devices` setting from the pipeline configuration
2. It sets this environment variable, affecting all libraries that respect CUDA_VISIBLE_DEVICES
3. Components check device availability at runtime using `is_gpu_available()`
4. If GPU is available, GPU-capable components use it; otherwise, they fall back to CPU

## Command-Line Override

You can also override the configuration setting via the command line:

```bash
CUDA_VISIBLE_DEVICES="" python -m your_script.py  # Force CPU
CUDA_VISIBLE_DEVICES="0" python -m your_script.py  # Use only first GPU
```

This approach provides maximum flexibility without requiring code changes to switch between CPU and GPU modes.
