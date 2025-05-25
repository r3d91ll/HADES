# Device Configuration Implementation

## Overview

This document explains the changes made to ensure that all components (document processing, chunking, and embedding) 
properly respect the device configuration settings in the pipeline configuration file.

## Problem

Previously, even though `cuda:1` was specified in the pipeline configuration, some components defaulted to `cuda:0`,
leading to inefficient GPU resource utilization and potential device conflicts.

## Solution

We've implemented a unified approach to device configuration that ensures all components respect the settings
in the pipeline configuration file:

1. Created a centralized configuration loader (`src/config/config_loader.py`)
2. Updated all components to properly read and use the device settings
3. Added proper logging for improved transparency and debugging
4. Created unit and integration tests to verify the device configuration

## Implementation Details

### 1. Configuration Loader

Created a new configuration loader module (`src/config/config_loader.py`) that:
- Loads the pipeline configuration from YAML
- Provides utilities to get component-specific settings
- Resolves device settings based on configured priorities

### 2. Document Processing

Updated `DoclingAdapter` to:
- Check for device configuration in the pipeline settings
- Properly respect the `cuda:1` device setting
- Add detailed logging about device selection

### 3. Chunking Component

Updated `chonky_chunker.py` to:
- Check pipeline configuration for device settings 
- Use the configured device in both the model engine and the splitter
- Propagate the device setting throughout the chunking process

### 4. Embedding Component

Updated `ModernBERTEmbeddingAdapter` to:
- Read device configuration from pipeline settings
- Prioritize configurations correctly (explicit parameter > pipeline config > adapter config)
- Adjust device mappings for CUDA_VISIBLE_DEVICES support

### 5. Testing

Added tests to verify the device configuration implementation:
- Unit tests for the configuration loader
- Integration tests for all components
- Mocked GPU environment for testing without actual GPU hardware

## Usage

The components now respect the device configuration in `pipeline_config.yaml`:

```yaml
gpu_execution:
  enabled: true
  docproc:
    device: "cuda:1"
    batch_size: 8
  chunking:
    device: "cuda:1" 
    batch_size: 8
  embedding:
    device: "cuda:1"
    batch_size: 8
```

## Future Work

- Add support for multi-GPU configurations with device mapping
- Implement dynamic device selection based on memory availability
- Add monitoring for device usage during the ISNE training phase

## Testing

Run unit tests to verify the implementation:

```bash
pytest tests/unit/utils/test_device_config.py -v
```

Run integration tests to verify the end-to-end behavior:

```bash
pytest tests/integration/test_pipeline_device_config.py -v
```
