# Haystack Model Engine

The Haystack Model Engine is a robust implementation of the `ModelEngine` interface that provides efficient management of large language models using the Haystack framework. It features a client-server architecture with a Unix domain socket for communication, memory-efficient model loading, and a Linux-style service management interface.

## Overview

The Haystack Model Engine is designed to efficiently manage multiple language models in memory, with automatic loading/unloading based on usage patterns. It supports various model types (embedding, classification, etc.) and provides a clean API for model operations.

### Key Features

- **Memory-Efficient Model Management**: Uses an LRU cache to automatically unload least-recently-used models when memory limits are reached
- **Multiple Model Support**: Can load and manage multiple models simultaneously
- **Service-Oriented Architecture**: Client-server design with Unix domain socket communication
- **Linux-Style Service Management**: Familiar `start`, `stop`, `restart`, and `status` commands

## Architecture

The engine is composed of the following components:

### HaystackModelEngine

The main entry point for clients, implementing the `ModelEngine` interface defined in `src/model_engine/base.py`. This provides high-level operations for model management and handles service lifecycle operations.

### Client-Server Communication

- **ModelClient**: A lightweight client that communicates with the model manager server over a Unix domain socket.
- **Server**: A long-running process that manages model loading, unloading, and inference. Runs as a background service.

### LRU Cache

The server uses an LRU (Least Recently Used) cache to efficiently manage models in memory. When memory limits are reached, the least recently used models are automatically unloaded.

## Usage

### Service Management

The Haystack Model Engine follows Linux service conventions with the following operations:

```python
from src.model_engine.engines.haystack import HaystackModelEngine

# Create an engine instance
engine = HaystackModelEngine()

# Start the service
engine.start()

# Check service status
status = engine.status()
print(status)

# Restart the service
engine.restart()

# Stop the service
engine.stop()
```

### Model Operations

Once the service is running, you can manage models with the following operations:

```python
# Load a model
result = engine.load_model("mirth/chonky_modernbert_large_1")

# Get information about loaded models
loaded_models = engine.get_loaded_models()

# Unload a model
engine.unload_model("mirth/chonky_modernbert_large_1")

# Check engine health
health = engine.health_check()
```

### Using the Command-Line Interface

A command-line interface is provided through the `scripts/model_service_manager.py` script:

```bash
# Start the model manager service
python scripts/model_service_manager.py start

# Load a model
python scripts/model_service_manager.py load mirth/chonky_modernbert_large_1

# List loaded models
python scripts/model_service_manager.py list

# Get service status
python scripts/model_service_manager.py status

# Unload a model
python scripts/model_service_manager.py unload mirth/chonky_modernbert_large_1

# Stop the service
python scripts/model_service_manager.py stop
```

### Context Manager Interface

The engine can be used as a context manager for automatic resource management:

```python
with HaystackModelEngine() as engine:
    # Load model
    engine.load_model("mirth/chonky_modernbert_large_1")
    
    # Use the model
    # ...
    
    # The service will be automatically stopped when exiting the context
```

## Configuration

The engine can be configured using the following environment variables:

- `HADES_MAX_MODELS`: Maximum number of models to keep in memory (default: 3)
- `HADES_DEFAULT_DEVICE`: Default device for model loading (default: "cuda:0")
- `HADES_MODEL_MGR_SOCKET`: Path for the Unix socket (default: "/tmp/hades_model_mgr.sock")
- `HADES_RUNTIME_AUTOSTART`: Whether to automatically start the server (default: "1")

## Multiple Model Types

The engine supports loading multiple models of different types simultaneously:

```python
# Load a chunking model
engine.load_model("mirth/chonky_modernbert_large_1")

# Load an embedding model
engine.load_model("sentence-transformers/all-MiniLM-L6-v2")

# Both models are now available for use
```

## Implementation Details

### Model Loading

Models are loaded using Hugging Face's `AutoModel` and `AutoTokenizer` classes. When a model ID is provided (e.g., "mirth/chonky_modernbert_large_1"), the engine will:

1. Check if the model is already loaded in memory
2. If not, download the model from Hugging Face (if not already cached locally)
3. Load the model into memory, moving it to the specified device
4. Store the model and tokenizer in the LRU cache

### Memory Management

The engine uses an LRU cache to manage models in memory. When the number of loaded models exceeds `HADES_MAX_MODELS`, the least recently used model is automatically unloaded to free up memory.

### Resource Cleanup

When the service is stopped, all models are unloaded and resources are freed. This ensures that GPU memory is properly released.

## Troubleshooting

### Service Not Starting

If the service fails to start, check:

- If the socket file already exists (`/tmp/hades_model_mgr.sock` by default)
- If another instance of the service is already running
- If you have sufficient permissions to create the socket file

### Model Loading Failures

If models fail to load, check:

- If you have sufficient GPU memory
- If the model ID is correct
- If you have internet connectivity (for downloading models)
- If the model is compatible with Hugging Face's `AutoModel`

### Common Errors

- "Connection refused": The server is not running or the socket path is incorrect
- "Out of memory": Not enough GPU memory to load the model
- "Model not found": The specified model ID doesn't exist on Hugging Face
