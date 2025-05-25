# Model Engine

## Overview

The Model Engine is a foundational component of HADES-PathRAG that provides a unified interface for language model operations, including embedding generation, text completion, and chat functionality. It supports multiple model backends (vLLM, Ollama, HuggingFace) and enables specialized models for different tasks.

## Architecture

The Model Engine is designed around the adapter pattern, with a clean separation of concerns:

```text
src/model_engine/
├── adapters/
│   ├── base.py             - Abstract base classes
│   ├── vllm_adapter.py     - vLLM implementation
│   └── __init__.py
├── server_manager.py       - Manages model server lifecycle
└── __init__.py             - Package exports
```

### Core Components

#### Adapter Interfaces

The base interfaces define consistent APIs for model operations:

- `ModelAdapter`: Base interface with common functionality
- `EmbeddingAdapter`: Interface for generating embeddings from text
- `CompletionAdapter`: Interface for text completion
- `ChatAdapter`: Interface for chat-based interactions

Each adapter implementation must adhere to these interfaces, ensuring consistent behavior regardless of the underlying model backend.

#### Configuration System

Model configurations are defined in:

- `src/config/model_config.py`: Pydantic models for type-safe configuration
- `src/config/model_config.yaml`: YAML configuration file with model settings

The configuration system is designed to be model-agnostic, supporting multiple backend types and specialized models for different tasks.

#### ServerManager Class

The `ServerManager` class handles the lifecycle of model servers, including:

- Starting and stopping model servers
- Ensuring the correct model is loaded
- Managing server resources

It provides a clean API for components that need model services without worrying about server details.

## Using the Model Engine

### Embedding Generation

```python
from src.model_engine.adapters.vllm_adapter import VLLMAdapter

# Create adapter
adapter = VLLMAdapter(
    model_name="BAAI/bge-large-en-v1.5",
    server_url="http://localhost:8000",
    normalize_embeddings=True
)

# Generate embeddings
texts = ["This is an example text.", "Another example text."]
embeddings = adapter.get_embeddings(texts)
```

### Text Completion

```python
from src.pathrag.model_engine_adapter import model_complete
import asyncio

async def get_completion():
    prompt = "Explain the concept of embeddings in machine learning."
    completion = await model_complete(prompt, model_alias="default")
    return completion

# Run asynchronously
result = asyncio.run(get_completion())
```

### Using the Server Manager

```python
from src.model_engine.server_manager import get_server_manager
import asyncio

async def ensure_model_running():
    manager = get_server_manager()
    # Start the 'code' model in inference mode
    server_running = await manager.ensure_server_running("code", mode="inference")
    return server_running

# Run asynchronously
is_running = asyncio.run(ensure_model_running())
```

## Model Configuration

The `model_config.yaml` file defines all available models and their settings. Models are organized into two categories:

### Ingestion Models

Used during document processing for:

- Embedding generation
- Text chunking
- Relationship extraction

Example configuration:

```yaml
ingestion:
  embedding:
    model_id: BAAI/bge-large-en-v1.5
    max_tokens: 512
    context_window: 4096
    truncate_input: true
    backend: vllm
    batch_size: 32
```

### Inference Models

Used during query time for:

- Generating responses
- Path planning
- Code generation

Example configuration:

```yaml
inference:
  default:
    model_id: meta-llama/Llama-3-8b
    max_tokens: 2048
    temperature: 0.7
    backend: vllm
    batch_size: 1
```

## Supported Backends

Currently, the Model Engine supports the following backends:

1. **vLLM**: High-performance inference engine with tensor parallelism
2. **Ollama** (planned): Lightweight local model server
3. **HuggingFace** (planned): Direct integration with Transformers

## Server Management

### Starting a Model Server

The Model Engine provides a script to start a model server:

```bash
python scripts/start_vllm_server.py --model-type embedding
```

This uses the configuration in `model_config.yaml` to start a server with the specified model.

### Server Lifecycle

The `ServerManager` class handles server lifecycle management:

1. Check if a server is already running
2. Start the server if needed
3. Monitor server health
4. Stop the server when done

This ensures efficient resource usage and seamless model switching.

## Extending the Model Engine

### Adding a New Backend

To add a new model backend:

1. Create a new adapter class that implements the required interfaces
2. Update the configuration system to support the new backend
3. Add server management support if needed

### Adding Specialized Models

To add a specialized model for a specific task:

1. Add the model configuration to `model_config.yaml`
2. Use the model by alias in your code

## Best Practices

1. **Use Model Aliases**: Reference models by their aliases in the configuration, not by specific model IDs
2. **Batch Processing**: When generating embeddings for large datasets, use appropriate batch sizes
3. **Resource Management**: Consider GPU memory requirements when configuring model servers
4. **Error Handling**: Implement proper error handling for model server failures

## Troubleshooting

### Common Issues

1. **Server Not Starting**: Check GPU availability and memory requirements
2. **Embedding Differences**: Ensure normalization settings are consistent
3. **Performance Issues**: Adjust batch sizes and tensor parallelism settings

### Logging

The Model Engine uses the standard Python logging module. To enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```
