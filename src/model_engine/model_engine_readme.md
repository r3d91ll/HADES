# Model Engine Module Documentation

## Overview

The model engine module provides a unified architecture for loading, managing, and using machine learning models across different backends. It abstracts away the implementation details of various ML frameworks and allows for consistent model usage throughout HADES-PathRAG. A flexible architecture that allows for multiple backend implementations.

## Architecture

### Core Components

1. **ModelEngine Interface** (`base.py`):
   - Abstract base class that defines the standard interface for all model engines
   - Includes methods for loading/unloading models and running inference

2. **Engine Implementations**:
   - Located in the `engines/` directory
   - Each implementation extends the `ModelEngine` base class
   - Current implementations:
     - **VLLMModelEngine**: Uses vLLM for high-performance inference
     - **HaystackModelEngine**: Uses Haystack for document processing pipelines

3. **Adapters** (Legacy/Deprecated):
   - Located in the `adapters/` directory
   - Older pattern that should be migrated to the engines pattern
   - New code should use the engines pattern instead

## Usage Guidelines

### Preferred Usage Pattern

```python
from src.model_engine.engines.vllm import VLLMModelEngine
from src.model_engine.engines.haystack import HaystackModelEngine

# Initialize the appropriate engine
engine = VLLMModelEngine()

# Start the engine service
engine.start()

# Load a model
engine.load_model("meta-llama/Llama-2-7b-chat-hf")

# Use the engine for inference
result = await engine.generate_completion("Your prompt here")
```

### Legacy Adapter Pattern (To Be Refactored)

```python
from src.model_engine.adapters.vllm_adapter import VLLMAdapter

# Initialize the adapter
adapter = VLLMAdapter(model_name="BAAI/bge-large-en-v1.5")

# Use the adapter directly
embeddings = await adapter.generate_embeddings(["Your text here"])
```

## Migration Path

Code that currently uses the adapter pattern should be gradually migrated to use the engine pattern. The adapter pattern is being maintained for backward compatibility but will eventually be deprecated.

When extending or modifying model functionality:

1. Implement new features using the engine pattern
2. Update existing adapter-based code to use engines when possible

## Engine-specific Notes

### VLLMModelEngine

- Manages vLLM processes for inference
- Provides a high-performance implementation for both embedding and text generation
- Handles batch processing efficiently

### HaystackModelEngine

- Integrates with Haystack for document processing pipelines
- Well-suited for RAG (Retrieval Augmented Generation) applications
- Includes integration with document processing components

## Testing

### Test Coverage Requirements

All model engines must meet a minimum of 85% unit test coverage, focusing on the public API methods. Tests should verify:

- Engine lifecycle (start, stop, restart)
- Model management (load, unload)
- Status and health checking
- Error handling and edge cases
- Context manager functionality

### Testing Strategy

- **Unit Testing**: Each engine is tested in isolation with mocked dependencies
- **Integration Testing**: Inter-module communication is tested with real or realistic test fixtures
- **Performance Benchmarks**: Critical operations are benchmarked for performance comparison

### Current Coverage Status

- **HaystackModelEngine**: 89% coverage of core implementation
- **VLLMModelEngine**: *In progress* - to be completed with the same standard

To run the tests and check coverage:

```bash
python -m pytest tests/unit/model_engine/test_haystack_final.py --cov=src.model_engine.engines.haystack --cov-report=term
```
