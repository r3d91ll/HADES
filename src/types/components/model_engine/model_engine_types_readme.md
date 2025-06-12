# Model Engine Types

This directory contains centralized type definitions for the model engine module in HADES.

## Overview

The model engine types provide consistent interfaces and type definitions for working with different language model backends. The centralized type system ensures type safety and consistency across the codebase. Recent improvements include:

1. Created a dedicated `results.py` module defining standardized adapter result types
2. Implemented comprehensive test suite with mock adapters
3. Successfully migrated the VLLM adapter to use the centralized type system
4. Fixed type incompatibilities throughout the model engine module

## Files

- `engine.py`: Core model engine interface types
  - `ModelEngineProtocol`: Protocol defining the required interface for model engines
  - `EngineType`, `ModelType`, `ModelStatus`, `EngineStatus`: Enums for consistent status and type definitions
  - `ModelInfo`, `HealthStatus`: TypedDict classes for model and health information

- `adapters.py`: Types for model adapters that provide a standardized interface to different backends
  - `ModelAdapterProtocol`: Protocol defining the required interface for model adapters
  - `ModelAdapterConfig`: TypedDict class for adapter configuration
  - `ModelAdapterRegistry`: Type definition for adapter registry

- `results.py`: Standardized result types for adapter operations
  - `EmbeddingResult`: TypedDict for embedding operation results
  - `CompletionResult`: TypedDict for text completion results
  - `ChatCompletionResult`: TypedDict for chat completion results
  - `StreamingChatCompletionResult`: TypedDict for streaming chat results
  - `ChatMessage`, `ChatRole`: Common chat message types and enums

- `inference.py`: Request and response types for model inference
  - Request types: `CompletionRequest`, `ChatCompletionRequest`, `EmbeddingRequest`
  - Response types: `CompletionResponse`, `ChatCompletionResponse`, `EmbeddingResponse`
  - Common types: `InferenceRequest`, `InferenceResponse`

- `config.py`: Configuration types for model engines
  - Engine-specific configs: `HaystackEngineConfigType`, `VLLMConfigType`
  - Common type: `EngineConfigType`

## Usage

Import types directly from these modules rather than defining them inline or importing from engine-specific modules:

```python
# CORRECT:
from src.types.model_engine.engine import ModelEngineProtocol, EngineType
from src.types.model_engine.inference import CompletionRequest

# INCORRECT:
# from src.model_engine.engines.vllm.vllm_engine import SomeType
# from src.types.vllm_types import ModelMode  # Use types.model_engine.engine instead
```

## Migration

When working with existing code:

1. Replace inline type definitions with imports from these centralized modules
2. Update imports from engine-specific modules to use these centralized types
3. Add deprecation warnings to legacy type locations that point to these new types
4. Ensure consistent type usage across the codebase

## Type Safety Guidelines

- Use Protocols and TypedDicts rather than custom classes where appropriate
- Use Union types for values that can have multiple types
- Use Optional rather than Union[X, None]
- Include descriptive docstrings for all type definitions
- Use `total=False` for TypedDict when not all fields are required
