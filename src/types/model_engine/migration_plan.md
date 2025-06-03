# Model Engine Type System Migration Plan

## Overview

This document outlines the step-by-step process to migrate existing model engine code to use the new centralized type system.

## Migration Steps

### 1. Update Base Model Engine Interface

- [x] Modify `src/model_engine/base.py` to import types from `src/types/model_engine/engine.py`
- [x] Replace any inline type definitions with imported types
- [x] Ensure the `ModelEngine` abstract base class matches the `ModelEngineProtocol`

### 2. Update VLLM Engine Implementation

- [x] Modify `src/model_engine/engines/vllm/engine.py` to import from centralized type system
- [x] Replace key methods to use new type definitions (get_loaded_models, health_check)
- [ ] Complete conversion of remaining methods to use centralized types
- [x] Add deprecation warning to `src/types/vllm_types.py` pointing to new location

### 3. Update Haystack Engine Implementation

- [x] Modify `src/model_engine/engines/haystack/__init__.py` to import from centralized type system
- [x] Update `get_loaded_models` method to return standardized `ModelInfo` objects
- [x] Update `health_check` method to return standardized `HealthStatus` TypedDict
- [x] Complete conversion of remaining methods to use centralized types
- [x] Update type annotations to match new interface definitions
- [x] Fix type incompatibilities discovered during migration

### 4. Update Factory Implementation

- [x] Modify `src/model_engine/factory.py` to use centralized engine types
- [x] Update `ENGINE_REGISTRY` to use proper type definitions
- [x] Fix compatibility issues in engine creation functions

### 5. Update Adapter Implementations

- [x] Modify base adapter interfaces in `src/model_engine/adapters/base.py` to use centralized types
- [x] Update `ModelAdapter`, `EmbeddingAdapter`, `CompletionAdapter`, and `ChatAdapter` base classes
- [x] Update concrete adapter implementation (VLLMAdapter)
  - [x] Add required `model_id` property implementation
  - [x] Add required `is_available` property implementation
  - [x] Implement `get_config` method returning `ModelAdapterConfig`
  - [x] Update `get_embeddings` method to return standardized `EmbeddingResult`
  - [x] Update `complete` and `complete_async` methods to return standardized `CompletionResult`
  - [x] Update `chat_complete` and `chat_complete_async` methods to return standardized `ChatCompletionResult`
  - [x] Implement `chat_complete_stream_async` with standardized `StreamingChatCompletionResult`
- [ ] Update remaining adapter implementations with centralized types

### 6. Create Centralized Result Types

- [x] Create `src/types/model_engine/results.py` for standardized adapter result types
- [x] Define `EmbeddingResult`, `CompletionResult`, `ChatCompletionResult`, and `StreamingChatCompletionResult`
- [x] Define option types for adapter operations
- [x] Update imports in adapters to use centralized result types

### 7. Run Type Validation and Testing

- [x] Create test suite to verify adapter implementations with new type system
- [x] Run tests with mock adapters to verify type compatibility
- [x] Fix type incompatibilities discovered during validation
- [x] Run mypy on updated adapter and engine implementations
- [x] Fix type issues in Haystack engine implementation
- [x] Run mypy on VLLM adapter implementation to verify type compatibility
- [x] Fix type issues in VLLM adapter implementation
- [ ] Run mypy on remaining adapter implementations
- [ ] Run integration tests to ensure runtime behavior is preserved
- [ ] Run mypy across the entire model_engine module
- [ ] Run all model engine tests to ensure runtime compatibility
- [ ] Update tests to use the new centralized types

### 8. Final Cleanup

- [ ] Remove any duplicated type definitions
- [ ] Ensure consistent import patterns across the codebase
- [ ] Update documentation to reference the new type system

## Best Practices for Migration

1. Make incremental changes, testing each step
2. Commit frequently to track progress
3. Use mypy to verify type correctness after each change
4. Document any compatibility issues or backward compatibility requirements
5. Follow the migration pattern:
   - Import new types
   - Replace old types in annotations
   - Fix implementations to match new interfaces
   - Test with mypy
   - Run unit tests
