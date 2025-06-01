# HADES-PathRAG Development Tasks

## Pipeline Architecture Enhancements

This section outlines the planned improvements to the pipeline architecture for enhanced modularity, scalability, and maintainability. These changes are designed to prepare the ingestion pipeline for separation into dedicated services.

### High Priority

#### 1. Pipeline Stage Interface Implementation

- [x] Create `src/pipeline/stages` directory with appropriate module structure
- [x] Implement abstract `PipelineStage` base class with:
  - [x] `run(input_data: Dict) -> Dict` method
  - [x] `validate(input_data: Dict) -> bool` method
  - [x] Standardized error handling interface
  - [x] Logging mechanisms
- [x] Create comprehensive documentation for stage interface
- [x] Implement unit tests for stage interface validation

#### 2. Concrete Stage Implementations

- [x] Implement `DocumentProcessingStage` class wrapping document processing functionality
  - [x] Add support for different document adapter types
  - [x] Implement validation for document processing inputs/outputs
- [x] Implement `ChunkingStage` class for document chunking
  - [x] Support both text and code chunking
  - [x] Add chunk validation mechanisms
- [x] Implement `EmbeddingStage` for generating embeddings
  - [x] Support multiple embedding models
  - [x] Include embedding validation
- [x] Implement `ISNEStage` for graph-based embedding enhancement
  - [x] Support relationship preservation
  - [x] Include validation for ISNE embeddings and relationships
- [x] Create comprehensive tests for each stage

#### 3. Pydantic Schema Definition

- [x] Create `src/pipeline/schema.py` module with Pydantic models
- [x] Define core data models:
  - [x] `ChunkSchema` model with embeddings and relationships
  - [x] `DocumentSchema` model with chunks and metadata
  - [x] `ValidationResult` model for tracking validation status
- [x] Integrate schema validation with pipeline stages
- [x] Create tests for schema validation
- [x] Refactor to dedicated `src/schemas` directory with domain-specific schema organization
- [x] Fix type issues in schema validators for improved type safety
  - [x] Replace untyped decorators with typed equivalents (`typed_field_validator` and `typed_model_validator`)
  - [x] Properly handle parameter types and return types in validators
  - [x] Add documentation on validator type safety best practices

#### 4. Storage Stage Implementation

- [x] Implement `StorageStage` for ArangoDB interaction
  - [x] Support different storage modes (create/append/upsert)
  - [x] Add retry logic with exponential backoff
  - [x] Implement transactional operations
- [x] Add validation for database operations
- [x] Create tests for storage stage

### Medium Priority

#### 5. Schema Type Safety & Code Quality

- [x] Fix type issues across schema files:
  - [x] Create utility scripts for automating type fixes (`fix_type_issues.py`, `fix_schemas_type_issues.py`)
  - [x] Fix validator method type annotations and return types
  - [x] Correct parameter types in validator methods (`self` → `cls`, typed values dictionaries)
  - [x] Fix unreachable code issues and missing return statements
  - [x] Update schemas_readme.md with type safety guidelines
- [ ] Address remaining type issues in other modules
  - [x] Fix typing issues in adapter modules
    - [x] Fix Python code adapter typing (`python_code_adapter.py`)
    - [x] Fix Docling adapter typing (`docling_adapter.py`)
    - [x] Fix Python adapter typing (`python_adapter.py`)
    - [x] Improve test coverage for adapters to ≥85% standard
  - [ ] Implement centralized type system
    - [x] Create src/types/docproc/ directory (rename from src/types/documents/)
    - [x] Create src/types/chunking/ directory for chunking types
    - [x] Create src/types/storage/ directory for storage types
    - [x] Create src/types/pathrag/ directory for PathRAG types
    - [x] Create src/types/indexing/ directory for indexing types
    - [x] Create src/types/model_engine/ directory for model engine types
    - [x] Create src/types/embedding/ directory for embedding types
    - [x] Create src/types/isne/ directory for ISNE types
    - [x] Update type migration plan document in src/types/types_readme.md

    ### Type System Refactoring Strategy

    Since we're in active development, we'll use a **replace-don't-migrate** approach to establish a clean type system without complex migrations. Each module will be refactored completely with proper type annotations.

    ### Git Strategy for Type Refactoring

    For each module:
    1. Create a feature branch: `git checkout -b feature/types-{module_name}`
    2. Complete all type refactoring for that module
    3. Run type checks with mypy
    4. Create a pull request for review
    5. After approval, merge to main branch

    ### Module Refactoring Order and Approach

    #### 1. docproc (First Priority)

    - **Branch**: `feature/types-docproc`
    - **Type Definitions**:
      - `src/types/docproc/document.py`: Document type definitions
      - `src/types/docproc/metadata.py`: Metadata type definitions
      - `src/types/docproc/adapter.py`: Adapter interface types
      - `src/types/docproc/processing.py`: Processing result types
    - **Implementation Steps**:
      - Replace all internal types with centralized types
      - Fix any syntax errors in type annotations
      - Ensure consistent usage of types across the module
      - Run mypy to verify type correctness

    #### 2. model_engine (Second Priority)

    - **Branch**: `feature/types-model-engine`
    - **Type Definitions**:
      - `src/types/model_engine/engine.py`: Engine interface types
      - `src/types/model_engine/adapters.py`: Model adapter types
      - `src/types/model_engine/inference.py`: Inference request/response types
      - `src/types/model_engine/config.py`: Configuration types
    - **Implementation Steps**:
      - Refactor all engine implementations to use centralized types
      - Standardize interface for all engine types
      - Run mypy to verify type correctness

    #### 3. chunking (Third Priority)

    - **Branch**: `feature/types-chunking`
    - **Type Definitions**:
      - `src/types/chunking/chunk.py`: Chunk data types
      - `src/types/chunking/strategy.py`: Chunking strategy interfaces
      - `src/types/chunking/text.py`: Text-specific chunking types
      - `src/types/chunking/code.py`: Code-specific chunking types
    - **Implementation Steps**:
      - Replace local type definitions with centralized types
      - Ensure compatibility with document types
      - Run mypy to verify type correctness

    #### 4. embedding (Fourth Priority)

    - **Branch**: `feature/types-embedding`
    - **Type Definitions**:
      - `src/types/embedding/vector.py`: Embedding vector types
      - `src/types/embedding/model.py`: Embedding model interfaces
      - `src/types/embedding/service.py`: Embedding service types
    - **Implementation Steps**:
      - Standardize embedding vector representation
      - Ensure compatibility with both text and code embedding needs
      - Run mypy to verify type correctness

    #### 5. isne (Fifth Priority)

    - **Branch**: `feature/types-isne`
    - **Type Definitions**:
      - `src/types/isne/graph.py`: Graph structure types
      - `src/types/isne/layers.py`: Neural network layer types
      - `src/types/isne/training.py`: Training types
      - `src/types/isne/inference.py`: Inference types
    - **Implementation Steps**:
      - Consolidate types from multiple locations
      - Ensure compatibility with embedding types
      - Run mypy to verify type correctness

    #### 6. storage (Sixth Priority)

    - **Branch**: `feature/types-storage`
    - **Type Definitions**:
      - `src/types/storage/repository.py`: Repository interface types
      - `src/types/storage/document.py`: Document storage types
      - `src/types/storage/connection.py`: Database connection types
      - `src/types/storage/query.py`: Query types
    - **Implementation Steps**:
      - Refactor storage interfaces to use centralized types
      - Ensure compatibility with document and embedding types
      - Run mypy to verify type correctness

    #### Common Type Utilities

    - **Branch**: `feature/types-common`
    - **Type Definitions**:
      - `src/types/common/result.py`: Success/failure result types
      - `src/types/common/validation.py`: Validation result types
      - `src/types/common/config.py`: Configuration types
      - `src/types/common/utils.py`: Type utilities and helpers
    - **Implementation Steps**:
      - Create shared type utilities used across modules
      - Ensure compatibility with all module-specific types
      - Run mypy to verify type correctness

    #### Final Integration

    - **Branch**: `feature/types-integration`
    - **Implementation Steps**:
      - Resolve any cross-module type compatibility issues
      - Run comprehensive mypy checks across the entire codebase
      - Create final documentation for the type system

    After all type issues are addressed, we will circle back for unit and integration testing to ensure the refactored code meets our test coverage requirements.

- [ ] Improve test coverage to ≥85% for schema modules

#### 6. Streaming Document Processing

- **DEPENDENCY**: requires an existing database. Initial database pipeline must be run first.
- [ ] Modify pipeline to support streaming processing mode
  - [ ] Add batch processing with configurable batch size
  - [ ] Implement document streaming to storage
  - [ ] Add memory usage monitoring
- [ ] Create tests for streaming document processing

#### 7. Error Handling & Retry Logic

- [ ] Enhance `AlertManager` to handle stage-specific failures
- [ ] Implement retry mechanisms for each stage
- [ ] Add detailed error reporting with context
- [ ] Create tests for error handling and recovery

#### 8. Multiprocessing Optimization

- [ ] Refactor to use `concurrent.futures.ProcessPoolExecutor`
- [ ] Improve exception handling in multiprocessing
- [ ] Optimize resource allocation
- [ ] Create tests for multiprocessing performance

### Lower Priority

#### 9. Dry Run Mode

- [ ] Implement dry run mode for testing without storage side effects
- [ ] Add detailed validation reporting in dry run mode
- [ ] Create tests for dry run functionality

#### 10. Logging and Debugging

- [ ] Add `--debug` flag for detailed pipeline execution information
- [ ] Implement intermediate artifact saving
- [ ] Create visualization tools for pipeline execution
- [ ] Add tests for debugging functionality

#### 11. Directory Scanning Improvements

- [ ] Add `--exclude-dir` and `--include-ext` CLI options
- [ ] Implement file manifest caching
- [ ] Optimize directory traversal for large repositories
- [ ] Create tests for directory scanning

## Implementation Approach

The implementation will follow an incremental approach:

1. First, implement the `PipelineStage` interface and refactor existing code
2. Then define Pydantic models for data structures
3. Implement the storage stage with streaming support
4. Add error handling and retry logic
5. Implement debugging and dry run features

This approach allows for continuous testing and validation as new components are added, ensuring the system remains functional throughout the development process.

## Completion Criteria

Each task will be considered complete when:

- Implementation passes all unit and integration tests
- Documentation is complete and accurate
- Code meets quality standards (type hints, docstrings, etc.)
- Performance benchmarks show no degradation (or document acceptable tradeoffs)
