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
  - [ ] Implement centralized type system
    - [x] Create src/types/docproc/ directory (rename from src/types/documents/)
    - [x] Create src/types/chunking/ directory for chunking types
    - [x] Create src/types/storage/ directory for storage types
    - [x] Create src/types/pathrag/ directory for PathRAG types
    - [x] Create src/types/indexing/ directory for indexing types
    - [x] Update type migration plan document in src/types/types_readme.md

    ### Current Type System Status By Module

    | Module | Type Status | Current Location | Migration Needs |
    |--------|-------------|------------------|------------------|
    | **docproc** | Mixed | `src/docproc/schemas/` and imports from `src/types/common` | Migrate internal types to `src/types/docproc/` |
    | **chunking** | Local only | Types defined within module | Create and migrate to `src/types/chunking/` |
    | **storage** | Mostly centralized | Uses types from `src/types/common` | Complete migration to `src/types/storage/` |
    | **pathrag** | Mixed | Internal types + imports from `src/types/common` and `model_types` | Create and migrate to `src/types/pathrag/` |
    | **pipeline** | Mixed | Has schemas in `src/schemas/pipeline/` and types in `src/types/pipeline/` | Consolidate to `src/types/pipeline/` |
    | **embedding** | Mostly centralized | Uses `EmbeddingVector` from `src/types/common` | Complete migration to `src/types/embedding/` |
    | **isne** | Mixed | Has types in both `src/isne/types/` and `src/types/isne/` | Consolidate to `src/types/isne/` |
    | **model_engine** | Mostly centralized | Uses types from `src/types/vllm_types` | Create `src/types/model_engine/` |

    ### Migration Priority Order

    1. **docproc**: Fix syntax issues, then migrate from internal schemas
    2. **chunking**: Create and migrate to centralized types
    3. **pathrag**: Create and migrate to centralized types
    4. **storage**: Complete migration to centralized types
    5. **isne**: Consolidate types from multiple locations
    6. **model_engine**: Create and migrate to centralized types
  - [ ] Fix type issues in docproc module
    - [ ] Fix syntax errors in type annotations
    - [ ] Migrate module-specific types to centralized types in src/types/docproc/
    - [ ] Update imports to use centralized types
    - [ ] Ensure consistent type usage across the module
  - [ ] Fix type issues in storage module
    - [ ] Fix syntax errors in type annotations
    - [ ] Migrate module-specific types to centralized types in src/types/storage/
    - [ ] Update imports to use centralized types
  - [ ] Fix type issues in pathrag module
    - [ ] Fix syntax errors in type annotations
    - [ ] Migrate module-specific types to centralized types in src/types/pathrag/
    - [ ] Update imports to use centralized types
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
