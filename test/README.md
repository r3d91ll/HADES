# HADES Test Suite Organization

This directory contains all tests for the HADES project, organized by test type and component scope.

## Directory Structure

### `/unit/`
**Unit tests for individual components and functions:**

- `test_api_metrics.py` - API metrics collection testing
- `test_arango_connection.py` - ArangoDB connection testing
- `test_arangodb_storage_v2.py` - ArangoDB storage layer testing
- `test_docproc_functionality.py` - Document processing functionality
- `test_docproc_monitoring.py` - Document processing monitoring
- `test_docproc_simple.py` - Simple document processing tests
- `test_haystack_engine.py` - Haystack engine testing
- `test_haystack_monitoring.py` - Haystack monitoring tests
- `test_training_simple.py` - Simple training pipeline tests

### `/integration/`
**Integration tests for component interactions:**

- `integration_test_chunking.py` - Chunking pipeline integration
- `integration_test_docproc.py` - Document processing integration
- `integration_test_embedding.py` - Embedding pipeline integration

### `/system/`
**System-level and validation tests:**

- `test_isne_graph_validation.py` - ISNE graph validation
- `test_sequential_isne_schema.py` - Sequential ISNE schema validation

### `/components/`
**Component-specific test suites:**

- `chunking/` - Chunking component tests
- `docproc/` - Document processing tests
- `embedding/` - Embedding component tests
- `graph_enhancement/` - Graph enhancement (ISNE) tests
- `model_engine/` - Model engine tests
- `storage/` - Storage layer tests

### `/pipelines/`
**Pipeline-specific tests:**

- `bootstrap/` - Bootstrap pipeline tests

## Running Tests

### All Tests
```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=src --cov-report=html
```

### By Category
```bash
# Unit tests only
poetry run pytest test/unit/

# Integration tests only  
poetry run pytest test/integration/

# System tests only
poetry run pytest test/system/

# Component tests
poetry run pytest test/components/
```

### By Component
```bash
# Document processing tests
poetry run pytest test/components/docproc/

# ISNE/Graph enhancement tests
poetry run pytest test/components/graph_enhancement/

# Embedding tests
poetry run pytest test/components/embedding/
```

### Specific Tests
```bash
# Single test file
poetry run pytest test/unit/test_arango_connection.py

# Specific test function
poetry run pytest test/unit/test_api_metrics.py::test_metrics_collection
```

## Test Markers

Tests are marked with categories for selective execution:

```bash
# Run only unit tests
poetry run pytest -m unit

# Run only integration tests
poetry run pytest -m integration

# Run specific component tests
poetry run pytest -m docproc
poetry run pytest -m embedding
poetry run pytest -m isne
```

## Test Configuration

- `pytest.ini` - Pytest configuration with markers and settings
- Test data in `/test-data/` directory
- Test outputs in `/test-out/` directory

## Coverage Requirements

- **Minimum 85% unit test coverage** for all modules
- All new functions must have unit tests
- Integration tests for module interactions
- System tests for end-to-end workflows

## Test Development Guidelines

1. **Unit Tests**: Test individual functions and classes in isolation
2. **Integration Tests**: Test component interactions and data flow
3. **System Tests**: Test complete workflows and pipelines
4. **Mock External Dependencies**: Use mocks for external services in unit tests
5. **Use Fixtures**: Create reusable test data with pytest fixtures
6. **Clear Test Names**: Use descriptive test function names
7. **Test Documentation**: Include docstrings explaining test purpose

## Migration from Root Directory

**Before:** Test files scattered in project root  
**After:** Organized test structure with clear categorization

### Benefits:
1. **Clear separation** between unit, integration, and system tests
2. **Component organization** for targeted testing
3. **Better maintainability** and discoverability
4. **Scalable structure** for growing test suite
5. **Parallel execution** capabilities for different test categories