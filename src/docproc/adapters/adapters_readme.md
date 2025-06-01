# Document Processing Adapters Documentation

This document provides information about the adapters module in the HADES document processing system.

## Overview

The adapters module contains specialized adapters for processing different document types:
- `python_adapter.py`: For processing general Python files and extracting code structure information
- `python_code_adapter.py`: A wrapper around PythonAdapter with additional functionality
- `docling_adapter.py`: For processing documents using the Docling library

## Type Safety

All adapter modules have been reviewed and updated to ensure proper typing:
- Explicit return types for all methods
- Proper handling of optional parameters
- Type narrowing where appropriate
- Elimination of unreachable code issues

## Test Coverage

As of May 2025, all adapter modules combined have achieved >85% test coverage:
- `PythonAdapter`: 95.7% line coverage
- `PythonCodeAdapter`: 89.8% line coverage
- `CallFinder`: 86.4% line coverage
- `DoclingAdapter`: 75.5% line coverage
- `EntityExtractor`: 75.0% line coverage
- **Overall**: 85.5% line coverage

## Usage

All adapters follow a common interface with the following primary methods:

### `process(file_path, options)`
Processes a document from a file path, returning a ProcessedDocument object.

### `process_text(content, options)`
Processes the content of a document directly, returning a dictionary with analysis results.

### `extract_entities(content, options)`
Extracts entities (functions, classes, etc.) from document content.

### `extract_metadata(content, options)`
Extracts metadata from document content.

## Testing

Test files for adapters can be found in the `tests/docproc/adapters/` directory:
- `test_python_adapter.py` - Base tests for the Python adapter
- `test_python_adapter_improvements.py` - Additional tests for higher coverage
- `test_python_code_adapter.py` - Tests for the Python code adapter
- `test_docling_adapter.py` - Base tests for the Docling adapter
- `test_docling_adapter_improvements.py` - Additional tests for higher coverage

## Future Work

While overall coverage meets the 85% threshold, there are still opportunities for improvement:
- Individual coverage for `DoclingAdapter` (75.5%) and `EntityExtractor` (75.0%) could be improved
- Initialization methods (`__init__`) are under-tested across most adapter classes
- Further tests could be added for edge cases and error conditions
