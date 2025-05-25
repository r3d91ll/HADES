# Test Plan for the docproc Module

## Current Progress

- ✅ Fixed all mypy errors in the docproc module
- ✅ Created isolated test utilities to mock external dependencies
- ✅ Implemented format detector tests with 91% coverage
- ✅ Enhanced format detection logic for better YAML vs Markdown distinction

## Test Strategy

Given the constraints with the Docling dependency, we'll use the following approach:

1. **Isolated Testing**: Create isolated tests that don't depend on problematic external libraries
2. **Mock Dependencies**: Use mock implementations for external dependencies
3. **Component Testing**: Focus on testing each component individually
4. **Integration Testing**: For core functionality, create tests that verify the integration between components

## Components to Test

### Core Components

- [x] Format Detector (91% coverage)
- [ ] Core Document Processing
  - [ ] process_document function
  - [ ] process_text function
  - [ ] get_format_for_document function
  - [ ] detect_format function

### Adapter Components

- [ ] Base Adapter
  - [ ] Abstract interface
  - [ ] Key methods
- [ ] Registry
  - [ ] register_adapter function
  - [ ] get_adapter_for_format function
  - [ ] get_adapter_class function
- [ ] Format-Specific Adapters
  - [ ] HTMLAdapter
  - [ ] JSONAdapter
  - [ ] XMLAdapter
  - [ ] YAMLAdapter
  - [ ] CSVAdapter
  - [ ] TextAdapter
  - [ ] PythonAdapter (replacing CodeAdapter)
  - [ ] Mock implementation for PDFAdapter (to avoid Docling dependency)

## Coverage Goals

- Target minimum 85% code coverage for each component
- Focus on testing edge cases and error handling

## Implementation Plan

1. Create base test utilities and mocks for all dependencies
2. Implement isolated tests for each component
3. Implement integration tests for core functionality
4. Measure coverage and address any gaps

## Challenges and Solutions

- **Docling Dependency**: Using mocks and wrapper classes to isolate Docling
- **Abstract Base Classes**: Create concrete implementations for testing
- **External File Operations**: Use temporary files and directories
