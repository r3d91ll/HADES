# ISNE Module Integration Tests

This directory contains integration tests for the ISNE (Inductive Shallow Node Embedding) module, verifying that the various components work together correctly.

## Test Files

- **`test_isne_real_data_pipeline.py`**: Tests the ISNE pipeline using real output data from previous tests.
  - Validates proper loading of JSON test output files (complete_ISNE_paper_output.json and complete_PathRAG_paper_output.json)
  - Verifies document structure and metadata extraction
  - Ensures embeddings are correctly loaded and have appropriate dimensions (768d for ModernBERT)
  - Tests basic IngestDocument and DocumentRelation creation for model compatibility

## Testing Approach

The integration tests focus on:

1. **Data Validation**: Ensuring the output files contain the expected format and data
2. **Document Processing**: Verifying that `IngestDocument` objects can be correctly created from the output
3. **Relationship Validation**: Checking that document relationships are properly extracted and processed

## Running Tests

Run the tests from the project root directory:

```bash
python -m unittest tests/integration/isne/test_isne_real_data_pipeline.py
```

## Performance Considerations

- The tests are designed to be lightweight and avoid unnecessary model training or inference
- Tests use minimal sample data to prevent memory issues
- Timeouts are implemented where appropriate to prevent hanging tests

## Test Quality Standards

- Unit test coverage maintained at >85% for all ISNE module functions
- Integration tests validate core functionality without expensive operations
- Error handling includes detailed logging and traceback reports
- Assertions verify both data structure and semantic correctness

## Known Issues

- Some tests may be skipped if real test output data files are not found
- Integration tests focus on data loading and validation rather than full model training/inference 
  due to performance considerations
