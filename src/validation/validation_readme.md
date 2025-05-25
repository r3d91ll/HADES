# Validation Module

## Overview

The validation module provides tools for ensuring data consistency and quality throughout the HADES-PathRAG pipeline, with a current focus on embedding validation for the ingestion process.

## Components

### Embedding Validator

The `embedding_validator.py` file contains utilities for validating embeddings during the ISNE application process:

- `validate_embeddings_before_isne`: Checks documents before ISNE embedding application
- `validate_embeddings_after_isne`: Checks documents after ISNE embedding application
- `create_validation_summary`: Creates a comprehensive validation report
- `attach_validation_summary`: Attaches validation data to document collections

## Usage

To validate embeddings during the ingestion pipeline:

```python
from src.validation import (
    validate_embeddings_before_isne,
    validate_embeddings_after_isne,
    create_validation_summary,
    attach_validation_summary
)

# Before ISNE application
pre_validation = validate_embeddings_before_isne(documents)

# Apply ISNE embeddings here...

# After ISNE application  
post_validation = validate_embeddings_after_isne(documents, pre_validation)

# Create and attach summary
validation_summary = create_validation_summary(pre_validation, post_validation)
documents = attach_validation_summary(documents, validation_summary)
```

## Validation Checks

The validation performs the following checks:

1. Pre-ISNE validation:
   - Document and chunk counts
   - Base embedding presence
   - Existing ISNE embeddings (which shouldn't be present)
   - Missing base embeddings

2. Post-ISNE validation:
   - ISNE embedding presence
   - Missing ISNE embeddings
   - Duplicate ISNE embeddings
   - Document-level embeddings (should only be at chunk level)

3. Discrepancy detection:
   - Differences between chunk counts and embedding counts
   - Duplicate embeddings
   - Misplaced embeddings

## Output

Validation results are output in three ways:

1. Logs: Warnings and information are logged using the standard logging system
2. Validation Report: A JSON file with detailed validation results
3. Document Attributes: Validation data is attached to the document collection

## Performance

The validation module is optimized for high performance, with benchmark results showing:

| Dataset Size | Processing Speed |
|--------------|------------------|
| 100 chunks   | ~830,000 chunks/sec |
| 1,000 chunks | ~1.6 million chunks/sec |
| 10,000 chunks | ~1.9 million chunks/sec |
| 100,000 chunks | ~1.3 million chunks/sec |

The benchmarks demonstrate that the validation scales well with increasing dataset size, making it suitable for production use with large document collections.

## Testing Status

- Unit tests: Complete with 100% test coverage
- Integration tests: Implemented in pipeline_multiprocess_test.py and pipeline_validation_test.py
- Type checking: Validated with mypy
- Benchmarks: Implemented in benchmark/validation/benchmark_embedding_validator.py

## Integration Examples

### CLI Validation Tool

The validation module includes a standalone CLI tool for validating existing datasets:

```bash
python scripts/validate_isne_embeddings.py --input path/to/isne_enhanced_documents.json --output validation_report.json
```

### Integration Test

A dedicated integration test is available that demonstrates validation in action:

```bash
python tests/integration/pipeline_validation_test.py --input path/to/isne_enhanced_documents.json
```
