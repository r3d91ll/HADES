# Simplified ISNE Model Adapter

## Overview

The SimplifiedISNEModel adapter enables integration of code-enabled ISNE embeddings with the ingestion pipeline. It bridges the gap between the simpler linear projection model used in code-enabled training and the full ISNE architecture expected by the pipeline.

## Components

### `SimplifiedISNEModel` Class

- Wraps a single linear projection layer to mimic the interface of the full ISNE model
- Adapts between different state dictionary formats
- Handles error conditions gracefully with appropriate logging
- Ensures backward compatibility with existing models

## Usage

```python
from src.isne.models.simplified_isne_model import SimplifiedISNEModel

# Create the model
model = SimplifiedISNEModel(
    in_features=768,  # Input dimension (e.g., from ModernBERT or CodeBERT)
    hidden_features=128,  # Not used but kept for API compatibility
    out_features=64  # Output dimension for ISNE embeddings
)

# Load a state dictionary
model.load_state_dict(state_dict)

# Generate embeddings
enhanced_embeddings = model(input_embeddings)
```

## Validation

The `pipeline_multiprocess_test.py` file contains validation functionality to check for embedding inconsistencies:

1. Pre-validation: Ensures base embeddings are present and consistent
2. Post-validation: Verifies ISNE embeddings are correctly applied to all chunks
3. Discrepancy detection: Identifies issues such as:
   - Missing ISNE embeddings
   - Duplicate ISNE embeddings
   - Document-level embeddings (should only be at chunk level)

## Performance

The simplified model offers significantly faster inference than the full ISNE model while preserving the core projection functionality.

## Testing Status

- Unit tests: In progress (target ≥85% coverage)
- Integration tests: Tested with pipeline_multiprocess_test.py
- Type checking: Validated with mypy
