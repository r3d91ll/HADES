# ISNE Pipeline Module Documentation

## Overview

The `isne_pipeline.py` module provides a production-ready implementation of the ISNE (Incremental Stochastic Neighbor Embedding) pipeline. This module handles the entire processing flow from loading pre-computed embeddings to applying ISNE transformations and validating the results.

## Type Safety Improvements

The module has been enhanced with the following type safety features:

1. **Union Type for Models**: Added `ISNEModelTypes = Union[ISNEModel, SimplifiedISNEModel]` to ensure proper type safety when handling different model implementations.

2. **Proper Type Casting**: Implemented explicit type casting using `cast()` to ensure correct type handling throughout the pipeline.

3. **Modern Tensor Type Checking**: Replaced deprecated `torch.FloatTensor` checks with modern type checking using `isinstance(x, torch.Tensor) and x.dtype == torch.float`.

4. **Return Type Guarantees**: Enhanced model methods to ensure consistent return types even when internal implementation details change.

## Classes

### ISNEPipeline

The `ISNEPipeline` class provides a robust implementation for applying ISNE embeddings to documents with integrated validation and alerting.

#### Usage Example

```python
from src.isne.pipeline.isne_pipeline import ISNEPipeline

# Initialize the pipeline with a pre-trained model
pipeline = ISNEPipeline(
    model_path="/path/to/model.pt",
    validate=True,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Process a batch of embeddings
input_embeddings = load_embeddings_from_source()
enhanced_embeddings = pipeline.process_embeddings(input_embeddings)

# Apply to a graph structure
graph_data = load_graph_data()
enhanced_graph = pipeline.process_graph(graph_data)
```

## Configuration

The pipeline can be configured with the following parameters:

- `model_path`: Path to a pre-trained ISNE model
- `validate`: Whether to perform validation on the embeddings
- `alert_threshold`: Severity threshold for alerts ("low", "medium", "high")
- `device`: Device to run computations on ("cpu", "cuda")

## Testing

The module includes comprehensive unit tests in `test/isne/pipeline/test_pipeline_types.py` that verify type safety, proper tensor handling, and consistent return types.

## Type Checking

The module has been verified with mypy to ensure type safety. To check types, run:

```bash
mypy src/isne/pipeline/ --config-file mypy.ini
```
