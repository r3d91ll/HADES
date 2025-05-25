# Graph Dataset Loader

## Overview

The Graph Dataset Loader is a critical component of the ISNE pipeline that transforms document collections and their relationships into PyTorch Geometric graph structures. This enables the ISNE model to perform inductive node embedding on document networks.

## Features

- **Multi-modal Support**: Handles different document types (text, code, PDF, etc.) with specialized processing
- **Flexible Loading Options**: 
  - In-memory loading from document collections 
  - File-based loading for debugging and testing
  - Direct integration with pipeline data
- **Graph Structure Options**:
  - Homogeneous graphs (single node and edge type)
  - Heterogeneous graphs (multiple node and edge types based on document modalities)
- **Data Preparation Utilities**:
  - Node feature extraction from document embeddings
  - Edge attribute handling based on relationship types
  - Document type encoding with one-hot vectors
- **Dataset Splitting**: Tools for creating training, validation, and test splits

## Usage Examples

### Loading from Document Collections

```python
from isne.loaders import GraphDatasetLoader
from isne.types.models import IngestDocument, DocumentRelation

# Initialize the loader
loader = GraphDatasetLoader(use_heterogeneous_graph=False)

# Load from document collections
graph_data = loader.load_from_documents(
    documents=document_list,  # List of IngestDocument objects
    relations=relation_list,  # List of DocumentRelation objects
)

# Use for ISNE model training
model.train(graph_data)
```

### Loading from Files

```python
# Load from a previously saved JSON file
graph_data = loader.load_from_file(
    file_path="/path/to/documents.json",
    include_node_types=True,
    include_edge_attributes=True
)
```

### Creating Heterogeneous Graphs

```python
# Create a heterogeneous graph with node and edge types
loader = GraphDatasetLoader(use_heterogeneous_graph=True)

# Node types will be based on document_type
# Edge types will be based on relation_type
hetero_graph = loader.load_from_documents(documents, relations)
```

## Integration Points

The Graph Dataset Loader integrates with:

1. The document processing pipeline via the `IngestDocument` and `DocumentRelation` models
2. The PyTorch Geometric library for graph neural networks
3. The ISNE model training process
4. The evaluation and inference workflows

## Performance Considerations

- For large document collections, consider using batched loading
- GPU acceleration is automatically used when available
- Heterogeneous graphs have higher memory requirements but may provide better model performance

## Future Extensions

- Add support for dynamic graphs that can be updated incrementally
- Implement more sophisticated dataset splitting for heterogeneous graphs
- Add data visualization tools for inspecting graph structures
