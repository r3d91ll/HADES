# ISNE (Inductive Shallow Node Embedding) Module

## Overview

The ISNE module implements the Inductive Shallow Node Embedding algorithm for HADES-PathRAG, enabling graph-based embedding enhancement of document chunks. This implementation is based on the methodology described in the [original research paper](https://link.springer.com/article/10.1007/s40747-024-01545-6), ensuring accurate representation of the authors' approach for inductive learning on graphs. The module enables the system to generalize embeddings to unseen nodes without requiring retraining.

## Architecture

The module follows the original paper's architecture with the following components:

### 1. Data Loading & Graph Construction

Responsible for loading document chunks from various inputs, constructing PyTorch Geometric graph structures, and establishing document relationships and metadata.

**Key Components:**

- `ModernBERTLoader`: Loads chunked documents with original embeddings (maintained from original implementation)
- `GraphDatasetLoader`: Converts document collections into PyTorch Geometric datasets
- `RelationshipExtractor`: Identifies semantic and structural connections between documents

### 2. ISNE Layers & Models

Implements the core ISNE algorithm as described in the research paper, using message passing and attention mechanisms.

**Key Components:**

- `ISNELayer`: Neural network layer implementing neighborhood aggregation with attention-based weights
- `ISNEAttention`: Attention mechanism for weighting neighborhood nodes
- `ISNEModel`: Complete model architecture combining multiple ISNE layers for effective embedding propagation

### 3. Loss Functions & Training

Implements the multi-objective loss functions required for proper model training as specified in the paper.

**Key Components:**

- `StructuralLoss`: Preserves graph structure through random walk-based sampling
- `FeatureLoss`: Maintains feature similarity between initial and enhanced embeddings
- `ContrastiveLoss`: Encourages similar nodes to have similar embeddings while pushing dissimilar nodes apart
- `Sampler`: Implements efficient neighborhood sampling techniques
- `Trainer`: Orchestrates the training process with proper optimization techniques

### 4. Evaluation & Utilities

Provides tools for evaluating embedding quality and visualizing results.

**Key Components:**

- `Metrics`: Implements evaluation metrics from the paper (link prediction, node classification, etc.)
- `Visualizers`: Creates visualizations of embedding spaces
- `GeometricUtils`: Provides utility functions for working with graph data structures

## Implementation Phases

Implementation follows the three-phase approach outlined in the main TODO.md file:

### Phase 1: Core Architecture (Week 1)

- Implementation of the ISNE layer and model architecture
- Integration with PyTorch Geometric for efficient graph operations
- Creation of basic neighborhood sampling functions
- Implementation of foundational attention mechanisms

### Phase 2: Training & Evaluation (Week 2)

- Implementation of all loss functions from the original paper
- Development of training loops and optimization strategies
- Creation of evaluation metrics matching those in the paper
- Implementation of visualization utilities for embedding inspection

### Phase 3: Pipeline & Integration (Week 3)

- Integration with ModernBERT outputs
- Database storage implementation
- Comprehensive testing and validation
- Documentation and performance optimization

## Usage

### Basic Usage

```python
from src.isne.pipeline import ISNEPipeline

# Initialize pipeline with default configuration
pipeline = ISNEPipeline()

# Process ModernBERT output
result = pipeline.process_modernbert_output("path/to/modernbert_output.json")

# Access enhanced embeddings
enhanced_documents = result["documents"]
```

### Training Mode

```python
from src.isne.pipeline import ISNEPipeline
from src.isne.training.trainer import ISNETrainer

# Initialize pipeline with training components
pipeline = ISNEPipeline(training_mode=True)

# Load data and create graph dataset
graph_data = pipeline.load_data("path/to/documents")

# Initialize trainer with model configuration
trainer = ISNETrainer(
    embedding_dim=768,
    hidden_dim=256, 
    num_layers=2,
    learning_rate=0.001
)

# Train model
trainer.train(
    graph_data=graph_data,
    epochs=100,
    batch_size=32,
    early_stopping_patience=10
)

# Save trained model
trainer.save_model("path/to/model.pt")
```

### Simplified Training for Development

A simplified ISNE training approach has been implemented specifically for development and testing scenarios where graph construction might be problematic:

```python
# Using the simplified ISNE training approach
from tests.integration.pipeline_multiprocess_test import PipelineMultiprocessTest

# Initialize test with custom configuration
test = PipelineMultiprocessTest(config_path="config/test_config.yaml")

# Process documents and prepare them for ISNE
documents = test.process_documents(input_path)
doc_list = test.prepare_documents_for_isne(documents)

# Use simplified ISNE training that avoids graph construction issues
results = test.simplified_isne_training(doc_list)
```

This simplified approach:
- Bypasses the complex graph construction process
- Uses a linear projection model that preserves cosine similarities
- Processes embeddings directly without requiring a graph structure
- Provides faster training times for development iterations
- Is particularly useful when working with small document sets or when testing CodeBERT integration

## Testing

Following the project's standard protocol, this module includes comprehensive testing:

- **Unit Tests**: All functions tested with ≥85% coverage
- **Integration Tests**: End-to-end tests with real document examples
- **Performance Benchmarks**: Measures of training time, inference time, and memory usage
- **Quality Metrics**: Evaluation against metrics defined in the original paper

## Implementation Standards

This implementation adheres to the project's standards:

- Type annotations on all functions and classes
- Comprehensive docstrings with parameter descriptions and examples
- MyPy validation for type correctness
- Code organization following the project structure guidelines
- Performance considerations for large document collections

## Operational Modes

The ISNE module supports multiple operational modes for training and inference:

### Data Processing Options

1. **In-Memory Pipeline Integration**
   - Direct integration with the embedding pipeline
   - No intermediate file I/O for normal operation
   - Data passed directly between pipeline stages
   - File-based loading only used for debugging/troubleshooting

2. **File-Based Processing**
   - Available through the ModernBERT loader for offline testing
   - Processes JSON outputs from previous pipeline stages
   - Not recommended for production use

### Training and Inference Workflow

#### Two-Phase Approach

1. **Training Phase**
   - Process documents through the pipeline (docproc → chunking → embedding)
   - Collect document embeddings and relationships into a dataset
   - Train ISNE model using the trainer module
   - Save trained model weights

2. **Inference/Ingestion Phase**
   - Process documents through the same initial pipeline
   - Pass data through trained ISNE model
   - Store enhanced embeddings in the database
   - Use for retrieval and downstream tasks

#### Configuration Options

The pipeline can be configured to operate in three modes:

1. **Training Collection Mode**
   - Process documents and collect data for later ISNE training
   - No model training or inference occurs
   - Embeddings are stored in their original form

2. **Training Mode**
   - Process documents and immediately train the ISNE model
   - Useful for initial setup or retraining sessions
   - Can be computationally intensive

3. **Inference Mode**
   - Process documents through a pre-trained ISNE model
   - Normal production operation mode
   - Requires a previously trained model

This separation of training and inference allows for efficient processing during normal operation while enabling periodic model updates as new data becomes available.

## References

- Original Research Paper: "Unsupervised Graph Representation Learning with Inductive Shallow Node Embedding" (2024)
- PyTorch Geometric: [https://pytorch-geometric.readthedocs.io/](https://pytorch-geometric.readthedocs.io/)
- Related Work: GraphSAGE, Node2Vec, and Graph Attention Networks
