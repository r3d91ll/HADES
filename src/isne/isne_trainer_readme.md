# ISNE Trainer Module

## Overview

The ISNE Trainer Module provides a complete implementation for training the Inductive Shallow Node Embedding (ISNE) model as part of the HADES-PathRAG system. This module bridges the output from the document processing pipeline to the ISNE model training process, enabling enhanced document embeddings through graph-based learning.

## Core Components

### ISNETrainingOrchestrator

The primary class responsible for coordinating the end-to-end ISNE training workflow:

- **Input Processing**: Takes processed documents with ModernBERT embeddings from the document processing pipeline
- **Graph Construction**: Builds a document graph from the embeddings and multiple types of relationships:
  - Sequential proximity within documents
  - Directory-based relationships (documents in the same directory)
  - Semantic similarity between chunks
- **ISNE Model Training**: Trains the ISNE model on the constructed graph
- **Model Evaluation**: Evaluates the trained model using appropriate metrics
- **Model Persistence**: Saves trained models for later use in inference

### Configuration System

Extends the existing pipeline configuration system to include ISNE training parameters:

- **Training Parameters**: Epochs, learning rate, batch size, model architecture settings
- **Resource Allocation**: CPU/GPU usage settings with dynamic device selection
  - Configurable device selection (e.g., 'cuda:0', 'cuda:1', 'cpu')
  - Memory optimization settings to prevent CUDA out-of-memory errors
- **File Handling**: Input/output directory configurations with support for nested directory structures
- **Chunking Parameters**: Optimized for ModernBERT token limits (max_tokens_per_chunk = 4096)

### Workflow Integration

Seamlessly integrates with the existing document processing pipeline:

1. **Pipeline Output Consumption**: Reads the output from the multiprocessing pipeline test
2. **Document Graph Creation**: Converts document relationships into a graph structure
3. **Training Loop Execution**: Implements efficient training with progress tracking
4. **Model Serialization**: Stores models with version history

## Usage

### Basic Command-line Usage

```bash
python -m src.isne.trainer.training_orchestrator \
    --input-dir ./test-output/pipeline-mp-test \
    --model-output-dir ./models/isne \
    --epochs 50 \
    --device cpu
```

### Programmatic Usage

```python
from src.isne.trainer.training_orchestrator import ISNETrainingOrchestrator

# Initialize the orchestrator
orchestrator = ISNETrainingOrchestrator(
    input_dir="./test-output/pipeline-mp-test",
    model_output_dir="./models/isne",
    config_override={
        "training": {
            "epochs": 50,
            "learning_rate": 0.001,
            "batch_size": 32
        }
    }
)

# Run the training process
training_results = orchestrator.train()

# Get trained model for inference
model = orchestrator.load_model()
```

### Integration with Pipeline Test

```bash
# First run the document processing pipeline
python -m tests.integration.pipeline_multiprocess_test --num-files 20

# Then train the ISNE model on the processed documents
python -m src.isne.trainer.training_orchestrator \
    --input-dir ./test-output/pipeline-mp-test
```

## Implementation

### Graph Construction

The trainer converts document chunks and their relationships into a PyTorch Geometric graph structure:

1. **Node Features**: Document chunk embeddings from ModernBERT
2. **Edge Construction**: Relationships between chunks based on:
   - Sequential proximity within documents (chunks that follow each other)
   - Directory structure relationships (documents in the same folder)
   - Semantic similarity between chunks
   - Explicit relationships like citations or references

### Training Procedure

The training procedure follows the approach defined in the ISNE paper:

1. **Batching**: Mini-batch training with neighborhood sampling
2. **Loss Computation**: Multi-objective loss combining:
   - Feature preservation loss (maintaining original semantic information)
   - Structural preservation loss (preserving graph structure)
   - Contrastive loss (differentiating between related and unrelated chunks)
3. **Optimization**: Adam optimizer with learning rate scheduling

### Evaluation Metrics

The trainer evaluates model performance using:

- **Embedding Quality**: Cosine similarity between related documents
- **Link Prediction**: Ability to predict connections between documents
- **Ablation Studies**: Measuring contribution of different loss components

## Performance Reporting

The module includes comprehensive performance reporting for the complete pipeline:

- **Document Processing Statistics**:
  - Processing time for each stage (extraction, chunking, embedding)
  - Number of documents and chunks processed
  - Total tokens processed
  
- **ISNE Training Statistics**:
  - Training time and epochs
  - Loss values and convergence metrics
  - Memory usage during training
  - GPU utilization metrics

- **Benchmarks**:
  - Training time across different document set sizes
  - Embedding quality comparison (ISNE-enhanced vs. original)
  - Memory usage profiling during training

## Testing

Following the project's standard protocol, this module includes comprehensive testing:

- **Unit Tests**: Individual components tested with ≥85% coverage
- **Integration Tests**: End-to-end tests with real document examples
- **Performance Tests**: Measurement of training efficiency

## Implementation Notes

### Directory Traversal

The system supports recursive directory traversal to locate documents:

- All supported document formats are discovered in nested subdirectories
- The directory structure is incorporated into the ISNE graph construction
- Documents in the same directory receive additional edge connections
- Configurable directory relationship weight (default: 0.5)

### GPU Optimization

The system includes several optimizations for GPU processing:

- Dynamic device selection based on configuration
- Reduced embedding batch size (default: 4) to prevent CUDA out-of-memory errors
- Optimized token limits for ModernBERT (max_tokens_per_chunk: 4096)
- Proper memory management during embedding generation

## Future Improvements

Planned enhancements for this module include:

1. **Distributed Training**: Support for training across multiple nodes
2. **Dynamic Graph Updates**: Incremental training for evolving document collections
3. **Advanced Relationship Extraction**: More sophisticated methods for identifying document relationships
4. **Hyperparameter Optimization**: Automated tuning of model parameters
5. **Further GPU Optimizations**: Additional memory management strategies for larger document collections
