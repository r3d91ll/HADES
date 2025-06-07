# ISNE Bootstrap Process

## Overview

The ISNE Bootstrap Process solves the "chicken-and-egg" problem of initializing Inductive Shallow Node Embedding (ISNE) models from scratch when no existing graph database or trained model is available. This is particularly crucial for new RAG systems starting with a corpus of documents.

## The Bootstrap Challenge

When implementing ISNE for the first time, you face a fundamental problem:

- **ISNE requires a graph structure** to learn meaningful neighborhood aggregation patterns
- **Creating a meaningful graph requires understanding semantic relationships** between documents/chunks
- **Understanding semantic relationships requires embeddings** that capture domain-specific knowledge
- **Domain-specific embeddings require a trained ISNE model**

This circular dependency is what the bootstrap process resolves.

## Bootstrap Strategy: 7-Phase Approach

### Phase 1: Document Processing
**Objective**: Convert raw documents into structured, processable chunks

```bash
# Process all documents in corpus
python -m src.isne.bootstrap --corpus-dir ./corpus --output-dir ./bootstrap-output
```

**What happens**:
- Documents are processed using DocProc adapters (Docling for PDFs, AST for Python code)
- Content is extracted, validated, and prepared for chunking
- Metadata is preserved for downstream relationship building

**Output**: Structured document objects with extracted content

### Phase 2: Chunking
**Objective**: Segment documents into semantic units

**What happens**:
- Documents are chunked using appropriate strategies (paragraph-based for PDFs, AST-based for code)
- Chunks maintain references to source documents
- Cross-document chunking enables relationship discovery
- Each chunk becomes a potential graph node

**Output**: Flat list of chunks ready for relationship building

### Phase 3: Initial Graph Construction
**Objective**: Create initial graph structure using text similarity

**Method**: TF-IDF + Cosine Similarity
```python
# Text similarity computation
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words='english', 
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)
similarity_matrix = cosine_similarity(tfidf_matrix)
```

**Graph Building Rules**:
- Similarity threshold: 0.3 (configurable)
- Max connections per chunk: 10 (prevents dense graphs)
- Bidirectional edges with similarity weights

**Output**: NetworkX graph with nodes=chunks, edges=similarity relationships

### Phase 4: Initial ISNE Training
**Objective**: Train ISNE model to learn neighborhood aggregation patterns

**Training Configuration**:
```python
model_config = {
    'hidden_dim': 128,
    'output_dim': 64, 
    'num_layers': 2,
    'dropout': 0.1
}
```

**What the model learns**:
- How to aggregate information from neighboring chunks
- Domain-specific patterns in your corpus
- Inductive capabilities for future unseen chunks

**Output**: Trained ISNE model (`initial_isne_model.pt`)

### Phase 5: Production Embedding Generation
**Objective**: Generate embeddings using trained ISNE model

**Process**:
- Use trained ISNE encoder on all chunks
- Generate embeddings that capture both content and structural relationships
- Embeddings reflect learned neighborhood patterns

**Output**: NumPy array of embeddings (`production_embeddings.npy`)

### Phase 6: Refined Graph Construction
**Objective**: Build improved graph using ISNE embeddings

**Improvement over Phase 3**:
- ISNE embeddings capture learned patterns vs. raw TF-IDF
- Better semantic understanding of relationships
- More meaningful connections between chunks

**Process**:
```python
# Compute similarities in ISNE embedding space
embedding_similarities = cosine_similarity(isne_embeddings)
# Build refined graph with same rules but better similarities
```

**Output**: Improved NetworkX graph (`refined_isne_graph.json`)

### Phase 7: Final Training & Production Embeddings
**Objective**: Retrain ISNE on improved graph and generate final embeddings

**Retraining Benefits**:
- Model learns from improved graph structure
- Better optimization for domain-specific patterns
- Enhanced inductive capabilities

**Final Output**:
- Refined ISNE model (`refined_isne_model.pt`)
- Production-ready embeddings (`final_embeddings.npy`)
- Chunk metadata with quality metrics (`chunk_metadata.json`)

## File Structure After Bootstrap

```
bootstrap-output/
├── models/
│   ├── initial_isne_model.pt      # Phase 4: Initial trained model
│   └── refined_isne_model.pt      # Phase 7: Final trained model
├── embeddings/
│   ├── production_embeddings.npy  # Phase 5: Initial embeddings
│   ├── final_embeddings.npy       # Phase 7: Final embeddings
│   └── chunk_metadata.json        # Chunk info + quality metrics
├── graphs/
│   ├── initial_similarity_graph.json  # Phase 3: TF-IDF graph
│   └── refined_isne_graph.json        # Phase 6: ISNE graph
├── debug/
│   └── (various debugging outputs)
└── bootstrap_results.json         # Complete process metrics
```

## Usage Instructions

### Basic Bootstrap Command
```bash
python -m src.isne.bootstrap \
  --corpus-dir ./your-corpus \
  --output-dir ./bootstrap-output \
  --similarity-threshold 0.3
```

### Advanced Configuration
```python
from src.isne.bootstrap import ISNEBootstrapper

bootstrapper = ISNEBootstrapper(
    corpus_dir=Path("./corpus"),
    output_dir=Path("./output"),
    similarity_threshold=0.3,          # Min similarity for connections
    max_connections_per_chunk=10,      # Max edges per node
    min_cluster_size=5                 # Min cluster size for analysis
)

results = bootstrapper.bootstrap_full_corpus()
```

### Loading Results into Database
```python
import numpy as np
import json

# Load embeddings and metadata
embeddings = np.load("./bootstrap-output/embeddings/final_embeddings.npy")
with open("./bootstrap-output/embeddings/chunk_metadata.json") as f:
    metadata = json.load(f)

# Insert into your database
for i, chunk in enumerate(metadata['chunks']):
    database.insert_chunk(
        id=chunk['id'],
        text=chunk['text'], 
        embedding=embeddings[i],
        source=chunk['source_document'],
        metadata=chunk['metadata']
    )
```

## Bootstrap Results Analysis

The `bootstrap_results.json` provides comprehensive metrics:

### Corpus Statistics
```json
{
  "corpus_stats": {
    "total_documents": 300,
    "total_chunks": 15420,
    "avg_chunk_length": 847.3,
    "corpus_directory": "./corpus"
  }
}
```

### Graph Evolution
```json
{
  "graph_stats": {
    "initial_graph": {
      "nodes": 15420,
      "edges": 89342,
      "avg_degree": 5.8,
      "density": 0.00075
    },
    "refined_graph": {
      "nodes": 15420, 
      "edges": 124567,
      "avg_degree": 8.1,
      "density": 0.00105
    }
  }
}
```

### Training Progress
```json
{
  "training_results": {
    "initial_training": {
      "final_loss": 0.234,
      "epochs": 100,
      "validation_score": 0.856
    },
    "final_training": {
      "final_loss": 0.187,
      "epochs": 50, 
      "validation_score": 0.892
    }
  }
}
```

## Quality Metrics

The bootstrap process includes automatic quality validation:

- **Embedding Consistency**: Measures how well embeddings capture semantic relationships
- **Graph Quality**: Analyzes graph structure properties
- **Training Convergence**: Monitors loss reduction and validation scores
- **Cross-Document Relationships**: Validates that chunks from different documents are appropriately connected

## Post-Bootstrap Operations

### Loading Trained Model
```python
from src.isne.pipeline.isne_pipeline import ISNEPipeline

# Load the trained model
isne_pipeline = ISNEPipeline.load_from_file(
    "./bootstrap-output/models/refined_isne_model.pt"
)

# Use for new documents
new_embeddings = isne_pipeline.encode_chunks(new_chunks)
```

### Adaptive Training Setup
```python
from src.isne.adaptive_training import AdaptiveISNETrainer

# Set up adaptive training with bootstrap model
trainer = AdaptiveISNETrainer(
    isne_pipeline=isne_pipeline,
    retrain_threshold=0.15,
    min_retrain_interval=24,
    max_incremental_updates=1000
)

# Process new documents
results = trainer.process_new_chunks(new_chunks, current_graph_stats)
```

## Best Practices

### Corpus Preparation
1. **Clean Documents**: Remove corrupted or irrelevant files
2. **Balanced Content**: Ensure diverse representation of your domain
3. **Sufficient Size**: Minimum 100 documents, ideally 300+ for robust training

### Parameter Tuning
1. **Similarity Threshold**: Start with 0.3, adjust based on graph density
2. **Max Connections**: 10 is good for most corpora, increase for dense domains
3. **Training Epochs**: 100 initial + 50 refinement works well

### Monitoring
1. **Check Graph Density**: Should be 0.001-0.01 for good balance
2. **Validation Scores**: Should improve from initial to final training
3. **Embedding Quality**: Overall score should be >0.8

## Troubleshooting

### Common Issues

**Graph Too Dense**:
- Reduce similarity_threshold (e.g., 0.2 → 0.4)
- Reduce max_connections_per_chunk

**Graph Too Sparse**:
- Increase similarity_threshold (e.g., 0.4 → 0.2)
- Check corpus diversity - may need more varied content

**Poor Training Convergence**:
- Increase training epochs
- Adjust learning rate
- Check graph connectivity

**Low Embedding Quality**:
- Review chunk quality - may be too short/long
- Check for document processing errors
- Consider domain-specific preprocessing

### Debug Information

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check debug outputs in `./bootstrap-output/debug/` for:
- Chunk processing logs
- Graph construction details
- Training metrics per epoch
- Embedding validation results

## Integration with HADES-PathRAG Training Infrastructure

The bootstrap process is designed to integrate seamlessly with the existing HADES-PathRAG training system:

### Core Integration Points

1. **Document Processing**: Uses existing `DocumentProcessorStage` and `ChunkingStage`
2. **Training Infrastructure**: Leverages `ISNETrainingOrchestrator` and `ISNETrainer` 
3. **Embedding Pipeline**: Compatible with HADES embedding framework
4. **Graph Construction**: Prepares data for PathRAG graph operations
5. **Quality Validation**: Uses HADES validation framework

### Training Orchestrator Usage

The bootstrap process uses the existing training infrastructure:

```python
# Bootstrap creates training orchestrator
self.training_orchestrator = ISNETrainingOrchestrator(
    documents=documents,
    output_dir=self.output_dir / "initial_training",
    model_output_dir=self.output_dir / "models",
    config_override=config_override
)

# Trains using existing trainer
training_results = self.training_orchestrator.train()
self.isne_trainer = self.training_orchestrator.trainer
```

### Post-Bootstrap Production Use

After bootstrap, you have access to:

1. **Trained ISNE Model**: `refined_isne_model.pt` 
2. **Training Infrastructure**: Ready for `AdaptiveISNETrainer`
3. **Production Embeddings**: `final_embeddings.npy`
4. **Training Orchestrator**: For ongoing model updates

### Seamless Transition to Adaptive Training

```python
from src.isne.adaptive_training import AdaptiveISNETrainer

# Load bootstrap results
bootstrapper = ISNEBootstrapper.load_from_results("./bootstrap-output")

# Set up adaptive training with bootstrap model
adaptive_trainer = AdaptiveISNETrainer(
    isne_pipeline=bootstrapper.training_orchestrator,
    retrain_threshold=0.15,
    min_retrain_interval=24,
    max_incremental_updates=1000
)

# Process new documents using existing infrastructure
results = adaptive_trainer.process_new_chunks(new_chunks, current_graph_stats)
```

### Training Infrastructure Benefits

1. **Consistent Architecture**: Same models, losses, and training procedures
2. **Configuration Management**: Uses existing ISNE config system
3. **Monitoring & Validation**: Built-in quality metrics and alerts
4. **Checkpoint Management**: Automatic model versioning and saving
5. **Device Management**: GPU/CPU handling through existing infrastructure

After bootstrap, your ISNE model is ready for production use with the full adaptive training system for ongoing updates.