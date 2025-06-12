# Graph Enhancement Component

The graph enhancement component provides ISNE (Inductive Shallow Node Embedding) functionality for the HADES-PathRAG system, enhancing text embeddings with graph-based relationships and structural information to improve semantic representation quality.

## Architecture Overview

The graph enhancement system follows a hierarchical component architecture implementing different ISNE strategies and deployment modes:

```
src/components/graph_enhancement/
└── isne/                    # ISNE implementation variants
    ├── core/               # Core orchestration and factory
    │   └── processor.py    # CoreISNE - Main coordinator
    ├── training/           # Training-capable ISNE
    │   └── processor.py    # ISNETrainingEnhancer - Model training
    ├── inference/          # Pre-trained model inference
    │   └── processor.py    # ISNEInferenceEnhancer - Fast inference
    ├── inductive/          # Neural network-based enhancement
    │   └── processor.py    # InductiveISNE - PyTorch models
    └── none/               # Passthrough for baselines
        └── processor.py    # NoneISNE - No enhancement
```

## What is ISNE?

ISNE (Inductive Shallow Node Embedding) is a graph neural network technique that enhances text embeddings by incorporating structural relationships between documents. It works by:

1. **Graph Construction**: Creating similarity graphs from embedding neighborhoods
2. **Structural Learning**: Learning from graph topology and node relationships  
3. **Inductive Enhancement**: Applying learned transformations to improve embeddings
4. **Semantic Preservation**: Maintaining original semantic meaning while adding structural context

## Component Types

### Core ISNE (`isne/core/processor.py`)
- **Purpose**: Factory and coordinator for different ISNE enhancement methods
- **Key Features**:
  - Dynamic method selection (isne, similarity, basic, none)
  - Graph construction from embedding similarities
  - K-nearest neighbor relationship modeling
  - Configurable enhancement strength
- **Methods**: ISNE, similarity-based, basic noise, passthrough
- **Best For**: General-purpose enhancement with automatic fallbacks

### Training ISNE (`isne/training/processor.py`)
- **Purpose**: ISNE model training and enhancement
- **Key Features**:
  - Online training during processing
  - Learned transformation matrices
  - Training history and metrics tracking
  - Automatic training triggers
- **Training**: Simulated neural network with weight/bias learning
- **Best For**: Adaptive enhancement that improves over time

### Inference ISNE (`isne/inference/processor.py`)
- **Purpose**: Fast inference with pre-trained ISNE models
- **Key Features**:
  - Model loading from disk
  - High-speed inference (1ms per embedding)
  - Cached model parameters
  - Batch processing optimization
- **Models**: Pre-trained transformation matrices
- **Best For**: Production deployments with consistent enhancement

### Inductive ISNE (`isne/inductive/processor.py`)
- **Purpose**: Deep neural network-based enhancement
- **Key Features**:
  - PyTorch neural network models
  - Multi-layer feedforward architectures
  - GPU/CUDA acceleration support
  - Incremental training capabilities
- **Models**: Configurable neural networks with dropout and regularization
- **Best For**: Research applications and custom model architectures

### None ISNE (`isne/none/processor.py`)
- **Purpose**: Passthrough component for baselines and A/B testing
- **Key Features**:
  - Zero-latency passthrough processing
  - Input/output validation
  - Perfect embedding preservation
  - Statistical tracking
- **Processing**: Direct passthrough with optional metadata
- **Best For**: Control groups, performance baselines, debugging

## Configuration

### Core Configuration
```yaml
graph_enhancement:
  isne:
    # Core ISNE configuration
    core:
      enhancement_method: "isne"  # isne | similarity | basic | none
      model_config:
        embedding_dim: 768
        hidden_dim: 256
        num_layers: 3
        dropout: 0.1
        learning_rate: 0.001
      graph_config:
        k_neighbors: 10
        similarity_threshold: 0.5
        edge_weighting: "cosine"  # cosine | euclidean | dot
        enhancement_strength: 0.5
    
    # Training-specific configuration
    training:
      training_config:
        epochs: 100
        batch_size: 64
        learning_rate: 0.001
        patience: 10
        validation_split: 0.2
      model_config:
        embedding_dim: 768
        hidden_dim: 256
        num_layers: 3
        dropout: 0.1
    
    # Inference-specific configuration
    inference:
      model_path: "./models/isne_trained.pt"
      inference_config:
        batch_size: 128
        use_cache: true
        enhancement_strength: 0.5
      model_config:
        embedding_dim: 768
        hidden_dim: 256
        num_layers: 3
    
    # Inductive neural network configuration
    inductive:
      hidden_dim: 256
      num_layers: 2
      dropout: 0.1
      device: "auto"  # cuda | cpu | auto
      learning_rate: 0.001
      max_epochs: 100
      early_stopping_patience: 10
    
    # Passthrough configuration
    none:
      preserve_original_embeddings: true
      add_metadata: true
      validate_inputs: true
      validate_outputs: true
      copy_tensors: false
```

### Enhancement Methods

#### Core ISNE Methods
1. **ISNE**: Full graph-based enhancement with learned transformations
2. **Similarity**: K-nearest neighbor similarity-based enhancement
3. **Basic**: Simple noise addition with normalization
4. **None**: Passthrough without modification

#### Graph Construction Parameters
- **k_neighbors**: Number of most similar embeddings to connect (default: 10)
- **similarity_threshold**: Minimum similarity for edge creation (0.0-1.0)
- **edge_weighting**: Similarity metric (cosine, euclidean, dot product)
- **enhancement_strength**: Blend ratio between original and enhanced (0.0-1.0)

## Usage Examples

### Basic Graph Enhancement
```python
from src.components.graph_enhancement.isne.core.processor import CoreISNE
from src.types.components.contracts import GraphEnhancementInput, ChunkEmbedding

# Initialize core ISNE enhancer
enhancer = CoreISNE({
    'enhancement_method': 'similarity',
    'graph_config': {
        'k_neighbors': 10,
        'enhancement_strength': 0.5,
        'edge_weighting': 'cosine'
    }
})

# Prepare embeddings
embeddings = [
    ChunkEmbedding(
        chunk_id="chunk_001",
        embedding=[0.1, 0.2, 0.3, ...],  # 768-dim vector
        embedding_dimension=768,
        model_name="all-MiniLM-L6-v2",
        confidence=1.0
    ),
    ChunkEmbedding(
        chunk_id="chunk_002", 
        embedding=[0.4, 0.5, 0.6, ...],
        embedding_dimension=768,
        model_name="all-MiniLM-L6-v2",
        confidence=1.0
    )
]

input_data = GraphEnhancementInput(
    embeddings=embeddings,
    enhancement_options={'method': 'similarity'}
)

# Enhance embeddings
result = enhancer.enhance(input_data)
print(f"Enhanced {len(result.enhanced_embeddings)} embeddings")
for enhanced in result.enhanced_embeddings:
    print(f"Chunk {enhanced.chunk_id}: score {enhanced.enhancement_score:.3f}")
```

### Training ISNE Models
```python
from src.components.graph_enhancement.isne.training.processor import ISNETrainingEnhancer

# Initialize training enhancer
trainer = ISNETrainingEnhancer({
    'training_config': {
        'epochs': 50,
        'learning_rate': 0.001,
        'batch_size': 32
    },
    'model_config': {
        'embedding_dim': 768,
        'hidden_dim': 256
    }
})

# Training will happen automatically during enhancement
result = trainer.enhance(input_data)

# Check training status
metrics = trainer.get_metrics()
print(f"Model trained: {metrics['is_trained']}")
print(f"Training epochs: {metrics['training_epochs_completed']}")
```

### High-Performance Inference
```python
from src.components.graph_enhancement.isne.inference.processor import ISNEInferenceEnhancer

# Initialize inference enhancer with pre-trained model
inferencer = ISNEInferenceEnhancer({
    'model_path': './models/isne_production.pt',
    'inference_config': {
        'batch_size': 128,
        'enhancement_strength': 0.3
    }
})

# Fast inference
result = inferencer.enhance(input_data)
stats = result.graph_stats
print(f"Inference time: {stats['processing_time']:.3f}s")
print(f"Throughput: {len(result.enhanced_embeddings)/stats['processing_time']:.0f} emb/sec")
```

### Neural Network Enhancement
```python
from src.components.graph_enhancement.isne.inductive.processor import InductiveISNE

# Initialize neural network enhancer
neural_enhancer = InductiveISNE({
    'hidden_dim': 512,
    'num_layers': 3,
    'dropout': 0.1,
    'device': 'cuda',
    'learning_rate': 0.001
})

# Train on data
neural_enhancer.train(input_data)

# Save trained model
neural_enhancer.save_model('./models/neural_isne.pt')

# Enhance with trained model
result = neural_enhancer.enhance(input_data)
```

### Baseline Comparison
```python
from src.components.graph_enhancement.isne.none.processor import NoneISNE

# Initialize passthrough for baseline
baseline = NoneISNE({
    'preserve_original_embeddings': True,
    'add_metadata': True
})

# Process without enhancement
baseline_result = baseline.enhance(input_data)

# Compare with enhanced results
enhanced_result = enhancer.enhance(input_data)

print("Enhancement comparison:")
for base, enh in zip(baseline_result.enhanced_embeddings, enhanced_result.enhanced_embeddings):
    similarity = cosine_similarity(base.enhanced_embedding, enh.enhanced_embedding)
    print(f"Chunk {base.chunk_id}: similarity {similarity:.3f}")
```

## Performance Characteristics

### Core ISNE
- **Throughput**: ~50-200 enhancements/sec
- **Memory**: Low (similarity matrix caching)
- **Latency**: 20-50ms per batch
- **Quality**: Good similarity-based enhancement

### Training ISNE
- **Throughput**: ~20-100 enhancements/sec (including training)
- **Memory**: Medium (training parameters)
- **Latency**: 50-200ms per batch (first time)
- **Quality**: Improves over time with more data

### Inference ISNE
- **Throughput**: ~500-1000 enhancements/sec
- **Memory**: Low (cached parameters)
- **Latency**: 1-5ms per batch
- **Quality**: Consistent with trained model

### Inductive ISNE
- **Throughput**: ~100-500 enhancements/sec (GPU dependent)
- **Memory**: High (neural network parameters)
- **Latency**: 5-20ms per batch
- **Quality**: Highest with proper training

### None ISNE
- **Throughput**: ~10,000+ passthroughs/sec
- **Memory**: Minimal
- **Latency**: <1ms per batch
- **Quality**: Perfect preservation (no enhancement)

## Integration Patterns

### Pipeline Integration
```python
from src.orchestration.pipelines.data_ingestion.pipeline import DataIngestionPipeline

# Configure graph enhancement stage
pipeline_config = {
    'stages': {
        'graph_enhancement': {
            'component_type': 'graph_enhancement',
            'component_name': 'isne_core',
            'config': {
                'enhancement_method': 'similarity',
                'graph_config': {
                    'k_neighbors': 15,
                    'enhancement_strength': 0.4
                }
            }
        }
    }
}
```

### Dynamic Enhancement Selection
```python
def select_enhancement_method(
    embedding_count: int,
    quality_requirement: str,
    latency_requirement: str
) -> str:
    """Select optimal enhancement method based on requirements."""
    
    if latency_requirement == "ultra_low":
        return "none"  # Passthrough
    elif embedding_count < 100 and quality_requirement == "highest":
        return "inductive"  # Neural network
    elif quality_requirement == "adaptive":
        return "training"  # Learning enhancer
    elif latency_requirement == "low":
        return "inference"  # Pre-trained model
    else:
        return "similarity"  # Core method

# Use in pipeline
enhancement_method = select_enhancement_method(
    embedding_count=len(embeddings),
    quality_requirement="standard",
    latency_requirement="medium"
)
```

### Multi-Stage Enhancement
```python
class ProgressiveEnhancer:
    """Multi-stage enhancement with increasing sophistication."""
    
    def __init__(self):
        self.similarity = CoreISNE({'enhancement_method': 'similarity'})
        self.training = ISNETrainingEnhancer()
        self.neural = InductiveISNE()
    
    def enhance_progressive(self, input_data: GraphEnhancementInput) -> GraphEnhancementOutput:
        """Apply progressive enhancement stages."""
        
        # Stage 1: Similarity-based enhancement
        stage1_result = self.similarity.enhance(input_data)
        
        # Stage 2: Training enhancement on stage 1 results
        stage1_as_input = GraphEnhancementInput(
            embeddings=[
                ChunkEmbedding(
                    chunk_id=enh.chunk_id,
                    embedding=enh.enhanced_embedding,
                    embedding_dimension=len(enh.enhanced_embedding)
                )
                for enh in stage1_result.enhanced_embeddings
            ]
        )
        stage2_result = self.training.enhance(stage1_as_input)
        
        # Stage 3: Neural enhancement (if trained)
        if self.neural.is_trained():
            final_result = self.neural.enhance(stage1_as_input)
        else:
            final_result = stage2_result
        
        return final_result
```

## Model Training and Management

### Training Workflow
```python
class ISNETrainingWorkflow:
    """Complete ISNE training and deployment workflow."""
    
    def __init__(self):
        self.trainer = ISNETrainingEnhancer()
        self.inferencer = ISNEInferenceEnhancer()
    
    def train_and_deploy(self, training_data: GraphEnhancementInput, model_path: str):
        """Train model and prepare for inference deployment."""
        
        # Training phase
        print("Starting ISNE training...")
        self.trainer.train(training_data)
        
        # Validation phase
        validation_result = self.trainer.enhance(training_data)
        avg_score = sum(e.enhancement_score or 0 for e in validation_result.enhanced_embeddings) / len(validation_result.enhanced_embeddings)
        print(f"Average enhancement score: {avg_score:.3f}")
        
        # Model export (simulated)
        print(f"Saving model to {model_path}")
        # In real implementation: self.trainer.save_model(model_path)
        
        # Inference setup
        self.inferencer.configure({'model_path': model_path})
        print("Model ready for inference")
        
        return {
            'training_epochs': len(self.trainer.get_metrics()['training_epochs_completed']),
            'enhancement_score': avg_score,
            'model_path': model_path
        }
```

### Model Evaluation
```python
def evaluate_enhancement_quality(
    original_embeddings: List[List[float]],
    enhanced_embeddings: List[List[float]]
) -> Dict[str, float]:
    """Evaluate enhancement quality metrics."""
    
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    
    orig_matrix = np.array(original_embeddings)
    enh_matrix = np.array(enhanced_embeddings)
    
    # Semantic preservation (cosine similarity)
    semantic_preservation = []
    for orig, enh in zip(orig_matrix, enh_matrix):
        sim = cosine_similarity([orig], [enh])[0][0]
        semantic_preservation.append(sim)
    
    # Embedding diversity (average pairwise distance)
    orig_diversity = np.mean(1 - cosine_similarity(orig_matrix))
    enh_diversity = np.mean(1 - cosine_similarity(enh_matrix))
    
    # Enhancement magnitude
    enhancement_magnitude = []
    for orig, enh in zip(orig_matrix, enh_matrix):
        magnitude = np.linalg.norm(enh - orig)
        enhancement_magnitude.append(magnitude)
    
    return {
        'semantic_preservation': float(np.mean(semantic_preservation)),
        'diversity_improvement': float(enh_diversity - orig_diversity),
        'average_enhancement_magnitude': float(np.mean(enhancement_magnitude)),
        'enhancement_consistency': float(np.std(enhancement_magnitude))
    }
```

## Error Handling and Fallbacks

### Robust Enhancement Pipeline
```python
class RobustGraphEnhancer:
    """Graph enhancer with comprehensive fallback strategies."""
    
    def __init__(self):
        self.primary = InductiveISNE()  # Best quality
        self.secondary = ISNETrainingEnhancer()  # Good fallback
        self.tertiary = CoreISNE()  # Reliable fallback
        self.fallback = NoneISNE()  # Always works
    
    def enhance_with_fallbacks(self, input_data: GraphEnhancementInput) -> GraphEnhancementOutput:
        """Attempt enhancement with automatic fallback chain."""
        
        enhancers = [
            ("primary", self.primary),
            ("secondary", self.secondary), 
            ("tertiary", self.tertiary),
            ("fallback", self.fallback)
        ]
        
        for name, enhancer in enhancers:
            try:
                if enhancer.health_check():
                    result = enhancer.enhance(input_data)
                    if result.enhanced_embeddings:
                        print(f"Successfully enhanced using {name}")
                        return result
            except Exception as e:
                print(f"{name} enhancer failed: {e}")
                continue
        
        raise RuntimeError("All enhancement methods failed")
```

### Health Monitoring
```python
def monitor_enhancement_health():
    """Monitor graph enhancement component health."""
    
    enhancers = {
        'core': CoreISNE(),
        'training': ISNETrainingEnhancer(),
        'inference': ISNEInferenceEnhancer(),
        'inductive': InductiveISNE(),
        'none': NoneISNE()
    }
    
    health_report = {}
    
    for name, enhancer in enhancers.items():
        try:
            is_healthy = enhancer.health_check()
            metrics = enhancer.get_metrics()
            
            health_report[name] = {
                'healthy': is_healthy,
                'avg_processing_time': metrics.get('avg_processing_time', 0),
                'total_processed': metrics.get('total_processed', 0),
                'component_version': enhancer.version
            }
        except Exception as e:
            health_report[name] = {
                'healthy': False,
                'error': str(e)
            }
    
    return health_report
```

## Best Practices

### Enhancement Strategy Selection
1. **Development/Testing**: Use `none` for baselines, `similarity` for quick enhancement
2. **Production Low-Latency**: Use `inference` with pre-trained models
3. **Production High-Quality**: Use `training` for adaptive improvement
4. **Research/Custom**: Use `inductive` with neural networks

### Configuration Guidelines
1. **K-Neighbors**: 5-15 for most applications, higher for diverse datasets
2. **Enhancement Strength**: 0.3-0.7, lower for preservation, higher for improvement
3. **Similarity Threshold**: 0.3-0.7, higher for stricter graph connectivity
4. **Neural Architecture**: Start with 2-3 layers, 256-512 hidden units

### Performance Optimization
1. **Batch Processing**: Process embeddings in batches for efficiency
2. **Model Caching**: Cache trained models and parameters
3. **GPU Utilization**: Use CUDA for inductive neural networks
4. **Fallback Chains**: Always implement fallback strategies

### Quality Assurance
1. **Validation**: Monitor enhancement scores and semantic preservation
2. **A/B Testing**: Compare enhanced vs original embeddings in downstream tasks
3. **Training Monitoring**: Track training loss and convergence
4. **Health Checks**: Regular component health monitoring

## Related Components

- **Embedding**: Provides base embeddings for graph enhancement
- **Storage**: Stores enhanced embeddings and graph structures  
- **Pipeline**: Orchestrates enhancement within processing workflows
- **Validation**: Validates enhancement quality and consistency