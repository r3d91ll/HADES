# Preserved Components from bootstrap_graph_builder

This directory contains valuable components preserved from the original bootstrap_graph_builder implementation. These components complement the new supra-weight bootstrap pipeline.

## Components

### 1. spectral_sparsifier.py
**Purpose**: Implements Spielman-Srivastava spectral graph sparsification algorithm (2011)
**Key Features**:
- Preserves spectral properties while reducing edge count by 95%
- Uses effective resistance-based edge sampling
- Maintains edge type distributions
- Mathematical guarantees on spectral approximation

**Integration**: Can be used as a post-processing step after supra-weight graph construction to further reduce density while preserving graph properties essential for ISNE training.

### 2. thermal_aware_batch_group_extractor.py
**Purpose**: Manages I/O operations with thermal awareness to prevent NVMe overheating
**Key Features**:
- Real-time thermal monitoring
- Dynamic I/O throttling (1x, 1.5x, 3x delays)
- Batch write accumulation
- Compression support
- Memory buffer management

**Integration**: Can replace the BatchWriter in our storage layer when thermal management is critical.

### 3. raid1_thermal_monitor.py & nvme_thermal_monitor.py
**Purpose**: Hardware-specific thermal monitoring
**Key Features**:
- RAID1 array monitoring (Crucial T700 4TB drives)
- General NVMe temperature monitoring
- Temperature thresholds and alerts
- Integration with thermal-aware components

**Integration**: Can be used by storage components to monitor hardware health during intensive graph construction.

### 4. lcc_classifier.py
**Purpose**: Library of Congress Classification for semantic document categorization
**Key Features**:
- Hierarchical classification system
- Domain-specific categorization
- Confidence scoring
- Integration with graph building

**Integration**: Can enhance theory-practice bridge detection by providing additional semantic categorization.

### 5. simple_chunker.py
**Purpose**: AST-based code chunking without embedding dependencies
**Key Features**:
- Python AST parsing
- Function/class extraction
- Docstring preservation
- Simple text chunking fallback

**Integration**: Can be used as an alternative chunker in our processing pipeline for code-heavy datasets.

## Usage Examples

### Spectral Sparsification
```python
from preserved_components.spectral_sparsifier import SpectralSparsifier

# After building dense graph with supra-weights
sparsifier = SpectralSparsifier(config={'target_density': 0.05})
sparse_graph = sparsifier.sparsify_graph(dense_graph, preserve_edge_types=True)
```

### Thermal-Aware Processing
```python
from preserved_components.thermal_aware_batch_group_extractor import ThermalAwareBatchGroupManager

# Replace standard batch writer with thermal-aware version
manager = ThermalAwareBatchGroupManager(
    total_chunks=10000,
    num_workers=4,
    chunks_per_batch=100,
    thermal_threshold_c=70,
    thermal_critical_c=80
)
```

### LCC Classification Enhancement
```python
from preserved_components.lcc_classifier import LCCClassifier

# Add to theory-practice detector
classifier = LCCClassifier()
lcc_class, confidence = classifier.classify_document(content, file_path)
# Use classification to enhance bridge detection
```

## Notes

- These components were part of a sophisticated graph building system that achieved 6 hours → 45 minutes performance improvement
- The embedding-free approach aligns with our understanding that ISNE creates embeddings FROM graph structure
- Thermal monitoring is particularly valuable for large-scale ML workloads on consumer hardware
- Spectral sparsification provides mathematical guarantees important for preserving graph properties