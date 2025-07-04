# ISNE (Inductive Shallow Node Embedding)

## Overview
This directory contains the ISNE implementation for graph-based embedding enhancement, providing structure-aware improvements to base embeddings.

## Contents
- `models/` - ISNE model implementations
  - `isne_model.py` - Core ISNE model
  - `directory_aware_isne.py` - Filesystem-aware variant
- `training/` - Training pipeline and utilities
  - `pipeline.py` - Main training pipeline
  - `directory_aware_trainer.py` - Specialized trainer
  - `hierarchical_batch_sampler.py` - Directory-aware batch sampling
- `pipeline/` - Inference pipeline components

## Related Resources
- **Theory Documentation**: `/docs/concepts/ISNE_THEORY.md`
- **Bootstrap Architecture**: `/docs/architecture/ISNE_BOOTSTRAP_ARCHITECTURE.md`
- **Training Guide**: `/docs/guides/isne_training_guide.md`
- **Configuration**: `/src/config/isne/`
- **PathRAG Integration**: `/src/pathrag/` (uses ISNE for query placement)
- **Storage Schema**: `/src/storage/` (graph structure for ISNE)

## Key Concepts
- **Graph Enhancement**: Improves embeddings using graph structure
- **Directory Awareness**: Leverages filesystem hierarchy for better representations
- **Hierarchical Training**: Preserves directory relationships during training
- **Multi-hop Encoding**: Captures relationships at various distances

## Model Architecture
- Multi-layer Graph Neural Network (GNN)
- Attention mechanisms for relationship weighting
- Contrastive learning objectives
- Directory-aware positional encoding

## Dependencies
- PyTorch for deep learning
- PyTorch Geometric for graph operations
- NetworkX for graph utilities
- NumPy for numerical operations

## Training Requirements
- GPU recommended (supports multi-GPU)
- Minimum 16GB RAM
- Pre-computed embeddings from Jina v4

## Research Background
ISNE builds on shallow node embedding techniques but adds inductive capabilities and hierarchical awareness. This implementation extends the core concepts with filesystem-aware enhancements specific to code and documentation retrieval.