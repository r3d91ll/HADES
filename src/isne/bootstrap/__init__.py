"""
ISNE Bootstrap Pipeline Module

This module provides the complete bootstrap pipeline for ISNE model creation.
It handles the one-time process of:
- Document processing
- Text chunking  
- Embedding generation
- Graph construction
- ISNE model training

The bootstrap pipeline is separate from ongoing training which is handled
by the training module and API endpoints.

Key Components:
- ISNEBootstrapPipeline: Main pipeline orchestrator
- Individual pipeline stages with comprehensive validation and monitoring
- BootstrapMonitor: Real-time monitoring and alerting
- BootstrapConfig: Comprehensive configuration system

Usage:
    from src.isne.bootstrap import ISNEBootstrapPipeline, BootstrapConfig
    
    config = BootstrapConfig.from_yaml("config/bootstrap.yaml")
    pipeline = ISNEBootstrapPipeline(config)
    
    result = pipeline.run(
        input_files=[Path("/path/to/docs")],
        output_dir=Path("/path/to/output"),
        model_name="my_isne_model"
    )

Command Line Interface:
    python -m src.isne.bootstrap.cli --input-dir /path/to/docs --output-dir /path/to/output
"""

from .pipeline import ISNEBootstrapPipeline, BootstrapResult
from .config import BootstrapConfig
from .monitoring import BootstrapMonitor

# Export stages for advanced usage
from .stages import (
    BaseBootstrapStage,
    DocumentProcessingStage,
    ChunkingStage,
    EmbeddingStage,
    GraphConstructionStage,
    ISNETrainingStage
)

__all__ = [
    'ISNEBootstrapPipeline',
    'BootstrapResult',
    'BootstrapConfig', 
    'BootstrapMonitor',
    # Stages
    'BaseBootstrapStage',
    'DocumentProcessingStage',
    'ChunkingStage',
    'EmbeddingStage',
    'GraphConstructionStage',
    'ISNETrainingStage'
]