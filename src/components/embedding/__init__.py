"""
Embedding Components

This module provides embedding components that implement the
Embedder protocol for various embedding models and backends.

Available components:
- core: Core embedder using HADES embedding registry (auto-selection)
- cpu: CPU-based embedder (uses model engine for CPU models)
- gpu: High-throughput GPU embedder (uses model engine for GPU models)
- encoder: Transformer encoder embedder (custom models)
"""

from .core.processor import CoreEmbedder
from .cpu.processor import CPUEmbedder
from .gpu.processor import GPUEmbedder
from .encoder.processor import EncoderEmbedder

__all__ = [
    "CoreEmbedder",
    "CPUEmbedder", 
    "GPUEmbedder",
    "EncoderEmbedder"
]