"""
Chunking Components

This module provides chunking components that implement the
Chunker protocol for various content types and chunking strategies.

Available components:
- core: Core chunker using HADES chunking registry (auto-selection)
- text: Specialized text chunker (CPU, Chonky, Chonky Batch)
- code: Specialized code chunker (AST-based, language-aware)
"""

from .core.processor import CoreChunker
from .chunkers.text.processor import TextChunker
from .chunkers.code.processor import CodeChunker
from .chunkers.cpu.processor import CPUChunker
from .chunkers.chonky.processor import ChonkyChunker
from .factory import create_chunking_component, get_available_chunking_components

__all__ = [
    "CoreChunker",
    "TextChunker", 
    "CodeChunker",
    "CPUChunker",
    "ChonkyChunker",
    "create_chunking_component",
    "get_available_chunking_components"
]