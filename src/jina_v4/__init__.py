"""
Jina v4 Unified Document Processing Component

This component replaces the entire document processing pipeline with a single
Jina v4 model that handles:
- Document parsing (multimodal)
- Embedding generation
- Late-chunking
- Keyword extraction
- Semantic structuring

All processing happens on GPU with minimal overhead.
"""

from .jina_processor import JinaV4Processor
from .factory import create_jina_component
from .isne_adapter import (
    convert_jina_to_isne,
    convert_jina_to_embedding_input,
    extract_graph_structure,
    prepare_for_storage
)

__all__ = [
    'JinaV4Processor', 
    'create_jina_component',
    'convert_jina_to_isne',
    'convert_jina_to_embedding_input',
    'extract_graph_structure',
    'prepare_for_storage'
]