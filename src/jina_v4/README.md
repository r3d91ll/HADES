# Jina v4 Integration

## Overview
This directory contains the Jina v4 integration for unified document processing, embedding generation, and keyword extraction.

## Contents
- `jina-embeddings-v4.pdf` - Research paper on Jina v4's multimodal capabilities
- `jina_processor.py` - Main processor handling all document operations
- `vllm_integration.py` - vLLM backend for efficient inference
- `isne_adapter.py` - Adapter to convert Jina output for ISNE
- `factory.py` - Component factory for initialization

## Related Resources
- **User Documentation**: `/docs/concepts/CORE_CONCEPTS.md#theory-practice-bridges`
- **Configuration Guide**: `/docs/configuration/jina_v4_config.md`
- **API Reference**: `/docs/api/jina_v4_api.md`
- **Configuration**: `/config/jina_v4/config.yaml`
- **Migration Guide**: `/docs/MIGRATION_FROM_ORIGINAL.md`
- **Bootstrap Scripts**: `/scripts/bootstrap/bootstrap_jinav4.py`

## Key Features
- **Multimodal Processing**: Handles text, images, and code in unified pipeline
- **Late Chunking**: Preserves document context during chunking
- **Attention-based Keywords**: Extracts keywords using model attention weights
- **Filesystem Awareness**: Maintains directory hierarchy information

## Implementation Status
⚠️ **Note**: Several methods have placeholder implementations:
- `_parse_document()` - Needs Docling integration
- `_extract_embeddings_local()` - Needs vLLM direct access
- `_extract_keywords()` - Needs attention weight extraction
- Semantic operations - Needs actual calculations
- Image processing - Needs multimodal implementation

## Dependencies
- Transformers/Hugging Face
- vLLM for inference
- Docling for document parsing
- PyTorch for tensor operations

## Research Context
The Jina v4 paper demonstrates state-of-the-art performance on multimodal retrieval tasks, with particular strength in handling visually rich documents. Our implementation leverages these capabilities for unified document processing.