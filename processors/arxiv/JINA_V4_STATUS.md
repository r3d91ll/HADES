# Jina v4 Implementation Status

## ✅ Successfully Completed

### 1. Cleaned Up ArXiv Directory

- Removed all shell scripts (.sh files)
- Moved deprecated code to Acheron with timestamps
- Kept only working components

### 2. Updated to Jina v4

- **Model**: `jinaai/jina-embeddings-v4` (NOT v3!)
- **Context**: 32,768 tokens (4x larger than v3's 8k)
- **Dimensions**: 2048 (double v3's 1024)
- **Precision**: fp16 for GPU efficiency
- **Base Model**: Qwen2.5-VL-3B-Instruct (completely different from v3)

### 3. Working Components

#### Core Modules

- `core/jina_v4_embedder.py` - Clean Jina v4 implementation with correct API
- `core/batch_embed_jina.py` - Updated for v4 (needs API fixes)
- `core/enhanced_docling_processor_v2.py` - PDF processor (Docling API fixed)

#### Scripts

- `scripts/process_pdf_on_demand.py` - On-demand PDF processing
- `scripts/daily_arxiv_update.py` - Daily updates
- `scripts/catchup_arxiv.py` - Catch up missing papers

### 4. Key Differences from v3

| Feature | Jina v3 | Jina v4 |
|---------|---------|---------|
| Context Length | 8,192 tokens | 32,768 tokens |
| Embedding Dims | 1,024 | 2,048 |
| Base Model | BERT-based | Qwen2.5-VL-3B |
| API Method | `model.encode()` | `model.encode_text()` |
| Task Parameter | Not required | Required (`retrieval`, `code`, etc.) |
| Precision | fp32 default | fp16 recommended |
| Late Chunking | Limited | Full support |

### 5. Python Commands (No Shell Scripts!)

```python
# Process single PDF
CUDA_VISIBLE_DEVICES=1 python3 scripts/process_pdf_on_demand.py 2301.00234

# Daily update
python3 scripts/daily_arxiv_update.py --date 2025-01-15

# Catch up
python3 scripts/catchup_arxiv.py --limit 100

# Test Jina v4
CUDA_VISIBLE_DEVICES=1 python3 core/jina_v4_embedder.py
```

## 🔧 Issues Fixed

1. **Import Errors**: Fixed `DocumentConversionInput` and pipeline_options
2. **API Changes**: Updated to use `encode_text()` with task parameter
3. **Tensor Conversion**: Properly handle CUDA to CPU conversion
4. **fp16 Support**: Now using fp16 for efficiency

## 📝 Late Chunking

With Jina v4's 32k context, late chunking works as follows:

1. Process entire document (up to 32k tokens) through transformer
2. Get token-level embeddings with full context
3. Apply chunking AFTER embeddings generated
4. Each chunk maintains awareness of surrounding context

This preserves long-range dependencies and improves semantic coherence.

## ⚠️ Important Notes

1. **Always use fp16** on GPU for efficiency
2. **Task parameter required** for all encode operations
3. **2048 dimensions** - update any dimension checks
4. **32k context** - most documents fit in single chunk

## Next Steps

- [ ] Update batch_embed_jina.py to use jina_v4_embedder.py
- [ ] Test on-demand PDF processing with real ArXiv papers
- [ ] Implement GitHub repository processing with code task
- [ ] Set up MCP tools for GitHub integration
