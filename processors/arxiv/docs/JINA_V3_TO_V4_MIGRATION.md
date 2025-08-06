# Jina v3 to v4 Migration Guide

## Executive Summary

The migration from Jina v3 to v4 represents a fundamental architectural shift, not just a version upgrade. This document captures the lessons learned during our ArXiv database rebuild.

## Timeline

- **Discovery**: January 2025 - Found all code using v3 instead of v4
- **Migration**: Completed in 1 day
- **Rebuild Start**: Database rebuild initiated with dual-GPU setup
- **Expected Completion**: ~27 hours for 2.8M papers

## Critical Differences

### Model Architecture

| Aspect | Jina v3 | Jina v4 |
|--------|---------|---------|
| **Base Model** | BERT-based | Qwen2.5-VL-3B |
| **Architecture** | Traditional transformer | Vision-Language model |
| **Parameters** | ~100M | ~3B |
| **Multimodal** | No | Yes |
| **Code Understanding** | Limited | Enhanced with LoRA |

### Technical Specifications

| Feature | v3 | v4 | Impact |
|---------|-----|-----|---------|
| **Dimensions** | 1024 | 2048 | 2x storage, better semantics |
| **Context** | 8,192 tokens | 32,768 tokens | 4x longer documents |
| **Precision** | fp32 | fp16 recommended | 50% memory savings |
| **Processing** | ~86 papers/sec | ~29 papers/sec | 3x slower but worth it |
| **API Method** | `encode()` | `encode_text()` | Breaking change |

## Code Migration

### Old Code (v3)
```python
from transformers import AutoModel
import torch

# Jina v3
model = AutoModel.from_pretrained(
    "jinaai/jina-embeddings-v3",
    trust_remote_code=True
)
model.eval()

# Simple encoding
embeddings = model.encode(texts)  # 1024 dimensions
```

### New Code (v4)
```python
from transformers import AutoModel
import torch

# Jina v4 with fp16
model = AutoModel.from_pretrained(
    "jinaai/jina-embeddings-v4",
    trust_remote_code=True,
    torch_dtype=torch.float16  # Critical for efficiency
).to("cuda")
model.eval()

# Must use encode_text with task parameter
embeddings = model.encode_text(
    texts=texts,
    task="retrieval",  # or "separation" for clustering
    batch_size=len(texts)
)  # 2048 dimensions
```

## Performance Impact

### Processing Time
- **v3**: 9 hours for 2.8M papers
- **v4**: ~27 hours for 2.8M papers
- **Reason**: 3B parameter model vs 100M, double dimensions

### Memory Usage
- **v3**: ~8 GB for embeddings storage
- **v4**: ~16 GB for embeddings storage
- **GPU Memory**: 8-12 GB with fp16 (would be 16-24 GB with fp32)

### Quality Improvements
- **Semantic Understanding**: Significantly better with vision-language foundation
- **Code Comprehension**: Enhanced through code LoRA adapter
- **Late Chunking**: Native support for context-aware chunking
- **Multimodal**: Can process images alongside text

## Migration Checklist

### 1. Update Model Names
```bash
# Find all v3 references
grep -r "jina-embeddings-v3" .
grep -r "jina-bert-v2" .

# Replace with v4
sed -i 's/jina-embeddings-v3/jina-embeddings-v4/g' file.py
```

### 2. Update API Calls
```python
# Old
embeddings = model.encode(texts)

# New
embeddings = model.encode_text(
    texts=texts,
    task="retrieval"  # Required!
)
```

### 3. Update Dimensions
```python
# Old
expected_dim = 1024
chunk_size = 6000  # For 8k context

# New
expected_dim = 2048
chunk_size = 28000  # For 32k context
```

### 4. Add fp16 Support
```python
# Always use fp16 for v4
model = AutoModel.from_pretrained(
    "jinaai/jina-embeddings-v4",
    trust_remote_code=True,
    torch_dtype=torch.float16  # Critical
)
```

### 5. Update Database Schema
```python
# Add new fields
{
    "embedding_model": "jinaai/jina-embeddings-v4",
    "embedding_dim": 2048,
    "embedding_version": "v4_dual_gpu",
    "embedding_date": datetime.utcnow().isoformat()
}
```

## Common Errors and Fixes

### Error 1: AttributeError: 'JinaBertModel' object has no attribute 'encode'
**Fix**: Use `encode_text()` instead of `encode()`

### Error 2: Missing task parameter
**Fix**: Always provide `task="retrieval"` or `task="separation"`

### Error 3: CUDA OOM with fp32
**Fix**: Use `torch_dtype=torch.float16`

### Error 4: Wrong embedding dimensions (1024 instead of 2048)
**Fix**: Model might be cached, clear cache and re-download

### Error 5: Slow processing
**Fix**: This is expected - v4 is 3x slower but provides much better quality

## Rollback Plan

If you need to rollback to v3:
```python
# Keep v3 code in Acheron for reference
# Database has embedding_model field to track versions
# Can query specifically for v3 embeddings:
FOR doc IN base_arxiv
  FILTER doc.embedding_model == "jinaai/jina-embeddings-v3"
  RETURN doc
```

## Lessons Learned

1. **Always verify model version** - We ran v3 for months thinking it was v4
2. **fp16 is essential** - Without it, GPU memory usage doubles
3. **Task parameter is required** - v4 won't work without it
4. **Processing time triples** - But quality improvements justify it
5. **Late chunking is powerful** - Process full document before chunking
6. **Multimodal capability** - Future-proofs for image processing

## Future Considerations

### Phase 2: PDF Processing
- v4's 32k context perfect for full papers
- Can process paper + images together
- Late chunking maintains document coherence

### GitHub Integration
- v4's code LoRA adapter understands code better
- Can embed entire files without chunking
- Better theory-practice bridge discovery

### Performance Optimization
- Investigate 8-bit quantization
- Explore batching strategies
- Consider caching frequent queries

## Database Migration Status

As of January 2025:
- **Papers to process**: 2,792,339
- **Processing rate**: ~29 papers/second
- **Time estimate**: ~27 hours
- **Storage needed**: ~16 GB for embeddings alone

## Validation

To verify successful migration:
```python
# Check embedding dimensions
from arango import ArangoClient
client = ArangoClient(hosts='http://192.168.1.69:8529')
db = client.db('academy_store', username='root', password=password)

# Sample a document
doc = db.collection('base_arxiv').random()
assert len(doc['abstract_embeddings']) == 2048
assert doc['embedding_model'] == 'jinaai/jina-embeddings-v4'
```

## References

- [Jina v4 Announcement](https://jina.ai/news/jina-embeddings-v4)
- [Late Chunking Paper](https://arxiv.org/abs/2409.04701)
- [Jina Documentation](https://jina.ai/embeddings)

---

*Migration completed: January 2025*
*Database rebuild: In progress*
*Expected completion: ~19 hours remaining*