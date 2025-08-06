# ArXiv Processing Pipeline

## Overview

This module implements the ArXiv paper processing pipeline for the Information Reconstructionism theory validation. It handles the ingestion, processing, and embedding of ~2.8M ArXiv papers using state-of-the-art Jina v4 embeddings.

## Current Status (January 2025)

- **Database Rebuild**: In progress with Jina v4 (2048-dimensional embeddings)
- **Processing Rate**: ~29 papers/second on dual A6000 GPUs with NVLink
- **Source Data**: `/fastpool/temp/arxiv-metadata-oai-snapshot.json` (2.8M papers)
- **Database**: ArangoDB `academy_store` database, `base_arxiv` collection
- **Progress**: 834,347 / 2,792,339 papers (29.9% complete after 8 hours)

## Directory Structure

```
arxiv/
├── core/                          # Core processing components
│   ├── jina_v4_embedder.py       # Clean Jina v4 implementation with fp16
│   ├── batch_embed_jina.py       # Batch processing for embeddings
│   └── enhanced_docling_processor_v2.py  # PDF processing with Docling
├── scripts/                       # Executable scripts (minimal set)
│   ├── rebuild_dual_gpu.py       # Dual-GPU rebuild with Ray (ACTIVE)
│   ├── daily_arxiv_update.py     # Daily updates with Jina v4 (CRON DISABLED)
│   └── process_pdf_on_demand.py  # On-demand PDF processing
├── monitoring/                    # Monitoring tools
│   └── monitor_system_resources.py
├── utils/                         # Utilities
│   └── count_pending_pdfs.py
├── tests/                         # Test suite
│   ├── test_jina_v4.py
│   ├── test_docling.py
│   ├── test_equation_extraction.py
│   └── test_image_extraction.py
├── docs/                          # Documentation
│   └── EMBEDDING_GUIDE.md
├── checkpoints/                   # Processing checkpoints
│   └── backups/
└── logs/                          # Log files
    └── old/
```

## Key Components

### 1. Jina v4 Embeddings

The migration from v3 to v4 brings significant improvements:

| Feature | Jina v3 | Jina v4 |
|---------|---------|---------|
| **Dimensions** | 1024 | 2048 |
| **Context Length** | 8,192 tokens | 32,768 tokens |
| **Model Base** | BERT | Qwen2.5-VL-3B |
| **Precision** | fp32 | fp16 |
| **Processing Time** | 9 hours | ~27 hours |
| **Rate** | ~86 papers/sec | ~29 papers/sec |
| **Storage** | ~8 GB | ~16 GB |

### 2. Database Schema

```python
{
    "_key": "2310_08560",              # Normalized arxiv_id
    "arxiv_id": "2310.08560",          # Original ID
    "title": "Paper Title",
    "abstract": "Paper abstract...",
    "authors": "Author1, Author2",
    "categories": "cs.AI cs.CL",
    "update_date": "2023-10-15",
    "authors_parsed": [["Last", "First"], ...],
    "abstract_embeddings": [...],      # 2048-dim float array
    "embedding_model": "jinaai/jina-embeddings-v4",
    "embedding_dim": 2048,
    "embedding_date": "2025-01-20T...",
    "embedding_version": "v4_dual_gpu",
    "pdf_status": null,                # Future: "not_downloaded", "downloaded", "processed"
    "gpu_id": 0                        # Which GPU processed this
}
```

### 3. Processing Pipeline

#### Phase 1: Abstract-Only Processing (Current)
- Process metadata and abstracts from JSON source
- Generate 2048-dim Jina v4 embeddings
- Store in ArangoDB for fast retrieval

#### Phase 2: PDF Enhancement (Future)
- On-demand PDF download from ArXiv
- Full-text extraction with Docling
- Equation and image extraction with PyMuPDF
- Chunk and embed full content

## Usage Guide

### Running the Dual-GPU Rebuild

```bash
# Set environment
export ARANGO_PASSWORD='your_password'
export CUDA_VISIBLE_DEVICES=0,1  # Use both GPUs

# Test first (100 papers, ~5 minutes)
python3 scripts/rebuild_dual_gpu.py \
    --source /fastpool/temp/arxiv-metadata-oai-snapshot.json \
    --limit 100 \
    --batch-size 32 \
    --num-gpus 2 \
    --db-host 192.168.1.69

# Full rebuild (overnight, ~27 hours)
nohup python3 scripts/rebuild_dual_gpu.py \
    --source /fastpool/temp/arxiv-metadata-oai-snapshot.json \
    --batch-size 256 \
    --num-gpus 2 \
    --db-host 192.168.1.69 \
    > rebuild_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```


### Monitoring Progress

```bash
# Check paper count in database
python3 -c "
from arango import ArangoClient
import os
client = ArangoClient(hosts='http://192.168.1.69:8529')
db = client.db('academy_store', username='root', password=os.environ['ARANGO_PASSWORD'])
count = db.collection('base_arxiv').count()
print(f'Papers processed: {count:,} / 2,792,339')
"

# Monitor GPU usage
watch -n 1 nvidia-smi

# Check running process
ps aux | grep rebuild_dual_gpu
```

### On-Demand PDF Processing

```bash
# Process specific paper with PDF
python3 scripts/process_pdf_on_demand.py 2310.08560
```

### Daily Updates (After Rebuild Completes)

```bash
# Re-enable cron job after testing
crontab -e
# Uncomment the line to re-enable at 01:00 UTC daily

# Manual test run
export ARANGO_PASSWORD='your_password'
poetry run python3 scripts/daily_arxiv_update.py --days-back 1

# Catch up multiple days if needed
for i in {1..7}; do
    poetry run python3 scripts/daily_arxiv_update.py --days-back $i
done
```

Note: Cron job is currently disabled. See `docs/CRON_SETUP.md` for re-enabling instructions.

## Performance Optimization

### GPU Configuration
- **Dual A6000s**: 48GB VRAM each
- **NVLink**: 600 GB/s GPU-to-GPU bandwidth
- **Batch Size**: 256 optimal for dual-GPU
- **Workers**: 2 Ray actors, one per GPU

### Memory Management
- **fp16 Precision**: Half memory usage vs fp32
- **Batch Processing**: Efficient GPU utilization
- **Checkpoint Recovery**: Resume from failures

### Expected Performance
- **Single GPU**: ~15-20 papers/second
- **Dual GPU (PCIe)**: ~25-35 papers/second  
- **Dual GPU + NVLink**: ~29-45 papers/second

## Theory Connection

This pipeline implements core dimensions of Information Reconstructionism:

- **WHERE**: Location in database, accessibility via AQL queries
- **WHAT**: Semantic content in 2048-dimensional embeddings
- **CONVEYANCE**: Abstract clarity and implementability
- **Zero Propagation**: Papers without abstracts → no embeddings (WHAT = 0)
- **Context Amplification**: Measured through embedding similarity (Context^α)

The doubled dimensionality (1024→2048) provides richer semantic representation for validating the multiplicative dependency: WHERE × WHAT × CONVEYANCE × TIME.

## Migration Notes

### From Jina v3 to v4

```python
# Old (v3)
model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3")
embeddings = model.encode(texts)  # 1024 dims

# New (v4)
model = AutoModel.from_pretrained(
    "jinaai/jina-embeddings-v4",
    torch_dtype=torch.float16
)
embeddings = model.encode_text(
    texts=texts,
    task="retrieval"  # Required parameter
)  # 2048 dims
```

### API Changes
- Method: `encode()` → `encode_text()`
- Task parameter: Now required (`"retrieval"` or `"separation"`)
- Dimensions: 1024 → 2048
- Context: 8k → 32k tokens

## Troubleshooting

### Ray Initialization Issues
If Ray fails to initialize, check:
- CUDA_VISIBLE_DEVICES is set correctly
- Ray version compatibility
- GPU memory availability

### CUDA Out of Memory
```bash
# Reduce batch size
--batch-size 16  # Instead of 32+
```

### Database Connection
```bash
# Test connection
curl -u root:$ARANGO_PASSWORD http://192.168.1.69:8529/_api/version
```

### Verify GPU Setup
```bash
python3 scripts/verify_dual_gpu.py
```

## Future Enhancements

1. **PDF Processing Pipeline**
   - Full-text extraction with Docling
   - Mathematical equation preservation
   - Figure and table extraction
   - Semantic chunking with late chunking

2. **GitHub Integration** 
   - Code repository analysis
   - Implementation linking to papers
   - Theory-practice bridge discovery

3. **Citation Networks**
   - Build citation graphs in ArangoDB
   - Influence propagation analysis
   - Research lineage tracking

4. **Real-time Updates**
   - ArXiv feed integration
   - Incremental daily processing
   - Version tracking for updates

## Related Documentation

- **Main Theory**: `/home/todd/olympus/CLAUDE.md`
- **HADES Overview**: `/home/todd/olympus/HADES/CLAUDE.md`
- **Embedding Guide**: `docs/EMBEDDING_GUIDE.md`
- **Dual-GPU Setup**: `DUAL_GPU_REBUILD.md`
- **Overnight Instructions**: `OVERNIGHT_REBUILD.md`

## Contact

Part of the Olympus/HADES Information Reconstructionism implementation.
See parent documentation for theoretical framework and architectural decisions.