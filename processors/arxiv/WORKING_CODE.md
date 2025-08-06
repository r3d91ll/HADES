# Working ArXiv Processor Code

## Active Components (Jina v4)

### Core Modules
- `core/batch_embed_jina.py` - Jina v4 embedder with late chunking (32k context)
- `core/enhanced_docling_processor_v2.py` - PDF text/image extraction with validation

### Scripts
- `scripts/process_pdf_on_demand.py` - On-demand PDF processing
- `scripts/daily_arxiv_update.py` - Daily ArXiv updates
- `scripts/catchup_arxiv.py` - Catch up missing papers

## Python Commands

### 1. Process a Single PDF On-Demand
```bash
# Basic usage
ARANGO_PASSWORD=$ARANGO_PASSWORD CUDA_VISIBLE_DEVICES=1 python3 scripts/process_pdf_on_demand.py 2301.00234

# With custom database
ARANGO_PASSWORD=$ARANGO_PASSWORD CUDA_VISIBLE_DEVICES=1 python3 scripts/process_pdf_on_demand.py 2301.00234 --db-host 192.168.1.69 --db-name academy_store
```

### 2. Daily ArXiv Update
```bash
# Check for new papers (default: yesterday)
ARANGO_PASSWORD=$ARANGO_PASSWORD python3 scripts/daily_arxiv_update.py

# Specific date
ARANGO_PASSWORD=$ARANGO_PASSWORD python3 scripts/daily_arxiv_update.py --date 2025-01-15

# Date range
ARANGO_PASSWORD=$ARANGO_PASSWORD python3 scripts/daily_arxiv_update.py --start-date 2025-01-10 --end-date 2025-01-15
```

### 3. Catch Up Missing Papers
```bash
# Catch up all missing papers
ARANGO_PASSWORD=$ARANGO_PASSWORD python3 scripts/catchup_arxiv.py

# Limit number of papers
ARANGO_PASSWORD=$ARANGO_PASSWORD python3 scripts/catchup_arxiv.py --limit 100

# Custom source directory
ARANGO_PASSWORD=$ARANGO_PASSWORD python3 scripts/catchup_arxiv.py --source-dir /mnt/data/arxiv-kaggle
```

### 4. Test Jina v4 Embeddings
```bash
# Run tests
CUDA_VISIBLE_DEVICES=1 python3 tests/test_jina_v4.py

# Test Docling processor
python3 tests/test_docling.py /path/to/test.pdf
```

### 5. Monitor Pending PDFs
```bash
# Count pending PDFs
ARANGO_PASSWORD=$ARANGO_PASSWORD python3 utils/count_pending_pdfs.py
```

## Key Features with Jina v4

1. **32k Token Context** - Process entire documents without splitting
2. **Late Chunking** - Maintains context across chunks
3. **2048 Dimensions** - Double the embedding space of v3
4. **Code-Aware** - Better understanding of code structure

## Environment Setup

```bash
# Required environment variables
export ARANGO_PASSWORD="your_password"
export CUDA_VISIBLE_DEVICES=1  # For GPU processing

# Python dependencies
pip install torch transformers sentence-transformers
pip install docling docling-core
pip install python-arango
pip install tiktoken
```

## Database Configuration

Default settings:
- Host: `192.168.1.69`
- Port: `8529`
- Database: `academy_store`
- Collection: `base_arxiv`

## Processing Workflow

1. **Daily Updates**: Run `daily_arxiv_update.py` via cron
2. **On-Demand**: Use `process_pdf_on_demand.py` when full text needed
3. **Batch Catchup**: Use `catchup_arxiv.py` to process missing papers

## Performance Notes

With Jina v4:
- Small papers (<10 pages): ~30-60 seconds
- Medium papers (10-30 pages): ~1-2 minutes  
- Large papers (30+ pages): ~2-5 minutes

Late chunking provides better semantic coherence at the cost of slightly longer processing time.