# GitHub Processor

Theory-practice bridge discovery through code repository analysis.

## Status

✅ **Code Complete** - Implementation and tests ready
⏳ **Waiting** - ArXiv rebuild must complete before running (12-15 hours remaining)
🚫 **Not Run** - No embeddings generated yet (GPU occupied)

## Architecture

### Core Components

1. **github_processor.py** - Main implementation
   - Repository cloning with full history
   - Jina v4 embeddings (2048-dimensional)
   - Atomic database transactions
   - Rate limiting and error recovery

2. **test_github_processor.py** - Unit tests
   - 11 tests, all passing
   - Full mock coverage (no GPU/DB required)
   - Validates key generation, cloning, embedding logic

3. **run_mvp.py** - CLI interface
   - Prerequisite checking
   - Clone-only mode (no GPU required)
   - Embed-only mode (requires GPU)
   - Full processing mode

## Database Schema

### base_github_repos
- Repository metadata
- Clone information
- Embedding status tracking

### base_github_embeddings  
- File embeddings
- MD5 keys (avoid path issues)
- Chunk management for large files

## Word2Vec MVP

Target repositories:
- `dav/word2vec` - Original C implementation (2.3 MB)
- `tmikolov/word2vec` - Google's code (~5 MB)
- `danielfrg/word2vec` - Python wrapper (~10 MB)
- `RaRe-Technologies/gensim` - Production library (~450 MB)

Total storage: ~482 MB
Processing time: ~15 minutes (when GPU available)

## Usage

### Check Prerequisites
```bash
python3 run_mvp.py --check
```

### Clone Repositories (No GPU Required)
```bash
export ARANGO_PASSWORD='your_password'
python3 run_mvp.py --clone-only
```

### Generate Embeddings (GPU Required)
```bash
export ARANGO_PASSWORD='your_password'
export CUDA_VISIBLE_DEVICES=0  # Or 1
python3 run_mvp.py --embed-only
```

### Full Processing (After ArXiv Complete)
```bash
export ARANGO_PASSWORD='your_password'
python3 run_mvp.py --full
```

## Current Limitations

1. **GPUs Busy** - ArXiv rebuild using both A6000s
2. **No Embeddings** - Cannot generate until GPU available
3. **No Integration Tests** - Require actual database/GPU

## Next Steps

After ArXiv rebuild completes:

1. Run `--clone-only` to get repositories
2. Run `--embed-only` to generate embeddings
3. Verify with integration tests
4. Process Word2Vec ecosystem
5. Measure conveyance gradients

## Theory Connection

This processor handles the PRACTICE dimension of theory-practice bridges:
- **Maximum CONVEYANCE** - Code is directly executable
- **WHERE** = Repository location, file paths
- **WHAT** = Semantic content via embeddings
- **Zero Propagation** - Missing files → no information

## Testing

Run unit tests (no GPU/DB required):
```bash
python3 test_github_processor.py
```

All 11 tests passing:
- Database initialization
- Repository cloning
- File listing
- Key generation
- Language detection
- Mock embedding

## Configuration

Environment variables:
- `ARANGO_HOST` - Database host (default: 192.168.1.69:8529)
- `ARANGO_DATABASE` - Database name (default: academy_store)
- `ARANGO_USERNAME` - Username (default: root)
- `ARANGO_PASSWORD` - **Required**
- `GITHUB_CLONE_PATH` - Repository storage (default: /data/repos)

## Implementation Notes

- **Jina v4 API** - Using `encode()` not `encode_text()`
- **fp16 Precision** - Half memory usage
- **MD5 Keys** - Avoid "/" in ArangoDB keys
- **Transactions** - Atomic multi-collection updates
- **Rate Limiting** - 1 second between clones, 0.5 between files

---

*Ready for deployment once GPUs are available*