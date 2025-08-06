# Jina Late Chunking Embeddings Guide

## Overview

This guide covers the production-ready late chunking embedding system for arXiv papers using Jina v3 embeddings with dual RTX A6000 GPUs.

## Quick Start

### Basic Usage
```bash
# Run with default settings
./scripts/run_late_chunking_embeddings_production.sh

# Dry run to see what would be executed
./scripts/run_late_chunking_embeddings_production.sh --dry-run

# Process limited number of documents
./scripts/run_late_chunking_embeddings_production.sh --max-docs 1000

# Verbose mode with GPU monitoring
./scripts/run_late_chunking_embeddings_production.sh --verbose
```

### Health Check
```bash
# Check system health
./scripts/check_embedding_health.sh
```

## Architecture

### Late Chunking vs Traditional Chunking

**Traditional Approach:**
1. Split document into fixed-size chunks
2. Embed each chunk independently
3. Each chunk has no context of others

**Late Chunking Approach:**
1. Embed entire document (up to 32k tokens)
2. Split token embeddings into chunks
3. Apply mean pooling to create chunk embeddings
4. Each chunk retains full document context

### Dual GPU Configuration

- **Model Distribution**: Automatic layer splitting across GPUs
- **Context Window**: 32,768 tokens (4x standard)
- **NVLink**: High-bandwidth GPU communication
- **Memory**: ~20GB per GPU for model + activations

## Configuration

### Environment Variables
```bash
# Required
export ARANGO_PASSWORD="your_password"

# Optional
export MARKDOWN_DIR="/path/to/markdown/files"
export CHECKPOINT_DIR="/path/to/checkpoints"
```

### Script Parameters
- `--dry-run`: Show what would be executed without running
- `--verbose`: Enable detailed logging and GPU monitoring
- `--markdown-dir PATH`: Override markdown directory
- `--max-docs N`: Limit number of documents to process
- `--context-window N`: Context window size (default: 32768)
- `--chunk-size N`: Chunk size in tokens (default: 512)

## Production Deployment

### As a Systemd Service

1. Copy service file:
```bash
sudo cp scripts/jina-embeddings.service /etc/systemd/system/
```

2. Edit service file and set ARANGO_PASSWORD:
```bash
sudo systemctl edit jina-embeddings.service
```

3. Enable and start service:
```bash
sudo systemctl enable jina-embeddings.service
sudo systemctl start jina-embeddings.service
```

4. Check status:
```bash
sudo systemctl status jina-embeddings.service
journalctl -u jina-embeddings.service -f
```

### Monitoring

The production script includes:
- Automatic GPU monitoring (30s intervals)
- Comprehensive logging with timestamps
- Error tracking and recovery
- Progress checkpointing

Monitor files:
- Logs: `logs/jina_embedding_YYYYMMDD_HHMMSS.log`
- GPU metrics: `logs/gpu_metrics_YYYYMMDD_HHMMSS.csv`
- Checkpoint: `checkpoints/embedding_checkpoint_late_chunking.json`

### Resource Requirements

**Minimum:**
- 2x GPUs with 24GB VRAM each
- 64GB system RAM
- NVLink connection recommended

**Recommended:**
- 2x RTX A6000 (48GB) or similar
- 128GB system RAM
- NVMe storage for data

### Performance Tuning

1. **Memory Optimization:**
```bash
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:1024,expandable_segments:True"
```

2. **CPU Thread Limiting:**
```bash
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
```

3. **NVLink Optimization:**
```bash
export NCCL_P2P_LEVEL=NVL
```

## Troubleshooting

### Common Issues

1. **Out of Memory:**
   - Reduce context window
   - Process fewer documents at once
   - Check for memory leaks with `nvidia-smi`

2. **NVLink Not Active:**
   - Verify with `nvidia-smi nvlink -s`
   - Check BIOS settings
   - Ensure GPUs are in correct slots

3. **Slow Performance:**
   - Check GPU utilization
   - Verify NVLink is active
   - Monitor CPU bottlenecks

4. **Checkpoint Corruption:**
   - Backups are created automatically
   - Restore from `embedding_checkpoint_late_chunking_backup_*.json`

### Debug Commands

```bash
# Check GPU status
nvidia-smi

# Monitor GPU in real-time
watch -n 1 nvidia-smi

# Check NVLink bandwidth
nvidia-smi nvlink -g 0

# View recent errors
grep -i error logs/jina_embedding_*.log | tail -20

# Parse checkpoint
jq '.processed_ids | length' checkpoints/embedding_checkpoint_late_chunking.json
```

## Database Schema

Documents are updated with:
```json
{
  "embeddings": {
    "model": "jinaai/jina-embeddings-v3",
    "context_window": 32768,
    "chunk_size_tokens": 512,
    "total_tokens": 15234,
    "late_chunking": true,
    "chunks": [
      {
        "chunk_index": 0,
        "chunk_text": "...",
        "start_token": 0,
        "end_token": 512,
        "embedding": [...],
        "metadata": {
          "context_aware": true,
          "num_tokens": 512,
          "chunk_hash": "..."
        }
      }
    ],
    "num_chunks": 30,
    "processing_time": 12.5,
    "embedded_date": "2024-01-20T15:30:00Z"
  },
  "pdf_status": "embedded"
}
```

## Best Practices

1. **Always run health check before starting**
2. **Use dry-run mode for testing**
3. **Monitor GPU memory during initial runs**
4. **Keep checkpoint backups**
5. **Review logs for errors regularly**
6. **Process in batches for very large datasets**

## Recovery Procedures

### From Checkpoint
```bash
# Resume from checkpoint automatically
./scripts/run_late_chunking_embeddings_production.sh
```

### From Backup
```bash
# List backups
ls -la checkpoints/embedding_checkpoint_late_chunking_backup_*.json

# Restore specific backup
cp checkpoints/embedding_checkpoint_late_chunking_backup_20240120_143000.json \
   checkpoints/embedding_checkpoint_late_chunking.json
```

### Clean Start
```bash
# Move old checkpoint
mv checkpoints/embedding_checkpoint_late_chunking.json \
   checkpoints/embedding_checkpoint_late_chunking.old

# Start fresh
./scripts/run_late_chunking_embeddings_production.sh
```