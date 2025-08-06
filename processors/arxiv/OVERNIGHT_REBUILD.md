# Overnight Database Rebuild with Jina v4

## 🚀 Command to Run Overnight

```bash
# Set environment variables
export ARANGO_PASSWORD='your_password_here'
export CUDA_VISIBLE_DEVICES=1

# Run rebuild from source with larger batch size for efficiency
nohup python3 scripts/rebuild_from_source.py \
    --source /fastpool/temp/arxiv-metadata-oai-snapshot.json \
    --batch-size 64 \
    > rebuild_jina_v4_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Note the PID
echo "Process started with PID: $!"
```

## 📊 What This Does

1. **Reads from source**: `/fastpool/temp/arxiv-metadata-oai-snapshot.json`
2. **Processes each paper**: Extracts title, abstract, metadata
3. **Generates embeddings**: Jina v4 with 2048 dimensions (double v3's 1024)
4. **Stores in ArangoDB**: `academy_store` database, `base_arxiv` collection

## ⚠️ Test First (Recommended)

Before running overnight, test with 1000 papers:

```bash
export ARANGO_PASSWORD='your_password_here'
export CUDA_VISIBLE_DEVICES=1

python3 scripts/rebuild_from_source.py \
    --source /fastpool/temp/arxiv-metadata-oai-snapshot.json \
    --limit 1000 \
    --batch-size 32
```

This should take ~2-3 minutes and will verify everything works.

## 📈 Monitor Progress

```bash
# Watch the log file
tail -f rebuild_jina_v4_*.log

# Check last 100 lines
tail -100 rebuild_jina_v4_*.log

# Count processed papers
grep "Progress:" rebuild_jina_v4_*.log | tail -1
```

## ⏱️ Time Estimates

Based on GPU and batch size:
- **RTX 3090/4090 (24GB)**: ~10-15 papers/second
- **A6000 (48GB)**: ~20-30 papers/second

For ~2 million papers:
- **RTX 3090/4090**: 37-56 hours
- **A6000**: 18-28 hours

## 💾 Storage Requirements

- **Old (Jina v3)**: 1024 dimensions × 4 bytes × 2M papers ≈ 8 GB
- **New (Jina v4)**: 2048 dimensions × 4 bytes × 2M papers ≈ 16 GB
- **Total increase**: ~8 GB additional storage needed

## 🔄 Clean Start Option

If you want to completely rebuild from scratch:

```bash
python3 scripts/rebuild_from_source.py \
    --source /fastpool/temp/arxiv-metadata-oai-snapshot.json \
    --batch-size 64 \
    --clean-start
```

⚠️ This will DELETE all existing data and start fresh!

## 🛑 Stop the Process

If you need to stop:

```bash
# Find the process
ps aux | grep rebuild_from_source

# Kill it
kill <PID>
```

The script saves after each batch, so you can resume later.

## ✅ Verify Success

After completion, verify with:

```bash
python3 scripts/check_and_rebuild.py
```

This will show:
- Total papers processed
- Papers with 2048-dim embeddings
- Any remaining papers without embeddings

## 📝 Key Changes from v3 to v4

| Feature | Jina v3 | Jina v4 |
|---------|---------|---------|
| Dimensions | 1024 | 2048 |
| Context Length | 8,192 tokens | 32,768 tokens |
| Model | BERT-based | Qwen2.5-VL-3B |
| Precision | fp32 | fp16 |
| Late Chunking | Limited | Full support |

The new embeddings will provide:
- Better semantic understanding
- Improved search accuracy
- Support for longer documents
- Better code understanding (for future GitHub integration)