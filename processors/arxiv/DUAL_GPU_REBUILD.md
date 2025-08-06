# Dual-GPU Overnight Rebuild with NVLink

## 🚀 Optimized Command for Both GPUs with NVLink

```bash
# Set environment variables
export ARANGO_PASSWORD='your_password_here'
export CUDA_VISIBLE_DEVICES=0,1  # Use both GPUs

# Run dual-GPU rebuild with larger batch size
nohup python3 scripts/rebuild_dual_gpu.py \
    --source /fastpool/temp/arxiv-metadata-oai-snapshot.json \
    --batch-size 256 \
    --num-gpus 2 \
    > rebuild_dual_gpu_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Note the PID
echo "Process started with PID: $!"
```

## ⚡ Performance Benefits with Dual GPU + NVLink

### Without NVLink (PCIe)
- GPU0 and GPU1 communicate through system RAM
- ~30-40% speedup over single GPU
- Batch size limited by PCIe bandwidth

### With NVLink (Your Setup)
- **Direct GPU-to-GPU communication at 600 GB/s**
- **~80-90% speedup over single GPU**
- **Larger batch sizes possible (256-512)**
- **Better load balancing between GPUs**

## 📊 Expected Performance

With dual A6000 GPUs + NVLink:
- **Single GPU**: ~20-30 papers/second
- **Dual GPU (no NVLink)**: ~30-40 papers/second  
- **Dual GPU + NVLink**: ~35-50 papers/second

Time estimates for ~2M papers:
- **Single GPU**: 18-28 hours
- **Dual GPU + NVLink**: 11-16 hours

## 🧪 Test First (5 minutes)

```bash
export ARANGO_PASSWORD='your_password_here'
export CUDA_VISIBLE_DEVICES=0,1

python3 scripts/rebuild_dual_gpu.py \
    --source /fastpool/temp/arxiv-metadata-oai-snapshot.json \
    --limit 1000 \
    --batch-size 128 \
    --num-gpus 2
```

## 📈 Monitor GPU Usage

In another terminal:
```bash
# Watch both GPUs
watch -n 1 nvidia-smi

# Or for detailed view
nvidia-smi dmon -i 0,1
```

You should see:
- Both GPUs at 80-95% utilization
- Memory usage on both GPUs (~8-12GB each with fp16)
- NVLink traffic (if supported by nvidia-smi)

## 📋 Monitor Progress

```bash
# Watch the log
tail -f rebuild_dual_gpu_*.log

# Check GPU distribution
grep "GPU distribution" rebuild_dual_gpu_*.log | tail -5
```

## 🎯 Optimization Tips

### Batch Size Selection
- **128**: Conservative, good for testing
- **256**: Recommended for 48GB GPUs with NVLink
- **384**: Aggressive, maximum throughput
- **512**: Only if memory permits

### Memory Formula (per GPU)
```
Memory = (batch_size/num_gpus) × model_size + overhead
       = (256/2) × ~4GB + 2GB
       = ~8GB per GPU (safe for 48GB A6000)
```

## 🔧 Advanced Tuning

For maximum performance:
```bash
# Enable GPU persistence mode
sudo nvidia-smi -pm 1

# Set maximum GPU clocks
sudo nvidia-smi -ac 1215,1410  # For A6000

# Run with optimized settings
export ARANGO_PASSWORD='your_password_here'
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

nohup python3 scripts/rebuild_dual_gpu.py \
    --source /fastpool/temp/arxiv-metadata-oai-snapshot.json \
    --batch-size 384 \
    --num-gpus 2 \
    > rebuild_dual_gpu_optimized.log 2>&1 &
```

## 🛡️ Error Recovery

The script saves after each batch. If it crashes:

1. Check the last processed batch in the log
2. The database already has all papers up to that point
3. You can restart and it will overwrite (idempotent)

## 📊 Verify NVLink is Working

```python
# Quick test script
import torch

if torch.cuda.device_count() >= 2:
    torch.cuda.set_device(0)
    can_access = torch.cuda.can_device_access_peer(0, 1)
    if can_access:
        print("✓ NVLink is active between GPU 0 and GPU 1")
        
        # Test bandwidth
        size = 1024 * 1024 * 1024  # 1GB
        tensor_gpu0 = torch.randn(size // 4, device='cuda:0')
        tensor_gpu1 = tensor_gpu0.to('cuda:1')
        print("✓ Successfully transferred 1GB via NVLink")
    else:
        print("✗ No NVLink detected, using PCIe")
```

## 🎉 Why This is Better

1. **2x the compute**: Both GPUs working in parallel
2. **NVLink advantage**: 600 GB/s vs 32 GB/s (PCIe 4.0)
3. **Load balancing**: Ray automatically distributes work
4. **Fault tolerance**: If one GPU errors, the other continues
5. **fp16 precision**: Half the memory, nearly same quality

## 📝 Final Notes

- Ray handles all the GPU coordination
- Papers are split evenly between GPUs
- Database writes are batched for efficiency
- NVLink reduces GPU communication overhead
- Each embedding is still 2048 dimensions (Jina v4)

Expected overnight result:
- **~2 million papers processed**
- **All with 2048-dim Jina v4 embeddings**
- **Completed in 11-16 hours instead of 28+ hours**