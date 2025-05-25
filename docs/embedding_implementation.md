# Embedding Implementation Plan

> Revision 0.1 – *2025-04-30*

This document outlines the **end-to-end strategy** for adding semantic
(chunk-level) and graph-level embeddings to the PathRAG ingestion pipeline
while optimally utilising two × A6000 GPUs.

---

## 1  Scope & Objectives

1. Generate **semantic vectors** for every chunk produced by the
   *chunking* stage:
   - **Code chunks** → microsoft/codebert-base (vLLM-served) – 768 dims.
   - **Text chunks** → mirth/chonky_modernbert_large_1 (vLLM-served Chonky-recommended) – 768 dims.
2. Produce **graph embeddings** with ISNE + GraphSAGE using the semantic
   vectors as initial node features.
3. Persist vectors to Arango collections and expose them for search.
4. Provide CLI/orchestrator flags to enable/disable (a) semantic, (b)
   graph embeddings.
5. Ensure GPU memory is managed cleanly and models are hot-swapped only
   when needed.

---

## 2  Hardware Assumptions

| Device | Memory | Role |
|--------|--------|------|
| **GPU-0** | 48 GB | vLLM server hosting CodeBERT & serving requests |
| **GPU-1** | 48 GB | Transformers inference for ModernBERT + GraphSAGE |

> If RAM permits, GraphSAGE can train on CPU and **infer** on GPU-1.

---

## 3  Models & Endpoints

| Modality | Model | Engine | Endpoint / Loader |
|----------|-------|--------|-------------------|
| Code | `microsoft/codebert-base` | **vLLM** | `http://127.0.0.1:8000` |
| Text | `mirth/chonky_modernbert_large_1` | **HF Transformers** | loaded in-process |
| Graph | ISNE (Shallow Node Embedding) | custom script | offline batch |
| Graph | GraphSAGE | PyG / DGL | offline (training), online (inference) |

---

## 4  Runtime Architecture

```ascii
                +------------------+           +---------------------+
 Ingest Chunks  |  Semantic Queue  | ——→——→——→ |  vLLM  (GPU-0)      |
 (code/text)    +------------------+           +---------------------+
                        |                        ^  REST (OpenAI-like)
                        |                        |
                        |   +-----------------+  |
                        +→→ |  HF Runner      |——+
                             |  ModernBERT     | (GPU-1)
                             +-----------------+

After all chunk vectors are ready:
    +--------------+        +--------------------+
    |  ISNE Prep   | ——→→→→ |  ISNE Embedding    |
    +--------------+        +--------------------+
                                        |
                                        v
                             +--------------------+
                             |  GraphSAGE Trainer |
                             +--------------------+
```

Communication:

- **Semantic Queue** = Python async worker pool; batches requests to
  vLLM or local model based on `chunk.type`.
- GPU assignment enforced via `CUDA_VISIBLE_DEVICES` env.

---

## 5  Public Python API

```python
from src.ingest.embedding import (
    embed_documents,           # semantic level
    embed_graph_with_isne,     # ISNE + GraphSAGE
)

vectors = embed_documents(chunks)                  # returns List[EmbeddedChunk]
embed_graph_with_isne(entities, relationships)     # persists graph vectors
```

### EmbeddedChunk schema

```json
{
  "id": "chunk:abcd1234",
  "vector": [0.1, 0.2, ...],    # 768 dims
  "model": "codebert-base"      # or "modernbert-large-1"
}
```

---

## 6  Configuration Files

`src/config/embedding_config.yaml`

```yaml
code:
  engine: vllm
  endpoint: http://127.0.0.1:8000
  model_id: microsoft/codebert-base
  batch_size: 32
  max_concurrent: 8

text:
  engine: transformers
  model_id: mirth/chonky_modernbert_large_1
  device: cuda:1
  batch_size: 16

isne:
  walk_length: 80
  num_walks: 10
  dimensions: 128

graphsage:
  epochs: 10
  hidden_dims: 256
  device: cuda:1
```

---

## 7  Orchestrator Flags

| Flag | Description |
|------|-------------|
| `--embed` / `--no-embed` | toggle semantic embedding stage |
| `--graph-embed`          | run ISNE + GraphSAGE after semantic stage |
| `--fresh-models`         | force reload models (clears caches) |

---

## 8  Implementation Milestones

| # | Milestone | Deliverables |
|---|-----------|-------------|
| 1 | **Scaffold**           | folder structure, config loader, dataclasses |
| 2 | **vLLM client wrapper**| start/attach to vLLM, batcher, retries |
| 3 | **ModernBERT runner**  | half-precision inference + batching |
| 4 | **embed_documents()**  | unified batching, async gather, error handling |
| 5 | **ISNE wrapper**       | convert graph JSON, run script, load vectors |
| 6 | **GraphSAGE trainer**  | minimal training loop, save `.pt` weights |
| 7 | **Arango integration** | write vectors to collections with indices |
| 8 | **CLI integration**    | new flags + progress bars |
| 9 | **Tests & Benchmarks** | unit tests; throughput benchmark on GPUs |
| 10| **Docs & Examples**    | update README; provide example notebook |

---

## 9  Model-Lifecycle Management

- **Lazy loading**: each embedder maintains a *singleton* model handle.
- **Reference counting**: orchestrator increments/decrements when tasks
  start/finish; last consumer triggers `model.cpu(); torch.cuda.empty_cache()`.
- **Process isolation**: vLLM runs in its own process on GPU-0; ModernBERT
  runs in the main orchestrator process pinned to GPU-1.
- **Health checks**: every 60 s ping vLLM `/v1/models` endpoint; restart if
  unreachable.
- **Checkpointing**: GraphSAGE trainer saves checkpoints every epoch.

---

## 10  Risk & Mitigation

| Risk | Mitigation |
|------|------------|
| GPU OOM from large batches | dynamic batch sizer + grad-accum for GraphSAGE |
| vLLM process crash | supervisor restart (systemd / pm2) |
| Model version drift | pin model hashes in `embedding_config.yaml` |
| Ingest backlog | adaptive rate-limit on semantic queue |

---

## 11  Timeline (est.)

| Week | Tasks |
|------|-------|
| 1 | Milestones 1-3 |
| 2 | Milestones 4-5 |
| 3 | Milestones 6-7 + initial benchmarks |
| 4 | Milestones 8-10 & documentation polish |

---

## 12  Appendix

- **vLLM** docs: <https://github.com/vllm-project/vllm>
- **Chonky** paper: <https://arxiv.org/abs/2310.01627>
- **ISNE**: <https://link.springer.com/article/10.1007/s40747-024-01545-6>
- **GraphSAGE**: <https://arxiv.org/pdf/1706.02216.pdf>
