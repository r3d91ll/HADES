# Qwen3-30B-A3B-Thinking Deployment Notes

- **Model ID**: `Qwen/Qwen3-30B-A3B-Thinking-2507` (AWQ quantized weights).
- **Serving expectations**: load via `transformers` + `autoawq` with AWQ weights, `accelerate` for device placement. Target device default `cuda:0` (adjust via `QWEN_DEVICE`).
- **Config surface** (see `.env.example`):
  - `QWEN_MODEL_ID` — Hugging Face model identifier.
  - `QWEN_AWQ_WEIGHTS` — optional local path to AWQ weights; leave blank to download (requires HF auth).
  - `QWEN_DEVICE` — GPU device (e.g., `cuda:0`).
  - `QWEN_DEVICE_MAP` — override placement; use `balanced` or `auto` to span NVLinked GPUs.
  - `QWEN_MAX_MEMORY` — optional limits per device (`cuda:0=44GiB,cuda:1=44GiB` or `0=44GiB,1=44GiB`); keys are normalized for AWQ multi-GPU loading.
  - `QWEN_USE_FLASH_ATTN` — set to `1` to request FlashAttention-2 (`flash_attn` must be installed).
  - `QWEN_MAX_NEW_TOKENS` — generation cap.
- **Dependencies** (Poetry extra `qwen`): `transformers>=4.44.0`, `accelerate>=0.33.0`, `autoawq>=0.1.0`.
- **Runtime integration**: `core/runtime/memgpt/model_engine.py` handles lazy loading, so tests run without model packages. Production deployments must install the `qwen` extra and call `QwenModelEngine.load()` before generation.
- **Memory access**: Use existing Unix sockets (`/run/hades/readonly/arangod.sock`, `/run/hades/readwrite/arangod.sock`) through `ArangoMemoryClient` for recall/persistence.
