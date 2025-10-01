#!/usr/bin/env python3
"""Utility to load the Qwen model engine and verify context window."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.runtime.memgpt import (
    MemGPTOrchestrator,
    OrchestratorConfig,
    QwenModelConfig,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Load Qwen model and report context window")
    parser.add_argument("--dry-run", action="store_true", help="Only print planned settings")
    args = parser.parse_args()

    defaults = QwenModelConfig()
    model_id = os.getenv("QWEN_MODEL_ID", defaults.model_id)
    device = os.getenv("QWEN_DEVICE", defaults.device)
    device_map_env = os.getenv("QWEN_DEVICE_MAP", "") or None
    awq = os.getenv("QWEN_AWQ_WEIGHTS", defaults.awq_weights or "") or None
    max_ctx = int(os.getenv("QWEN_MAX_CONTEXT", str(defaults.max_context_length)))
    max_memory_env = os.getenv("QWEN_MAX_MEMORY", "") or None
    use_flash_attn_env = os.getenv("QWEN_USE_FLASH_ATTN", "")

    max_memory = None
    if max_memory_env:
        entries = [chunk.strip() for chunk in max_memory_env.split(",") if chunk.strip()]
        if entries:
            max_memory = {}
            for entry in entries:
                try:
                    key, value = entry.split("=", 1)
                except ValueError as exc:  # pragma: no cover - defensive parsing
                    raise ValueError(
                        "Invalid QWEN_MAX_MEMORY entry. Expected format 'cuda:0=40GiB'."
                    ) from exc
                max_memory[key.strip()] = value.strip()

    if use_flash_attn_env:
        lowered = use_flash_attn_env.strip().lower()
        use_flash_attn = lowered in {"1", "true", "yes", "on"}
    else:
        use_flash_attn = defaults.use_flash_attn

    config = QwenModelConfig(
        model_id=model_id,
        device=device,
        device_map=device_map_env,
        awq_weights=awq,
        max_memory=max_memory,
        max_context_length=max_ctx,
        use_flash_attn=use_flash_attn,
    )
    orchestrator = MemGPTOrchestrator(
        config=OrchestratorConfig(auto_load_model=False),
        model_config=config,
    )
    engine = orchestrator.model_engine

    if args.dry_run:
        plan = f"[plan] model={config.model_id} device={config.device}"
        if config.device_map:
            plan += f" device_map={config.device_map}"
        if config.max_memory:
            mem = ",".join(f"{k}={v}" for k, v in config.max_memory.items())
            plan += f" max_memory={mem}"
        if config.use_flash_attn:
            plan += " flash_attn=on"
        print(f"{plan} ctx={config.max_context_length}")
        if not engine.dependencies_available():
            print("[warn] Required packages (transformers/accelerate/autoawq) not installed.")
        return 0

    try:
        engine.load()
    except Exception as exc:  # pragma: no cover - runtime check
        print(f"[error] Failed to load model: {exc}")
        return 1

    tokenizer = engine._tokenizer  # noqa: SLF001
    model = engine._model  # noqa: SLF001
    ctx = getattr(tokenizer, "model_max_length", None)
    model_ctx = getattr(getattr(model, "config", None), "max_position_embeddings", None)
    print(f"Loaded model: {config.model_id}")
    print(f"Tokenizer max length: {ctx}")
    print(f"Model max position embeddings: {model_ctx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
