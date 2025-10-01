#!/usr/bin/env python3
"""Probe utility to exercise long-context prompts against the Qwen orchestrator."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.runtime.memgpt import (  # noqa: E402  (import after sys.path tweak)
    ConsoleTelemetryClient,
    MemGPTOrchestrator,
    OrchestratorConfig,
    QwenModelConfig,
)


def build_payload(chars: int) -> str:
    seed = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
    repeats = max(1, (chars // len(seed)) + 1)
    payload = (seed * repeats)[:chars]
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe long-context handling for Qwen")
    parser.add_argument("--chars", type=int, default=262_144, help="Approximate prompt size in characters")
    parser.add_argument("--max-new-tokens", type=int, default=64, dest="max_new_tokens")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without loading the model")
    parser.add_argument("--system", default="", help="Optional system directive to prepend")
    parser.add_argument("--stop", action="append", default=[], help="Stop sequence (repeatable)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--budget", type=int, default=-1, help="Token budget; -1 disables budget (default)")
    parser.add_argument("--telemetry", action="store_true", help="Print telemetry spans to stdout")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    defaults = QwenModelConfig()
    model_id = os.getenv("QWEN_MODEL_ID", defaults.model_id)
    device = os.getenv("QWEN_DEVICE", defaults.device)
    device_map = os.getenv("QWEN_DEVICE_MAP", "") or None
    awq = os.getenv("QWEN_AWQ_WEIGHTS", defaults.awq_weights or "") or None
    max_ctx = int(os.getenv("QWEN_MAX_CONTEXT", str(defaults.max_context_length)))
    max_memory_env = os.getenv("QWEN_MAX_MEMORY", "") or None

    max_memory = None
    if max_memory_env:
        entries = [chunk.strip() for chunk in max_memory_env.split(",") if chunk.strip()]
        if entries:
            max_memory = {}
            for entry in entries:
                key, value = entry.split("=", 1)
                max_memory[key.strip()] = value.strip()

    config = QwenModelConfig(
        model_id=model_id,
        device=device,
        device_map=device_map,
        awq_weights=awq,
        max_memory=max_memory,
        max_context_length=max_ctx,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    telemetry = ConsoleTelemetryClient() if args.telemetry else None
    token_budget = None if args.budget < 0 else args.budget
    orchestrator = MemGPTOrchestrator(
        config=OrchestratorConfig(auto_load_model=False, token_budget=token_budget),
        model_config=config,
        telemetry=telemetry,
    )

    if args.system:
        orchestrator.ingest_turn(role="system", content=args.system)

    payload = build_payload(args.chars)
    orchestrator.ingest_turn(role="user", content=payload, tokens=len(payload.split()))

    envelope = orchestrator.prepare_envelope(stop=args.stop, overrides={})

    if args.dry_run:
        print("[plan] Ready to load model with:")
        print(f"  model_id={config.model_id}")
        print(f"  device={config.device} device_map={config.device_map or 'default'}")
        if config.max_memory:
            mem = ",".join(f"{k}={v}" for k, v in config.max_memory.items())
            print(f"  max_memory={mem}")
        print(f"  prompt_chars={len(payload)} stop_count={len(args.stop)} token_budget={token_budget}")
        print(f"  overrides={{'max_new_tokens': {args.max_new_tokens}, 'temperature': {args.temperature}, 'top_p': {args.top_p}}}")
        return 0

    engine = orchestrator.model_engine
    if not engine.is_loaded():
        try:
            engine.load()
        except Exception as exc:  # pragma: no cover - runtime path
            print(f"[error] Failed to load model: {exc}")
            return 1

    overrides = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }

    start = time.perf_counter()
    reply_turn = orchestrator.generate_turn(envelope=envelope, **overrides)
    duration = time.perf_counter() - start

    print("[result] Generation completed")
    print(f"  duration_ms={duration * 1000:.2f}")
    print(f"  response_chars={len(reply_turn.content)}")
    preview = reply_turn.content[:256]
    print(f"  preview={preview!r}{'...' if len(reply_turn.content) > 256 else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
