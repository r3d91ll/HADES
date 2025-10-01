#!/usr/bin/env python3
"""
Paper-Code Pair Ingestion Workflow

WHAT: Orchestrates extraction, embedding, and storage of paper/code pairs
WHERE: core/workflows/embed_paper_code_pair.py - high-level workflow
WHO: CLI tool for ingesting research papers with associated code
TIME: Full pipeline <30s for typical paper/code pair

Phase 1 implementation: ingests a single paper/code pair into ArangoDB.
Coordinates extractors, embedders, and database operations to build the
knowledge graph with semantic embeddings.

Boundary Notes:
- Workflow boundaries define transaction scope
- Each phase measured for performance optimization
- Idempotent operations support re-runs without data corruption
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from core.paper_code_pipeline import (
    PairBundle,
    build_object_chunk_embeddings,
    build_windowed_embeddings,
    clear_paper_bundle,
    ensure_collections,
    extract_pdf,
    parse_code_directory,
    upsert_chunk_embeddings,
    upsert_paper_bundle,
)
from core.paper_code_pipeline.embedding_runner import JinaBundleEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest a paper/code pair for multimodal embeddings")
    parser.add_argument("input_dir", type=Path, help="Directory containing paper assets")
    parser.add_argument("--paper-id", dest="paper_id", help="Identifier for the paper (defaults to folder name)")
    parser.add_argument("--paper-title", dest="paper_title", help="Optional human-readable title")
    parser.add_argument("--paper-pdf", dest="paper_pdf", type=Path, help="Explicit path to the paper PDF")
    parser.add_argument("--code-dir", dest="code_dir", type=Path, help="Explicit path to the code directory")
    parser.add_argument("--adapter", default="code-v4", choices=["code-v4", "retrieval-v4", "text-matching-v4"], help="LoRA adapter to use for later embedding phases")
    parser.add_argument("--db", default="paper_code_pairs", help="Target Arango database")
    parser.add_argument("--reset", action="store_true", help="Remove existing documents for this paper before ingest")
    parser.add_argument("--dry-run", action="store_true", help="Extract and print metadata without writing to Arango")
    parser.add_argument("--embed", action="store_true", help="Run unified embedding after ingestion")
    parser.add_argument("--device", default=None, help="Device for embedding (e.g., cuda:0)")
    parser.add_argument("--dtype", default=None, help="Torch dtype (float16, bfloat16, float32)")
    parser.add_argument("--max-tokens", type=int, default=None, help="Optional safety cap for tokenised bundle")
    parser.add_argument("--local-files-only", action="store_true", help="Use cached models/tokenizer only")
    parser.add_argument("--window-tokens", type=int, default=4096, help="Token count for sliding window chunks")
    parser.add_argument("--window-stride", type=int, default=2048, help="Stride between sliding windows")
    return parser.parse_args()


def sanitize_paper_key(paper_id: str) -> str:
    return paper_id.replace("/", "_").replace(".", "_")


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    if not input_dir.exists():
        raise SystemExit(f"Input directory not found: {input_dir}")

    pdf_path = _resolve_pdf_path(input_dir, args.paper_pdf)
    code_dir = _resolve_code_dir(input_dir, args.code_dir)

    paper_id = args.paper_id or input_dir.name
    paper_key = sanitize_paper_key(paper_id)

    ensure_collections(args.db)

    paper = extract_pdf(pdf_path, paper_id=paper_id, title=args.paper_title)
    code_assets = parse_code_directory(code_dir)
    bundle = PairBundle.from_extracted(
        paper_result=paper,
        code_assets=code_assets,
        paper_key=paper_key,
        adapter=args.adapter,
    )

    if args.dry_run:
        payload = {
            "paper_key": paper_key,
            "word_count": bundle.paper.word_count,
            "code_files": len(bundle.code),
            "adapter": args.adapter,
        }
        print(json.dumps(payload, indent=2))
        return

    if args.reset:
        clear_paper_bundle(args.db, paper_key)

    upsert_paper_bundle(args.db, bundle)

    if args.embed:
        encoder = JinaBundleEncoder(
            device=args.device,
            dtype=args.dtype,
            local_files_only=args.local_files_only,
        )
        embedding_result = encoder.encode_bundle(
            bundle,
            max_tokens=args.max_tokens,
        )
        token_count = embedding_result.input_ids.shape[1]
        budget = args.max_tokens or encoder.max_tokens_auto
        print(f"Tokenised bundle length: {token_count} tokens (budget {budget or 'unbounded'})")
        object_embeddings = build_object_chunk_embeddings(bundle, embedding_result)
        window_embeddings = build_windowed_embeddings(
            bundle,
            embedding_result,
            window_tokens=args.window_tokens,
            stride_tokens=args.window_stride,
        )
        upsert_chunk_embeddings(args.db, [*object_embeddings, *window_embeddings])
        code_objects = sum(1 for emb in object_embeddings if emb.source_type == "code")
        paper_objects = len(object_embeddings) - code_objects
        print(
            f"✓ Embedded bundle for '{paper_id}' -> "
            f"{len(object_embeddings)} object chunks (paper={paper_objects}, code={code_objects}) "
            f"+ {len(window_embeddings)} windows"
        )
    else:
        print(f"✓ Ingested paper '{paper_id}' with {len(bundle.code)} code files into {args.db}")


def _resolve_pdf_path(input_dir: Path, override: Optional[Path]) -> Path:
    if override is not None:
        candidate = (input_dir / override) if not override.is_absolute() else override
        candidate = candidate.expanduser().resolve()
        if candidate.exists():
            return candidate
        raise SystemExit(f"Provided --paper-pdf does not exist: {candidate}")

    default = input_dir / "paper.pdf"
    if default.exists():
        return default

    pdf_candidates = sorted(p for p in input_dir.glob("*.pdf"))
    if len(pdf_candidates) == 1:
        return pdf_candidates[0]
    if not pdf_candidates:
        raise SystemExit(f"No PDF found in {input_dir}. Use --paper-pdf to specify the file.")
    raise SystemExit(
        "Multiple PDFs found. Use --paper-pdf to choose one:\n" +
        "\n".join(str(p) for p in pdf_candidates)
    )


def _resolve_code_dir(input_dir: Path, override: Optional[Path]) -> Path:
    if override is not None:
        candidate = (input_dir / override) if not override.is_absolute() else override
        candidate = candidate.expanduser().resolve()
        if candidate.exists() and candidate.is_dir():
            return candidate
        raise SystemExit(f"Provided --code-dir does not exist or is not a directory: {candidate}")

    default = input_dir / "code"
    if default.exists() and default.is_dir():
        return default

    subdirs = [p for p in input_dir.iterdir() if p.is_dir()]
    if len(subdirs) == 1:
        return subdirs[0]
    if not subdirs:
        raise SystemExit(f"No code directory found in {input_dir}. Use --code-dir to specify the folder.")
    raise SystemExit(
        "Multiple candidate directories found. Use --code-dir to choose one:\n" +
        "\n".join(str(p) for p in subdirs)
    )


if __name__ == "__main__":
    main()
