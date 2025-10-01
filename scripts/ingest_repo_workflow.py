#!/usr/bin/env python3
"""
Module: scripts/ingest_repo_workflow.py
Summary: Unified repo ingestion: normalize → Jina v4 embed → late‑chunk → write.
Owners: @todd, @hades-runtime
Last-Updated: 2025-09-30
Inputs: repo root, .hades.ignore patterns, RO/RW sockets via ARANGO_*, device/batch
Outputs: repo_docs/doc_chunks with embeddings and metadata; ArangoSearch view links
Data-Contracts: repo_docs {_key,path,dir,name,ext,kind,lang,content_hash,metadata,text,embedding};
                doc_chunks {_key,doc_key,path,chunk_index,start_token,end_token,total_chunks,text,embedding}
Related: Docs/Program_Plan.md, core/extractors/*, core/embedders/*, core/database/arango/*
Stability: beta; Security: DB-scoped endpoints only; honors .hades.ignore
Boundary: C_ext assumes local UDS; P_ij requires analyzer/view present

Notes
- Adds rich metadata (Tree‑sitter for code, Docling for PDFs) and directory path info.
- Performs late chunking after document embeddings (Jina v4).
- Idempotent writes (onDuplicate=update); batch queue semantics.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import os
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.database.arango.admin import (
    ensure_arangosearch_view,
    ensure_persistent_index,
    ensure_text_analyzer,
    ensure_vector_index,
)
from core.database.arango.memory_client import (
    ArangoMemoryClient,
    CollectionDefinition,
    resolve_memory_config,
)
from core.embedders import EmbeddingConfig, create_embedder
from core.extractors.extractors_code import CodeExtractor


def sha1_hex(data: bytes) -> str:
    h = hashlib.sha1()
    h.update(data)
    return h.hexdigest()


def sanitize_key(relpath: str, max_len: int = 254) -> str:
    s = relpath.replace("/", "_").replace(" ", "_")
    s = "".join(ch if ch.isalnum() or ch in "-_.:@()+" else "_" for ch in s)
    s = s.strip("._-") or "item"
    if len(s.encode("utf-8")) <= max_len:
        return s
    h = sha1_hex(relpath.encode("utf-8"))[:10]
    base = s[: max_len - 1 - len(h)]
    return f"{base}_{h}"


def _load_ignore(root: Path) -> List[str]:
    """Load .hades.ignore patterns (gitignore-like, minimal).

    - Lines starting with # are comments.
    - Blank lines ignored.
    - Trailing slash means directory prefix.
    - Patterns are matched against posix relative paths.
    """
    path = root / ".hades.ignore"
    patterns: List[str] = []
    if not path.exists():
        return patterns
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        patterns.append(s)
    return patterns


def _is_ignored(rel_posix: str, parts: Tuple[str, ...], patterns: List[str]) -> bool:
    for pat in patterns:
        # Directory prefix pattern
        if pat.endswith('/'):
            pref = pat.rstrip('/')
            if rel_posix == pref or rel_posix.startswith(pref + '/'):
                return True
        else:
            # Anchored at root
            p = pat[1:] if pat.startswith('/') else pat
            if fnmatch.fnmatch(rel_posix, p):
                return True
    return False


def iter_files(root: Path, globs: List[str]) -> Iterator[Path]:
    seen: set[Path] = set()
    ignore = _load_ignore(root)
    for pat in globs:
        for p in root.glob(pat):
            if not p.is_file() or p in seen:
                continue
            parts = p.relative_to(root).parts
            # Skip hidden and archived content
            if any(part.startswith(".") for part in parts):
                continue
            if "acheron" in parts:
                continue
            if any(part in {".git", "node_modules", "__pycache__", ".venv"} for part in parts):
                continue
            rel_posix = p.relative_to(root).as_posix()
            if ignore and _is_ignored(rel_posix, parts, ignore):
                continue
            yield p
            seen.add(p)


@dataclass
class NormalizedDoc:
    key: str
    path: str
    dir: str
    name: str
    ext: str
    kind: str  # 'code' | 'text'
    lang: Optional[str]
    text: str
    metadata: Dict
    content_hash: str


def normalize_file(p: Path, root: Path, code_extractor: CodeExtractor, text_exts: set[str]) -> Optional[NormalizedDoc]:
    rel = p.relative_to(root).as_posix()
    ext = p.suffix.lower()
    name = p.name
    dirpart = p.parent.relative_to(root).as_posix() if p.parent != root else ""
    key = sanitize_key(rel)

    if ext in code_extractor.supported_formats:
        res = code_extractor.extract(p)
        text = res.text or ""
        meta = dict(res.metadata)
        meta.update({
            "extractor": meta.get("extractor", "code_extractor"),
            "source_path": rel,
            "source_dir": dirpart,
        })
        lang = meta.get("language")
        content_hash = sha1_hex(text.encode("utf-8", errors="ignore"))
        return NormalizedDoc(key, rel, dirpart, name, ext, "code", lang, text, meta, content_hash)

    # PDFs via Docling
    if ext == ".pdf":
        try:
            from core.extractors.extractors_docling import DoclingExtractor  # lazy import
            de = DoclingExtractor()
            out = de.extract(p)
            text = (out.get("markdown") or out.get("full_text") or "").strip()
            meta = dict(out.get("metadata") or {})
            meta.update({
                "extractor": meta.get("extractor", "docling"),
                "source_path": rel,
                "source_dir": dirpart,
            })
            content_hash = sha1_hex(text.encode("utf-8", errors="ignore"))
            return NormalizedDoc(key, rel, dirpart, name, ext, "pdf", None, text, meta, content_hash)
        except Exception:
            return None

    # Treat markdown/plaintext as text
    if ext in text_exts or ext in {".md", ".markdown", ".txt"}:
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return None
        meta = {
            "line_count": len(text.splitlines()),
            "char_count": len(text),
            "extractor": "text",
            "source_path": rel,
            "source_dir": dirpart,
        }
        content_hash = sha1_hex(text.encode("utf-8", errors="ignore"))
        return NormalizedDoc(key, rel, dirpart, name, ext, "text", None, text, meta, content_hash)

    # Skip binaries/unknown
    return None


def ensure_collections(client: ArangoMemoryClient, analyzer: str) -> None:
    defs = [
        CollectionDefinition(name="repo_docs", type="document", indexes=[{"type": "persistent", "fields": ["path"], "unique": True}]),
        CollectionDefinition(name="doc_chunks", type="document", indexes=[{"type": "persistent", "fields": ["path", "doc_key", "chunk_index"], "unique": False}]),
    ]
    client.create_collections(defs)
    ensure_persistent_index(client, collection="repo_docs", fields=["content_hash"], unique=False, sparse=False)
    # Analyzer + view for text search over doc_chunks
    ensure_text_analyzer(client, name=analyzer)
    ensure_arangosearch_view(client, view_name="repo_text", links={
        "doc_chunks": {"fields": {"text": {"analyzers": [analyzer]}}}
    })


def main() -> int:
    ap = argparse.ArgumentParser(description="Unified repo ingestion workflow")
    ap.add_argument("--root", default=".")
    ap.add_argument("--glob", action="append", default=["**/*.py", "**/*.go", "**/*.md", "**/*.pdf"], help="Glob(s) relative to root")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default=os.getenv("EMBED_DEVICE", "cuda"))
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--analyzer", default="text_en")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-tree-sitter", action="store_true")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    files = list(iter_files(root, args.glob))
    if args.limit > 0:
        files = files[: args.limit]
    print(f"[plan] files={len(files)} device={args.device} batch={args.batch} write={args.write}")

    # Stage 0: DB + model
    if not args.dry_run:
        cfg = resolve_memory_config()
        client = ArangoMemoryClient(cfg)
        try:
            ensure_collections(client, analyzer=args.analyzer)
            # Best-effort vector index create (Arango 3.12+); ignore if unsupported
            try:
                ensure_vector_index(client._write_client, collection="repo_docs", field="embedding", dimensions=2048)  # type: ignore[attr-defined]
                ensure_vector_index(client._write_client, collection="doc_chunks", field="embedding", dimensions=2048)  # type: ignore[attr-defined]
            except Exception:
                pass
        finally:
            client.close()

    # Only initialize embedder if not in dry-run mode
    if not args.dry_run:
        emb_config = EmbeddingConfig(model_name="jinaai/jina-embeddings-v4", device=args.device, batch_size=args.batch, trust_remote_code=True)
        embedder = create_embedder(config=emb_config)
    else:
        embedder = None  # type: ignore[assignment]

    code_extractor = CodeExtractor(use_tree_sitter=not args.skip_tree_sitter)
    text_exts = {".md", ".markdown", ".txt", ".rst"}

    # Stage 1: normalize -> queue
    q_norm: deque[NormalizedDoc] = deque()
    for p in files:
        nd = normalize_file(p, root, code_extractor, text_exts)
        if nd and nd.text.strip():
            q_norm.append(nd)
    print(f"[stage1] normalized={len(q_norm)}")

    if args.dry_run or not args.write:
        print("[dry-run] stopping before embedding/writes")
        return 0

    # Stage 2: embed docs in batches (with adaptive sizing to prevent OOM)
    docs_to_write: List[dict] = []
    chunks_to_write: List[dict] = []
    client = ArangoMemoryClient(cfg)

    # Helper to estimate tokens and determine safe batch size
    def estimate_tokens(text: str) -> int:
        """Rough token estimate: 1 token ≈ 4 characters."""
        return len(text) // 4

    def get_adaptive_batch_size(texts: List[str], max_batch: int) -> int:
        """Reduce batch size for large documents to prevent OOM."""
        max_tokens = max(estimate_tokens(t) for t in texts)

        if max_tokens > 16000:  # Very large docs
            return 1
        elif max_tokens > 8000:  # Large docs
            return min(2, max_batch)
        elif max_tokens > 4000:  # Medium docs
            return min(4, max_batch)
        else:  # Small docs
            return max_batch

    try:
        while q_norm:
            batch: List[NormalizedDoc] = []
            while q_norm and len(batch) < args.batch:
                batch.append(q_norm.popleft())

            texts = [d.text for d in batch]

            # Adaptive batch sizing to prevent OOM
            safe_batch_size = get_adaptive_batch_size(texts, args.batch)
            if safe_batch_size < len(batch):
                print(f"[adaptive] Reducing batch size from {len(batch)} to {safe_batch_size} due to large documents")

            doc_embs = embedder.embed_documents(texts, batch_size=safe_batch_size)
            for d, emb in zip(batch, doc_embs):
                doc = {
                    "_key": d.key,
                    "path": d.path,
                    "dir": d.dir,
                    "name": d.name,
                    "ext": d.ext,
                    "kind": d.kind,
                    "lang": d.lang,
                    "content_hash": d.content_hash,
                    "metadata": d.metadata,
                    "text": d.text,
                    "embedding": emb.tolist(),
                }
                docs_to_write.append(doc)

                # Stage 3: late-chunking per doc
                chunks = getattr(embedder, "embed_with_late_chunking", None)
                if callable(chunks):
                    for c in chunks(d.text):
                        chunk_key = sanitize_key(f"{d.key}::{c.chunk_index}")
                        chunks_to_write.append({
                            "_key": chunk_key,
                            "doc_key": d.key,
                            "path": d.path,
                            "chunk_index": int(c.chunk_index),
                            "start_token": int(c.start_token),
                            "end_token": int(c.end_token),
                            "total_chunks": int(c.total_chunks),
                            "text": c.text,
                            "embedding": c.embedding.tolist(),
                        })
            # Flush periodically to bound memory
            if len(docs_to_write) >= 500:
                client.bulk_import("repo_docs", docs_to_write, on_duplicate="update")
                docs_to_write.clear()
            if len(chunks_to_write) >= 1000:
                client.bulk_import("doc_chunks", chunks_to_write, on_duplicate="update")
                chunks_to_write.clear()

        # Final flush
        if docs_to_write:
            client.bulk_import("repo_docs", docs_to_write, on_duplicate="update")
        if chunks_to_write:
            client.bulk_import("doc_chunks", chunks_to_write, on_duplicate="update")
    finally:
        client.close()

    print("[done] repo_docs and doc_chunks written with doc + late-chunk embeddings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
