#!/usr/bin/env python3
"""
Module: scripts/clear_hades_db.py
Summary: Danger tool to drop collections or the entire hades_memories database.
Owners: @todd
Last-Updated: 2025-09-30
Inputs: ARANGO_* env (admin socket/auth), CLI flags
Outputs: Drops `repo_text` view and collections {files, funcs, doc_chunks, contains, imports, chunk_code_edges} or DB
Related: core/database/arango/README.md (bootstrap/admin policy)
Stability: admin-only; Security: must use admin socket for DB drop

Usage:
  poetry run python scripts/clear_hades_db.py --dry-run
  poetry run python scripts/clear_hades_db.py --collections
  ARANGO_ADMIN_SOCKET=/run/arangodb3/arangodb.sock poetry run python scripts/clear_hades_db.py --drop-db
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.database.arango.memory_client import (  # noqa: E402
    ArangoMemoryClient,
    resolve_memory_config,
)
from core.database.arango.optimized_client import ArangoHttp2Client, ArangoHttp2Config  # noqa: E402

COLLS_DOC = ["files", "funcs", "doc_chunks"]
COLLS_EDGE = ["contains", "imports", "chunk_code_edges"]
VIEW = "repo_text"


def drop_collections(c: ArangoMemoryClient, names: Iterable[str]) -> None:
    for n in names:
        try:
            c.drop_collections([n], ignore_missing=True)
            print(f"[drop] collection={n}")
        except Exception as exc:
            print(f"[warn] failed to drop {n}: {exc}")


def build_admin_client() -> ArangoHttp2Client:
    admin_socket = os.getenv("ARANGO_ADMIN_SOCKET") or os.getenv("ARANGO_RW_SOCKET") or "/run/arangodb3/arangodb.sock"
    skip_auth = os.getenv("ARANGO_SKIP_AUTH", "").strip().lower() in {"1", "true", "yes", "on"}
    username = os.getenv("ARANGO_USERNAME") if not skip_auth else None
    password = os.getenv("ARANGO_PASSWORD") if not skip_auth else None
    if username and not password:
        username = None
        password = None
    return ArangoHttp2Client(
        ArangoHttp2Config(
            database="_system",
            socket_path=admin_socket,
            base_url=os.getenv("ARANGO_HTTP_BASE_URL", "http://localhost"),
            username=username,
            password=password,
        )
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Clear hades_memories collections or drop DB")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--collections", action="store_true", help="Drop bootstrap collections in hades_memories")
    ap.add_argument("--drop-db", action="store_true", help="Drop the entire hades_memories database (admin)")
    args = ap.parse_args()

    if not args.collections and not args.drop_db:
        print("[plan] no-op (use --collections or --drop-db). Nothing changed.")
        return 0

    if args.drop_db and args.collections:
        print("[error] choose one of --collections or --drop-db")
        return 2

    cfg = resolve_memory_config()
    if args.dry_run:
        print(f"[plan] database={cfg.database} drop_db={args.drop_db} drop_collections={args.collections}")
        return 0

    if args.drop_db:
        admin = build_admin_client()
        try:
            admin.request("DELETE", f"/_api/database/{cfg.database}")
            print(f"[drop] database={cfg.database}")
        finally:
            admin.close()
        return 0

    # collections only
    c = ArangoMemoryClient(cfg)
    try:
        # Drop view first (admin client preferred, but RW proxy may allow)
        try:
            admin = build_admin_client()
            admin.request("DELETE", f"/_db/{cfg.database}/_api/view/{VIEW}")
            print(f"[drop] view={VIEW}")
            admin.close()
        except Exception as exc:
            print(f"[warn] failed to drop view {VIEW}: {exc}")
        drop_collections(c, COLLS_EDGE + COLLS_DOC)
    finally:
        c.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
