#!/usr/bin/env python3
"""UDS microbenchmarks for Arango upstream vs proxies.

Runs p50/p95/p99 for two endpoints:
  - GET /_api/version
  - POST /_api/cursor with a tiny query (RETURN 1)

Usage:
  poetry run python tests/benchmarks/uds_microbench.py \
    --upstream /run/arangodb3/arangodb.sock \
    --ro /tmp/hades-test/ro.sock \
    --rw /tmp/hades-test/rw.sock \
    --database _system --iters 1000
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.database.arango.optimized_client import ArangoHttp2Client, ArangoHttp2Config


def quantiles(samples: List[float]) -> Tuple[float, float, float]:
    if not samples:
        return 0.0, 0.0, 0.0
    xs = sorted(samples)
    def q(p: float) -> float:
        k = int(round((len(xs) - 1) * p))
        return xs[k]
    return q(0.50), q(0.95), q(0.99)


def bench_socket(socket: str, database: str, iters: int, username: str | None, password: str | None) -> Dict[str, Tuple[float, float, float]]:
    cfg = ArangoHttp2Config(database=database, socket_path=socket, username=username, password=password)
    client = ArangoHttp2Client(cfg)
    try:
        # version
        v_times: List[float] = []
        for _ in range(iters):
            t0 = time.perf_counter()
            client.request("GET", "/_api/version")
            v_times.append((time.perf_counter() - t0) * 1000.0)

        # tiny cursor
        c_times: List[float] = []
        for _ in range(iters):
            t0 = time.perf_counter()
            client.request("POST", f"/_db/{database}/_api/cursor", json={"query": "RETURN 1", "batchSize": 1})
            c_times.append((time.perf_counter() - t0) * 1000.0)

        return {
            "version_ms": quantiles(v_times),
            "cursor_ms": quantiles(c_times),
        }
    finally:
        client.close()


def main() -> int:
    ap = argparse.ArgumentParser(description="UDS microbenchmarks")
    ap.add_argument("--upstream", default="/run/arangodb3/arangodb.sock")
    ap.add_argument("--ro", default="/run/hades/readonly/arangod.sock")
    ap.add_argument("--rw", default="/run/hades/readwrite/arangod.sock")
    ap.add_argument("--database", default="_system")
    ap.add_argument("--iters", type=int, default=1000)
    args = ap.parse_args()

    username = os.getenv("ARANGO_USERNAME")
    password = os.getenv("ARANGO_PASSWORD")
    skip_auth = os.getenv("ARANGO_SKIP_AUTH", "").strip().lower() in {"1", "true", "yes", "on"}
    if skip_auth:
        username = None
        password = None

    print(f"[plan] database={args.database} iters={args.iters}")
    for label, sock in [("upstream", args.upstream), ("ro_proxy", args.ro), ("rw_proxy", args.rw)]:
        try:
            stats = bench_socket(sock, args.database, args.iters, username, password)
            v50, v95, v99 = stats["version_ms"]
            c50, c95, c99 = stats["cursor_ms"]
            print(f"[{label}] version p50/p95/p99: {v50:.3f}/{v95:.3f}/{v99:.3f} ms | cursor p50/p95/p99: {c50:.3f}/{c95:.3f}/{c99:.3f} ms")
        except Exception as exc:
            print(f"[{label}] error: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
