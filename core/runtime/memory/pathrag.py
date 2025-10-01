"""
Module: core/runtime/memgpt/pathrag.py
Summary: PathRAG heuristic scaffold (budgeted neighbors with caps and reasons).
Owners: @todd, @hades-runtime
Last-Updated: 2025-09-30
Inputs: seed node IDs, caps {hops, fanout, beam, timeout_ms}
Outputs: PathResult nodes/edges with scores and reason codes
Data-Contracts: AQL neighbor query over edge collections (contains/imports)
Related: core/runtime/memgpt/plan_executor.py, Docs/PRD_PathRAG_Integration.md
Stability: experimental; Security: parameterized AQL only
Boundary: C_ext unaffected; P_ij requires DB read via RO socket

Implements a budgeted single-hop expansion with caps {H,F,B} and reason codes.
Learned ranker and multi-hop PRUNE will replace this heuristic later.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic
from typing import Any, Dict, List, Tuple

from ...database.arango.memory_client import ArangoMemoryClient


@dataclass(slots=True)
class PathCaps:
    hops: int = 1
    fanout: int = 2
    beam: int = 8
    timeout_ms: int = 500


@dataclass(slots=True)
class PathResult:
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    score: float
    reasons: List[str]


def _time_exceeded(start: float, timeout_ms: int) -> bool:
    return (monotonic() - start) * 1000.0 >= timeout_ms


def _neighbors_once(client: ArangoMemoryClient, node_ids: List[str], fanout: int) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """Return up to `fanout` outgoing neighbors via known edge collections.

    This is a simplified expansion to keep the scaffold minimal; it looks at
    `contains` and `imports` if present.
    """
    results: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for nid in node_ids:
        aql = """
        LET out = (
          FOR e IN contains FILTER e._from == @nid LIMIT @fanout RETURN { e, v: DOCUMENT(e._to) }
        )
        RETURN out
        """
        try:
            rows = client.execute_query(aql, {"nid": nid, "fanout": fanout})
        except Exception:
            rows = []
        for sub in rows:
            for pair in sub:
                results.append((pair.get("v", {}), pair.get("e", {})))
    return results[:fanout]


def find_paths(
    client: ArangoMemoryClient,
    *,
    seed_nodes: List[str],
    caps: PathCaps,
) -> Tuple[List[PathResult], List[str]]:
    """Return PathRAG candidates and reason codes under budgeted caps.

    For MVP, perform H=1 expansion with top-F neighbors per seed; scoring is
    a simple heuristic: prefer direct neighbors, break ties by name length.
    """
    reasons: List[str] = []
    start = monotonic()
    if caps.hops <= 0 or not seed_nodes:
        return [], ["empty_seeds"]

    # Single hop expansion for scaffold
    pairs = _neighbors_once(client, seed_nodes, caps.fanout)
    if _time_exceeded(start, caps.timeout_ms):
        reasons.append("timeout")
        pairs = pairs[: max(1, caps.fanout // 2)]

    # Score: simple length prior on name, favor shorter names
    scored: List[Tuple[float, Dict[str, Any], Dict[str, Any]]] = []
    for v, e in pairs:
        name = str(v.get("name") or v.get("path") or v.get("_key", ""))
        score = 1.0 / (1.0 + len(name))
        scored.append((score, v, e))
    scored.sort(key=lambda t: t[0], reverse=True)

    # Beam cap
    if len(scored) > caps.beam:
        reasons.append("cap_exceeded")
        scored = scored[: caps.beam]

    results: List[PathResult] = []
    for s, v, e in scored:
        results.append(PathResult(nodes=[v], edges=[e], score=s, reasons=[]))
    return results, reasons


__all__ = ["PathCaps", "PathResult", "find_paths"]
