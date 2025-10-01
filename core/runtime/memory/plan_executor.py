"""
Module: core/runtime/memgpt/plan_executor.py
Summary: Query Plan executor normalization and cap enforcement scaffold.
Owners: @todd, @hades-runtime
Last-Updated: 2025-09-30
Inputs: plan dict {retriever, graph, reranker, budget}
Outputs: NormalizedPlan with reasons (trims/limits), validated caps/timeouts
Data-Contracts: plan JSON schema as per Docs/Program_Plan.md
Related: core/runtime/memgpt/pathrag.py, Docs/PRD_PathRAG_Integration.md
Stability: experimental
Boundary: C_ext N/A; P_ij requires caps not exceed configured bounds
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass(slots=True)
class NormalizedPlan:
    k: int
    hops: int
    fanout: int
    beam: int
    rerank_m: int
    final_context: int
    docs_max: int
    latency_ms: int


def normalize_plan(plan: Dict[str, Any]) -> Tuple[NormalizedPlan, List[str]]:
    reasons: List[str] = []
    retr = plan.get("retriever", {})
    graph = plan.get("graph", {})
    rer  = plan.get("reranker", {})
    budget = plan.get("budget", {})

    def clamp(v: int, lo: int, hi: int, reason: str) -> int:
        if v < lo:
            reasons.append(f"{reason}:min")
            return lo
        if v > hi:
            reasons.append(f"{reason}:max")
            return hi
        return v

    k = int(retr.get("k", 6))
    hops = clamp(int(graph.get("hops", 1)), 0, 2, "hops")
    fanout = clamp(int(graph.get("fanout", 2)), 1, 3, "fanout")
    beam = clamp(int(graph.get("beam", 8)), 1, 16, "beam")
    rerank_m = clamp(int(rer.get("m", 20)), 1, 24, "rerank_m")
    final_ctx = clamp(int(plan.get("final_context", 6)), 1, 12, "final_context")
    docs_max = clamp(int(budget.get("docs_max", 160)), 16, 512, "docs_max")
    latency_ms = clamp(int(budget.get("latency_ms", 2000)), 200, 10000, "latency_ms")

    return NormalizedPlan(k, hops, fanout, beam, rerank_m, final_ctx, docs_max, latency_ms), reasons


__all__ = ["NormalizedPlan", "normalize_plan"]
