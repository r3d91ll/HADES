# Program Plan v1.1 — HADES Retrieval & Experiential Memory Runtime

Version: 1.1
Date: 2025-09-30
Status: Updated terminology (v1.0 frozen baseline archived)

1) Executive Summary

- Unify sockets SLOs, PathRAG retrieval, and agent runtime with experiential memory under a single plan and budgets.
- Retrieval = vector seeds → PathRAG (beam, hops, fanout) → rerank → compact final context.
- Ingestion split into nodes → chunks → embeddings → edges, each idempotent.
- Security boundary: RO/RW UDS sockets; admin endpoints only via admin socket.
- Conveyance measured at boundary only: C = (W·R·H / T) · C_ext^α (α learned online).

2) Query Plan Contract (defaults)

```json
{
  "q": 1,
  "retriever": {"k": 6, "mmr": 0.4, "timeout_ms": 200},
  "graph": {"hops": 1, "fanout": 2, "beam": 8, "timeout_ms": 500},
  "reranker": {"m": 20, "timeout_ms": 500},
  "final_context": 6,
  "budget": {"docs_max": 160, "latency_ms": 2000},
  "reasons": []
}
```

- Beam B is a global cap (not per hop). Executor must enforce caps/timeouts and append reason codes.
- Reasons: cap_exceeded, coverage_reached, timeout, budget_docs, dedupe_drop, fanout_pruned.
- coverage_score v1: token‑weighted query term coverage over selected chunks (BM25‑based) in [0,1].

3) Phases & Gates (critical path)

- A. Transport (T gate):
  - SLOs (UDS, warm): version p50 ≤ 0.4 ms / p95 ≤ 0.9 ms; small cursor p50 ≤ 0.8 ms / p95 ≤ 1.6 ms.
  - Proxy overhead ≤ 0.2 ms p50. RO socket 0660 (group usable), RW 0600 (owner‑only).
  - RO denylist expanded (INSERT/UPDATE/UPSERT/REMOVE/REPLACE/TRUNCATE/DROP/CREATE/ALTER/RENAME/GRANT/REVOKE/GRAPH). RW DB‑scoped only.
  - Gate: SLOs met; perms validated; access logs/counters visible.
- B. Ingestion (W/R hygiene):
  - Nodes (files, funcs; no edges by default) → Chunks (doc_chunks) → Embeddings (vectors) → Edges (contains/imports/chunk_code_edges).
  - Gate: idempotent re‑runs; no analyzer/collection collisions.
- C. PathRAG v1 (R at fixed T):
  - Directional/filtered AQL with PRUNE; caps H ≤ 2, F ≤ 2–3, B ≤ 8–16; MMR diversity; parameterized inputs.
  - Enforce docs_max and staged timeouts; emit reasons on trims/early stop.
  - Gate: caps never exceeded; ≥95% runs meet 2 s global SLO.
- D. Telemetry (Ctx measurability):
  - Per‑stage: reads_stage, latency_stage, dedupe_rate, coverage_score, capped/early_stop.
  - Boundary Conveyance mapping (RO/RW). Gate: W·R·H uplift at constant/better T visible.
- E. Agent Runtime with Experiential Memory (H and C_ext^α online):
  - Integrate memory operations (store/retrieve/consolidate) into agent chat loop.
  - PathRAG-powered retrieval with plan executor; α̂ estimator for context amplification.
  - Experiential memory: observations, reflections, entities, relationships (not MemGPT paper architecture).
  - Gate: α̂ 95% CI in [1.5, 2.0]; recall/latency SLOs met; zero‑prop only when P_ij=0 or C_ext=0.
- F. Learned scoring swap (R ≤ same T):
  - Train GraphSAGE + small ranker; replace heuristic scoring under identical caps/timeouts.
  - Gate: ≥ coverage_score with ≤ reads/latency.

4) Unified Acceptance

- Transport: SLOs met; proxy overhead in bounds; RO/RW perms/logs validated.
- Ingestion: phases idempotent; no analyzer/collection collisions.
- Retrieval: never exceed {B,H,F} or docs_max; staged timeouts enforced; global SLO ≥95%; reasons emitted.
- Telemetry: per‑stage reads/latency/coverage + boundary W·R·H vs T visible; boundary‑only conveyance events complete.
- Agent Runtime: recall p99 ≤ 250 ms; vector p99 ≤ 750 ms; user‑agent avg < 100 ms; α̂ CI in band; memory operations integrated.

5) Ownership

- Platform: sockets, proxies, admin ops, perf CI.
- Runtime: plan executor, PathRAG, reranker, packer.
- ML Ops: exporters (Phoenix/Arize), α̂, dashboards.
- QA/Perf: perf CI, canary, breakers.

6) References

- PathRAG PRD: Docs/PRD_PathRAG_Integration.md
- Sockets PRD: Docs/PRD_Sockets_and_Proxy_Perf.md
- Pulse notes: Docs/pulse_articles/*
