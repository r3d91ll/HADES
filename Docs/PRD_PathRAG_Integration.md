# HADES — PathRAG Integration PRD (v0.2)

Author: HADES team
Date: 2025-09-29
Status: Draft for review

0) Executive Summary

- Adopt PathRAG (path-centric retrieval) to bound graph fanout and read amplification.
- Enforce strict, measurable budgets across stages; retrieval becomes: vector seeds → budgeted path search → rerank → compact final context.
- Split ingestion into separable phases (nodes → chunks → embeddings → edges) to de-risk rollout and testing.
- Keep security boundary (RO/RW sockets) and ops (admin socket provisioning) intact.

1) Problem Statement

- Naive multi-hop traversals explode fanout; chained stages (planner→retriever→graph→reranker→generator) amplify reads and latency.
- Our bootstrap conflated steps, and retrieval lacks budget enforcement and grounding.
- We need a graph-aware, budgeted retrieval design that is safe to operate and measurable.

2) PathRAG Module (Centerpiece)

- Objective: return a small set of high-value paths from seeds within hard caps.
- Inputs: seed IDs; caps {H: max_hops, B: beam_width_global, F: neighbors_per_hop}; per-turn doc+latency budgets.
- Algorithm (beam search):
  1. Initialize beam with seeds (score = seed retrieval score).
  2. Expand each path by up to F neighbors using directional, filtered AQL; apply PRUNE early.
  3. Score new paths; keep top B paths (global cap, not per hop).
  4. Stop at depth H or when coverage/confidence reached or budget/time exhausted.
- Scoring v1 (heuristic):
  score(path) = α·seed_sim + β·edge_weight + γ·lex_match − δ·depth_penalty − ρ·redundancy
  - Add diversity penalty ρ via MMR-style redundancy against already selected paths.
  - Later swap α..ρ to a trained ranker (GraphSAGE embeddings + small MLP / cross-encoder).
- Outputs: ranked paths with node/edge lists, snippets for packing; reason codes for pruning/early stop.

3) Query Plan Contract (strict budgets)

  ```json
  {
    "q": 2,
    "retriever": {"k": 6, "mmr": 0.4, "timeout_ms": 200},
    "graph": {"hops": 1, "fanout": 2, "beam": 8, "timeout_ms": 500},
    "reranker": {"m": 20, "timeout_ms": 500},
    "final_context": 6,
    "budget": {"docs_max": 160, "latency_ms": 2000},
    "reasons": []
  }
  ```

- B is a global beam cap per turn (not per-hop). Executor must obey caps and timeouts.
- Emit reason codes on trims and early stops (e.g., "cap_exceeded", "coverage_reached", "timeout").

4) Requirements

- R1 Vector-first retrieval (K ≤ 6–8).
- R2 PathRAG graph step optional; H ≤ 2, F ≤ 2–3, B ≤ 8–16; all enforced as hard caps.
- R3 Rerank M ≤ 24; final context C ≤ 8.
- R4 Per-turn doc budget (e.g., 160) and overall latency SLO (2 s) with staged timeouts.
- R5 RO/RW sockets separation; admin provisioning only via admin socket.
- R6 Telemetry per stage: reads_stage, latency_stage, dedupe_rate, coverage, capped flags.
- R7 Idempotent ingestion: nodes → chunks → embeddings → edges; dedupe order explicit.

5) Architecture & Ingestion Phases

- A Nodes (files, funcs)
  - Inputs: workspace files; sanitize/dedupe; no edges by default (`--edges=0`).
  - Dedupe order: by `path` for files; by `_key` for funcs; collisions resolved with stable suffixes.
- B Chunks (doc_chunks)
  - Split Docs/*.md + code docstrings/comments; store text, headings, token counts.
- C Embeddings
  - Compute embeddings for `doc_chunks` (and optional `funcs`); write to `embedding: float[]`.
  - Index: Arango vector index (3.12+) or defer to text-only search fallback.
- D Edges (explicit step)
  - contains (files→funcs), imports (funcs→funcs), chunk_code_edges (doc_chunks→funcs) built after embeddings.

Mermaid (ingestion & retrieval overview)

```mermaid
flowchart LR
  A[Nodes] --> B[Chunks] --> C[Embeddings] --> D[Edges]
  subgraph Retrieval
    S[Vector Seeds (K)] --> P[PathRAG Beam (B,H,F)] --> R[Rerank (M)] --> F[Final Context (C)]
  end
```

6) Telemetry & Conveyance

- Boundary metrics per tool call: {mode: RO|RW, path, status, ms} mapped to Conveyance (W,R,H,T,C_ext,P_ij).
- Per-stage: `reads_stage`, `latency_stage`, `dedupe_rate`, `coverage_score`, flags {capped, early_stop}.
- Acceptance ties to Conveyance: show improved W·R·H at constant/better T; record C_ext assumptions.

7) Security & Ops

- Keep admin endpoints off RW proxy; manual DB creation (see core/database/arango/README.md).
- Analyzer naming: prefer reuse of cluster `text_en`, fallback to namespaced `hades_text_en`.
- Socket modes: RO 0660 (group connect), RW 0600; ensure agent in socket group.
- AQL hygiene: parameterize all user inputs; avoid injecting literals into AQL PATH expansions.

8) Testing Strategy

- Unit
  - PathRAG expansion with mocked neighbors; cap enforcement; diversity penalty.
  - Plan executor caps/timeouts; error reasons.
  - Chunker tokenization; embedding writer schema.
- Integration
  - Vector seeds → PathRAG within caps → rerank → final pack.
  - RO socket access tests (permission model) and RW writes limited to explicit tools.
- Negative/Stress
  - Empty graph / missing embeddings; high-degree nodes; deep chains; analyzer mismatch.
  - Timeouts at each stage; ensure graceful early stop with reason codes.

9) Rollout Plan

- Phase A (Nodes): finalize bootstrap with `--edges=0`; sanitize/dedupe. [DONE]
- Phase B (Chunks+Embeddings): chunker and embedding runner; write vectors.
- Phase C (Edges): add imports & chunk↔code edges.
- Phase D (PathRAG Heuristic): beam search with AQL neighbors + heuristic scoring.
- Phase E (GraphSAGE): export features/edges, train; swap PathRAG scoring to learned model.
- Phase F (Agent Runtime Integration): integrate plan executor + retrieval into agent chat; full telemetry.

10) Acceptance Criteria

- A1 Ingestion phases run independently and idempotently; no analyzer/collection collisions.
- A2 PathRAG never exceeds caps (B,H,F) nor `budget.docs_max`; emits reasons on trims/early stops.
- A3 Staged timeouts enforced (retriever, graph, reranker); global SLO respected in ≥95% runs.
- A4 Telemetry dashboards show per-stage reads/latency, coverage_score, and Conveyance improvements (W·R·H vs T).
- A5 RO socket retrieval passes; RW writes only through explicit tools; audit logs separate RO/RW.

11) Open Questions

- Default analyzer? (proposal: `hades_text_en` to avoid cluster collisions.)
- Embedding model & dims (Jina v4 vs ModernBERT) and vector index availability.
- Edge weights: heuristic vs computed (e.g., import frequency, file proximity).
- Reranker choice and timeout policy (bi‑encoder first vs cross‑encoder with m, timeout_ms).

12) References & Appendix

- Graph fanout patterns: Docs/pulse_articles/GraphTraversalConceptWithExpandingFanout.md
- Read amplification budgeting: Docs/pulse_articles/ReadAmplificationInMulti‑stagePipelines.md
- Socket design & ops: Docs/pulse_articles/SocketPatterns.md and core/database/arango/README.md
