# Program Plan — SoP Quickstart (One‑Pager)

Version: 1.0
Date: 2025-09-29

Plan schema (defaults)

```json
{"q":1,"retriever":{"k":6,"mmr":0.4,"timeout_ms":200},"graph":{"hops":1,"fanout":2,"beam":8,"timeout_ms":500},"reranker":{"m":20,"timeout_ms":500},"final_context":6,"budget":{"docs_max":160,"latency_ms":2000},"reasons":[]}
```

Owners

- Platform: UDS sockets, proxies, admin ops, perf CI
- Runtime: plan executor, PathRAG, reranker, packer
- ML Ops: exporters, α̂ estimator, dashboards
- QA/Perf: perf CI, canaries, breakers

Phase checklist

- A Transport
  - [ ] RO 0660, RW 0600; agent in socket group
  - [ ] RO denylist expanded; RW DB‑scoped only
  - [ ] Benchmarks: version, small cursor (p50/p95/p99); proxy overhead ≤ 0.2 ms p50
  - [ ] Access logs + counters on
- B Ingestion
  - [ ] Nodes (files, funcs) — idempotent; no edges by default
  - [ ] Chunks (doc_chunks) — docs & code comments/docstrings
  - [ ] Embeddings — vectors written; index if 3.12+
  - [ ] Edges — contains/imports/chunk_code_edges
- C PathRAG v1
  - [ ] Caps set: H ≤ 2, F ≤ 2–3, B ≤ 8–16; docs_max ≤ 160
  - [ ] PRUNE early; parameterized AQL; MMR diversity
  - [ ] Staged timeouts enforced; reasons emitted
- D Telemetry
  - [ ] reads_stage, latency_stage, dedupe_rate, coverage_score, capped flags
  - [ ] Boundary events map to W,R,H,T,C_ext
- E MemGPT
  - [ ] Plan executor integrated; decay/redundancy/max tokens enforced
  - [ ] α̂ 95% CI in [1.5, 2.0]; recall/latency SLOs met
- F Learned scoring
  - [ ] GraphSAGE export/train; swap scoring with same caps/timeouts
  - [ ] Coverage↑ at reads/latency≤

Acceptance snapshot

- [ ] Transport SLOs met; RO/RW perms/logs verified
- [ ] Ingestion runs clean/idempotent
- [ ] Retrieval never exceeds caps; global SLO ≥95%
- [ ] Telemetry dashboards show W·R·H uplift at constant/better T
- [ ] MemGPT SLOs + α̂ in band; zero‑prop only on true boundary zeros

References: Docs/Program_Plan.md, Docs/PRD_PathRAG_Integration.md, Docs/PRD_Sockets_and_Proxy_Perf.md
