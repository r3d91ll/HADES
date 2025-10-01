HADES — Unix Socket & Proxy Performance PRD (v0.1)

Author: HADES team
Date: 2025-09-29
Status: Draft for review

0) Context & Motivation

- All DB access uses Unix Domain Sockets (UDS), not TCP. Prior tests showed sub‑millisecond latency for small queries. We must baseline current RO/RW proxy latencies, harden proxy policy, and lock SLOs so retrieval stages (PathRAG, etc.) can rely on predictable costs.

1) Goals (In Scope)

- G1: Establish latency SLOs for UDS paths (upstream socket vs RO/RW proxies) at p50/p95/p99 under realistic loads.
- G2: Verify and fix socket permissions/ownership (RO 0660, RW 0600) and group membership for the agent user.
- G3: Harden proxy allow/deny lists and scope (AQL write keywords; DB‑scoped endpoints for RW).
- G4: Add lightweight boundary metrics (mode=RO|RW, path, status, ms) for Conveyance and ops dashboards.
- G5: Document repeatable benchmarks and acceptance gates; include in CI perf jobs (sanity subset).

2) Non‑Goals

- Remote networking, cross‑host proxies, or TLS—UDS only.
- Full Arango cluster tuning; focus is app‑side transport and proxy behavior.

3) Target SLOs (UDS, warm cache, single digit concurrency)

- T1: Version endpoint (`/_api/version`) p50 ≤ 0.4 ms, p95 ≤ 0.9 ms, p99 ≤ 1.2 ms (upstream);
     RO/RW proxy overhead ≤ 0.2 ms absolute at p50.
- T2: Small AQL cursor (batchSize 100, result ≤ 100 rows) p50 ≤ 0.8 ms, p95 ≤ 1.6 ms.
- T3: Single doc GET (hot cache) p50 ≤ 0.5 ms, p95 ≤ 1.0 ms.
- T4: Under concurrency 10, p95 inflation ≤ 2× single‑shot for the same endpoint.

4) Benchmark Methodology

- Endpoints: `/_api/version`, `/_api/document/{coll}/{key}`, `/_api/cursor` with tiny AQL.
- Sockets: upstream admin UDS vs RO proxy vs RW proxy.
- Loads: warm (repeat hits), micro cold (random keys), concurrency {1, 4, 10}.
- Metrics: end‑to‑end latency per request, CPU %, RSS for proxy, error rate.
- Runs: N ≥ 5,000 samples per cell; report p50/p95/p99.
- Tooling: Python httpx over UDS (existing transport), single file script executed via poetry; optional psutil for process stats.

5) Proxy Policy Hardening

- RO: allow GET/HEAD/OPTIONS and POST only to `/_api/cursor`;
      AQL write keyword scan must include: INSERT, UPDATE, UPSERT, REMOVE, REPLACE, TRUNCATE, DROP, CREATE, ALTER, RENAME, GRANT, REVOKE, GRAPH.
- RW: restrict to DB‑scoped endpoints `/_db/{db}/_api/{document,index,collection,import,cursor}`; disallow admin endpoints (`/database`, `/view`, `/analyzer`).
- Body peek limit: env `AQL_PEEK_LIMIT_BYTES` (default 128 KiB); over‑limit -> 413.
- Socket perms: RO 0660 (group connect), RW 0600; systemd units to enforce `SocketUser/Group` and `SocketMode`.

6) Observability

- Access log prefix by mode: `RO:` and `RW:`; fields: method, path (redacted query), status, duration_ms.
- Counter gauges: requests_total by mode and status, latency histograms; export to stdout or file for ingestion.
- Boundary Conveyance mapping: record `{mode, T, W='mem:query'|'mem:write', R='arangodb', H='memgpt-tool'}` per call.

7) Acceptance Criteria

- A1: Benchmarks show SLOs met on upstream UDS and via proxies; RO/RW overhead ≤ 0.2 ms p50 on version and ≤ 0.3 ms p50 on small cursor.
- A2: RO socket is usable by agent group (connect succeeds without admin socket); RW remains owner‑only.
- A3: RO denylist blocks write AQL (unit tests with crafted queries); RW scope blocks admin endpoints.
- A4: Access logs present; latency histograms visible; Conveyance boundary events include RO vs RW split.

8) Rollout Plan

- Phase S1: Add microbenchmark script (poetry runnable) and baseline results saved under `benchmarks/reports/`.
- Phase S2: Apply proxy hardening (denylist additions, RW scoping), set RO mode to 0660; restart units; re‑benchmark.
- Phase S3: Add access logging and minimal counters; re‑benchmark and confirm SLOs.
- Phase S4: CI perf smoke (reduced N) to prevent regressions in proxy builds.

9) Risks & Mitigations

- R: Analyzer/view admin operations needed during bootstrap. M: keep via admin socket only; do not expose via RW proxy.
- R: Proxy logging increases overhead. M: default to low‑cost structured log and sampling if needed.
- R: Pathological AQLs exceed peek limit. M: return 413 with clear message; adjust via env.

10) References

- Socket patterns: Docs/pulse_articles/SocketPatterns.md
- Proxy code: core/database/arango/proxies/*
- Prior latencies: core/database/arango/README.md benchmark table
