# Code Triage (Initial)

Scope: `core/` in this production repo. Goal: decide keep/change/drop ahead of MemGPT orchestration and model engine buildout.

Keep (adopt as-is)

- `core/database/arango/optimized_client.py` — HTTP/2 over Unix sockets; minimal, production-ready.
- `core/database/arango/memory_client.py` — high-level wrapper; now defaults to `hades_memories`.
- `core/database/arango/proxies/*.go` and `cmd/*` — RO/RW UDS proxies to upstream Arango.
- `core/logging/conveyance.py` — Conveyance utilities aligned with PRD metrics.
- `core/config/*.py` and `core/config/embedders/*.yaml` — lightweight, useful scaffolding.

Change (surgical updates required)

- `core/extractors/extractors_code.py` — keep Tree-sitter optional, fall back gracefully; no external heavy deps.
- Docs referring to `arxiv_repository` — update references to `hades_memories` over time.

Archive/Drop (not for production)

- `core/database/arango/php_unix_bridge.php` — dev-only TCP fallback; wrong DB name.
- `core/workflows/embed_paper_code_pair.py` — depends on missing `core.paper_code_pipeline`; port later or retire.

Delivered scaffolding / Next steps

- `core/runtime/memgpt/` now provides turn management, boundary metrics wrappers, model engine, and an orchestrator facade; extend with function-calling harness, queue manager orchestration, and production streaming integration.
- Seed script (`scripts/seed_from_workspace.py`) ingests Docs/ + code; add embedding + vector backfill when models are ready.
- Implement alpha-hat estimation and P_ij scoring config when telemetry layer is ready.

Notes

- Standardize Python 3.12 via Poetry; run `poetry install` then `make lint test`.
- UDS sockets and DB name documented in `AGENTS.md` and `.env.example`.
