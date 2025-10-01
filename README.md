# HADES — Retrieval + MemGPT Runtime

Parent: (root)

Children
- `Docs/` — PRDs, program plan, specs, pulse notes.
- `core/` — runtime code (Arango client, embedders, extractors, MemGPT glue).
- `scripts/` — operational CLIs (ingest, chat, admin helpers).
- `infra/` — systemd units and runtime config snippets.
- `tests/` — smoke and benchmarks.
- `acheron/` — archived (legacy) code/docs (excluded from ingestion).

Related Docs
- Docs/Program_Plan.md
- Docs/PRD_PathRAG_Integration.md
- Docs/PRD_Sockets_and_Proxy_Perf.md
- Docs/SoP_Program_Quickstart.md

Entry Points
- `scripts/ingest_repo_workflow.py` — normalize → embed (Jina v4) → late‑chunk → write.
- `scripts/hades_chat.py` — Agent chat with experiential memory and PathRAG retrieval.

Sockets
- RO: `/run/hades/readonly/arangod.sock` (0660 group hades)
- RW: `/run/hades/readwrite/arangod.sock` (0660 group hades; 0600 in prod)

Collections / Views
- repo_docs, doc_chunks; view: repo_text (links doc_chunks.text → analyzer).

Notes
- `acheron/` is excluded from ingestion.
- See `core/database/arango/README.md` for bootstrap/admin and SLOs.
