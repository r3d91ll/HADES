# scripts/ — Operational CLIs

Parent: `../README.md`

Children
- `ingest_repo_workflow.py` — unified repo ingestion (normalize → Jina v4 embed → late‑chunk → write).
- `hades_chat.py` — HADES agent chat with experiential memory (MCP pattern) and PathRAG retrieval.
- `clear_hades_db.py` — drop collections or DB (admin socket required for DB drop).
- `probe_qwen_longctx.py`, `test_qwen_engine.py` — model diagnostics.

Archived
- Legacy bootstrap scripts moved to `../acheron/scripts/`.

Usage
- Ingestion (GPU):
  - `poetry run python scripts/ingest_repo_workflow.py --root . --write --device cuda --batch 16`
- Chat:
  - `poetry run python scripts/hades_chat.py --session demo --no-schema`

Notes
- `acheron/` is excluded by default from ingestion.
