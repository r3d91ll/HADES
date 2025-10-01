# core/ — Runtime Code

Parent: `../README.md`

Children
- database/ — Arango HTTP/2 client, proxies, admin helpers
- embedders/ — Jina v4 wrapper and factory
- extractors/ — Tree‑sitter and Docling extractors
- runtime/memgpt/ — orchestration, prompts, path‑rag scaffolds

Related Docs
- Docs/Program_Plan.md
- Docs/PRD_PathRAG_Integration.md
- Docs/PRD_Sockets_and_Proxy_Perf.md

APIs
- `core/database/arango/optimized_client.py` — HTTP/2 over UDS
- `core/database/arango/memory_client.py` — high‑level Arango ops / bulk insert
- `core/embedders/embedders_jina.py` — Jina v4 embeddings (late‑chunk support)
- `core/extractors/extractors_code.py` — code + Tree‑sitter metadata
