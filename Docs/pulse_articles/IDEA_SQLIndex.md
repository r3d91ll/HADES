# IDEA - NOT APPORVED FOR IMPLEMENTATION

Here’s a quick win for your graph‑RAG stack: sometimes the bottleneck isn’t batch size or pooling—it’s the **query plan** (e.g., AQL forcing costly cross‑collection joins ArangoDB isn’t great at). A simple hybrid layout can cut tail latency a lot.

## Why this happens (fast background)

* ArangoDB rocks at **local traversals** on well‑indexed vertex/edge sets.
* It’s slower when an AQL query **joins large, semantically noisy sets first**, then traverses.
* Vector prefilters in the same query can push the planner toward **broad candidate sets**, inflating intermediate results.

## A hybrid strategy that works

**Step 1 — Semantic pre‑filter (SQLite or Postgres, outside Arango):**

* Maintain a lightweight table of `{doc_id, embedding, namespace, metadata…}`.
* Do vector search + cheap metadata predicates to get a **tight candidate list** (say 100–2,000 IDs).
* Return only `doc_id`s (and maybe a relevance score).

**Step 2 — Graph traversal (ArangoDB only on narrowed IDs):**

* Feed those IDs into AQL as a bind variable array; traverse from those nodes or filter edges by them.
* Keep Arango doing what it’s best at: **indexed filters + traversals**, not high‑cardinality joins.

## Practical checklist

* **Schema:**

  * Arango: `docs` (V), `RELATES_TO` (E), with persistent indexes on `_key`, `type`, and any hot metadata fields.
  * SQL: `embeddings(doc_id PK, vector, kind, lang, updated_at, tags JSONB…)` with `ivfflat/hnsw` index.
* **Data path:** ingestion → embeddings (GPU) → write to SQL → write doc+edges to Arango.
* **Query path:**

  1. `SELECT doc_id FROM embeddings WHERE … ORDER BY vector <-> :q LIMIT 1000;`
  2. AQL:

  ```aql
  LET cands = @doc_ids
  FOR v IN docs
    FILTER v._key IN cands
    FOR e, n, p IN 1..3 ANY v RELATES_TO
      OPTIONS { bfs: true, uniqueVertices: "global" }
      FILTER p.vertices[*].type ALL != "noise"
      RETURN DISTINCT {root:v._key, hop:n, id:p.vertices[-1]._key, edge:e._key}
  ```
* **Indexes to (re)check in Arango:**

  * Persistent on `docs.type`, `docs.namespace`, `edges._from`, `edges._to`.
  * If you filter by tags: inverted index on `docs.tags`.
* **AQL gotchas:** prefer `IN @bindVar` over building huge `OR` chains; avoid mixing vector‑like filtering and traversal in one step; enable `bfs:true` when you care about shortest‑hop discovery; use `uniqueVertices:"global"` to prevent path blowups.
* **Batch sizing:** keep `@doc_ids` under a few thousand to keep the traversal frontier small; paginate candidates if needed.
* **Explain your plan:** run `AQL EXPLAIN`—if you see scattered full scans or large gather nodes, you’re joining too much pre‑traversal.

## Minimal adapter pattern (pseudo)

* `retrieve_semantic(query) -> [ids]`  (SQL)
* `expand_graph(ids, hops=2) -> subgraph` (Arango)
* `rank_and_merge(subgraph, semantic_scores) -> final set`

## Monitoring tips

* Log `candidate_count`, `edges_touched`, `peak_memory`, `query_ms`.
* Alert when `candidate_count` > threshold or `edges_touched/query_ms` spikes—usually signals a planner detour or missing index.

If you want, I can draft a tiny reference repo skeleton: SQL DDL, Arango migrations, and two endpoints (`/semantic_prefilter`, `/graph_expand`) wired to your existing pipeline.

---
# repo skeleton for hybrid semantic prefilter + graph expansion


```python
# --- sql/schema.sql ---
"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE embeddings (
    doc_id TEXT PRIMARY KEY,
    embedding VECTOR(768),
    kind TEXT,
    lang TEXT,
    updated_at TIMESTAMP DEFAULT now(),
    tags JSONB
);

-- ivfflat index for fast ANN search
CREATE INDEX ON embeddings USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

CREATE INDEX ON embeddings (kind);
CREATE INDEX ON embeddings (lang);
"""

# --- arango/migrations/001_init_collections.js ---
"""
// create vertex + edge collections
db._createDatabase('hades_memories');
db._useDatabase('hades_memories');

db._createDocumentCollection('docs');
db._createEdgeCollection('RELATES_TO');

// indexes
db.docs.ensureIndex({ type: "persistent", fields: ["type"] });
db.docs.ensureIndex({ type: "persistent", fields: ["namespace"] });
db.RELATES_TO.ensureIndex({ type: "persistent", fields: ["_from", "_to"] });
"""

# --- services/semantic_prefilter.py ---
import psycopg2
import numpy as np
from typing import List

class SemanticPrefilter:
    def __init__(self, dsn: str):
        self.conn = psycopg2.connect(dsn)

    def query(self, embedding: np.ndarray, limit: int = 500) -> List[str]:
        """Return candidate doc_ids based on vector similarity."""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT doc_id
                FROM embeddings
                ORDER BY embedding <-> %s
                LIMIT %s;
                """,
                (embedding.tolist(), limit)
            )
            return [row[0] for row in cur.fetchall()]

# --- services/graph_expand.py ---
from pyArango.connection import Connection

class GraphExpand:
    def __init__(self, arango_url: str, username: str, password: str):
        self.conn = Connection(arangoURL=arango_url, username=username, password=password)
        self.db = self.conn["hades_memories"]

    def expand(self, doc_ids: List[str], depth: int = 2):
        query = """
        LET cands = @doc_ids
        FOR v IN docs
          FILTER v._key IN cands
          FOR e, n, p IN 1..@depth ANY v RELATES_TO
            OPTIONS { bfs: true, uniqueVertices: "global" }
            RETURN DISTINCT {
              root: v._key,
              hop: n,
              id: p.vertices[-1]._key,
              edge: e._key
            }
        """
        bind_vars = {"doc_ids": doc_ids, "depth": depth}
        return [d for d in self.db.AQLQuery(query, bindVars=bind_vars, rawResults=True)]

# --- api/main.py ---
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

from services.semantic_prefilter import SemanticPrefilter
from services.graph_expand import GraphExpand

app = FastAPI()

prefilter = SemanticPrefilter(dsn="postgresql://user:pass@localhost:5432/ragdb")
graph = GraphExpand(arango_url="http://localhost:8529", username="root", password="")

class QueryRequest(BaseModel):
    embedding: list
    limit: int = 500
    depth: int = 2

@app.post("/semantic_prefilter")
def semantic_prefilter(req: QueryRequest):
    ids = prefilter.query(np.array(req.embedding), limit=req.limit)
    return {"candidates": ids}

@app.post("/graph_expand")
def graph_expand(req: QueryRequest):
    ids = prefilter.query(np.array(req.embedding), limit=req.limit)
    subgraph = graph.expand(ids, depth=req.depth)
    return {"candidates": ids, "subgraph": subgraph}
```
