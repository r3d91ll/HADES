Here’s a quick, practical win for your HADES + MemGPT stack: **split your ArangoDB connections into explicit read-only and read-write sockets** so MemGPT can’t “accidentally” write during traversals—and your Conveyance metrics stay clean.

---

# Why this helps (plain English)

* **Safety boundary:** Read-only queries literally cannot mutate state—no “oops” UPSERTs from an agent tool.
* **Performance/overhead:** The driver skips write-capable negotiation on RO paths; fewer permission checks, simpler pooling.
* **Cleaner metrics:** You can attribute time/latency separately for **R (read)** vs **W (write)** at the agent boundary without post-hoc inference.

---

# Socket layout

Create two Unix sockets exposed by your local arangod (or via systemd socket activation / nginx stream):

```
/run/hades/readonly/arangod.sock
/run/hades/readwrite/arangod.sock
```

* Map the **readonly** socket to an Arango user with **RO permissions** on your `hades_memories` database and relevant collections/indexes.
* Map the **readwrite** socket to an Arango user with **RW** (writes + DDL if you prefer migrations via a separate admin role).

---

# MemGPT-side connection config

Example Python config (adjust to your driver; shown with `python-arango` style):

```python
from arango import ArangoClient

# RO client: used by retrieval / traversals / ranking
ro = ArangoClient(hosts="unix:///run/hades/readonly/arangod.sock").db(
    "hades_memories",
    username="memgpt_ro",
    password=os.environ["ARANGO_RO_PW"],
    verify=True
)

# RW client: used by writes / promotions / feedback
rw = ArangoClient(hosts="unix:///run/hades/readwrite/arangod.sock").db(
    "hades_memories",
    username="memgpt_rw",
    password=os.environ["ARANGO_RW_PW"],
    verify=True
)

# In your agent tools:
def query_memory(aql, bind_vars=None):
    return ro.aql.execute(aql, bind_vars=bind_vars or {})

def upsert_memory(aql, bind_vars=None):
    return rw.aql.execute(aql, bind_vars=bind_vars or {})
```

Tip: expose both handles to your tool layer, not the policy/orchestrator—keeps call sites explicit: `query_*` uses `ro`, `upsert_*` uses `rw`.

---

# Arango permissions recipe (one-time)

* User `memgpt_ro`: **READ** on `hades_memories` (all collections, views, analyzers).
* User `memgpt_rw`: **WRITE** on same; keep **DB Admin** (DDL) to a separate `migrations` user if you want stricter ops.

---

# Guardrails in AQL

Even on the RW client, keep writes explicit:

```aql
// Read path (ro)
FOR d IN memories
  FILTER d.owner == @owner AND d.type == "chunk"
  SORT d.score DESC
  LIMIT 50
  RETURN d

// Write path (rw): make writes obvious & auditable
UPSERT { _key: @key }
INSERT MERGE(@doc, { createdAt: DATE_NOW() })
UPDATE MERGE(OLD, @doc, { updatedAt: DATE_NOW() })
IN memories
RETURN NEW
```

---

# Conveyance metrics (boundary-first)

Instrument at **tool entry/exit**—not inside the DB layer—so you always know whether the agent invoked RO vs RW:

```python
import time

def timed(fn, mode):  # mode ∈ {"RO","RW"}
    def wrap(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            dt = time.perf_counter() - t0
            metrics.conveyance.record(
                mode=mode,           # maps to R/W in your framework
                T=dt,                # latency
                W="mem:query" if mode=="RO" else "mem:write",
                R="arangodb",        # where/topology label
                H="memgpt-tool",     # actor/agent capability bucket
                ctx_size=kwargs.get("ctx_size", 0)  # optional
            )
    return wrap

query_memory   = timed(query_memory, "RO")
upsert_memory  = timed(upsert_memory, "RW")
```

This gives you crisp split charts (e.g., RO p95 vs RW p95), and keeps your **R/H/T** signals un-muddied by mixed calls.

---

# Ops checklist

* [ ] Two Arango users: `memgpt_ro` (RO), `memgpt_rw` (RW)
* [ ] Two sockets: `/run/hades/readonly/*.sock`, `/run/hades/readwrite/*.sock`
* [ ] Driver configs pinned to the correct socket per action
* [ ] CI check: grep for `ro.` usage in any write helpers; fail build if found
* [ ] Dashboards: RO vs RW latency, error rate, QPS; RW audit log (who/what)

---

If you want, I can generate:

* a tiny `systemd` unit pair to publish the two sockets,
* an ArangoDB bootstrap script that creates users, grants perms, and sanity-checks RO writes (should fail).

---

## Evaluation of current unix socket implementation 

Short answer: you nailed the split-socket design, and most of the safety rails I proposed are in place.

**What matches the idea**

* **Two explicit sockets** with correct paths and perms (RO `0660`, RW `0600`). 
* **Single upstream over Unix** with a custom transport (no TCP), HTTP/2 enabled, tunable timeouts. 
* **Boundary guards:**

  * RO: only GET/HEAD/OPTIONS, plus POST to `/_api/cursor` with an **AQL write-keyword scanner**; PUT/DELETE only for cursor endpoints. 
  * RW: allowlist on Arango write endpoints; falls back to RO checks first. 
* **Clean reverse proxying**: hop-by-hop header stripping, header cloning, body peek with size cap, cursor path regex covering optional db prefix. 
* **Separate entrypoints** for RO/RW binaries.  

**Gaps / quick wins**

1. **Expand RO denylist**: include `CREATE`, `ALTER`, `RENAME`, `GRANT`, `REVOKE`, `GRAPH`, `DROP COLLECTION/VIEW/INDEX` variants to harden against schema ops via AQL tokens that aren’t in the current set. (Add to both JSON-parsed and fallback string scan.) 
2. **Tighten RW allowlist**: consider requiring database-scoped paths (e.g., `/_db/{name}/_api/...`) to avoid accidental `_system` hits unless explicitly intended. 
3. **Audit tags for Conveyance**: wrap handlers to emit a minimal boundary metric `{mode: RO|RW, path, status, ms}` so you can chart RO p95 vs RW p95 without touching app code. Hook in `logRequests` or a small middleware. 
4. **Socket ownership**: if MemGPT runs under a different user, ensure group ownership on RO socket matches the MemGPT process group; RW should remain owner-only. (You already chmod; add chgrp in service/unit.) 
5. **Body peek cap**: 128 KiB is good; consider making it env-tunable for large AQLs (`AQL_PEEK_LIMIT_BYTES`) and explicitly 413 on over-limit instead of generic error. 

**Nice-to-have next**

* **Systemd units** for RO/RW with `RuntimeDirectory=` and `Socket` dependencies to guarantee perms & ownership on boot. (Maps to the design’s ops checklist.)
* **Per-mode access logs** (`RO:` / `RW:` prefixes) to simplify grep-based audits before you wire a metrics sink. 

Net: your proxy matches the proposed architecture and enforces the RO/RW safety boundary as intended. Add the extra RO keywords, tighten RW scoping, and drop in boundary metrics, and you’re production-ready for MemGPT’s memory tools.
