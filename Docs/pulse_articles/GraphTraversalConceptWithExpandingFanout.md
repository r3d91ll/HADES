# ![Graph traversal concept with expanding fanout]

### Why traversals “blow up”

Graph queries often bottleneck on **fanout** (how many neighbors each hop touches), not your initial vector top‑k. Unbounded multi‑hop traversals expand combinatorially, causing latency spikes and RAM churn.

### Core constraints that tame fanout

* **Direction**: traverse `OUTBOUND`/`INBOUND` when your edge semantics allow it; avoid `ANY` unless necessary.
* **Depth**: set tight `min..max` hop ranges (e.g., `1..2`, not `1..5`).
* **Filters early**: restrict vertices/edges during expansion (labels, types, timestamps, scores).
* **PRUNE**: cut branches as soon as they can’t lead to valid results.
* **indexHint**: nudge the optimizer toward the right edge/vertex index for high‑cardinality filters.
* **Limit branching**: cap per‑level expansions (e.g., keep top‑k neighbors by weight).
* **De‑duplicate**: avoid revisiting nodes; prefer acyclic paths where you can.
* **Pre‑stage candidates**: run vector search first, then traverse only from those seeds.

### AQL patterns (copy‑paste friendly)

**1) Bounded, directional, filtered traversal**

```aql
FOR v, e, p IN 1..2 OUTBOUND @seed GRAPH "codegraph"
  OPTIONS { uniqueVertices: "path" }
  FILTER v.type IN ["Function","Class"]
  FILTER e.kind == "calls"
  LIMIT 200
  RETURN { id: v._id, hop: LENGTH(p.vertices) - 1 }
```

**2) Prune early to stop useless branches**

```aql
FOR v, e, p IN 1..3 OUTBOUND @seed GRAPH "codegraph"
  PRUNE v.lang != "python" OR v.lastSeen < @since
  FILTER v.lang == "python"
  RETURN v
```

**3) Keep top‑k per hop to control branching**

```aql
FOR v1, e1 IN 1 OUTBOUND @seed Edges
  SORT e1.weight DESC
  LIMIT 10
  FOR v2, e2 IN 1 OUTBOUND v1 Edges
    SORT e2.weight DESC
    LIMIT 5
    RETURN DISTINCT v2
```

**4) Hint the index the edge scan should use**

```aql
FOR v, e IN 1..2 OUTBOUND @seed Edges
  OPTIONS { indexHint: "by_kind_weight" }
  FILTER e.kind == "references" AND e.weight >= 0.7
  RETURN v
```

**5) Seed from vector search, then traverse**

```aql
LET seeds = (
  FOR d IN embeddings
    SEARCH ANALYZER(SIMILARITY(d.vec, @qvec) > 0.75, "vector")
    SORT SIMILARITY(d.vec, @qvec) DESC
    LIMIT 20
    RETURN d._id
)

FOR s IN seeds
  FOR v, e IN 1..2 OUTBOUND s Edges
    PRUNE v.score < 0.6
    FILTER v.domain == @domain
    RETURN DISTINCT v
```

### Quick checklist

* Set **direction** and **max depth**.
* Add **PRUNE** conditions aligned to your goal (lang/type/time/score).
* **Sort + LIMIT** neighbors per hop by a relevance signal (e.g., edge weight).
* Use **indexHint** when filters hit large edge sets.
* Start from **small seed sets** (vector/rule‑based prefilter).

If you want, tell me your edge names, indexes, and the traversal objective (e.g., “functions likely implementing section §2.3”), and I’ll draft the exact AQL tailored for HADES.
