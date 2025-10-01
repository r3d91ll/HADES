# Read amplification in multi‑stage pipelines (and how to tame it)

When you chain stages—planner → retriever → graph hops → reranker → generator—each stage often **multiplies** the number of reads. In storage terms, this is **read amplification**: extra I/O beyond the single logical question you asked. Quality may go up (better candidates, cleaner ranking), but latency and cost rise fast.

## Why it happens (plain English)

* **Planner** emits multiple sub‑queries → each triggers retrieval.
* **Retriever** pulls k docs per query → downstream stages now see k×q items.
* **Graph hops** expand each doc via neighbors → k×q×h nodes/edges.
* **Reranker** must score them all → O(N) or worse if cross‑encoders.
* **Generator** now juggles a bloated context → longer prompt, slower decode.

## The cure: budgets + caps (set them per turn)

Think like a DB admin with a query planner:

* **Per‑turn hop/document budget (hard cap)**
  `N_total ≤ 200` (example). Everything below must respect this.
* **Per‑stage caps**

  * Planner: `q ≤ 3` sub‑queries
  * Retriever: `k ≤ 8` per sub‑query (use MMR/ANN to diversify)
  * Graph: `hops ≤ 1` (default), `fanout ≤ 3`, dedupe aggressively
  * Reranker: `top_m ≤ 24` cross‑encode; never rerank the full set
  * Generator: `final_context ≤ 4–8` chunks, **compress** the rest
* **Early stopping**
  If confidence/coverage is met at an earlier stage, **skip** the rest.
* **Dedupe & coalesce**
  Normalize IDs, merge near‑duplicates before scoring.
* **Measure and enforce**
  Track `reads_stage_i`, `latency_stage_i`, and reject over‑budget plans.

## A minimal “query plan” contract

Have your planner emit a plan object the executor must obey:

```json
{
  "q": 2,
  "retriever": {"k": 6, "mmr": 0.4},
  "graph": {"hops": 1, "fanout": 2},
  "reranker": {"m": 20, "model": "cross-encoder"},
  "final_context": 6,
  "budget": {"docs_max": 160, "latency_ms": 1800}
}
```

## Back‑of‑napkin math (keep yourself honest)

* **Total candidates after retrieval**: `N0 = q × k`
* **After graph**: `N1 ≈ min(N0 × fanout, budget.docs_max)` (dedupe!)
* **Rerank set**: `N2 = min(N1, m)`
* **Final context**: `N_out = min(N2, final_context)`

If `N1` blows past the budget, **shrink fanout** or **drop graph** this turn.

## Practical defaults for your HADES/ArangoDB stack

* Planner: 1–3 sub‑queries max
* Vector retriever (ModernBERT/Jina v4): k=6–8, MMR on
* Graph step (ArangoDB): 1 hop, fanout ≤2–3, edge‑type whitelist
* Cross‑encoder rerank: m=16–24 (batch), timeout 300–500 ms
* Final pack: 4–8 chunks (compress extras to notes)
* Guardrails: hard doc cap 150–200; per‑stage timeouts; global 2 s target

## Latency levers (fastest wins first)

* **Cap early** (retriever) > **cap late** (reranker)
* Prefer **bi‑encoder rerank light** before **cross‑encoder heavy**
* **Cache** everything (sub‑queries, neighbor sets, rerank scores)
* **Sketchy networks, strict caps**: reduce k before touching hops

## Observability (what to log)

* `plan.budget`, `reads_per_stage`, `latency_per_stage`, `dedupe_rate`
* `coverage` (topic/entity hit rate), `answerable?` flag
* Drop/keep decisions with reason codes (e.g., `cap_exceeded`, `low_gain`)

## Quick checklist

* [ ] Enforce a **per‑turn doc budget** (e.g., 160)
* [ ] Cap **q**, **k**, **hops**, **fanout**, **m**, **final_context**
* [ ] Dedupe before reranking
* [ ] Early‑stop when coverage is good
* [ ] Log reads/latency by stage; block plans that exceed budget
* [ ] Cache candidate sets and rerank scores

## Code example
```Python
from typing import List, Dict, Any
import time

class PlanExecutor:
    def __init__(self, budget_docs: int = 200, budget_latency_ms: int = 2000):
        self.budget_docs = budget_docs
        self.budget_latency_ms = budget_latency_ms

    def execute_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        start = time.time()
        reads_stage = {}

        # Stage 1: Planner
        q = plan.get("q", 1)
        reads_stage["planner"] = q

        # Stage 2: Retriever
        k = plan.get("retriever", {}).get("k", 6)
        n0 = q * k
        reads_stage["retriever"] = n0

        # Stage 3: Graph expansion
        hops = plan.get("graph", {}).get("hops", 1)
        fanout = plan.get("graph", {}).get("fanout", 2)
        n1 = min(n0 * (fanout ** hops), self.budget_docs)
        reads_stage["graph"] = n1

        # Stage 4: Reranker
        m = plan.get("reranker", {}).get("m", 20)
        n2 = min(n1, m)
        reads_stage["reranker"] = n2

        # Stage 5: Final context
        final_context = plan.get("final_context", 6)
        n_out = min(n2, final_context)
        reads_stage["generator"] = n_out

        latency = int((time.time() - start) * 1000)

        # Enforce budgets
        over_docs = n1 > self.budget_docs
        over_latency = latency > self.budget_latency_ms

        return {
            "reads_stage": reads_stage,
            "n_final": n_out,
            "latency_ms": latency,
            "over_docs_budget": over_docs,
            "over_latency_budget": over_latency,
            "plan_budget": plan.get("budget", {})
        }


if __name__ == "__main__":
    sample_plan = {
        "q": 2,
        "retriever": {"k": 8},
        "graph": {"hops": 1, "fanout": 3},
        "reranker": {"m": 24},
        "final_context": 6,
        "budget": {"docs_max": 160, "latency_ms": 1800}
    }

    executor = PlanExecutor(budget_docs=200, budget_latency_ms=2000)
    report = executor.execute_plan(sample_plan)
    print(report)
```
