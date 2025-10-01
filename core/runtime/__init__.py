"""
Runtime Orchestration Module

WHAT: Runtime subsystem for HADES experiential memory and retrieval operations
WHERE: core/runtime/ - orchestration layer above database/extractors/embedders
WHO: Agents with experiential memory and PathRAG retrieval systems
TIME: Query-time orchestration with SLOs (recall p99 ≤250ms, vector p99 ≤750ms)

Provides the execution layer for experiential memory operations (observations,
reflections, entities), retrieval pipelines, and agent interactions. All
measurements follow Conveyance Framework principles - metrics collected only
at user↔agent boundaries.

Memory Architecture:
- observations: Raw experiences/events the agent witnesses
- reflections: Higher-order insights derived from observations
- entities: People/concepts/components extracted from experiences
- relationships: Graph edges connecting entities and memories

Boundary Notes:
- User-agent interactions tracked for conveyance metrics
- Internal operations (DB queries, embeddings) not counted in boundary metrics
- Context amplification α learned online (target CI: [1.5, 2.0])
"""

__all__ = ["memory"]
