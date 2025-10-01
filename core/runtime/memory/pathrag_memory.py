"""
PathRAG for Memory Graph - Graph Traversal over Experiential Memories

WHAT: PathRAG adaptation for observations/reflections/entities graph
WHERE: core/runtime/memory/pathrag_memory.py - retrieval layer
WHO: Memory operations requiring graph-augmented semantic search
TIME: p99 <500ms for graph traversal (within 2s global SLO)

Extends core PathRAG with memory-specific graph patterns:
- Vector seeds → entity expansion → related memories
- Observation → reflection consolidation links
- Entity → relationship → related entities
- Time-aware traversal with recency boosting

Boundary Notes:
- Enforces caps (H≤2, F≤3, B≤8) per PathRAG PRD
- Emits reason codes (cap_exceeded, timeout, coverage_reached)
- RO socket access only
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from time import monotonic
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

import numpy as np

from ...database.arango.memory_client import ArangoMemoryClient

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class MemoryPathCaps:
    """PathRAG caps for memory graph traversal."""

    hops: int = 2  # Max traversal depth
    fanout: int = 3  # Max neighbors per node
    beam: int = 8  # Global beam width
    timeout_ms: int = 500  # Per-stage timeout
    mmr_lambda: float = 0.4  # MMR diversity parameter (0=diversity, 1=relevance)


@dataclass(slots=True)
class MemoryPathNode:
    """Node in memory graph path."""

    doc_id: str  # Full _id like "observations/obs_123"
    collection: str  # observations|reflections|entities
    content: str  # Text content
    embedding: Optional[List[float]] = None
    score: float = 0.0  # Path score at this node
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MemoryPath:
    """Complete path through memory graph."""

    nodes: List[MemoryPathNode]
    edges: List[Dict[str, Any]]  # Edge documents
    final_score: float
    reasoning: List[str]  # Why this path was selected/pruned


@dataclass
class MemoryPathResult:
    """Result from PathRAG memory traversal."""

    paths: List[MemoryPath]
    total_nodes: int
    total_edges: int
    reasons: List[str]  # Global reason codes
    latency_ms: float


def _time_exceeded(start: float, timeout_ms: int) -> bool:
    """Check if time budget exceeded."""
    return (monotonic() - start) * 1000.0 >= timeout_ms


def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not vec1 or not vec2:
        return 0.0
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float(dot / norm) if norm > 0 else 0.0


def _mmr_score(
    candidate_emb: List[float],
    query_emb: List[float],
    selected_embs: List[List[float]],
    lambda_param: float = 0.4,
) -> float:
    """
    Compute MMR (Maximal Marginal Relevance) score for diversity.

    MMR = λ * sim(candidate, query) - (1-λ) * max(sim(candidate, selected))
    """
    relevance = _cosine_similarity(candidate_emb, query_emb)

    if not selected_embs:
        return relevance

    max_similarity = max(_cosine_similarity(candidate_emb, sel) for sel in selected_embs)
    return lambda_param * relevance - (1 - lambda_param) * max_similarity


class MemoryPathRAG:
    """
    PathRAG traversal over experiential memory graph.

    Supports multiple traversal patterns:
    1. Observation → Entity → Related Observations
    2. Observation → Reflection (consolidation links)
    3. Entity → Relationship → Related Entities
    4. Time-aware filtering and recency boosting
    """

    def __init__(self, client: ArangoMemoryClient):
        self.client = client

    def find_memory_paths(
        self,
        *,
        seed_nodes: List[MemoryPathNode],
        query_embedding: Optional[List[float]] = None,
        caps: MemoryPathCaps,
        collections: List[str] = ["observations", "reflections", "entities"],
        edge_collections: List[str] = ["relationships"],
        time_filter_unix: Optional[float] = None,
    ) -> MemoryPathResult:
        """
        Execute PathRAG traversal from seed nodes.

        Args:
            seed_nodes: Initial vector search results
            query_embedding: Query vector for relevance scoring
            caps: Traversal caps (H, F, B, timeout)
            collections: Document collections to traverse
            edge_collections: Edge collections to follow
            time_filter_unix: Filter memories after this timestamp

        Returns:
            MemoryPathResult with paths, nodes, edges, and reason codes
        """
        start_time = monotonic()
        reasons: List[str] = []
        all_paths: List[MemoryPath] = []

        if not seed_nodes:
            return MemoryPathResult(
                paths=[], total_nodes=0, total_edges=0, reasons=["empty_seeds"], latency_ms=0.0
            )

        # Initialize beam with seed nodes as single-node paths
        beam: List[Tuple[float, MemoryPath]] = []
        for seed in seed_nodes:
            path = MemoryPath(
                nodes=[seed], edges=[], final_score=seed.score, reasoning=["seed"]
            )
            beam.append((seed.score, path))

        # Sort by score and apply beam cap
        beam.sort(key=lambda x: x[0], reverse=True)
        if len(beam) > caps.beam:
            reasons.append("cap_exceeded_initial")
            beam = beam[: caps.beam]

        # Multi-hop expansion
        for hop in range(caps.hops):
            if _time_exceeded(start_time, caps.timeout_ms * (hop + 1)):
                reasons.append(f"timeout_hop_{hop}")
                break

            # Expand each path in beam
            new_beam: List[Tuple[float, MemoryPath]] = []
            selected_embeddings: List[List[float]] = []

            for score, path in beam:
                # Get last node in path
                current_node = path.nodes[-1]

                # Expand neighbors
                neighbors = self._get_neighbors(
                    node_id=current_node.doc_id,
                    collections=collections,
                    edge_collections=edge_collections,
                    fanout=caps.fanout,
                    time_filter_unix=time_filter_unix,
                )

                if not neighbors:
                    # Dead end - keep path as-is
                    new_beam.append((score, path))
                    continue

                # Score neighbors with MMR for diversity
                for neighbor_node, edge in neighbors:
                    # Skip if already in path (no cycles)
                    if any(n.doc_id == neighbor_node.doc_id for n in path.nodes):
                        continue

                    # Compute new path score
                    if query_embedding and neighbor_node.embedding:
                        mmr = _mmr_score(
                            neighbor_node.embedding,
                            query_embedding,
                            selected_embeddings,
                            caps.mmr_lambda,
                        )
                    else:
                        mmr = 0.5  # Neutral score if no embedding

                    # Recency boost for recent observations
                    recency_boost = self._compute_recency_boost(neighbor_node)

                    # Composite score: path_score + neighbor_mmr + recency
                    new_score = score * 0.6 + mmr * 0.3 + recency_boost * 0.1

                    # Create extended path
                    new_path = MemoryPath(
                        nodes=path.nodes + [neighbor_node],
                        edges=path.edges + [edge],
                        final_score=new_score,
                        reasoning=path.reasoning + [f"hop_{hop+1}"],
                    )
                    new_beam.append((new_score, new_path))

                    # Track selected embeddings for MMR
                    if neighbor_node.embedding:
                        selected_embeddings.append(neighbor_node.embedding)

            # Apply beam cap and diversity
            new_beam.sort(key=lambda x: x[0], reverse=True)
            if len(new_beam) > caps.beam:
                reasons.append(f"cap_exceeded_hop_{hop}")
                new_beam = new_beam[: caps.beam]

            # Check for early stopping (coverage reached)
            if not new_beam or all(score < 0.3 for score, _ in new_beam):
                reasons.append("coverage_reached")
                break

            beam = new_beam

        # Extract final paths
        all_paths = [path for _, path in beam]
        total_nodes = sum(len(p.nodes) for p in all_paths)
        total_edges = sum(len(p.edges) for p in all_paths)
        latency_ms = (monotonic() - start_time) * 1000.0

        logger.info(
            f"PathRAG memory traversal: {len(all_paths)} paths, "
            f"{total_nodes} nodes, {total_edges} edges, {latency_ms:.1f}ms"
        )

        return MemoryPathResult(
            paths=all_paths,
            total_nodes=total_nodes,
            total_edges=total_edges,
            reasons=reasons,
            latency_ms=latency_ms,
        )

    def _get_neighbors(
        self,
        node_id: str,
        collections: List[str],
        edge_collections: List[str],
        fanout: int,
        time_filter_unix: Optional[float] = None,
    ) -> List[Tuple[MemoryPathNode, Dict[str, Any]]]:
        """
        Get neighbors of a node via edge collections.

        Traversal patterns:
        - observations -> entities (via mentions in metadata)
        - entities -> observations (inbound mentions)
        - entities -> entities (via relationships edge collection)
        - observations -> reflections (via consolidates array)
        """
        neighbors: List[Tuple[MemoryPathNode, Dict[str, Any]]] = []

        # Time filter clause
        time_clause = (
            f"FILTER neighbor.created_at_unix >= {time_filter_unix}" if time_filter_unix else ""
        )

        # Pattern 1: Follow explicit edges (relationships collection)
        if "relationships" in edge_collections:
            aql = f"""
            FOR edge IN relationships
                FILTER edge._from == @node_id
                FILTER edge.status == 'active'
                LET neighbor = DOCUMENT(edge._to)
                FILTER neighbor != null
                {time_clause}
                LIMIT @fanout
                RETURN {{neighbor: neighbor, edge: edge}}
            """
            try:
                results = self.client.execute_query(aql, {"node_id": node_id, "fanout": fanout})
                for row in results:
                    neighbor_doc = row["neighbor"]
                    edge_doc = row["edge"]
                    node = self._doc_to_path_node(neighbor_doc)
                    if node:
                        neighbors.append((node, edge_doc))
            except Exception as e:
                logger.warning(f"Failed to traverse relationships from {node_id}: {e}")

        # Pattern 2: Observations -> Reflections (consolidates backlink)
        if node_id.startswith("observations/"):
            aql = f"""
            FOR refl IN reflections
                FILTER @node_id IN refl.consolidates
                FILTER refl.status == 'active'
                {time_clause}
                LIMIT @fanout
                RETURN {{neighbor: refl, edge: {{_from: @node_id, _to: refl._id, type: 'consolidates'}}}}
            """
            try:
                results = self.client.execute_query(aql, {"node_id": node_id, "fanout": fanout})
                for row in results:
                    neighbor_doc = row["neighbor"]
                    edge_doc = row["edge"]
                    node = self._doc_to_path_node(neighbor_doc)
                    if node:
                        neighbors.append((node, edge_doc))
            except Exception as e:
                logger.warning(f"Failed to find consolidating reflections for {node_id}: {e}")

        return neighbors[:fanout]

    def _doc_to_path_node(self, doc: Dict[str, Any]) -> Optional[MemoryPathNode]:
        """Convert ArangoDB document to MemoryPathNode."""
        try:
            doc_id = doc.get("_id")
            if not doc_id:
                return None

            collection = doc_id.split("/")[0]
            content = doc.get("content") or doc.get("name") or doc.get("description", "")
            embedding = doc.get("embedding")

            return MemoryPathNode(
                doc_id=doc_id,
                collection=collection,
                content=content,
                embedding=embedding,
                score=0.0,
                metadata={
                    "created_at_unix": doc.get("created_at_unix"),
                    "tags": doc.get("tags", []),
                    "memory_type": doc.get("memory_type"),
                    "entity_type": doc.get("entity_type"),
                },
            )
        except Exception as e:
            logger.warning(f"Failed to convert doc to path node: {e}")
            return None

    def _compute_recency_boost(self, node: MemoryPathNode) -> float:
        """
        Compute recency boost for time-aware ranking.

        Recent observations get higher scores (exponential decay).
        """
        created_at = node.metadata.get("created_at_unix")
        if not created_at:
            return 0.0

        # Time since creation in days
        now = monotonic()
        age_days = (now - created_at) / 86400.0

        # Exponential decay: boost = exp(-age_days / 30)
        # Recent (< 1 day): ~0.97
        # Week old: ~0.78
        # Month old: ~0.37
        import math

        return math.exp(-age_days / 30.0)


__all__ = ["MemoryPathRAG", "MemoryPathCaps", "MemoryPathNode", "MemoryPath", "MemoryPathResult"]