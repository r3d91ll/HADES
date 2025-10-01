"""
PathRAG for Code Graph - Heuristic v1
======================================

WHAT: PathRAG beam search for code repository graph traversal
WHERE: core/runtime/memory/pathrag_code.py - retrieval layer
WHO: Code retrieval requiring graph-augmented semantic search
TIME: p95 <2s global SLO (vector search + graph + rerank + pack)

Graph structure:
- repo_docs: Code and text files with embeddings
- code_edges: imports, references between code files
- contains_edges: directories → files hierarchy
- directories: Directory nodes

Scoring v1 (Heuristic):
  score(path) = α·seed_sim + β·edge_weight + γ·lex_match − δ·depth_penalty − ρ·redundancy

No ML training required - uses static weights and keyword matching.

Boundary Notes:
- Enforces caps (H≤2, F≤3, B≤8-16) per PathRAG PRD
- Emits reason codes (cap_exceeded, timeout, coverage_reached)
- RO socket access only
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from time import monotonic
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

import numpy as np

from ...database.arango.memory_client import ArangoMemoryClient

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PathRAGCaps:
    """PathRAG caps for code graph traversal."""

    hops: int = 2  # Max traversal depth (H)
    fanout: int = 3  # Max neighbors per node (F)
    beam: int = 8  # Global beam width (B)
    timeout_ms: int = 500  # Per-stage timeout
    mmr_lambda: float = 0.4  # MMR diversity (0=diversity, 1=relevance)

    # Heuristic scoring weights
    alpha: float = 0.6  # seed_sim weight
    beta: float = 0.3   # edge_weight
    gamma: float = 0.1  # lex_match
    delta: float = 0.05  # depth_penalty per hop
    rho: float = 0.2    # redundancy penalty


@dataclass(slots=True)
class PathNode:
    """Node in code graph path."""

    doc_id: str  # Full _id like "repo_docs/core_embedders_jina.py"
    path: str  # File path like "core/embedders/embedders_jina.py"
    content: str  # Text content
    embedding: Optional[List[float]] = None
    score: float = 0.0  # Cumulative path score
    depth: int = 0  # Depth in path (0 = seed)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Path:
    """Complete path through code graph."""

    nodes: List[PathNode]
    edges: List[Dict[str, Any]]  # Edge documents
    final_score: float
    reasoning: List[str]  # Why this path was selected/pruned

    def __repr__(self) -> str:
        node_paths = " → ".join(n.path for n in self.nodes)
        return f"Path(score={self.final_score:.3f}, {node_paths})"


@dataclass
class PathRAGResult:
    """Result from PathRAG code graph traversal."""

    paths: List[Path]
    total_nodes: int
    total_edges: int
    reasons: List[str]  # Global reason codes
    latency_ms: float

    def get_unique_documents(self) -> List[PathNode]:
        """Get all unique documents from paths (deduplicated)."""
        seen = set()
        docs = []
        for path in self.paths:
            for node in path.nodes:
                if node.doc_id not in seen:
                    seen.add(node.doc_id)
                    docs.append(node)
        return docs


# ============================================================================
# Utility Functions
# ============================================================================

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
    Maximal Marginal Relevance score for diversity.

    MMR = λ·sim(candidate, query) − (1−λ)·max_sim(candidate, selected)
    """
    if not candidate_emb or not query_emb:
        return 0.0

    relevance = _cosine_similarity(candidate_emb, query_emb)

    if not selected_embs:
        return lambda_param * relevance

    # Max similarity to already selected items
    max_sim = max(_cosine_similarity(candidate_emb, sel) for sel in selected_embs)

    return lambda_param * relevance - (1 - lambda_param) * max_sim


def _keyword_overlap(query: str, text: str) -> float:
    """
    Simple keyword overlap score (BM25-lite).

    Returns: fraction of query keywords found in text
    """
    # Extract keywords (simple: lowercase, alphanumeric, length > 2)
    def extract_keywords(s: str) -> Set[str]:
        words = re.findall(r'\b[a-z]{3,}\b', s.lower())
        # Remove common stop words
        stopwords = {'the', 'and', 'for', 'with', 'from', 'that', 'this', 'have'}
        return set(w for w in words if w not in stopwords)

    query_kw = extract_keywords(query)
    text_kw = extract_keywords(text)

    if not query_kw:
        return 0.0

    overlap = len(query_kw & text_kw)
    return overlap / len(query_kw)


# ============================================================================
# PathRAG Implementation
# ============================================================================

class CodePathRAG:
    """
    PathRAG implementation for code graph retrieval.

    Pipeline (with GraphSAGE):
    1. GraphSAGE finds semantically relevant nodes (may be many)
    2. PathRAG prunes to paths within budget (H≤2, F≤3, B≤8)

    Pipeline (without GraphSAGE - fallback):
    1. Vector search finds seed documents
    2. PathRAG beam search expands paths
    """

    def __init__(
        self,
        client: ArangoMemoryClient,
        caps: Optional[PathRAGCaps] = None,
        graphsage_inference=None,  # Optional GraphSAGEInference instance
    ):
        """
        Initialize CodePathRAG.

        Args:
            client: ArangoDB memory client (read-only)
            caps: PathRAG caps (uses defaults if None)
            graphsage_inference: Optional GraphSAGE inference engine for GNN→PathRAG pipeline
        """
        self.client = client
        self.caps = caps or PathRAGCaps()
        self.graphsage_inference = graphsage_inference

    def retrieve(
        self,
        query: str,
        query_embedding: List[float],
        k_seeds: int = 6,
        edge_types: Optional[List[str]] = None
    ) -> PathRAGResult:
        """
        PathRAG retrieval with optional GraphSAGE.

        Pipeline (with GraphSAGE):
        1. GraphSAGE finds top-k candidate nodes (may be many, k=50)
        2. PathRAG prunes to paths within budget (H≤2, F≤3, B≤8)

        Pipeline (without GraphSAGE - fallback):
        1. Vector search finds seed documents (k=6)
        2. PathRAG beam search expands paths

        Args:
            query: Query text
            query_embedding: Query embedding vector (2048-dim Jina v4)
            k_seeds: Number of seed documents (used for vector search OR GraphSAGE pruning)
            edge_types: Edge types to traverse (None = all)

        Returns:
            PathRAGResult with ranked paths
        """
        start_time = monotonic()
        reasons = []

        # Step 1: Get candidate nodes
        if self.graphsage_inference:
            # GraphSAGE pipeline: find candidates (may be many)
            import numpy as np
            query_emb_np = np.array(query_embedding, dtype=np.float32)

            try:
                # Get top-50 candidates from GraphSAGE (PathRAG will prune)
                candidates = self.graphsage_inference.find_relevant_nodes(
                    query_embedding=query_emb_np,
                    top_k=50,  # More candidates for PathRAG to prune
                    min_score=0.3,  # Lower threshold, PathRAG handles quality
                )

                if not candidates:
                    logger.warning("GraphSAGE found no candidates, falling back to vector search")
                    seeds = self._vector_search_seeds(query_embedding, k_seeds)
                else:
                    # Convert GraphSAGE candidates to seed format
                    seeds = self._graphsage_candidates_to_seeds(candidates, k_seeds)
                    logger.info(f"PathRAG (GraphSAGE): Found {len(seeds)} candidate seeds from {len(candidates)} GNN results")
                    reasons.append("graphsage_candidates")
            except Exception as exc:
                logger.warning(f"GraphSAGE inference failed: {exc}, falling back to vector search")
                seeds = self._vector_search_seeds(query_embedding, k_seeds)
                reasons.append("graphsage_failed_fallback")
        else:
            # Fallback: Vector search for seeds
            seeds = self._vector_search_seeds(query_embedding, k_seeds)

        if not seeds:
            reasons.append("no_seeds_found")
            return PathRAGResult(
                paths=[],
                total_nodes=0,
                total_edges=0,
                reasons=reasons,
                latency_ms=(monotonic() - start_time) * 1000.0
            )

        if "graphsage_candidates" not in reasons:
            logger.info(f"PathRAG (vector search): Found {len(seeds)} seed documents")

        # Step 2: Beam search expansion
        paths, nodes_visited, edges_traversed = self._beam_search(
            seeds=seeds,
            query=query,
            query_embedding=query_embedding,
            edge_types=edge_types,
            start_time=start_time
        )

        if not paths:
            reasons.append("no_paths_found")

        if _time_exceeded(start_time, self.caps.timeout_ms):
            reasons.append("timeout")

        latency_ms = (monotonic() - start_time) * 1000.0

        logger.info(
            f"PathRAG complete: {len(paths)} paths, "
            f"{nodes_visited} nodes, {edges_traversed} edges, "
            f"{latency_ms:.1f}ms"
        )

        return PathRAGResult(
            paths=paths,
            total_nodes=nodes_visited,
            total_edges=edges_traversed,
            reasons=reasons,
            latency_ms=latency_ms
        )

    def _graphsage_candidates_to_seeds(
        self,
        candidates: List[Tuple[str, float]],
        top_k: int
    ) -> List[PathNode]:
        """
        Convert GraphSAGE candidates to PathNode seeds for beam search.

        Args:
            candidates: List of (node_id, score) from GraphSAGE
            top_k: Number of top candidates to convert to seeds

        Returns:
            List of PathNode seeds for beam search
        """
        # Take top-k candidates
        top_candidates = candidates[:top_k]

        seeds = []
        for node_id, score in top_candidates:
            # Fetch document from ArangoDB
            aql = """
                FOR doc IN repo_docs
                    FILTER doc._id == @doc_id
                    RETURN {
                        _id: doc._id,
                        path: doc.path,
                        text: doc.text,
                        embedding: doc.embedding,
                        metadata: doc.metadata
                    }
            """

            try:
                results = list(self.client.execute_query(aql, {"doc_id": node_id}))
                if results:
                    doc = results[0]
                    seeds.append(PathNode(
                        doc_id=doc["_id"],
                        path=doc.get("path", ""),
                        content=doc.get("text", "")[:1000],  # Truncate for memory
                        embedding=doc.get("embedding"),
                        score=float(score),  # GraphSAGE similarity score
                        depth=0,
                        metadata=doc.get("metadata", {}),
                    ))
            except Exception as exc:
                logger.warning(f"Failed to fetch GraphSAGE candidate {node_id}: {exc}")
                continue

        return seeds

    def _vector_search_seeds(
        self,
        query_embedding: List[float],
        k: int
    ) -> List[PathNode]:
        """
        Vector search to find seed documents.

        Uses cosine similarity over repo_docs embeddings.
        """
        # Get all documents with embeddings
        query_aql = """
            FOR doc IN repo_docs
                FILTER doc.embedding != null
                LET similarity = COSINE_SIMILARITY(doc.embedding, @query_emb)
                SORT similarity DESC
                LIMIT @k
                RETURN {
                    _id: doc._id,
                    path: doc.path,
                    text: doc.text,
                    embedding: doc.embedding,
                    score: similarity,
                    metadata: doc.metadata
                }
        """

        try:
            results = self.client.execute_query(
                query_aql,
                bind_vars={"query_emb": query_embedding, "k": k}
            )

            seeds = []
            for r in results:
                seeds.append(PathNode(
                    doc_id=r["_id"],
                    path=r["path"],
                    content=r["text"][:1000],  # Truncate for memory
                    embedding=r["embedding"],
                    score=r["score"],
                    depth=0,
                    metadata=r.get("metadata", {})
                ))

            return seeds

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    def _beam_search(
        self,
        seeds: List[PathNode],
        query: str,
        query_embedding: List[float],
        edge_types: Optional[List[str]],
        start_time: float
    ) -> Tuple[List[Path], int, int]:
        """
        Beam search expansion from seeds.

        Returns:
            (paths, nodes_visited, edges_traversed)
        """
        # Initialize beam with seed paths
        beam: List[Path] = [
            Path(nodes=[seed], edges=[], final_score=seed.score, reasoning=[])
            for seed in seeds
        ]

        nodes_visited = len(seeds)
        edges_traversed = 0

        # Expand beam for H hops
        for hop in range(self.caps.hops):
            if _time_exceeded(start_time, self.caps.timeout_ms):
                for path in beam:
                    path.reasoning.append(f"timeout_at_hop_{hop}")
                break

            # Expand all paths in current beam
            expanded_paths = []

            for path in beam:
                if _time_exceeded(start_time, self.caps.timeout_ms):
                    break

                # Get last node in path
                current_node = path.nodes[-1]

                # Find neighbors via edges
                neighbors = self._get_neighbors(
                    current_node.doc_id,
                    edge_types,
                    limit=self.caps.fanout
                )

                edges_traversed += len(neighbors)

                if not neighbors:
                    # Dead end - keep current path
                    expanded_paths.append(path)
                    continue

                # Create new paths by extending with each neighbor
                for neighbor_node, edge in neighbors:
                    nodes_visited += 1

                    # Calculate heuristic score for this extension
                    extension_score = self._score_path_extension(
                        path=path,
                        neighbor=neighbor_node,
                        edge=edge,
                        query=query,
                        query_embedding=query_embedding,
                        hop=hop + 1
                    )

                    new_path = Path(
                        nodes=path.nodes + [neighbor_node],
                        edges=path.edges + [edge],
                        final_score=extension_score,
                        reasoning=path.reasoning.copy()
                    )

                    expanded_paths.append(new_path)

            # Prune to beam width using MMR
            if len(expanded_paths) > self.caps.beam:
                beam = self._select_diverse_paths(
                    expanded_paths,
                    query_embedding,
                    self.caps.beam
                )

                # Mark pruned paths
                for path in expanded_paths:
                    if path not in beam:
                        path.reasoning.append(f"pruned_at_hop_{hop}")
            else:
                beam = expanded_paths

        # Final sort by score
        beam.sort(key=lambda p: p.final_score, reverse=True)

        return beam, nodes_visited, edges_traversed

    def _get_neighbors(
        self,
        doc_id: str,
        edge_types: Optional[List[str]],
        limit: int
    ) -> List[Tuple[PathNode, Dict[str, Any]]]:
        """
        Get neighbor nodes via edges.

        Returns:
            List of (neighbor_node, edge_doc) tuples
        """
        # Build edge type filter
        if edge_types:
            type_filter = "FILTER e.type IN @edge_types"
            bind_vars = {"start": doc_id, "edge_types": edge_types, "limit": limit}
        else:
            type_filter = ""
            bind_vars = {"start": doc_id, "limit": limit}

        query_aql = f"""
            FOR v, e IN 1..1 OUTBOUND @start code_edges, contains_edges
                {type_filter}
                LIMIT @limit
                RETURN {{
                    vertex: v,
                    edge: e
                }}
        """

        try:
            results = self.client.execute_query(query_aql, bind_vars=bind_vars)

            neighbors = []
            for r in results:
                v = r["vertex"]
                e = r["edge"]

                node = PathNode(
                    doc_id=v["_id"],
                    path=v.get("path", "unknown"),
                    content=v.get("text", "")[:1000],
                    embedding=v.get("embedding"),
                    score=0.0,  # Will be calculated
                    depth=0,  # Will be updated
                    metadata=v.get("metadata", {})
                )

                neighbors.append((node, e))

            return neighbors

        except Exception as e:
            logger.error(f"Neighbor query failed: {e}")
            return []

    def _score_path_extension(
        self,
        path: Path,
        neighbor: PathNode,
        edge: Dict[str, Any],
        query: str,
        query_embedding: List[float],
        hop: int
    ) -> float:
        """
        Score a path extension using heuristic v1.

        score = α·seed_sim + β·edge_weight + γ·lex_match − δ·depth_penalty − ρ·redundancy
        """
        # α·seed_sim: Original seed similarity (from first node)
        seed_sim = path.nodes[0].score

        # β·edge_weight: Edge weight from graph
        edge_weight = edge.get("weight", 0.5)

        # γ·lex_match: Keyword overlap with query
        lex_match = _keyword_overlap(query, neighbor.content)

        # δ·depth_penalty: Penalize deeper paths
        depth_penalty = hop * self.caps.delta

        # ρ·redundancy: MMR penalty vs already selected nodes
        selected_embs = [n.embedding for n in path.nodes if n.embedding]
        if neighbor.embedding and selected_embs:
            max_sim = max(
                _cosine_similarity(neighbor.embedding, emb)
                for emb in selected_embs
            )
            redundancy = max_sim * self.caps.rho
        else:
            redundancy = 0.0

        # Combine scores
        score = (
            self.caps.alpha * seed_sim +
            self.caps.beta * edge_weight +
            self.caps.gamma * lex_match -
            depth_penalty -
            redundancy
        )

        return max(0.0, score)  # Ensure non-negative

    def _select_diverse_paths(
        self,
        paths: List[Path],
        query_embedding: List[float],
        beam_size: int
    ) -> List[Path]:
        """
        Select diverse paths using MMR.

        Balances relevance (score) with diversity (different endpoints).
        """
        if len(paths) <= beam_size:
            return paths

        selected = []
        remaining = paths.copy()

        # Always take top-scored path first
        remaining.sort(key=lambda p: p.final_score, reverse=True)
        selected.append(remaining.pop(0))

        # Select remaining using MMR
        while len(selected) < beam_size and remaining:
            best_path = None
            best_mmr = -float('inf')

            # Get embeddings of selected path endpoints
            selected_embs = [
                p.nodes[-1].embedding
                for p in selected
                if p.nodes[-1].embedding
            ]

            for path in remaining:
                endpoint = path.nodes[-1]
                if endpoint.embedding:
                    mmr = _mmr_score(
                        endpoint.embedding,
                        query_embedding,
                        selected_embs,
                        self.caps.mmr_lambda
                    )

                    if mmr > best_mmr:
                        best_mmr = mmr
                        best_path = path

            if best_path:
                selected.append(best_path)
                remaining.remove(best_path)
            else:
                break

        return selected
