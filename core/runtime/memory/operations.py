"""
Memory Operations - Local Library for Experiential Memory

WHAT: High-level operations for storing, retrieving, and managing memories
WHERE: core/runtime/memory/operations.py - API layer
WHO: Agents and scripts performing memory operations (in-process, no network)
TIME: Store p99 <50ms, retrieve p99 <250ms (PathRAG-powered)

Local Python library for experiential memory operations powered by ArangoDB + PathRAG.
Prompt engineering and operation patterns inspired by:
https://github.com/doobidoo/mcp-memory-service

Operations:
- store_observation: Persist experiential memories with embeddings
- retrieve_memories: Semantic search with graph traversal
- search_by_tag: Category-based retrieval
- list_observations: List recent memories
- archive_observation: Soft delete

All operations run in-process via direct ArangoDB Unix socket connections.
No network services, no MCP protocol - pure local library.

Boundary Notes:
- Operations tracked for conveyance metrics
- Embeddings generated via Jina v4 (32k context)
- PathRAG enforces caps (H≤2, F≤3, B≤8)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Literal, Optional

import numpy as np

from ...database.arango.memory_client import ArangoMemoryClient, resolve_memory_config
from ...embedders import EmbedderBase, EmbeddingConfig, create_embedder
from .models import Entity, MemoryQueryResult, Observation, Reflection, Relationship, Session
from .pathrag_memory import MemoryPathRAG, MemoryPathCaps, MemoryPathNode

logger = logging.getLogger(__name__)


@dataclass
class MemoryStore:
    """
    Experiential memory store - local library for agent memory operations.

    Handles observations, reflections, entities, and relationships with
    PathRAG-powered semantic retrieval. All operations run in-process via
    direct ArangoDB connections (no network services).

    Inspired by: https://github.com/doobidoo/mcp-memory-service
    """

    client: ArangoMemoryClient
    embedder: Optional[EmbedderBase] = None

    # Collection names
    OBSERVATIONS = "observations"
    REFLECTIONS = "reflections"
    ENTITIES = "entities"
    RELATIONSHIPS = "relationships"
    SESSIONS = "sessions"

    @classmethod
    def from_env(
        cls, embedder_model: str = "jinaai/jina-embeddings-v4", device: str = "cuda"
    ) -> MemoryStore:
        """Create from environment configuration."""
        cfg = resolve_memory_config()
        client = ArangoMemoryClient(cfg)

        # Initialize embedder
        try:
            emb_config = EmbeddingConfig(
                model_name=embedder_model, device=device, batch_size=16, trust_remote_code=True
            )
            embedder = create_embedder(config=emb_config)
        except Exception as e:
            logger.warning(f"Failed to initialize embedder: {e}")
            embedder = None

        return cls(client=client, embedder=embedder)

    def close(self) -> None:
        """Close database connection."""
        self.client.close()

    # ============================================================
    # Schema Management
    # ============================================================

    def ensure_schema(self) -> None:
        """Create collections and indexes for experiential memory."""
        from ...database.arango.memory_client import CollectionDefinition

        definitions = [
            CollectionDefinition(
                name=self.OBSERVATIONS,
                type="document",
                indexes=[
                    {"type": "persistent", "fields": ["content_hash"], "unique": True, "sparse": True},
                    {"type": "persistent", "fields": ["created_at_unix"], "unique": False},
                    {"type": "persistent", "fields": ["session_id"], "unique": False},
                    {"type": "persistent", "fields": ["tags[*]"], "unique": False},
                    {"type": "persistent", "fields": ["memory_type"], "unique": False},
                ],
            ),
            CollectionDefinition(
                name=self.REFLECTIONS,
                type="document",
                indexes=[
                    {"type": "persistent", "fields": ["content_hash"], "unique": True, "sparse": True},
                    {"type": "persistent", "fields": ["created_at_unix"], "unique": False},
                    {"type": "persistent", "fields": ["tags[*]"], "unique": False},
                    {"type": "persistent", "fields": ["memory_type"], "unique": False},
                    {"type": "persistent", "fields": ["consolidates[*]"], "unique": False},
                ],
            ),
            CollectionDefinition(
                name=self.ENTITIES,
                type="document",
                indexes=[
                    {"type": "fulltext", "fields": ["name"], "minLength": 2},
                    {"type": "persistent", "fields": ["entity_type"], "unique": False},
                    {"type": "persistent", "fields": ["tags[*]"], "unique": False},
                    {"type": "persistent", "fields": ["mention_count"], "unique": False},
                ],
            ),
            CollectionDefinition(
                name=self.RELATIONSHIPS,
                type="edge",
                indexes=[
                    {"type": "persistent", "fields": ["relationship_type"], "unique": False},
                    {"type": "persistent", "fields": ["strength"], "unique": False},
                    {"type": "persistent", "fields": ["created_at"], "unique": False},
                ],
            ),
            CollectionDefinition(
                name=self.SESSIONS,
                type="document",
                indexes=[
                    {"type": "persistent", "fields": ["started_at"], "unique": False},
                    {"type": "persistent", "fields": ["session_type"], "unique": False},
                    {"type": "persistent", "fields": ["status"], "unique": False},
                ],
            ),
        ]
        self.client.create_collections(definitions)
        logger.info("Experiential memory schema ensured")

    # ============================================================
    # Core Operations (MCP-compatible)
    # ============================================================

    def store_observation(
        self,
        content: str,
        tags: List[str] = [],
        memory_type: Literal["observation", "decision", "learning", "event"] = "observation",
        session_id: Optional[str] = None,
        turn_index: Optional[int] = None,
        metadata: Dict[str, Any] = {},
    ) -> str:
        """
        Store a new observation with embedding.

        Args:
            content: The observation text
            tags: Categorization tags (e.g., ["debugging", "concurrency"])
            memory_type: Type of observation
            session_id: Associated conversation session
            turn_index: Turn number within session
            metadata: Additional context (project, component, confidence, etc.)

        Returns:
            Document key (_key) of stored observation

        Example:
            >>> store.store_observation(
            ...     content="Fixed race condition in auth by adding mutex",
            ...     tags=["debugging", "concurrency", "authentication"],
            ...     memory_type="decision",
            ...     session_id="chat_abc123",
            ...     metadata={"component": "auth_system", "confidence": 0.9}
            ... )
            'obs_2025-09-29T12-34-56.789Z_abc12345'
        """
        # Generate embedding if embedder available
        embedding = None
        if self.embedder:
            try:
                emb_result = self.embedder.embed_documents([content], batch_size=1)
                embedding = emb_result[0].tolist() if len(emb_result) > 0 else None
            except Exception as e:
                logger.warning(f"Failed to generate embedding: {e}")

        # Create observation model
        obs = Observation(
            content=content,
            memory_type=memory_type,
            tags=tags,
            embedding=embedding,
            session_id=session_id,
            turn_index=turn_index,
            metadata=metadata,
        )

        # Write to database
        doc = obs.to_arango_doc()
        self.client.insert_document(self.OBSERVATIONS, doc, overwrite_mode="ignore")

        logger.info(f"Stored observation {obs.key} with {len(tags)} tags")
        return obs.key

    def retrieve_memories(
        self,
        query: str,
        n_results: int = 5,
        min_similarity: float = 0.7,
        time_filter: Optional[str] = None,
        tags: Optional[List[str]] = None,
        memory_types: Optional[List[str]] = None,
        collections: Optional[List[str]] = None,
    ) -> List[MemoryQueryResult]:
        """
        Retrieve memories using semantic similarity + optional filters.

        Args:
            query: Natural language query
            n_results: Maximum number of results
            min_similarity: Minimum cosine similarity threshold
            time_filter: Natural language time (e.g., "last week", "yesterday")
            tags: Filter by tags
            memory_types: Filter by memory_type
            collections: Limit to specific collections (default: observations + reflections)

        Returns:
            List of MemoryQueryResult with relevance scores

        Example:
            >>> results = store.retrieve_memories(
            ...     query="authentication race conditions",
            ...     n_results=5,
            ...     time_filter="last month",
            ...     tags=["debugging", "authentication"]
            ... )
        """
        if not self.embedder:
            raise RuntimeError("Embedder not initialized - cannot perform semantic search")

        # Generate query embedding
        query_emb = self.embedder.embed_query(query)
        if isinstance(query_emb, np.ndarray):
            query_emb = query_emb.tolist()

        # Parse time filter
        time_constraint = self._parse_time_filter(time_filter) if time_filter else None

        # Build search collections
        search_colls = collections or [self.OBSERVATIONS, self.REFLECTIONS]

        # STAGE 1: Vector search for seed nodes
        seed_nodes: List[MemoryPathNode] = []

        for coll in search_colls:
            # Build filter conditions
            filters = []
            if time_constraint:
                filters.append(f"doc.created_at_unix >= {time_constraint}")
            if tags:
                tag_conditions = " OR ".join([f"'{tag}' IN doc.tags" for tag in tags])
                filters.append(f"({tag_conditions})")
            if memory_types:
                type_conditions = " OR ".join(
                    [f"doc.memory_type == '{mt}'" for mt in memory_types]
                )
                filters.append(f"({type_conditions})")

            filter_clause = " AND ".join(filters) if filters else "true"

            # Vector search for initial seeds (K=6-8)
            aql = f"""
            FOR doc IN {coll}
                FILTER {filter_clause}
                FILTER doc.embedding != null
                LET similarity = 1 - COSINE_DISTANCE(doc.embedding, @query_embedding)
                FILTER similarity >= @min_similarity
                SORT similarity DESC
                LIMIT @seed_limit
                RETURN {{doc: doc, similarity: similarity}}
            """

            try:
                cursor = self.client.execute_query(
                    aql,
                    bind_vars={
                        "query_embedding": query_emb,
                        "min_similarity": min_similarity,
                        "seed_limit": 6,  # PathRAG seed limit
                    },
                )
                for row in cursor:
                    doc_data = row["doc"]
                    seed_nodes.append(
                        MemoryPathNode(
                            doc_id=doc_data["_id"],
                            collection=coll,
                            content=doc_data.get("content", ""),
                            embedding=doc_data.get("embedding"),
                            score=row["similarity"],
                            metadata={
                                "created_at_unix": doc_data.get("created_at_unix"),
                                "tags": doc_data.get("tags", []),
                                "memory_type": doc_data.get("memory_type"),
                            },
                        )
                    )
            except Exception as e:
                logger.error(f"Failed to search {coll}: {e}")

        if not seed_nodes:
            logger.warning(f"No seed nodes found for query: {query}")
            return []

        # STAGE 2: PathRAG graph traversal
        pathrag = MemoryPathRAG(self.client)
        caps = MemoryPathCaps(hops=2, fanout=3, beam=8, timeout_ms=500, mmr_lambda=0.4)

        path_result = pathrag.find_memory_paths(
            seed_nodes=seed_nodes,
            query_embedding=query_emb,
            caps=caps,
            collections=search_colls + [self.ENTITIES],
            edge_collections=[self.RELATIONSHIPS],
            time_filter_unix=time_constraint,
        )

        # STAGE 3: Convert paths to MemoryQueryResult
        results: List[MemoryQueryResult] = []
        seen_ids: set = set()

        for path in path_result.paths:
            for node in path.nodes:
                if node.doc_id in seen_ids:
                    continue
                seen_ids.add(node.doc_id)

                # Fetch full document and convert to model
                try:
                    coll_name = node.collection
                    doc_key = node.doc_id.split("/")[1]
                    doc_data = self.client.get_document(coll_name, doc_key)

                    if not doc_data:
                        continue

                    if coll_name == self.OBSERVATIONS:
                        memory = Observation.from_arango_doc(doc_data)
                    elif coll_name == self.REFLECTIONS:
                        memory = Reflection.from_arango_doc(doc_data)
                    elif coll_name == self.ENTITIES:
                        memory = Entity.from_arango_doc(doc_data)
                    else:
                        continue

                    results.append(
                        MemoryQueryResult(
                            memory=memory,
                            relevance_score=path.final_score,
                            debug_info={
                                "collection": coll_name,
                                "path_length": len(path.nodes),
                                "path_reasoning": path.reasoning,
                                "pathrag_reasons": path_result.reasons,
                                "latency_ms": path_result.latency_ms,
                            },
                        )
                    )
                except Exception as e:
                    logger.warning(f"Failed to fetch document {node.doc_id}: {e}")

        # Sort by relevance and limit
        results.sort(key=lambda r: r.relevance_score, reverse=True)
        logger.info(
            f"PathRAG retrieval: {len(results)} memories from {len(path_result.paths)} paths "
            f"({path_result.latency_ms:.1f}ms)"
        )
        return results[:n_results]

    def search_by_tag(
        self,
        tags: List[str],
        match_all: bool = False,
        collection: Optional[str] = None,
        limit: int = 10,
    ) -> List[Observation | Reflection]:
        """
        Search memories by tags.

        Args:
            tags: Tags to search for
            match_all: If True, require all tags; if False, match any tag
            collection: Specific collection to search (default: observations)
            limit: Maximum results

        Returns:
            List of matching observations or reflections
        """
        coll = collection or self.OBSERVATIONS
        operator = "AND" if match_all else "OR"
        tag_conditions = f" {operator} ".join([f"'{tag}' IN doc.tags" for tag in tags])

        aql = f"""
        FOR doc IN {coll}
            FILTER {tag_conditions}
            SORT doc.created_at_unix DESC
            LIMIT @limit
            RETURN doc
        """

        results = []
        try:
            cursor = self.client.execute_query(aql, bind_vars={"limit": limit})
            for doc in cursor:
                if coll == self.OBSERVATIONS:
                    results.append(Observation.from_arango_doc(doc))
                elif coll == self.REFLECTIONS:
                    results.append(Reflection.from_arango_doc(doc))
        except Exception as e:
            logger.error(f"Failed to search by tags: {e}")

        return results

    def list_observations(
        self,
        session_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Observation]:
        """
        List observations with optional filtering.

        Args:
            session_id: Filter by session
            limit: Maximum results
            offset: Pagination offset

        Returns:
            List of observations
        """
        filter_clause = f"FILTER doc.session_id == '{session_id}'" if session_id else ""

        aql = f"""
        FOR doc IN {self.OBSERVATIONS}
            {filter_clause}
            SORT doc.created_at_unix DESC
            LIMIT @offset, @limit
            RETURN doc
        """

        results = []
        try:
            cursor = self.client.execute_query(aql, bind_vars={"limit": limit, "offset": offset})
            results = [Observation.from_arango_doc(doc) for doc in cursor]
        except Exception as e:
            logger.error(f"Failed to list observations: {e}")

        return results

    def archive_observation(self, key: str) -> bool:
        """
        Archive an observation (soft delete).

        Args:
            key: Observation _key

        Returns:
            True if successful
        """
        try:
            doc = self.client.get_document(self.OBSERVATIONS, key)
            if doc:
                doc["status"] = "archived"
                doc["updated_at"] = datetime.now(timezone.utc).isoformat()
                self.client.update_document(self.OBSERVATIONS, key, doc)
                return True
        except Exception as e:
            logger.error(f"Failed to archive observation {key}: {e}")
        return False

    # ============================================================
    # Helper Methods
    # ============================================================

    def _parse_time_filter(self, time_str: str) -> Optional[float]:
        """
        Parse natural language time filter to Unix timestamp.

        Supports:
        - "yesterday", "today"
        - "last week", "last month", "last year"
        - "last 7 days", "last 30 days"
        """
        now = datetime.now(timezone.utc)
        time_lower = time_str.lower().strip()

        if time_lower == "today":
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            return start.timestamp()
        elif time_lower == "yesterday":
            start = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            return start.timestamp()
        elif time_lower == "last week" or time_lower == "last 7 days":
            return (now - timedelta(days=7)).timestamp()
        elif time_lower == "last month" or time_lower == "last 30 days":
            return (now - timedelta(days=30)).timestamp()
        elif time_lower == "last year" or time_lower == "last 365 days":
            return (now - timedelta(days=365)).timestamp()
        else:
            # Try to parse "last N days"
            import re

            match = re.match(r"last\s+(\d+)\s+days?", time_lower)
            if match:
                days = int(match.group(1))
                return (now - timedelta(days=days)).timestamp()

        logger.warning(f"Could not parse time filter: {time_str}")
        return None