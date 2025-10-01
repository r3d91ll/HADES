"""
Memory Models - Type-safe data structures for experiential memory

WHAT: Pydantic models for observations, reflections, entities, relationships
WHERE: core/runtime/memory/models.py - data layer
WHO: Memory operations creating/validating memory instances
TIME: Model validation <1ms

Provides type-safe models matching the ArangoDB schema defined in
Docs/SCHEMA_Experiential_Memory.md. All models include:
- Timestamp handling (ISO 8601 + Unix)
- Content hashing for deduplication
- Embedding vectors (2048 dimensions, Jina v4)
- Metadata dictionaries for extensibility

Boundary Notes:
- Models enforce schema consistency
- Validation prevents malformed memories
- Hash-based deduplication reduces storage
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


def generate_content_hash(content: str) -> str:
    """Generate SHA-256 hash of content for deduplication."""
    return f"sha256:{hashlib.sha256(content.encode('utf-8')).hexdigest()}"


def generate_timestamp_key(prefix: str) -> str:
    """Generate timestamp-based key with UUID suffix."""
    now = datetime.now(timezone.utc)
    ts = now.isoformat().replace("+00:00", "Z").replace(":", "-")
    suffix = str(uuid.uuid4())[:8]
    return f"{prefix}_{ts}_{suffix}"


class Observation(BaseModel):
    """
    Raw experience/event witnessed by the agent.

    Examples:
    - "Fixed race condition in authentication by adding mutex locks"
    - "User requested dark mode feature for settings page"
    - "PathRAG retrieval completed in 1.2s, within SLO"
    """

    key: str = Field(default_factory=lambda: generate_timestamp_key("obs"))
    content: str = Field(min_length=1, max_length=10000)
    content_hash: str = Field(default="")
    memory_type: Literal["observation", "decision", "learning", "event"] = "observation"
    tags: List[str] = Field(default_factory=list)
    embedding: Optional[List[float]] = Field(default=None, max_length=2048, min_length=2048)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    created_at_unix: float = Field(default=0.0)
    session_id: Optional[str] = None
    turn_index: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    revision: int = 1
    status: Literal["active", "archived", "superseded"] = "active"

    def model_post_init(self, __context: Any) -> None:
        """Auto-generate hash and unix timestamp after initialization."""
        if not self.content_hash:
            self.content_hash = generate_content_hash(self.content)
        if self.created_at_unix == 0.0:
            self.created_at_unix = self.created_at.timestamp()

    def to_arango_doc(self) -> Dict[str, Any]:
        """Convert to ArangoDB document format."""
        return {
            "_key": self.key,
            "content": self.content,
            "content_hash": self.content_hash,
            "memory_type": self.memory_type,
            "tags": self.tags,
            "embedding": self.embedding,
            "created_at": self.created_at.isoformat(),
            "created_at_unix": self.created_at_unix,
            "session_id": self.session_id,
            "turn_index": self.turn_index,
            "metadata": self.metadata,
            "updated_at": self.updated_at.isoformat(),
            "revision": self.revision,
            "status": self.status,
        }

    @classmethod
    def from_arango_doc(cls, doc: Dict[str, Any]) -> Observation:
        """Create instance from ArangoDB document."""
        return cls(
            key=doc["_key"],
            content=doc["content"],
            content_hash=doc["content_hash"],
            memory_type=doc["memory_type"],
            tags=doc["tags"],
            embedding=doc.get("embedding"),
            created_at=datetime.fromisoformat(doc["created_at"].replace("Z", "+00:00")),
            created_at_unix=doc["created_at_unix"],
            session_id=doc.get("session_id"),
            turn_index=doc.get("turn_index"),
            metadata=doc.get("metadata", {}),
            updated_at=datetime.fromisoformat(doc["updated_at"].replace("Z", "+00:00")),
            revision=doc.get("revision", 1),
            status=doc.get("status", "active"),
        )


class Reflection(BaseModel):
    """
    Higher-order insight synthesized from multiple observations.

    Examples:
    - "Authentication bugs often involve race conditions in multi-threaded systems"
    - "Users prefer dark mode when working late; should be context-aware"
    - "PathRAG consistently performs better with H=2, F=3 for code queries"
    """

    key: str = Field(default_factory=lambda: generate_timestamp_key("refl"))
    content: str = Field(min_length=1, max_length=10000)
    content_hash: str = Field(default="")
    memory_type: Literal["reflection", "pattern", "principle", "insight"] = "reflection"
    tags: List[str] = Field(default_factory=list)
    embedding: Optional[List[float]] = Field(default=None, max_length=2048, min_length=2048)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    created_at_unix: float = Field(default=0.0)
    consolidates: List[str] = Field(default_factory=list)  # observation/_ids
    source_count: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: Literal["active", "archived", "superseded"] = "active"

    def model_post_init(self, __context: Any) -> None:
        """Auto-generate hash, unix timestamp, and source count."""
        if not self.content_hash:
            self.content_hash = generate_content_hash(self.content)
        if self.created_at_unix == 0.0:
            self.created_at_unix = self.created_at.timestamp()
        if self.source_count == 0:
            self.source_count = len(self.consolidates)

    def to_arango_doc(self) -> Dict[str, Any]:
        """Convert to ArangoDB document format."""
        return {
            "_key": self.key,
            "content": self.content,
            "content_hash": self.content_hash,
            "memory_type": self.memory_type,
            "tags": self.tags,
            "embedding": self.embedding,
            "created_at": self.created_at.isoformat(),
            "created_at_unix": self.created_at_unix,
            "consolidates": self.consolidates,
            "source_count": self.source_count,
            "metadata": self.metadata,
            "updated_at": self.updated_at.isoformat(),
            "status": self.status,
        }

    @classmethod
    def from_arango_doc(cls, doc: Dict[str, Any]) -> Reflection:
        """Create instance from ArangoDB document."""
        return cls(
            key=doc["_key"],
            content=doc["content"],
            content_hash=doc["content_hash"],
            memory_type=doc["memory_type"],
            tags=doc["tags"],
            embedding=doc.get("embedding"),
            created_at=datetime.fromisoformat(doc["created_at"].replace("Z", "+00:00")),
            created_at_unix=doc["created_at_unix"],
            consolidates=doc.get("consolidates", []),
            source_count=doc.get("source_count", 0),
            metadata=doc.get("metadata", {}),
            updated_at=datetime.fromisoformat(doc["updated_at"].replace("Z", "+00:00")),
            status=doc.get("status", "active"),
        )


class Entity(BaseModel):
    """
    Person, component, concept, project, or tool extracted from experiences.

    Examples:
    - name="ArangoDB HTTP/2 Client", type="component"
    - name="Todd", type="person"
    - name="PathRAG", type="concept"
    """

    key: str  # manually set, e.g., "ent_component_arango_client"
    name: str = Field(min_length=1, max_length=200)
    aliases: List[str] = Field(default_factory=list)
    entity_type: Literal["person", "component", "concept", "project", "tool"]
    tags: List[str] = Field(default_factory=list)
    description: str = ""
    embedding: Optional[List[float]] = Field(default=None, max_length=2048, min_length=2048)
    first_seen: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    mention_count: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    status: Literal["active", "archived", "merged"] = "active"

    def to_arango_doc(self) -> Dict[str, Any]:
        """Convert to ArangoDB document format."""
        return {
            "_key": self.key,
            "name": self.name,
            "aliases": self.aliases,
            "entity_type": self.entity_type,
            "tags": self.tags,
            "description": self.description,
            "embedding": self.embedding,
            "first_seen": self.first_seen.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "mention_count": self.mention_count,
            "metadata": self.metadata,
            "status": self.status,
        }

    @classmethod
    def from_arango_doc(cls, doc: Dict[str, Any]) -> Entity:
        """Create instance from ArangoDB document."""
        return cls(
            key=doc["_key"],
            name=doc["name"],
            aliases=doc.get("aliases", []),
            entity_type=doc["entity_type"],
            tags=doc.get("tags", []),
            description=doc.get("description", ""),
            embedding=doc.get("embedding"),
            first_seen=datetime.fromisoformat(doc["first_seen"].replace("Z", "+00:00")),
            last_updated=datetime.fromisoformat(doc["last_updated"].replace("Z", "+00:00")),
            mention_count=doc.get("mention_count", 0),
            metadata=doc.get("metadata", {}),
            status=doc.get("status", "active"),
        )


class Relationship(BaseModel):
    """
    Graph edge connecting two entities or an entity and a memory.

    Examples:
    - from=auth_system, to=arango_client, type="uses"
    - from=observation_123, to=entity_mutex, type="mentions"
    - from=reflection_456, to=entity_pathrag, type="explains"
    """

    key: str = Field(default_factory=lambda: f"rel_{uuid.uuid4().hex[:12]}")
    from_id: str  # full _id like "entities/ent_component_auth"
    to_id: str  # full _id like "entities/ent_component_client"
    relationship_type: Literal[
        "uses",
        "causes",
        "relates_to",
        "improves",
        "conflicts_with",
        "mentions",
        "explains",
    ]
    strength: float = Field(ge=0.0, le=1.0, default=0.5)
    evidence: List[str] = Field(default_factory=list)  # observation/_ids or reflection/_ids
    evidence_count: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_reinforced: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)
    status: Literal["active", "archived", "invalidated"] = "active"

    def model_post_init(self, __context: Any) -> None:
        """Auto-update evidence count."""
        if self.evidence_count == 0:
            self.evidence_count = len(self.evidence)

    def to_arango_doc(self) -> Dict[str, Any]:
        """Convert to ArangoDB edge document format."""
        return {
            "_key": self.key,
            "_from": self.from_id,
            "_to": self.to_id,
            "relationship_type": self.relationship_type,
            "strength": self.strength,
            "evidence": self.evidence,
            "evidence_count": self.evidence_count,
            "created_at": self.created_at.isoformat(),
            "last_reinforced": self.last_reinforced.isoformat(),
            "metadata": self.metadata,
            "status": self.status,
        }

    @classmethod
    def from_arango_doc(cls, doc: Dict[str, Any]) -> Relationship:
        """Create instance from ArangoDB edge document."""
        return cls(
            key=doc["_key"],
            from_id=doc["_from"],
            to_id=doc["_to"],
            relationship_type=doc["relationship_type"],
            strength=doc.get("strength", 0.5),
            evidence=doc.get("evidence", []),
            evidence_count=doc.get("evidence_count", 0),
            created_at=datetime.fromisoformat(doc["created_at"].replace("Z", "+00:00")),
            last_reinforced=datetime.fromisoformat(
                doc["last_reinforced"].replace("Z", "+00:00")
            ),
            metadata=doc.get("metadata", {}),
            status=doc.get("status", "active"),
        )


class MemoryQueryResult(BaseModel):
    """Result from memory retrieval with relevance scoring."""

    memory: Observation | Reflection | Entity
    relevance_score: float = Field(ge=0.0, le=1.0)
    debug_info: Dict[str, Any] = Field(default_factory=dict)


class Session(BaseModel):
    """Conversation session for grouping memories."""

    key: str = Field(default_factory=lambda: f"chat_{uuid.uuid4().hex[:8]}")
    session_name: str
    session_type: Literal["debugging", "design", "learning", "exploration"]
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    ended_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    stats: Dict[str, int] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    status: Literal["active", "completed", "archived"] = "active"

    def to_arango_doc(self) -> Dict[str, Any]:
        """Convert to ArangoDB document format."""
        return {
            "_key": self.key,
            "session_name": self.session_name,
            "session_type": self.session_type,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "duration_seconds": self.duration_seconds,
            "stats": self.stats,
            "metadata": self.metadata,
            "status": self.status,
        }

    @classmethod
    def from_arango_doc(cls, doc: Dict[str, Any]) -> Session:
        """Create instance from ArangoDB document."""
        return cls(
            key=doc["_key"],
            session_name=doc["session_name"],
            session_type=doc["session_type"],
            started_at=datetime.fromisoformat(doc["started_at"].replace("Z", "+00:00")),
            ended_at=datetime.fromisoformat(doc["ended_at"].replace("Z", "+00:00"))
            if doc.get("ended_at")
            else None,
            duration_seconds=doc.get("duration_seconds"),
            stats=doc.get("stats", {}),
            metadata=doc.get("metadata", {}),
            status=doc.get("status", "active"),
        )