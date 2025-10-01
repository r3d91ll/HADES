"""
Memory Store - ArangoDB Persistence for HADES Agent Memory

WHAT: Persistence adapter for agent conversations and recall events
WHERE: core/runtime/memory/memory_store.py - interfaces with ArangoDB via HTTP/2
WHO: HADES agents persisting conversation state and recall logs
TIME: Write latency p99 ≤100ms, read latency p99 ≤250ms (recall log SLO)

Provides a minimal MemoryStore interface with ArangoDB implementation using
the optimized HTTP/2 client. Focuses on Phase 1 needs: message persistence,
session restoration, and recall logging.

Collections:
- messages (document): conversation turns with metadata
- recall_log (document): recall event tracking for analysis

Boundary Notes:
- All operations via RW socket for persistence
- Recall log queries support conveyance metric extraction
- No boundary metrics collected here - those happen at orchestrator level
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional, Protocol, TypedDict

from ...database.arango.memory_client import (
    ArangoMemoryClient,
    ArangoMemoryClientConfig,
    CollectionDefinition,
    resolve_memory_config,
)
from .turn_manager import ConversationTurn

DEFAULT_DB = os.environ.get("ARANGO_DB_NAME", "hades_memories")


class MessageDoc(TypedDict, total=False):
    _key: str
    session_id: str
    role: str
    content: str
    tokens: int
    created_at: str
    metadata: dict


class RecallLogDoc(TypedDict, total=False):
    _key: str
    session_id: str
    query: str
    result_count: int
    duration_ms: float
    created_at: str
    notes: str


class MemoryStore(Protocol):
    """Abstract interface for MemGPT persistence."""

    def ensure_schema(self) -> None:
        """Create required collections and indexes if missing (idempotent)."""

    def append_turn(self, *, session_id: str, turn: ConversationTurn) -> str:
        """Persist a single turn; returns document key."""

    def list_recent_turns(self, *, session_id: str, limit: int = 50) -> List[MessageDoc]:
        """Retrieve most-recent-first message docs for a session."""

    def log_recall(self, *, session_id: str, query: str, result_count: int, duration_ms: float, notes: str = "") -> str:
        """Record a recall event for auditability; returns document key."""


@dataclass(slots=True)
class ArangoMemoryStore(MemoryStore):
    """Arango-backed implementation using the optimized HTTP/2 client."""

    client: ArangoMemoryClient
    database: str = DEFAULT_DB

    MESSAGES: str = "messages"
    RECALL_LOG: str = "recall_log"

    @staticmethod
    def from_env(*, database: Optional[str] = None) -> "ArangoMemoryStore":
        cfg: ArangoMemoryClientConfig = resolve_memory_config(
            database=database or DEFAULT_DB,
        )
        return ArangoMemoryStore(client=ArangoMemoryClient(cfg), database=cfg.database)

    def close(self) -> None:
        self.client.close()

    # ------------------ schema ------------------
    def ensure_schema(self) -> None:
        definitions = [
            CollectionDefinition(
                name=self.MESSAGES,
                type="document",
                indexes=[
                    {"type": "persistent", "fields": ["session_id", "created_at"], "unique": False, "sparse": False},
                ],
            ),
            CollectionDefinition(
                name=self.RECALL_LOG,
                type="document",
                indexes=[
                    {"type": "persistent", "fields": ["session_id", "created_at"], "unique": False, "sparse": False},
                ],
            ),
        ]
        self.client.create_collections(definitions)

    # ------------------ turns -------------------
    def append_turn(self, *, session_id: str, turn: ConversationTurn) -> str:
        doc: MessageDoc = {
            "_key": str(uuid.uuid4()),
            "session_id": session_id,
            "role": turn.role,
            "content": turn.content,
            "tokens": int(turn.tokens or 0),
            "created_at": turn.timestamp.isoformat(),
            "metadata": dict(turn.metadata or {}),
        }
        res = self.client.bulk_insert(self.MESSAGES, [doc])
        # bulk_insert returns count; we still return _key we generated
        return doc["_key"]

    def list_recent_turns(self, *, session_id: str, limit: int = 50) -> List[MessageDoc]:
        aql = (
            f"FOR m IN {self.MESSAGES} "
            "FILTER m.session_id == @sid "
            "SORT m.created_at DESC "
            "LIMIT @limit "
            "RETURN m"
        )
        rows: List[MessageDoc] = self.client.execute_query(aql, {"sid": session_id, "limit": limit})
        return rows

    # ------------------ recall ------------------
    def log_recall(self, *, session_id: str, query: str, result_count: int, duration_ms: float, notes: str = "") -> str:
        doc: RecallLogDoc = {
            "_key": str(uuid.uuid4()),
            "session_id": session_id,
            "query": query,
            "result_count": int(result_count),
            "duration_ms": float(duration_ms),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        if notes:
            doc["notes"] = notes
        _ = self.client.bulk_insert(self.RECALL_LOG, [doc])
        return doc["_key"]


__all__ = [
    "MemoryStore",
    "ArangoMemoryStore",
    "MessageDoc",
    "RecallLogDoc",
]
