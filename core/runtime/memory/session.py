"""
Session Manager - MemGPT Conversation State Coordination

WHAT: Session-level coordinator for MemGPT with ArangoDB persistence
WHERE: core/runtime/memgpt/session.py - bridges orchestrator and storage
WHO: MemGPT agents managing multi-turn conversations
TIME: Session restoration <100ms, turn persistence <50ms

Coordinates between MemGPTOrchestrator (prompt building/generation) and
ArangoMemoryStore (persistence). Handles history restoration, turn persistence,
and optional recall logging.

Boundary Notes:
- Session boundaries define conveyance measurement windows
- Turn-level metrics aggregated for session-level analysis
- Recall events logged for α learning
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Iterable

from .memory_store import MemoryStore
from .orchestrator import MemGPTOrchestrator, PromptEnvelope
from .prompting import compose_system_prompt
from .turn_manager import ConversationTurn


@dataclass(slots=True)
class MemGPTSession:
    orchestrator: MemGPTOrchestrator
    store: MemoryStore
    session_id: str

    @staticmethod
    def new(orchestrator: MemGPTOrchestrator, store: MemoryStore, *, session_id: str | None = None) -> "MemGPTSession":
        return MemGPTSession(orchestrator=orchestrator, store=store, session_id=session_id or str(uuid.uuid4()))

    # ---------------------- lifecycle ----------------------
    def restore_history(self, *, limit: int = 50) -> int:
        """Load recent turns from the store into the orchestrator.

        Returns the number of restored turns.
        """

        records = self.store.list_recent_turns(session_id=self.session_id, limit=limit)
        # Records are most-recent-first; ingest from oldest to newest
        for rec in reversed(records):
            self.orchestrator.ingest_turn(
                role=str(rec.get("role", "user")),
                content=str(rec.get("content", "")),
                tokens=int(rec.get("tokens", 0) or 0),
                metadata=dict(rec.get("metadata", {})),
            )
        return len(records)

    def ensure_system_core_memory(self, *, persona: str, human: str, previous_message_count: int = 0) -> None:
        """If no system turn exists, insert a system prompt with core memory blocks."""

        turns = self.orchestrator.turn_manager.turns
        has_system = any(t.role == "system" for t in turns)
        if has_system:
            return
        system_text = compose_system_prompt(
            persona=persona,
            human=human,
            previous_message_count=previous_message_count,
        )
        self.orchestrator.ingest_turn(role="system", content=system_text)

    # ---------------------- messaging ----------------------
    def send_user_message(self, content: str, *, tokens: int | None = None, stop: Iterable[str] | None = None, overrides: dict | None = None, metadata: dict | None = None) -> ConversationTurn:
        """Append a user message, generate a reply, persist both, and return the assistant turn."""

        user_turn = self.orchestrator.ingest_turn(role="user", content=content, tokens=tokens, metadata=metadata)
        try:
            self.store.append_turn(session_id=self.session_id, turn=user_turn)
        except Exception:
            # Persist errors should not block generation
            pass

        envelope: PromptEnvelope = self.orchestrator.prepare_envelope(stop=stop, overrides=overrides, metadata=metadata)
        reply_turn = self.orchestrator.generate_turn(envelope=envelope)
        try:
            self.store.append_turn(session_id=self.session_id, turn=reply_turn)
        except Exception:
            pass
        return reply_turn


__all__ = [
    "MemGPTSession",
]
