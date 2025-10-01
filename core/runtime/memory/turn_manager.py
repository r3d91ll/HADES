"""
Turn Manager - Conversation State and History Management

WHAT: Manages conversation turns, context windows, and message formatting
WHERE: core/runtime/memgpt/turn_manager.py - orchestration subsystem
WHO: MemGPT orchestrator for managing multi-turn dialogues
TIME: Turn operations O(1), context window operations O(n) where n=window_size

Handles conversation state, including turn tracking, context window management,
and message formatting. Implements FIFO queue semantics for memory management.

Boundary Notes:
- Turn boundaries mark measurement points for incremental conveyance
- Context window size affects R (structure) dimension
- Message truncation impacts W (semantic quality)
"""

from __future__ import annotations

import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Iterable, Literal

Role = Literal["user", "assistant", "system", "tool"]


@dataclass(slots=True)
class ConversationTurn:
    """Represents a single conversational event ingested by MemGPT."""

    turn_id: str
    role: Role
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tokens: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        role: Role,
        content: str,
        tokens: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "ConversationTurn":
        return cls(
            turn_id=str(uuid.uuid4()),
            role=role,
            content=content,
            tokens=tokens,
            metadata=metadata or {},
        )


class TurnManager:
    """Maintains the rolling conversation context and budget."""

    def __init__(self, *, max_turns: int = 50, token_budget: int | None = None) -> None:
        self._turns: Deque[ConversationTurn] = deque()
        self._max_turns = max_turns
        self._token_budget = token_budget

    @property
    def turns(self) -> tuple[ConversationTurn, ...]:
        return tuple(self._turns)

    def add_turn(self, turn: ConversationTurn) -> None:
        """Append a new turn and enforce retention policies."""

        self._turns.append(turn)
        self._enforce_limits()

    def extend(self, turns: Iterable[ConversationTurn]) -> None:
        for turn in turns:
            self.add_turn(turn)

    def summarize(self) -> dict[str, Any]:
        """Return a lightweight summary used for telemetry and context packing."""

        total_tokens = sum(t.tokens or 0 for t in self._turns)
        return {
            "turn_count": len(self._turns),
            "roles": [t.role for t in self._turns],
            "tokens": total_tokens,
            "token_budget": self._token_budget,
        }

    def _enforce_limits(self) -> None:
        while len(self._turns) > self._max_turns:
            self._turns.popleft()

        if self._token_budget is None:
            return

        total = sum(t.tokens or 0 for t in self._turns)
        while self._turns and total > self._token_budget:
            expired = self._turns.popleft()
            total -= expired.tokens or 0

