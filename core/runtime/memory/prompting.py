"""
Prompt Engineering - MemGPT System Prompt Composition

WHAT: Prompt templates and composition utilities for MemGPT agents
WHERE: core/runtime/memgpt/prompting.py - prompt generation layer
WHO: MemGPT orchestrator building system prompts
TIME: Prompt assembly <1ms

Adapted from Letta/MemGPT concepts. Provides minimal system prompts with
core-memory blocks and optional metadata headers. Backend-agnostic design
allows flexible memory system integration.

Boundary Notes:
- Prompt quality directly impacts H (agent capability)
- System prompt size affects context budget and R dimension
- Memory block structure influences agent coherence
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Optional

MEMGPT_CHAT_SYSTEM = (
    "You are a capable assistant. Converse naturally from the perspective of your persona.\n"
    "Never refer to yourself as an AI; stay in character. Keep any inner monologue short.\n\n"
    "Core memory (always visible):\n"
    "<persona>\n{persona}\n</persona>\n"
    "<human>\n{human}\n</human>\n\n"
    "Recall memory: Conversation history exists beyond this context.\n"
    "Archival memory: Long‑term notes live outside context; retrieve explicitly when needed.\n"
)


def format_memory_metadata(
    *,
    memory_edit_timestamp: datetime,
    previous_message_count: int,
    archival_memory_size: Optional[int] = 0,
    archive_tags: Iterable[str] | None = None,
) -> str:
    ts = memory_edit_timestamp.astimezone(timezone.utc).replace(microsecond=0).isoformat()
    lines = [
        "<memory_metadata>",
        f"- Memory last updated: {ts}",
        f"- {previous_message_count} previous messages stored in recall memory",
    ]
    if archival_memory_size and archival_memory_size > 0:
        lines.append(f"- {archival_memory_size} total items in archival memory")
    if archive_tags:
        tags = ", ".join(sorted(archive_tags))
        lines.append(f"- Available archival memory tags: {tags}")
    lines.append("</memory_metadata>")
    return "\n".join(lines)


@dataclass(slots=True)
class SystemPromptConfig:
    persona: str = "You are a helpful, concise engineer."
    human: str = "User: name=unknown. Preferences=unknown."
    include_metadata: bool = True


def compose_system_prompt(
    *,
    persona: str,
    human: str,
    previous_message_count: int = 0,
    archival_memory_size: int | None = None,
    archive_tags: Iterable[str] | None = None,
    edited_at: datetime | None = None,
) -> str:
    base = MEMGPT_CHAT_SYSTEM.format(persona=persona.strip(), human=human.strip())
    if edited_at is None:
        edited_at = datetime.now(timezone.utc)
    meta = format_memory_metadata(
        memory_edit_timestamp=edited_at,
        previous_message_count=previous_message_count,
        archival_memory_size=archival_memory_size or 0,
        archive_tags=archive_tags,
    )
    return f"{base}\n{meta}\n"


SUMMARY_PROMPT = (
    "Your job is to summarize previous messages between an AI persona and a human.\n"
    "Use first person from the AI's perspective. Keep the summary under 100 words.\n"
    "Only output the summary text."
)


__all__ = [
    "compose_system_prompt",
    "format_memory_metadata",
    "SystemPromptConfig",
    "SUMMARY_PROMPT",
]

