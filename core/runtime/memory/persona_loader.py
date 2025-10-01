"""
Persona Loader - Agent Identity and Configuration Management

WHAT: Loads and manages agent personas and human profiles from templates
WHERE: core/runtime/memgpt/persona_loader.py - configuration layer
WHO: MemGPT agents requiring identity configuration
TIME: Template loading <10ms

Manages agent personas and human profiles from template files. Supports
variable substitution for dynamic persona generation.

Boundary Notes:
- Persona quality affects H (agent capability) dimension
- Well-defined personas improve protocol compatibility (P_ij)
- Template variables enable context-specific agent configuration
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping


class _SafeDict(dict):
    def __missing__(self, key):  # type: ignore[override]
        return "{" + key + "}"


def render_template(text: str, variables: Mapping[str, str] | None = None) -> str:
    if not variables:
        return text
    return text.format_map(_SafeDict(variables))


def load_text(path: str | Path) -> str:
    p = Path(path).expanduser().resolve()
    return p.read_text(encoding="utf-8")


__all__ = [
    "render_template",
    "load_text",
]

