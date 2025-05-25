"""Utility helpers for ArangoDB naming compliance.

These helpers guarantee that collection names and document keys
follow the restrictions documented in the ArangoDB manual:
https://docs.arangodb.com/3.10/data-modeling/naming-conventions/

They are deliberately *pure* functions so they can be unit-tested
without touching a live database.
"""
from __future__ import annotations

import re
from typing import Final

__all__: list[str] = [
    "safe_name",
    "safe_key",
]

# ---------------------------------------------------------------------------
# Regular expressions taken from the ArangoDB docs
# ---------------------------------------------------------------------------

# Collection names: letters, numbers, underscores; must not start with underscore
_NAME_ALLOWED_RE: Final[re.Pattern[str]] = re.compile(r"[^A-Za-z0-9_]")

# Document keys – Arango allows a specific set of printable ASCII chars.
# We’ll remove anything outside that set; see docs for full list.
_KEY_ALLOWED_RE: Final[re.Pattern[str]] = re.compile(r"[^A-Za-z0-9_\-:.@()+,=;$!*'%]")

def safe_name(name: str) -> str:
    """Return a collection name that conforms to ArangoDB rules.

    - Replaces every disallowed char with an underscore.
    - Strips leading underscores to satisfy “must not start with ‘’_’’”.
    """
    cleaned = _NAME_ALLOWED_RE.sub("_", name)
    return cleaned.lstrip("_") or "collection"


def safe_key(key: str) -> str:
    """Return a document `_key` safe for ArangoDB.

    - Replaces spaces with underscores.
    - Removes all characters not whitelisted by the Arango spec.
    """
    key = key.replace(" ", "_")
    return _KEY_ALLOWED_RE.sub("", key) or "doc"
