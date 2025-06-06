"""
Type definitions for batching utilities.

This module contains TypedDict and other type definitions used by the
batching utilities in src/utils/batching/.
"""

from typing import Dict, TypedDict


class BatchStats(TypedDict):
    """Statistics about batched files."""
    total: int
    by_type: Dict[str, int]