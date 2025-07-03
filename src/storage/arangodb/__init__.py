"""
ArangoDB Storage Component

This module provides ArangoDB storage functionality implementing the
Storage protocol.
"""

from .storage import ArangoStorageV2
from .storage import ArangoStorageV2 as ArangoStorage

__all__ = [
    "ArangoStorage",
    "ArangoStorageV2"
]