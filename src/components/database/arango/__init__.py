"""
ArangoDB Database Component

This module provides ArangoDB database connectivity implementing the
DatabaseConnector protocol.
"""

from .client import ArangoConnector

__all__ = [
    "ArangoConnector"
]