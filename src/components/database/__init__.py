"""
Database Components

This module contains database connector components that implement the
DatabaseConnector protocol for different database backends.

Available components:
- arango: ArangoDB connector
"""

from .arango import ArangoConnector

__all__ = [
    "ArangoConnector"
]