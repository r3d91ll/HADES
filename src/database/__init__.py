"""
Database Module for HADES

This module provides database connectivity and operations for HADES.
Currently supports ArangoDB for graph-based storage and retrieval.
"""

from .arango_client import ArangoClient

__all__ = ['ArangoClient']