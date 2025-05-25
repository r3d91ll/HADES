"""
Database module for HADES-PathRAG.

This module provides database access and operations for the pipeline.
"""

from src.database.arango_client import ArangoClient

__all__ = ["ArangoClient"]
