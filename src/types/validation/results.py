"""
Type definitions for validation results and reports.

This module contains TypedDict definitions for structured validation
result data returned by validation functions.
"""

from typing import Dict, List, TypedDict

from .base import ChunkIdentifier


class PreValidationResult(TypedDict):
    """Results from pre-ISNE validation."""
    total_docs: int
    docs_with_chunks: int
    total_chunks: int
    chunks_with_base_embeddings: int
    existing_isne: int
    missing_base_embeddings: int
    missing_base_embedding_ids: List[ChunkIdentifier]


class PostValidationResult(TypedDict):
    """Results from post-ISNE validation."""
    chunks_with_isne: int
    chunks_missing_isne: int
    chunks_missing_isne_ids: List[ChunkIdentifier]
    chunks_with_relationships: int
    chunks_missing_relationships: int
    chunks_missing_relationship_ids: List[ChunkIdentifier]
    chunks_with_invalid_relationships: int
    chunks_with_invalid_relationship_ids: List[ChunkIdentifier]
    total_relationships: int
    doc_level_isne: int
    total_isne_count: int
    duplicate_isne: int
    duplicate_chunk_ids: List[ChunkIdentifier]


class ValidationDiscrepancies(TypedDict):
    """Discrepancies found between validation stages."""
    isne_vs_chunks: int
    missing_isne: int
    doc_level_isne: int
    duplicate_isne: int


class ValidationSummary(TypedDict):
    """Comprehensive validation summary comparing pre and post states."""
    pre_validation: PreValidationResult
    post_validation: PostValidationResult
    discrepancies: ValidationDiscrepancies