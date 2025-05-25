"""Tests for the repository ingestor orchestrator.

This module tests the RepositoryIngestor class, which orchestrates the ingestion
of code repositories into a knowledge graph with embeddings.
"""

import sys
import os
import pytest
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from unittest.mock import MagicMock, AsyncMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

from src.pipelines.ingest.orchestrator.ingestor import RepositoryIngestor, IngestionStats
from src.isne.types.models import DocumentRelation, RelationType


@pytest.fixture
def mock_repository():
    """Create a mock repository."""
    mock = MagicMock()
    mock.store_node = AsyncMock()
    mock.store_embedding = AsyncMock()
    mock.store_edge = AsyncMock()
    mock.initialize = MagicMock()
    return mock


@pytest.fixture
def mock_connection():
    """Create a mock connection."""
    mock = MagicMock()
    return mock


@pytest.fixture
def mock_doc_processor():
    """Create a mock document processor."""
    mock = MagicMock()
    mock.process_document = MagicMock(return_value={
        "id": "test-doc-id",
        "type": "python",
        "path": "test/file.py",
        "content": "def test(): pass",
        "metadata": {}
    })
    return mock


@pytest.fixture
def mock_embedding_adapter():
    """Create a mock embedding adapter."""
    mock = MagicMock()
    mock.embed = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    return mock


@pytest.fixture
def ingestor(mock_repository, mock_connection, mock_doc_processor, mock_embedding_adapter):
    """Create a repository ingestor with mocked dependencies."""
    ingestor = RepositoryIngestor(
        connection=mock_connection,
        doc_processor=mock_doc_processor,
        embedding_adapter=mock_embedding_adapter,
    )
    ingestor.repository = mock_repository
    return ingestor


def test_ingestion_stats():
    """Test the IngestionStats class."""
    # Create stats and check defaults
    stats = IngestionStats()
    assert stats.total_files == 0
    assert stats.processed_files == 0
    assert stats.ended_at == 0.0
    
    # Test duration calculation
    assert stats.duration() > 0  # Should be positive
    
    # Test marking completion
    stats.mark_complete()
    assert stats.ended_at > 0
    
    # Test to_dict conversion
    stats_dict = stats.to_dict()
    assert "duration_seconds" in stats_dict
    assert "total_files" in stats_dict
    assert "processed_files" in stats_dict


def test_ingestor_initialization(ingestor, mock_repository):
    """Test initializing the RepositoryIngestor."""
    # Test initialization without DB init
    assert ingestor.initialize_db is False
    assert not mock_repository.initialize.called
    
    # Test initialization with DB init
    ingestor_with_init = RepositoryIngestor(
        connection=mock_connection,
        doc_processor=mock_doc_processor,
        embedding_adapter=mock_embedding_adapter,
        initialize_db=True,
    )
    ingestor_with_init.repository = mock_repository
    assert ingestor_with_init.initialize_db is True
    assert mock_repository.initialize.called


def test_determine_doc_type(ingestor):
    """Test determining document type from file extension."""
    assert ingestor._determine_doc_type(Path("test.py")) == "python"
    assert ingestor._determine_doc_type(Path("test.md")) == "markdown"
    assert ingestor._determine_doc_type(Path("test.txt")) == "text"
    assert ingestor._determine_doc_type(Path("test.json")) == "json"
    assert ingestor._determine_doc_type(Path("test.unknown")) == "text"  # Default


@pytest.mark.asyncio
@patch("src.pipelines.ingest.orchestrator.ingestor.chunk_code")
@patch("src.pipelines.ingest.orchestrator.ingestor.chunk_text")
async def test_extract_entities_and_relationships_python(
    mock_chunk_text, mock_chunk_code, ingestor
):
    """Test extracting entities and relationships from a Python document."""
    # Mock chunk_code to return sample chunks
    mock_chunk_code.return_value = [
        {
            "id": "chunk1",
            "type": "python",
            "symbol_type": "function",
            "name": "test_func",
            "path": "test/file.py",
            "content": "def test_func(): pass",
            "line_start": 1,
            "line_end": 2,
            "parent": "file"
        }
    ]
    
    # Test with a Python document
    document = {
        "id": "doc1",
        "type": "python",
        "path": "test/file.py",
        "content": "def test_func(): pass",
        "metadata": {}
    }
    
    entities, relationships = ingestor._extract_entities_and_relationships(document)
    
    # Check that correct chunker was used
    assert mock_chunk_code.called
    assert not mock_chunk_text.called
    
    # Check entities
    assert len(entities) == 2  # Document and one chunk
    assert entities[0]["id"] == "doc1"
    assert entities[0]["type"] == "document"
    assert entities[1]["id"] == "chunk1"
    assert entities[1]["type"] == "chunk"
    
    # Check relationships
    assert len(relationships) == 1
    assert relationships[0].source_id == "doc1"
    assert relationships[0].target_id == "chunk1"
    assert relationships[0].relation_type == RelationType.CONTAINS
