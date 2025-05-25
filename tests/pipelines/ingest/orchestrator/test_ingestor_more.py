"""Additional tests for the repository ingestor orchestrator.

This module provides more test cases for the RepositoryIngestor class
to achieve comprehensive test coverage.
"""

import sys
import os
import pytest
import asyncio
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from unittest.mock import MagicMock, AsyncMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

from src.pipelines.ingest.orchestrator.ingestor import RepositoryIngestor, IngestionStats
from src.isne.types.models import DocumentRelation, RelationType


@pytest.mark.asyncio
@patch("src.pipelines.ingest.orchestrator.ingestor.chunk_code")
@patch("src.pipelines.ingest.orchestrator.ingestor.chunk_text")
async def test_extract_entities_and_relationships_text(
    mock_chunk_text, mock_chunk_code, mock_repository, mock_connection, 
    mock_doc_processor, mock_embedding_adapter
):
    """Test extracting entities and relationships from a text document."""
    ingestor = RepositoryIngestor(
        connection=mock_connection,
        doc_processor=mock_doc_processor,
        embedding_adapter=mock_embedding_adapter,
    )
    ingestor.repository = mock_repository
    
    # Mock chunk_text to return sample chunks
    mock_chunk_text.return_value = [
        {
            "id": "chunk1",
            "type": "markdown",
            "symbol_type": "paragraph",
            "name": "paragraph_1",
            "path": "test/file.md",
            "content": "This is a test paragraph.",
            "line_start": 1,
            "line_end": 1,
            "parent": "doc1"
        }
    ]
    
    # Test with a markdown document
    document = {
        "id": "doc1",
        "type": "markdown",
        "path": "test/file.md",
        "content": "This is a test paragraph.",
        "metadata": {}
    }
    
    entities, relationships = ingestor._extract_entities_and_relationships(document)
    
    # Check that correct chunker was used
    assert not mock_chunk_code.called
    assert mock_chunk_text.called
    
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


@pytest.mark.asyncio
@patch("src.pipelines.ingest.orchestrator.ingestor.chunk_code")
async def test_extract_entities_relationships_with_parent(
    mock_chunk_code, mock_repository, mock_connection, 
    mock_doc_processor, mock_embedding_adapter
):
    """Test extracting entities with parent-child relationships."""
    ingestor = RepositoryIngestor(
        connection=mock_connection,
        doc_processor=mock_doc_processor,
        embedding_adapter=mock_embedding_adapter,
    )
    ingestor.repository = mock_repository
    
    # Mock chunk_code to return chunks with parent relationships
    mock_chunk_code.return_value = [
        {
            "id": "class1",
            "type": "python",
            "symbol_type": "class",
            "name": "TestClass",
            "path": "test/file.py",
            "content": "class TestClass:\n    def method(self): pass",
            "line_start": 1,
            "line_end": 2,
            "parent": "file"
        },
        {
            "id": "method1",
            "type": "python",
            "symbol_type": "function",
            "name": "method",
            "path": "test/file.py",
            "content": "def method(self): pass",
            "line_start": 2,
            "line_end": 2,
            "parent": "class1"
        }
    ]
    
    # Test with a Python document
    document = {
        "id": "doc1",
        "type": "python",
        "path": "test/file.py",
        "content": "class TestClass:\n    def method(self): pass",
        "metadata": {}
    }
    
    entities, relationships = ingestor._extract_entities_and_relationships(document)
    
    # Check entities
    assert len(entities) == 3  # Document, class, and method
    
    # Check relationships - should have doc->class, doc->method, and class->method
    assert len(relationships) == 3
    
    # Check class->method relationship exists
    class_method_rels = [
        r for r in relationships 
        if r.source_id == "class1" and r.target_id == "method1"
    ]
    assert len(class_method_rels) == 1


@pytest.mark.asyncio
async def test_store_single_entity(mock_repository, mock_connection, 
                                  mock_doc_processor, mock_embedding_adapter):
    """Test storing a single entity."""
    ingestor = RepositoryIngestor(
        connection=mock_connection,
        doc_processor=mock_doc_processor,
        embedding_adapter=mock_embedding_adapter,
    )
    ingestor.repository = mock_repository
    
    # Test entity with embedding
    entity = {"id": "test1", "type": "document", "content": "Test content"}
    embedding = [0.1, 0.2, 0.3]
    
    await ingestor._store_single_entity(entity, embedding)
    
    # Check that both node and embedding were stored
    mock_repository.store_node.assert_called_once_with(entity)
    mock_repository.store_embedding.assert_called_once_with("test1", embedding)
    
    # Check stats
    assert ingestor.stats.nodes_created == 1
    assert ingestor.stats.vector_entries_created == 1
    
    # Test entity without embedding
    mock_repository.store_node.reset_mock()
    mock_repository.store_embedding.reset_mock()
    
    entity2 = {"id": "test2", "type": "document", "content": "Test content 2"}
    await ingestor._store_single_entity(entity2)
    
    # Check that only node was stored
    mock_repository.store_node.assert_called_once_with(entity2)
    mock_repository.store_embedding.assert_not_called()
    
    # Check stats
    assert ingestor.stats.nodes_created == 2
    assert ingestor.stats.vector_entries_created == 1


@pytest.mark.asyncio
async def test_store_single_entity_error(mock_repository, mock_connection, 
                                       mock_doc_processor, mock_embedding_adapter):
    """Test error handling when storing a single entity."""
    ingestor = RepositoryIngestor(
        connection=mock_connection,
        doc_processor=mock_doc_processor,
        embedding_adapter=mock_embedding_adapter,
    )
    ingestor.repository = mock_repository
    
    # Mock store_node to raise an exception
    mock_repository.store_node.side_effect = Exception("Test error")
    
    # Test entity with embedding
    entity = {"id": "test1", "type": "document", "content": "Test content"}
    embedding = [0.1, 0.2, 0.3]
    
    # This should handle the exception without raising it
    await ingestor._store_single_entity(entity, embedding)
    
    # Stats should not be incremented
    assert ingestor.stats.nodes_created == 0
    assert ingestor.stats.vector_entries_created == 0


@pytest.mark.asyncio
async def test_store_single_relationship(mock_repository, mock_connection, 
                                        mock_doc_processor, mock_embedding_adapter):
    """Test storing a single relationship."""
    ingestor = RepositoryIngestor(
        connection=mock_connection,
        doc_processor=mock_doc_processor,
        embedding_adapter=mock_embedding_adapter,
    )
    ingestor.repository = mock_repository
    
    # Test relationship
    relation = DocumentRelation(
        source_id="source1",
        target_id="target1",
        relation_type=RelationType.CONTAINS,
        weight=0.8,
        metadata={"test": "metadata"},
    )
    
    await ingestor._store_single_relationship(relation)
    
    # Check that edge was stored with correct parameters
    mock_repository.store_edge.assert_called_once_with(
        source_id="source1",
        target_id="target1",
        edge_type=RelationType.CONTAINS.value,
        properties={
            "weight": 0.8,
            "metadata": {"test": "metadata"},
        },
    )
    
    # Check stats
    assert ingestor.stats.edges_created == 1


@pytest.mark.asyncio
async def test_store_single_relationship_error(mock_repository, mock_connection, 
                                             mock_doc_processor, mock_embedding_adapter):
    """Test error handling when storing a single relationship."""
    ingestor = RepositoryIngestor(
        connection=mock_connection,
        doc_processor=mock_doc_processor,
        embedding_adapter=mock_embedding_adapter,
    )
    ingestor.repository = mock_repository
    
    # Mock store_edge to raise an exception
    mock_repository.store_edge.side_effect = Exception("Test error")
    
    # Test relationship
    relation = DocumentRelation(
        source_id="source1",
        target_id="target1",
        relation_type=RelationType.CONTAINS,
        weight=0.8,
        metadata={},
    )
    
    # This should handle the exception without raising it
    await ingestor._store_single_relationship(relation)
    
    # Stats should not be incremented
    assert ingestor.stats.edges_created == 0


@pytest.mark.asyncio
@patch("pathlib.Path.glob")
async def test_discover_files(mock_glob, mock_repository, mock_connection, 
                             mock_doc_processor, mock_embedding_adapter):
    """Test discovering files in a repository."""
    ingestor = RepositoryIngestor(
        connection=mock_connection,
        doc_processor=mock_doc_processor,
        embedding_adapter=mock_embedding_adapter,
    )
    ingestor.repository = mock_repository
    
    # Create mock file paths
    python_files = [Path("/test/file1.py"), Path("/test/file2.py")]
    markdown_files = [Path("/test/readme.md")]
    excluded_files = [Path("/test/__pycache__/cache.py")]
    
    # Configure mock glob to return different files for different patterns
    def glob_side_effect(pattern):
        if pattern == "**/*.py":
            return python_files + excluded_files
        elif pattern == "**/*.md":
            return markdown_files
        elif pattern == "**/__pycache__/**":
            return excluded_files
        else:
            return []
    
    mock_glob.side_effect = glob_side_effect
    
    # Test with default patterns
    with patch("pathlib.Path.exists", return_value=True):
        files = await ingestor.discover_files("/test")
    
    # Should have Python and Markdown files, but not excluded files
    assert set(files) == set(python_files + markdown_files)
    assert ingestor.stats.total_files == len(files)


@pytest.mark.asyncio
@patch("pathlib.Path.exists")
async def test_discover_files_nonexistent_path(mock_exists, mock_repository, mock_connection, 
                                             mock_doc_processor, mock_embedding_adapter):
    """Test discovering files with a non-existent path."""
    ingestor = RepositoryIngestor(
        connection=mock_connection,
        doc_processor=mock_doc_processor,
        embedding_adapter=mock_embedding_adapter,
    )
    ingestor.repository = mock_repository
    
    # Make Path.exists return False
    mock_exists.return_value = False
    
    # Should raise ValueError
    with pytest.raises(ValueError):
        await ingestor.discover_files("/nonexistent")


@pytest.mark.asyncio
@patch("src.pipelines.ingest.orchestrator.ingestor.batch_embed")
async def test_generate_embeddings(mock_batch_embed, mock_repository, mock_connection, 
                                 mock_doc_processor, mock_embedding_adapter):
    """Test generating embeddings for entities."""
    ingestor = RepositoryIngestor(
        connection=mock_connection,
        doc_processor=mock_doc_processor,
        embedding_adapter=mock_embedding_adapter,
    )
    ingestor.repository = mock_repository
    
    # Configure mock batch_embed
    mock_batch_embed.return_value = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]
    
    # Test entities
    entities = [
        {"id": "entity1", "content": "Content 1"},
        {"id": "entity2", "content": "Content 2"},
        {"id": "entity3", "type": "document"},  # No content
    ]
    
    embeddings = await ingestor.generate_embeddings(entities)
    
    # Only entities with content should be processed
    assert len(embeddings) == 2
    assert "entity1" in embeddings
    assert "entity2" in embeddings
    assert "entity3" not in embeddings
    
    # Check stats
    assert ingestor.stats.embeddings_created == 2


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
