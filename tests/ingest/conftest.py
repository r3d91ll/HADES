"""
Pytest fixtures for testing the ingest module components.

This file provides shared fixtures used by multiple test files
in the ingest module, ensuring consistent test setup.
"""
import os
import tempfile
from typing import Dict, Any, Generator, List
from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch

from src.types.common import StorageConfig, PreProcessorConfig, NodeData, EdgeData
from src.storage.arango.connection import ArangoConnection
from src.ingest.repository.arango_repository import ArangoRepository
from src.ingest.pre_processor.file_processor import FileProcessor
from src.ingest.pre_processor.manager import PreprocessorManager
from src.isne.processors.embedding_processor import EmbeddingProcessor


@pytest.fixture
def temp_test_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_storage_config() -> StorageConfig:
    """Provide a sample storage configuration."""
    return StorageConfig(
        storage_type="arango",
        host="localhost",
        port=8529,
        username="root",
        password="",
        database="test_pathrag",
        collection_prefix="test_",
        use_vector_index=True,
        vector_dimensions=768
    )


@pytest.fixture
def sample_preprocessor_config() -> PreProcessorConfig:
    """Provide a sample preprocessor configuration."""
    return PreProcessorConfig(
        max_workers=2,
        recursive=True,
        file_type_map={
            "py": ["python"],
            "md": ["markdown"]
        }
    )


@pytest.fixture
def mock_arango_connection() -> MagicMock:
    """Provide a mock ArangoConnection."""
    mock_conn = MagicMock()
    # Configure common mock behaviors
    mock_conn.create_collection.return_value = True
    mock_conn.create_edge_collection.return_value = True
    mock_conn.create_graph.return_value = True
    mock_conn.insert_document.return_value = "doc123"
    mock_conn.insert_edge.return_value = "edge123"
    mock_conn.update_document.return_value = True
    mock_conn.get_document.return_value = {"_id": "doc123", "content": "test content"}
    mock_conn.collection_count.return_value = 10
    mock_conn.query.return_value = [{"type": "code", "count": 5}]
    mock_conn.has_index.return_value = True
    mock_conn.vector_search.return_value = [{"_id": "doc1", "_score": 0.9}]
    mock_conn.traverse_graph.return_value = {"vertices": [], "edges": []}
    return mock_conn


@pytest.fixture
def mock_file_processor() -> MagicMock:
    """Provide a mock FileProcessor."""
    mock_processor = MagicMock(spec=FileProcessor)
    # Configure common mock behaviors
    return mock_processor


@pytest.fixture
def mock_preprocessor_manager() -> MagicMock:
    """Provide a mock PreprocessorManager."""
    mock_manager = MagicMock(spec=PreprocessorManager)
    # Configure common mock behaviors
    return mock_manager


@pytest.fixture
def mock_embedding_processor() -> MagicMock:
    """Provide a mock EmbeddingProcessor."""
    mock_embedding = MagicMock()
    # Configure embedding function to return simple embeddings
    mock_embedding.embedding_fn = MagicMock(return_value=[[0.1, 0.2, 0.3]])
    mock_embedding.embedding_config = MagicMock()
    mock_embedding.embedding_config.model_name = "test-model"
    return mock_embedding


@pytest.fixture
def sample_documents() -> List[Dict[str, Any]]:
    """Provide sample document data for testing."""
    return [
        {
            "type": "code_file",
            "content": "def hello(): print('world')",
            "title": "Sample Code",
            "source": "test_repo",
            "metadata": {"language": "python"}
        },
        {
            "type": "documentation",
            "content": "# Documentation\nThis is sample documentation.",
            "title": "Sample Doc",
            "source": "test_repo",
            "metadata": {"format": "markdown"}
        }
    ]


@pytest.fixture
def sample_relationships() -> List[Dict[str, Any]]:
    """Provide sample relationship data for testing."""
    return [
        {
            "source_id": "doc1",
            "target_id": "doc2",
            "type": "references",
            "weight": 0.8,
            "bidirectional": False,
            "metadata": {}
        },
        {
            "source_id": "doc2",
            "target_id": "doc3",
            "type": "contains",
            "weight": 1.0,
            "bidirectional": True,
            "metadata": {"details": "nested content"}
        }
    ]
