"""
Tests for the repository ingestor orchestrator.

This module tests the main orchestrator class that coordinates
the ingestion pipeline with a focus on type safety.
"""
import os
from typing import Dict, Any, List, cast
from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch, ANY

from src.types.common import StorageConfig, PreProcessorConfig, NodeData, EdgeData, IngestStats
from src.ingest.orchestrator.ingestor import RepositoryIngestor


class TestRepositoryIngestorInit:
    """Test the initialization of the RepositoryIngestor class."""

    @patch('src.ingest.orchestrator.ingestor.ArangoConnection')
    def test_init_with_defaults(self, mock_conn_cls):
        """Test initialization with default parameters."""
        # Mock the bootstrap method
        mock_conn = MagicMock()
        mock_conn_cls.bootstrap.return_value = mock_conn
        
        # Initialize with no configs
        ingestor = RepositoryIngestor()
        
        # Verify default configs were created - use dict checks instead of isinstance
        # (TypedDict doesn't support runtime instance checks)
        assert isinstance(ingestor.storage_config, dict)
        assert isinstance(ingestor.preprocessor_config, dict)
        
        # Verify components are initialized but might be None if they require external resources
        assert hasattr(ingestor, 'repository')
        assert hasattr(ingestor, 'embedding_service')
        assert hasattr(ingestor, 'file_processor')
        assert hasattr(ingestor, 'preprocessor_manager')

    def test_init_with_configs(self, sample_storage_config, sample_preprocessor_config):
        """Test initialization with provided configurations."""
        # Initialize with provided configs
        ingestor = RepositoryIngestor(
            storage_config=sample_storage_config,
            preprocessor_config=sample_preprocessor_config
        )
        
        # Verify configs are set correctly
        # Compare as dicts, not TypedDict objects
        assert dict(ingestor.storage_config) == dict(sample_storage_config)
        assert dict(ingestor.preprocessor_config) == dict(sample_preprocessor_config)

    @patch('src.ingest.orchestrator.ingestor.ArangoConnection')
    @patch('src.ingest.orchestrator.ingestor.ArangoRepository')
    def test_initialize_repository(self, mock_repo_cls, mock_conn_cls, sample_storage_config):
        """Test repository initialization."""
        # Create standalone mocks for this test
        mock_conn = MagicMock()
        mock_repo = MagicMock()
        mock_conn_cls.bootstrap.return_value = mock_conn
        mock_repo_cls.return_value = mock_repo
        
        # Create the ingestor with a dict copy of sample_storage_config
        config_dict = dict(sample_storage_config)
        ingestor = RepositoryIngestor(storage_config=config_dict)
        
        # Verify connection was initialized with correct params
        mock_conn_cls.bootstrap.assert_called_once()
        args, kwargs = mock_conn_cls.bootstrap.call_args
        assert kwargs['config']['database'] == config_dict.get('database')
        # Host might be modified with port in constructor, so just verify it contains the base host
        assert config_dict.get('host') in kwargs['config']['host']
        
        # Verify repository was initialized with connection
        mock_repo_cls.assert_called_once()
        args, kwargs = mock_repo_cls.call_args
        assert kwargs['connection'] == mock_conn
        
        # Verify repository is set
        assert ingestor.repository == mock_repo

    @patch('src.ingest.orchestrator.ingestor.EmbeddingProcessor')
    def test_initialize_embedding_service(self, mock_processor_cls, sample_storage_config):
        """Test embedding service initialization."""
        # Mock for embedding processor
        mock_processor = MagicMock()
        mock_processor_cls.return_value = mock_processor
        
        # Add embedding config
        config = dict(sample_storage_config)
        config['embedding'] = {
            'model_name': 'test-model',
            'model_dimension': 768,
            'normalize_embeddings': True
        }
        
        # Create the ingestor
        ingestor = RepositoryIngestor(storage_config=StorageConfig(**config))
        
        # Verify embedding processor was initialized
        mock_processor_cls.assert_called_once()
        
        # Verify embedding service is set
        assert ingestor.embedding_service == mock_processor


class TestRepositoryIngestorIngest:
    """Test the ingestion functionality."""
    
    @pytest.fixture
    def mock_ingestor(self, mock_file_processor, mock_preprocessor_manager, 
                     mock_arango_connection, mock_embedding_processor):
        """Provide a pre-configured ingestor with mocks."""
        with patch('src.ingest.orchestrator.ingestor.ArangoConnection') as mock_conn_cls, \
             patch('src.ingest.orchestrator.ingestor.ArangoRepository') as mock_repo_cls, \
             patch('src.ingest.orchestrator.ingestor.FileProcessor') as mock_fp_cls, \
             patch('src.ingest.orchestrator.ingestor.PreprocessorManager') as mock_pm_cls, \
             patch('src.ingest.orchestrator.ingestor.EmbeddingProcessor') as mock_ep_cls:
            # Set up the mocks
            mock_repo = MagicMock()
            mock_repo_cls.return_value = mock_repo
            mock_conn_cls.return_value = mock_arango_connection
            mock_fp_cls.return_value = mock_file_processor
            mock_pm_cls.return_value = mock_preprocessor_manager
            mock_ep_cls.return_value = mock_embedding_processor
            
            # Create ingestor
            ingestor = RepositoryIngestor()
            
            # Manually set up embedding service to have embedding_fn
            ingestor.embedding_service = mock_embedding_processor
            
            # Return the ingestor and main mock components
            yield ingestor

    def test_ingest_directory_empty(self, mock_ingestor, temp_test_dir):
        """Test ingestion with no files."""
        # Configure file processor to return empty batches
        mock_ingestor.file_processor.process_directory.return_value = []
        
        # Run ingestion
        stats = mock_ingestor.ingest(temp_test_dir)
        
        # Verify stats contain expected fields for empty case
        # Use dict check instead of TypedDict runtime check
        assert isinstance(stats, dict)
        assert 'start_time' in stats
        assert 'end_time' in stats
        assert stats.get('status') == 'aborted_no_files'
        assert stats.get('files_discovered') == 0

    def test_ingest_full_process(self, mock_ingestor, temp_test_dir, 
                                sample_documents, sample_relationships):
        """Test full ingestion process with mocked components."""
        # Configure file processor to return file batches
        file_batch = {
            'python': [Path('file1.py'), Path('file2.py')],
            'markdown': [Path('doc1.md')]
        }
        mock_ingestor.file_processor.process_directory.return_value = [file_batch]
        
        # Configure preprocessor manager to return processing results
        mock_ingestor.preprocessor_manager.preprocess_batch.return_value = {
            'python': [{'path': 'file1.py', 'content': 'code'}],
            'markdown': [{'path': 'doc1.md', 'content': 'docs'}]
        }
        
        # Configure extraction
        mock_ingestor.preprocessor_manager.extract_entities_and_relationships.return_value = {
            'entities': sample_documents,
            'relationships': sample_relationships
        }
        
        # Configure repository stats
        mock_ingestor.repository.collection_stats.return_value = {
            'nodes': 10,
            'edges': 5,
            'vector_coverage': 0.8
        }
        
        # Run ingestion
        stats = mock_ingestor.ingest(temp_test_dir, dataset_name="test_dataset")
        
        # Verify components were called with correct arguments
        mock_ingestor.file_processor.process_directory.assert_called_once_with(
            temp_test_dir, batch_size=ANY
        )
        
        mock_ingestor.preprocessor_manager.preprocess_batch.assert_called_once()
        
        # Verify extraction was performed
        mock_ingestor.preprocessor_manager.extract_entities_and_relationships.assert_called_once()
        
        # Verify entities were stored
        assert mock_ingestor.repository.store_document.call_count == len(sample_documents)
        
        # Verify relationships were stored
        assert mock_ingestor.repository.create_edge.call_count >= len(sample_relationships)
        
        # Verify stats are populated
        assert stats.get('entities_stored') == len(sample_documents)
        assert stats.get('relationships_stored') == len(sample_relationships)
        assert stats.get('status') == 'completed'
        assert 'repository_stats' in stats
        
    def test_store_entities_with_embeddings(self, mock_ingestor, sample_documents):
        """Test storing entities with embedding generation."""
        # Configure repository
        mock_ingestor.repository.store_document.return_value = "doc123"
        
        # Run method
        mock_ingestor._store_entities(sample_documents, "test_dataset")
        
        # Verify document storage
        assert mock_ingestor.repository.store_document.call_count == len(sample_documents)
        
        # Verify embedding generation and storage
        call_count = sum(1 for doc in sample_documents if doc.get('content'))
        assert mock_ingestor.embedding_service.embedding_fn.call_count == call_count
        assert mock_ingestor.repository.store_embedding.call_count == call_count

    def test_store_relationships(self, mock_ingestor, sample_relationships):
        """Test storing relationships."""
        # Run method
        mock_ingestor._store_relationships(sample_relationships)
        
        # Count bidirectional relationships to check for reciprocal edges
        bidirectional_count = sum(1 for rel in sample_relationships if rel.get('bidirectional'))
        
        # Verify edge creation (+1 for each bidirectional)
        expected_calls = len(sample_relationships) + bidirectional_count
        assert mock_ingestor.repository.create_edge.call_count == expected_calls


class TestRepositoryIngestorErrors:
    """Test error handling in the RepositoryIngestor class."""
    
    @patch('src.ingest.orchestrator.ingestor.ArangoConnection')
    def test_repository_initialization_error(self, mock_conn_cls, sample_storage_config):
        """Test error handling during repository initialization."""
        # Mock bootstrap to raise exception
        mock_conn_cls.bootstrap.side_effect = Exception("Connection failed")
        
        # Initialization should raise the error
        with pytest.raises(Exception) as exc_info:
            ingestor = RepositoryIngestor(storage_config=sample_storage_config)
            
        # Verify exception message
        assert "Connection failed" in str(exc_info.value)

    @pytest.fixture
    def partial_mock_ingestor(self):
        """Provide an ingestor with mocked components but real initialization."""
        with patch('src.ingest.orchestrator.ingestor.ArangoConnection'), \
             patch('src.ingest.orchestrator.ingestor.ArangoRepository'), \
             patch('src.ingest.orchestrator.ingestor.FileProcessor'), \
             patch('src.ingest.orchestrator.ingestor.PreprocessorManager'), \
             patch('src.ingest.orchestrator.ingestor.EmbeddingProcessor'):
                
            # Create ingestor without raising errors
            ingestor = RepositoryIngestor()
            
            # Mock specific components after initialization
            ingestor.repository = MagicMock()
            mock_embedding = MagicMock()
            mock_embedding.embedding_fn = MagicMock(return_value=[[0.1, 0.2, 0.3]])
            mock_embedding.embedding_config = MagicMock()
            mock_embedding.embedding_config.model_name = "test-model"
            ingestor.embedding_service = mock_embedding
            ingestor.file_processor = MagicMock()
            ingestor.preprocessor_manager = MagicMock()
            
            yield ingestor
            
    def test_embedding_error(self, partial_mock_ingestor, sample_documents):
        """Test graceful handling of embedding errors."""
        # Configure embedding to fail
        partial_mock_ingestor.embedding_service.embedding_fn.side_effect = Exception("Embedding failed")
        
        # Should not raise, just log error
        partial_mock_ingestor._store_entities(sample_documents, "test_dataset")
        
        # Documents should still be stored
        assert partial_mock_ingestor.repository.store_document.call_count == len(sample_documents)
        # But no embeddings stored
        assert partial_mock_ingestor.repository.store_embedding.call_count == 0
