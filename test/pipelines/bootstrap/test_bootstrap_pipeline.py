"""
Tests for Bootstrap Pipeline

Tests the complete Sequential-ISNE Bootstrap Pipeline including directory analysis,
file processing, and graph construction.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone

from src.pipelines.bootstrap import BootstrapPipeline, BootstrapConfig
from src.pipelines.bootstrap.config import FileTypeFilter, EdgeDiscoveryMethod
from src.storage.incremental.sequential_isne_types import FileType, EdgeType


@pytest.fixture
def temp_project_dir():
    """Create a temporary project directory with test files."""
    temp_dir = Path(tempfile.mkdtemp())
    
    # Create directory structure
    (temp_dir / "src").mkdir()
    (temp_dir / "src" / "core").mkdir()
    (temp_dir / "docs").mkdir()
    (temp_dir / "config").mkdir()
    
    # Create code files
    (temp_dir / "src" / "main.py").write_text("""
import os
from core.utils import helper_function

def main():
    '''Main entry point'''
    return helper_function()

class DataProcessor:
    def process(self, data):
        return data.upper()
""")
    
    (temp_dir / "src" / "core" / "utils.py").write_text("""
def helper_function():
    '''Helper function'''
    return "processed"

class Utils:
    pass
""")
    
    # Create documentation files
    (temp_dir / "docs" / "README.md").write_text("""
# Project Documentation

This project processes data using the DataProcessor class.

## Usage

```python
from src.main import DataProcessor
processor = DataProcessor()
```

See [utils](../src/core/utils.py) for helper functions.
""")
    
    (temp_dir / "docs" / "api.md").write_text("""
# API Reference

## main.py

- `main()`: Entry point function
- `DataProcessor`: Main processing class

## core/utils.py

- `helper_function()`: Utility function
""")
    
    # Create config files
    (temp_dir / "config" / "settings.json").write_text("""
{
    "app_name": "test_project",
    "main_module": "src.main",
    "documentation_path": "docs/README.md"
}
""")
    
    (temp_dir / "config" / "pipeline.yaml").write_text("""
pipeline:
  stages:
    - name: "processing"
      module: "src.core.utils"
""")
    
    yield temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def bootstrap_config(temp_project_dir):
    """Create bootstrap configuration for testing."""
    return BootstrapConfig(
        input_directory=temp_project_dir,
        output_database="test_bootstrap",
        file_type_filter=FileTypeFilter.ALL,
        max_file_size=1024 * 1024,  # 1MB
        batch_size=10,
        max_workers=2,
        enable_parallel_processing=False,  # Disable for deterministic tests
        enable_cross_modal_discovery=True,
        semantic_similarity_threshold=0.7,
        edge_discovery_methods=[
            EdgeDiscoveryMethod.DIRECTORY_STRUCTURE,
            EdgeDiscoveryMethod.IMPORT_ANALYSIS,
            EdgeDiscoveryMethod.CO_LOCATION,
            EdgeDiscoveryMethod.FILE_REFERENCES
        ],
        storage_component="arangodb_v2"
    )


@pytest.fixture
def mock_storage():
    """Create mock storage for testing."""
    storage = Mock()
    storage.connect.return_value = True
    storage.disconnect.return_value = True
    storage.health_check.return_value = True
    storage.store_modality_file.return_value = "test_file_id"
    storage.create_cross_modal_edge.return_value = "test_edge_id"
    storage.get_statistics.return_value = {
        "total_files": 6,
        "total_edges": 10
    }
    storage.get_modality_statistics.return_value = {
        "total_files": 6,
        "cross_modal_edges": 4,
        "modality_distribution": {
            "code": 2,
            "documentation": 2,
            "config": 2
        }
    }
    return storage


class TestBootstrapConfig:
    """Test BootstrapConfig validation and creation."""
    
    def test_config_creation_with_defaults(self, temp_project_dir):
        """Test config creation with default values."""
        config = BootstrapConfig(input_directory=temp_project_dir)
        
        assert config.input_directory == temp_project_dir
        assert config.output_database == "sequential_isne"
        assert config.file_type_filter == FileTypeFilter.ALL
        assert config.enable_cross_modal_discovery is True
        assert config.batch_size == 100
        assert config.max_workers == 4
    
    def test_config_validation_invalid_directory(self):
        """Test config validation with invalid directory."""
        with pytest.raises(ValueError):
            BootstrapConfig(input_directory="/nonexistent/directory")
    
    def test_config_edge_discovery_methods(self, temp_project_dir):
        """Test edge discovery methods configuration."""
        config = BootstrapConfig(
            input_directory=temp_project_dir,
            edge_discovery_methods=[
                EdgeDiscoveryMethod.IMPORT_ANALYSIS,
                EdgeDiscoveryMethod.CO_LOCATION
            ]
        )
        
        assert EdgeDiscoveryMethod.IMPORT_ANALYSIS in config.edge_discovery_methods
        assert EdgeDiscoveryMethod.CO_LOCATION in config.edge_discovery_methods
        assert EdgeDiscoveryMethod.DIRECTORY_STRUCTURE not in config.edge_discovery_methods


class TestBootstrapPipeline:
    """Test the main Bootstrap Pipeline functionality."""
    
    @patch('src.pipelines.bootstrap.pipeline.create_storage_component')
    def test_pipeline_initialization(self, mock_create_storage, bootstrap_config):
        """Test pipeline initialization."""
        pipeline = BootstrapPipeline(bootstrap_config)
        
        assert pipeline.config == bootstrap_config
        assert pipeline.directory_analyzer is not None
        assert pipeline.file_processor is not None
        assert pipeline.storage is None  # Not initialized until execution
    
    @patch('src.pipelines.bootstrap.pipeline.create_storage_component')
    def test_pipeline_dry_run(self, mock_create_storage, bootstrap_config):
        """Test pipeline dry run functionality."""
        pipeline = BootstrapPipeline(bootstrap_config)
        
        result = pipeline.dry_run()
        
        assert "analysis" in result
        assert "config" in result
        assert "estimated_processing_time" in result
        assert "recommendations" in result
        
        analysis = result["analysis"]
        assert analysis["total_files"] == 6  # 2 code + 2 docs + 2 config
        assert "file_types" in analysis
        assert "total_relationships" in analysis
        assert "edge_types" in analysis
    
    @patch('src.pipelines.bootstrap.pipeline.create_storage_component')
    def test_pipeline_execute_success(self, mock_create_storage, bootstrap_config, mock_storage):
        """Test successful pipeline execution."""
        mock_create_storage.return_value = mock_storage
        
        pipeline = BootstrapPipeline(bootstrap_config)
        result = pipeline.execute()
        
        assert result.success is True
        assert result.database_name == "test_bootstrap"
        assert result.node_count > 0
        assert result.edge_count > 0
        assert len(result.errors) == 0
        
        # Verify storage calls
        mock_storage.connect.assert_called_once()
        assert mock_storage.store_modality_file.call_count > 0
        mock_storage.disconnect.assert_called_once()
    
    @patch('src.pipelines.bootstrap.pipeline.create_storage_component')
    def test_pipeline_execute_storage_failure(self, mock_create_storage, bootstrap_config):
        """Test pipeline execution with storage failure."""
        mock_storage = Mock()
        mock_storage.connect.return_value = False
        mock_create_storage.return_value = mock_storage
        
        pipeline = BootstrapPipeline(bootstrap_config)
        result = pipeline.execute()
        
        assert result.success is False
        assert len(result.errors) > 0
        assert "Storage initialization failed" in result.errors[0]
    
    @patch('src.pipelines.bootstrap.pipeline.create_storage_component')
    def test_pipeline_create_from_directory(self, mock_create_storage, temp_project_dir):
        """Test pipeline creation from directory."""
        pipeline = BootstrapPipeline.create_from_directory(
            temp_project_dir,
            output_database="custom_db",
            batch_size=50
        )
        
        assert pipeline.config.input_directory == temp_project_dir
        assert pipeline.config.output_database == "custom_db"
        assert pipeline.config.batch_size == 50


class TestDirectoryAnalyzer:
    """Test directory analysis functionality."""
    
    def test_file_discovery(self, bootstrap_config):
        """Test file discovery and classification."""
        from src.pipelines.bootstrap.directory_analyzer import DirectoryAnalyzer
        
        analyzer = DirectoryAnalyzer(bootstrap_config)
        discovered_files, relationships = analyzer.analyze_directory(bootstrap_config.input_directory)
        
        # Should discover 6 files (2 code, 2 docs, 2 config)
        assert len(discovered_files) == 6
        
        # Check file types are classified correctly
        file_types = [info['file_type'] for info in discovered_files.values()]
        assert FileType.CODE in file_types
        assert FileType.DOCUMENTATION in file_types
        assert FileType.CONFIG in file_types
    
    def test_import_analysis(self, bootstrap_config):
        """Test import statement analysis."""
        from src.pipelines.bootstrap.directory_analyzer import DirectoryAnalyzer
        
        analyzer = DirectoryAnalyzer(bootstrap_config)
        discovered_files, relationships = analyzer.analyze_directory(bootstrap_config.input_directory)
        
        # Should find import relationships
        import_edges = [rel for rel in relationships if rel[2] == EdgeType.CODE_IMPORTS.value]
        assert len(import_edges) > 0
    
    def test_co_location_analysis(self, bootstrap_config):
        """Test co-location analysis."""
        from src.pipelines.bootstrap.directory_analyzer import DirectoryAnalyzer
        
        analyzer = DirectoryAnalyzer(bootstrap_config)
        discovered_files, relationships = analyzer.analyze_directory(bootstrap_config.input_directory)
        
        # Should find co-location relationships
        colocation_edges = [rel for rel in relationships 
                          if rel[4].get('source') == 'co_location']
        assert len(colocation_edges) > 0


class TestFileProcessor:
    """Test file processing functionality."""
    
    def test_sequential_processing(self, bootstrap_config):
        """Test sequential file processing."""
        from src.pipelines.bootstrap.file_processor import FileProcessor
        from src.pipelines.bootstrap.directory_analyzer import DirectoryAnalyzer
        
        # First discover files
        analyzer = DirectoryAnalyzer(bootstrap_config)
        discovered_files, _ = analyzer.analyze_directory(bootstrap_config.input_directory)
        
        # Process files
        processor = FileProcessor(bootstrap_config)
        processed_files = list(processor.process_files(discovered_files))
        
        assert len(processed_files) == 6
        
        # Check that different file types are created
        file_types = [type(file_obj).__name__ for _, file_obj, _ in processed_files]
        assert "CodeFile" in file_types
        assert "DocumentationFile" in file_types
        assert "ConfigFile" in file_types
    
    def test_chunk_creation(self, bootstrap_config):
        """Test text chunk creation."""
        from src.pipelines.bootstrap.file_processor import FileProcessor
        from src.pipelines.bootstrap.directory_analyzer import DirectoryAnalyzer
        
        # Use smaller chunk size for testing
        bootstrap_config.chunk_size = 100
        bootstrap_config.chunk_overlap = 20
        
        analyzer = DirectoryAnalyzer(bootstrap_config)
        discovered_files, _ = analyzer.analyze_directory(bootstrap_config.input_directory)
        
        processor = FileProcessor(bootstrap_config)
        processed_files = list(processor.process_files(discovered_files))
        
        # Check that chunks are created
        for file_path, file_obj, chunks in processed_files:
            if len(file_obj.content) > 100:  # Only files larger than chunk size
                assert len(chunks) > 1
                
                # Check chunk properties
                for chunk in chunks:
                    assert hasattr(chunk, 'content')
                    assert hasattr(chunk, 'chunk_index')
                    assert hasattr(chunk, 'start_pos')
                    assert hasattr(chunk, 'end_pos')


class TestGraphBuilder:
    """Test graph building functionality."""
    
    def test_graph_building(self, bootstrap_config, mock_storage):
        """Test graph construction from processed files."""
        from src.pipelines.bootstrap.graph_builder import GraphBuilder
        from src.pipelines.bootstrap.directory_analyzer import DirectoryAnalyzer
        from src.pipelines.bootstrap.file_processor import FileProcessor
        
        # Create test data
        analyzer = DirectoryAnalyzer(bootstrap_config)
        discovered_files, relationships = analyzer.analyze_directory(bootstrap_config.input_directory)
        
        processor = FileProcessor(bootstrap_config)
        processed_files = list(processor.process_files(discovered_files))
        
        # Build graph
        builder = GraphBuilder(bootstrap_config, mock_storage)
        metrics = builder.build_graph(processed_files, relationships)
        
        # Check metrics
        assert metrics.total_files_processed == 6
        assert metrics.total_nodes_created > 0
        assert metrics.total_edges_created > 0
        assert metrics.processing_errors == 0
    
    def test_cross_modal_edge_detection(self, bootstrap_config, mock_storage):
        """Test cross-modal edge detection."""
        from src.pipelines.bootstrap.graph_builder import GraphBuilder
        
        builder = GraphBuilder(bootstrap_config, mock_storage)
        
        # Test cross-modal edge type detection
        assert builder._is_cross_modal_edge(EdgeType.CODE_TO_DOC.value) is True
        assert builder._is_cross_modal_edge(EdgeType.DOC_TO_CODE.value) is True
        assert builder._is_cross_modal_edge(EdgeType.CODE_IMPORTS.value) is False
        assert builder._is_cross_modal_edge(EdgeType.DIRECTORY_COLOCATION.value) is False
    
    def test_graph_validation(self, bootstrap_config, mock_storage):
        """Test graph validation."""
        from src.pipelines.bootstrap.graph_builder import GraphBuilder
        
        builder = GraphBuilder(bootstrap_config, mock_storage)
        builder.metrics.total_nodes_created = 6
        builder.metrics.cross_modal_edges = 2
        builder.metrics.modality_distribution = {
            "code": 2,
            "documentation": 2,
            "config": 2
        }
        
        assert builder.validate_graph() is True
        
        # Test validation failure
        mock_storage.health_check.return_value = False
        assert builder.validate_graph() is False


@pytest.mark.integration
class TestBootstrapPipelineIntegration:
    """Integration tests for the complete pipeline."""
    
    @patch('src.pipelines.bootstrap.pipeline.create_storage_component')
    def test_end_to_end_pipeline(self, mock_create_storage, bootstrap_config, mock_storage):
        """Test complete end-to-end pipeline execution."""
        mock_create_storage.return_value = mock_storage
        
        pipeline = BootstrapPipeline(bootstrap_config)
        result = pipeline.execute()
        
        # Verify successful execution
        assert result.success is True
        assert result.node_count == 6  # From mock storage statistics
        assert result.edge_count > 0
        
        # Verify all phases completed
        assert mock_storage.connect.called
        assert mock_storage.store_modality_file.call_count == 6
        assert mock_storage.create_cross_modal_edge.call_count > 0
        assert mock_storage.disconnect.called
    
    @patch('src.pipelines.bootstrap.pipeline.create_storage_component')
    def test_pipeline_with_file_type_filter(self, mock_create_storage, temp_project_dir, mock_storage):
        """Test pipeline with file type filtering."""
        config = BootstrapConfig(
            input_directory=temp_project_dir,
            file_type_filter=FileTypeFilter.CODE_ONLY
        )
        
        mock_create_storage.return_value = mock_storage
        
        pipeline = BootstrapPipeline(config)
        result = pipeline.execute()
        
        assert result.success is True
        # Should process only code files
        assert mock_storage.store_modality_file.call_count == 2
    
    @patch('src.pipelines.bootstrap.pipeline.create_storage_component')
    def test_pipeline_progress_tracking(self, mock_create_storage, bootstrap_config, mock_storage):
        """Test pipeline progress tracking."""
        mock_create_storage.return_value = mock_storage
        
        pipeline = BootstrapPipeline(bootstrap_config)
        
        # Get initial progress
        initial_progress = pipeline.get_progress_info()
        assert initial_progress["pipeline_status"] == "running"
        
        # Execute pipeline
        result = pipeline.execute()
        
        # Get final progress
        final_progress = pipeline.get_progress_info()
        assert "file_processor" in final_progress
        assert "graph_builder" in final_progress