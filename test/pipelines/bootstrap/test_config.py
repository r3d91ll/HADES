"""
Tests for Bootstrap Pipeline Configuration

Tests configuration validation, serialization, and edge cases.
"""

import pytest
import tempfile
from pathlib import Path
from pydantic import ValidationError

from src.pipelines.bootstrap.config import (
    BootstrapConfig, BootstrapMetrics, BootstrapResult,
    FileTypeFilter, EdgeDiscoveryMethod
)


class TestBootstrapConfig:
    """Test BootstrapConfig validation and functionality."""
    
    def test_valid_config_creation(self, tmp_path):
        """Test creating a valid configuration."""
        config = BootstrapConfig(
            input_directory=tmp_path,
            output_database="test_db",
            file_type_filter=FileTypeFilter.ALL,
            max_file_size=1024 * 1024,
            batch_size=50,
            max_workers=2,
            enable_cross_modal_discovery=True,
            semantic_similarity_threshold=0.8
        )
        
        assert config.input_directory == tmp_path
        assert config.output_database == "test_db"
        assert config.file_type_filter == FileTypeFilter.ALL
        assert config.max_file_size == 1024 * 1024
        assert config.batch_size == 50
        assert config.max_workers == 2
        assert config.enable_cross_modal_discovery is True
        assert config.semantic_similarity_threshold == 0.8
    
    def test_config_with_defaults(self, tmp_path):
        """Test configuration with default values."""
        config = BootstrapConfig(input_directory=tmp_path)
        
        assert config.output_database == "sequential_isne"
        assert config.file_type_filter == FileTypeFilter.ALL
        assert config.max_file_size == 10 * 1024 * 1024  # 10MB
        assert config.batch_size == 100
        assert config.max_workers == 4
        assert config.enable_parallel_processing is True
        assert config.enable_cross_modal_discovery is True
        assert config.semantic_similarity_threshold == 0.7
    
    def test_invalid_directory_path(self):
        """Test validation with invalid directory path."""
        with pytest.raises(ValidationError) as exc_info:
            BootstrapConfig(input_directory="/nonexistent/directory")
        
        assert "Input directory does not exist" in str(exc_info.value)
    
    def test_invalid_similarity_threshold(self, tmp_path):
        """Test validation with invalid similarity threshold."""
        with pytest.raises(ValidationError):
            BootstrapConfig(
                input_directory=tmp_path,
                semantic_similarity_threshold=1.5  # > 1.0
            )
        
        with pytest.raises(ValidationError):
            BootstrapConfig(
                input_directory=tmp_path,
                semantic_similarity_threshold=-0.1  # < 0.0
            )
    
    def test_invalid_batch_size(self, tmp_path):
        """Test validation with invalid batch size."""
        with pytest.raises(ValidationError):
            BootstrapConfig(
                input_directory=tmp_path,
                batch_size=0  # Must be > 0
            )
    
    def test_invalid_max_workers(self, tmp_path):
        """Test validation with invalid max workers."""
        with pytest.raises(ValidationError):
            BootstrapConfig(
                input_directory=tmp_path,
                max_workers=0  # Must be > 0
            )
    
    def test_edge_discovery_methods(self, tmp_path):
        """Test edge discovery methods configuration."""
        config = BootstrapConfig(
            input_directory=tmp_path,
            edge_discovery_methods=[
                EdgeDiscoveryMethod.IMPORT_ANALYSIS,
                EdgeDiscoveryMethod.CO_LOCATION
            ]
        )
        
        assert len(config.edge_discovery_methods) == 2
        assert EdgeDiscoveryMethod.IMPORT_ANALYSIS in config.edge_discovery_methods
        assert EdgeDiscoveryMethod.CO_LOCATION in config.edge_discovery_methods
        assert EdgeDiscoveryMethod.DIRECTORY_STRUCTURE not in config.edge_discovery_methods
    
    def test_exclude_patterns(self, tmp_path):
        """Test exclude patterns configuration."""
        patterns = ["*.pyc", "*.log", "__pycache__", ".git"]
        config = BootstrapConfig(
            input_directory=tmp_path,
            exclude_patterns=patterns
        )
        
        assert config.exclude_patterns == patterns
    
    def test_include_extensions(self, tmp_path):
        """Test include extensions configuration."""
        extensions = {".py", ".md", ".json"}
        config = BootstrapConfig(
            input_directory=tmp_path,
            include_extensions=extensions
        )
        
        assert config.include_extensions == extensions
    
    def test_storage_config(self, tmp_path):
        """Test storage configuration."""
        storage_config = {
            "host": "localhost",
            "port": 8529,
            "database": "custom_db"
        }
        
        config = BootstrapConfig(
            input_directory=tmp_path,
            storage_config=storage_config
        )
        
        assert config.storage_config == storage_config
    
    def test_config_serialization(self, tmp_path):
        """Test configuration serialization and deserialization."""
        config = BootstrapConfig(
            input_directory=tmp_path,
            output_database="test_serialize",
            batch_size=25,
            max_workers=8
        )
        
        # Convert to dict
        config_dict = config.dict()
        assert config_dict["output_database"] == "test_serialize"
        assert config_dict["batch_size"] == 25
        assert config_dict["max_workers"] == 8
        
        # Recreate from dict
        config2 = BootstrapConfig(**config_dict)
        assert config2.output_database == config.output_database
        assert config2.batch_size == config.batch_size
        assert config2.max_workers == config.max_workers
    
    def test_file_type_filter_enum(self, tmp_path):
        """Test FileTypeFilter enum values."""
        # Test all enum values
        for filter_type in FileTypeFilter:
            config = BootstrapConfig(
                input_directory=tmp_path,
                file_type_filter=filter_type
            )
            assert config.file_type_filter == filter_type
    
    def test_chunking_config(self, tmp_path):
        """Test chunking configuration parameters."""
        config = BootstrapConfig(
            input_directory=tmp_path,
            chunk_size=256,
            chunk_overlap=25
        )
        
        assert config.chunk_size == 256
        assert config.chunk_overlap == 25
        
        # Test validation
        with pytest.raises(ValidationError):
            BootstrapConfig(
                input_directory=tmp_path,
                chunk_size=0  # Must be > 0
            )
        
        with pytest.raises(ValidationError):
            BootstrapConfig(
                input_directory=tmp_path,
                chunk_overlap=-1  # Must be >= 0
            )


class TestBootstrapMetrics:
    """Test BootstrapMetrics data structure."""
    
    def test_metrics_creation(self):
        """Test creating metrics with default values."""
        metrics = BootstrapMetrics()
        
        assert metrics.total_files_discovered == 0
        assert metrics.total_files_processed == 0
        assert metrics.total_nodes_created == 0
        assert metrics.total_edges_created == 0
        assert metrics.cross_modal_edges == 0
        assert metrics.processing_errors == 0
        assert metrics.total_processing_time == 0.0
        assert metrics.files_by_type == {}
        assert metrics.edges_by_type == {}
        assert metrics.modality_distribution == {}
    
    def test_metrics_with_values(self):
        """Test creating metrics with specific values."""
        metrics = BootstrapMetrics(
            total_files_discovered=10,
            total_files_processed=9,
            total_nodes_created=9,
            total_edges_created=15,
            cross_modal_edges=5,
            processing_errors=1,
            total_processing_time=25.5,
            files_by_type={"code": 4, "documentation": 3, "config": 2},
            edges_by_type={"imports": 6, "references": 4, "colocation": 5},
            modality_distribution={"code": 4, "documentation": 3, "config": 2}
        )
        
        assert metrics.total_files_discovered == 10
        assert metrics.total_files_processed == 9
        assert metrics.total_nodes_created == 9
        assert metrics.total_edges_created == 15
        assert metrics.cross_modal_edges == 5
        assert metrics.processing_errors == 1
        assert metrics.total_processing_time == 25.5
        assert metrics.files_by_type == {"code": 4, "documentation": 3, "config": 2}
        assert metrics.edges_by_type == {"imports": 6, "references": 4, "colocation": 5}
        assert metrics.modality_distribution == {"code": 4, "documentation": 3, "config": 2}
    
    def test_metrics_calculation_properties(self):
        """Test calculated properties in metrics."""
        metrics = BootstrapMetrics(
            total_files_processed=10,
            total_processing_time=20.0,
            total_nodes_created=8,
            total_edges_created=12
        )
        
        # Test average processing time calculation
        metrics.avg_file_processing_time = metrics.total_processing_time / metrics.total_files_processed
        assert metrics.avg_file_processing_time == 2.0
        
        # Test graph density (simplified calculation)
        max_edges = metrics.total_nodes_created * (metrics.total_nodes_created - 1)
        metrics.graph_density = metrics.total_edges_created / max_edges if max_edges > 0 else 0.0
        expected_density = 12 / (8 * 7)  # 12 / 56
        assert abs(metrics.graph_density - expected_density) < 0.001


class TestBootstrapResult:
    """Test BootstrapResult data structure."""
    
    def test_successful_result(self, tmp_path):
        """Test creating a successful result."""
        config = BootstrapConfig(input_directory=tmp_path)
        metrics = BootstrapMetrics(
            total_files_processed=5,
            total_nodes_created=5,
            total_edges_created=8
        )
        
        result = BootstrapResult(
            success=True,
            config=config,
            metrics=metrics,
            database_name="test_db",
            node_count=5,
            edge_count=8,
            errors=[],
            warnings=[]
        )
        
        assert result.success is True
        assert result.config == config
        assert result.metrics == metrics
        assert result.database_name == "test_db"
        assert result.node_count == 5
        assert result.edge_count == 8
        assert result.errors == []
        assert result.warnings == []
    
    def test_failed_result(self, tmp_path):
        """Test creating a failed result."""
        config = BootstrapConfig(input_directory=tmp_path)
        metrics = BootstrapMetrics()  # Empty metrics for failure
        
        result = BootstrapResult(
            success=False,
            config=config,
            metrics=metrics,
            database_name="test_db",
            node_count=0,
            edge_count=0,
            errors=["Storage connection failed", "Invalid configuration"],
            warnings=["Some files were skipped"]
        )
        
        assert result.success is False
        assert result.config == config
        assert result.metrics == metrics
        assert result.database_name == "test_db"
        assert result.node_count == 0
        assert result.edge_count == 0
        assert len(result.errors) == 2
        assert len(result.warnings) == 1
    
    def test_result_summary_properties(self, tmp_path):
        """Test result summary properties."""
        config = BootstrapConfig(input_directory=tmp_path)
        metrics = BootstrapMetrics(
            total_files_processed=10,
            processing_errors=2,
            total_processing_time=30.0
        )
        
        result = BootstrapResult(
            success=True,
            config=config,
            metrics=metrics,
            database_name="test_db",
            node_count=8,
            edge_count=15,
            errors=[],
            warnings=["File type detection failed for 2 files"]
        )
        
        # Calculate success rate
        success_rate = (metrics.total_files_processed - metrics.processing_errors) / metrics.total_files_processed
        assert success_rate == 0.8  # 8/10
        
        # Check warnings exist
        assert len(result.warnings) > 0
        assert result.warnings[0].startswith("File type detection failed")


class TestEdgeDiscoveryMethod:
    """Test EdgeDiscoveryMethod enum."""
    
    def test_all_methods_available(self):
        """Test that all expected discovery methods are available."""
        expected_methods = {
            "directory_structure",
            "import_analysis", 
            "file_references",
            "co_location",
            "semantic_similarity"
        }
        
        actual_methods = {method.value for method in EdgeDiscoveryMethod}
        assert actual_methods == expected_methods
    
    def test_method_string_values(self):
        """Test that enum values are correct strings."""
        assert EdgeDiscoveryMethod.DIRECTORY_STRUCTURE.value == "directory_structure"
        assert EdgeDiscoveryMethod.IMPORT_ANALYSIS.value == "import_analysis"
        assert EdgeDiscoveryMethod.FILE_REFERENCES.value == "file_references"
        assert EdgeDiscoveryMethod.CO_LOCATION.value == "co_location"
        assert EdgeDiscoveryMethod.SEMANTIC_SIMILARITY.value == "semantic_similarity"


class TestFileTypeFilter:
    """Test FileTypeFilter enum."""
    
    def test_all_filters_available(self):
        """Test that all expected filters are available."""
        expected_filters = {
            "all",
            "code_only",
            "docs_only", 
            "config_only",
            "code_and_docs"
        }
        
        actual_filters = {filter_type.value for filter_type in FileTypeFilter}
        assert actual_filters == expected_filters
    
    def test_filter_string_values(self):
        """Test that enum values are correct strings."""
        assert FileTypeFilter.ALL.value == "all"
        assert FileTypeFilter.CODE_ONLY.value == "code_only"
        assert FileTypeFilter.DOCS_ONLY.value == "docs_only"
        assert FileTypeFilter.CONFIG_ONLY.value == "config_only"
        assert FileTypeFilter.CODE_AND_DOCS.value == "code_and_docs"