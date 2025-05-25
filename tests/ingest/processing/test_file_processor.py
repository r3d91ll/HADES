"""
Tests for the file processor component.

This module tests the functionality of the FileProcessor class
which handles file discovery, filtering, and batching.
"""
import os
from typing import Dict, List
from pathlib import Path
import pytest
from unittest.mock import patch, mock_open

from src.types.common import PreProcessorConfig
from src.ingest.pre_processor.file_processor import FileProcessor


class TestFileProcessorInit:
    """Test initialization of the FileProcessor class."""
    
    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        processor = FileProcessor()
        
        # Verify defaults are set
        assert processor.exclude_patterns == []
        assert processor.recursive is True
        assert processor.max_workers == FileProcessor.DEFAULT_MAX_WORKERS
        assert isinstance(processor.file_type_map, dict)
        
    def test_init_with_config(self):
        """Test initialization with a configuration."""
        config = PreProcessorConfig(
            exclude_patterns=["*.tmp", ".git/"],
            recursive=False,
            max_workers=4,
            file_type_map={"py": ["python"]}
        )
        
        processor = FileProcessor(config)
        
        # Verify config values are applied
        assert processor.exclude_patterns == ["*.tmp", ".git/"]
        assert processor.recursive is False
        assert processor.max_workers == 4
        assert processor.file_type_map == {"py": ["python"]}
        
        # Verify exclude_regex is compiled
        # Just verify regex was created without asserting the exact number
        # which could change based on implementation
        assert len(processor.exclude_regex) > 0


class TestFileProcessorFiltering:
    """Test file filtering functionality."""
    
    @pytest.fixture
    def processor_with_filters(self):
        """Create a FileProcessor with specific filters."""
        config = PreProcessorConfig(
            exclude_patterns=[
                r"\.git/",
                r"__pycache__/",
                r".*\.pyc$",
                r"node_modules/"
            ]
        )
        return FileProcessor(config)
    
    def test_should_exclude(self, processor_with_filters):
        """Test the should_exclude method with various paths."""
        # These should be excluded
        assert processor_with_filters.should_exclude(".git/HEAD")
        assert processor_with_filters.should_exclude("src/__pycache__/module.pyc")
        assert processor_with_filters.should_exclude("test.pyc")
        assert processor_with_filters.should_exclude("node_modules/package.json")
        
        # These should not be excluded
        assert not processor_with_filters.should_exclude("src/main.py")
        assert not processor_with_filters.should_exclude("docs/README.md")
        assert not processor_with_filters.should_exclude("tests/test_main.py")
        
    def test_get_file_type(self, processor_with_filters):
        """Test the get_file_type method with various file paths."""
        # Set a custom file_type_map
        processor_with_filters.file_type_map = {
            "python": [".py", ".pyi"],
            "markdown": [".md", ".mdx"],
            "config": [".json", ".yaml", ".yml"]
        }
        
        # Test file type determination
        assert processor_with_filters.get_file_type("file.py") == "python"
        assert processor_with_filters.get_file_type("file.pyi") == "python"
        assert processor_with_filters.get_file_type("README.md") == "markdown"
        assert processor_with_filters.get_file_type("config.json") == "config"
        assert processor_with_filters.get_file_type("config.yml") == "config"
        
        # Test unknown extension
        assert processor_with_filters.get_file_type("file.txt") == "txt"
        
        # Test no extension
        assert processor_with_filters.get_file_type("Makefile") == "unknown"
        
        # Test with Path object
        assert processor_with_filters.get_file_type(Path("file.py")) == "python"


class TestFileProcessorCollection:
    """Test file collection functionality."""
    
    @pytest.fixture
    def temp_dir_with_files(self, temp_test_dir):
        """Create a temporary directory with test files."""
        # Create various file types
        (temp_test_dir / "main.py").write_text("print('hello')")
        (temp_test_dir / "README.md").write_text("# Test Project")
        (temp_test_dir / "config.json").write_text("{}")
        
        # Create a subdirectory with more files
        sub_dir = temp_test_dir / "sub"
        sub_dir.mkdir()
        (sub_dir / "module.py").write_text("def test(): pass")
        (sub_dir / "notes.md").write_text("Test notes")
        
        # Create some files that should be excluded
        (temp_test_dir / ".git").mkdir()
        (temp_test_dir / ".git" / "HEAD").write_text("ref: refs/heads/main")
        (temp_test_dir / "__pycache__").mkdir()
        (temp_test_dir / "__pycache__" / "main.cpython-39.pyc").write_text("binary data")
        
        return temp_test_dir
    
    def test_collect_files(self, temp_dir_with_files):
        """Test the collect_files method."""
        # Create processor with standard exclude patterns
        config = PreProcessorConfig(
            exclude_patterns=[r"\.git/", r"__pycache__/"]
        )
        processor = FileProcessor(config)
        
        # Collect files
        files_by_type = processor.collect_files(temp_dir_with_files)
        
        # Verify file types are correct
        assert "py" in files_by_type
        assert "md" in files_by_type
        assert "json" in files_by_type
        
        # Verify excluded files are not included
        all_files = [str(p) for file_list in files_by_type.values() for p in file_list]
        assert not any(".git" in f for f in all_files)
        assert not any("__pycache__" in f for f in all_files)
        
        # Verify correct number of files by type
        assert len(files_by_type["py"]) == 2
        assert len(files_by_type["md"]) == 2
        assert len(files_by_type["json"]) == 1
    
    def test_non_recursive_collection(self, temp_dir_with_files):
        """Test collect_files with recursive=False."""
        # Create processor with non-recursive option
        config = PreProcessorConfig(
            exclude_patterns=[r"\.git/", r"__pycache__/"],
            recursive=False
        )
        processor = FileProcessor(config)
        
        # Collect files
        files_by_type = processor.collect_files(temp_dir_with_files)
        
        # Verify only top-level files are included
        all_paths = [str(p) for file_list in files_by_type.values() for p in file_list]
        assert not any("sub" in p for p in all_paths)
        
        # Verify correct number of files
        assert len(files_by_type["py"]) == 1  # Only main.py, not sub/module.py
        assert len(files_by_type["md"]) == 1  # Only README.md, not sub/notes.md


class TestFileProcessorBatching:
    """Test file batching functionality."""
    
    @pytest.fixture
    def files_by_type(self):
        """Provide a sample files_by_type dictionary."""
        return {
            "py": [Path("file1.py"), Path("file2.py"), Path("file3.py")],
            "md": [Path("doc1.md"), Path("doc2.md")],
            "json": [Path("config.json")]
        }
    
    def test_create_batches(self, files_by_type):
        """Test the create_batches method."""
        # Create processor
        processor = FileProcessor()
        
        # Create batches with default batch size
        batches = processor.create_batches(files_by_type)
        
        # With default batch size, all files should be in one batch
        assert len(batches) == 1
        assert "py" in batches[0]
        assert "md" in batches[0]
        assert "json" in batches[0]
        
        # Verify all files are included
        assert len(batches[0]["py"]) == 3
        assert len(batches[0]["md"]) == 2
        assert len(batches[0]["json"]) == 1
    
    def test_create_batches_small_size(self, files_by_type):
        """Test create_batches with a small batch size."""
        # Create processor
        processor = FileProcessor()
        
        # Create batches with small batch size
        batches = processor.create_batches(files_by_type, batch_size=2)
        
        # Should create multiple batches
        assert len(batches) > 1
        
        # Verify all files are included across all batches
        total_py = sum(len(batch.get("py", [])) for batch in batches)
        total_md = sum(len(batch.get("md", [])) for batch in batches)
        total_json = sum(len(batch.get("json", [])) for batch in batches)
        
        assert total_py == 3
        assert total_md == 2
        assert total_json == 1
    
    def test_process_directory(self, temp_test_dir):
        """Test the process_directory method."""
        # Create test files
        (temp_test_dir / "file1.py").write_text("print('hello')")
        (temp_test_dir / "file2.py").write_text("print('world')")
        (temp_test_dir / "README.md").write_text("# Test")
        
        # Create processor with mocked collect_files
        processor = FileProcessor()
        
        # Call process_directory
        batches = processor.process_directory(temp_test_dir)
        
        # Verify batches are created correctly
        assert isinstance(batches, list)
        assert len(batches) > 0
        
        # Verify dictionary structure
        assert isinstance(batches[0], dict)
        assert "py" in batches[0] or "md" in batches[0]
