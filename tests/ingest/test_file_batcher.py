"""
Tests for the FileBatcher class in src/ingest/file_batcher.py
"""
import os
import pytest
import tempfile
import shutil
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

from src.ingest.file_batcher import FileBatcher


class TestFileBatcher:
    """Test suite for FileBatcher class."""

    @pytest.fixture
    def default_batcher(self):
        """Create a FileBatcher with default settings."""
        return FileBatcher()

    @pytest.fixture
    def custom_batcher(self):
        """Create a FileBatcher with custom settings."""
        file_type_map = {
            'python': ['.py'],
            'text': ['.txt'],
            'json': ['.json'],
        }
        exclude_patterns = ['__pycache__', '.git', 'test_dir']
        return FileBatcher(file_type_map, exclude_patterns)

    @pytest.fixture
    def test_directory(self):
        """Create a temporary directory with test files."""
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Create a normal structure
            # Main dir files
            with open(os.path.join(temp_dir, 'main.py'), 'w') as f:
                f.write('# Main Python file')
            
            with open(os.path.join(temp_dir, 'readme.md'), 'w') as f:
                f.write('# Readme file')
                
            with open(os.path.join(temp_dir, 'config.json'), 'w') as f:
                f.write('{"key": "value"}')
                
            # Create some subdirectories
            subdir1 = os.path.join(temp_dir, 'src')
            os.makedirs(subdir1)
            
            with open(os.path.join(subdir1, 'lib.py'), 'w') as f:
                f.write('# Library file')
                
            with open(os.path.join(subdir1, 'data.txt'), 'w') as f:
                f.write('Sample data')
                
            # Create an excluded directory
            excluded_dir = os.path.join(temp_dir, '__pycache__')
            os.makedirs(excluded_dir)
            
            with open(os.path.join(excluded_dir, 'compiled.pyc'), 'w') as f:
                f.write('# Compiled Python file')
                
            # Create a test_dir directory
            test_dir = os.path.join(temp_dir, 'test_dir')
            os.makedirs(test_dir)
            
            with open(os.path.join(test_dir, 'test.py'), 'w') as f:
                f.write('# Test file')
                
            # Create a .git directory
            git_dir = os.path.join(temp_dir, '.git')
            os.makedirs(git_dir)
            
            with open(os.path.join(git_dir, 'config'), 'w') as f:
                f.write('# Git config')
                
            # Create a .DS_Store file
            with open(os.path.join(temp_dir, '.DS_Store'), 'w') as f:
                f.write('# DS_Store file')
                
            yield temp_dir
        finally:
            # Clean up the temporary directory
            shutil.rmtree(temp_dir)

    def test_init_with_defaults(self, default_batcher):
        """Test initialization with default parameters."""
        assert isinstance(default_batcher.file_type_map, dict)
        assert 'python' in default_batcher.file_type_map
        assert '.py' in default_batcher.file_type_map['python']
        assert '__pycache__' in default_batcher.exclude_patterns
        assert isinstance(default_batcher.ext_to_type, dict)
        assert default_batcher.ext_to_type['.py'] == 'python'

    def test_init_with_custom_config(self, custom_batcher):
        """Test initialization with custom parameters."""
        assert list(custom_batcher.file_type_map.keys()) == ['python', 'text', 'json']
        assert custom_batcher.file_type_map['python'] == ['.py']
        assert 'test_dir' in custom_batcher.exclude_patterns
        assert custom_batcher.ext_to_type['.py'] == 'python'

    def test_collect_files_empty_dir(self):
        """Test collecting files from an empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            batcher = FileBatcher()
            result = batcher.collect_files(temp_dir)
            
            assert isinstance(result, dict)
            assert len(result) == 0

    def test_collect_files(self, test_directory, default_batcher):
        """Test collecting files from the test directory."""
        result = default_batcher.collect_files(test_directory)
        
        # Check that files are properly categorized
        assert 'python' in result
        assert 'markdown' in result
        assert 'json' in result
        assert any(file.endswith('main.py') for file in result['python'])
        assert any(file.endswith('lib.py') for file in result['python'])
        assert any(file.endswith('readme.md') for file in result['markdown'])
        assert any(file.endswith('config.json') for file in result['json'])
        
        # Check that excluded files are not included
        all_files = [item for sublist in result.values() for item in sublist]
        assert not any('__pycache__' in file for file in all_files)
        assert not any('.git' in file for file in all_files)
        assert not any('.DS_Store' in file for file in all_files)

    def test_collect_files_with_custom_config(self, test_directory, custom_batcher):
        """Test collecting files with custom file type map and exclusions."""
        result = custom_batcher.collect_files(test_directory)
        
        # Check that only files of specified types are included
        assert 'python' in result
        assert 'text' in result
        assert 'json' in result
        assert 'markdown' not in result  # Not in custom file types
        
        # Check that custom excluded directories are not included
        all_files = [item for sublist in result.values() for item in sublist]
        assert not any('test_dir' in file for file in all_files)

    def test_get_stats_empty(self):
        """Test get_stats with empty batches."""
        batcher = FileBatcher()
        stats = batcher.get_stats({})
        
        assert stats['total'] == 0

    def test_get_stats(self):
        """Test get_stats with sample data."""
        batcher = FileBatcher()
        batches = {
            'python': ['file1.py', 'file2.py', 'file3.py'],
            'json': ['file1.json', 'file2.json'],
            'text': ['file1.txt']
        }
        
        stats = batcher.get_stats(batches)
        
        assert stats['python'] == 3
        assert stats['json'] == 2
        assert stats['text'] == 1
        assert stats['total'] == 6

    def test_collect_files_nonexistent_dir(self):
        """Test collecting files from a non-existent directory."""
        batcher = FileBatcher()
        # The actual implementation returns an empty dict for non-existent directories
        # rather than raising an exception, so we'll test for that behavior
        result = batcher.collect_files('/nonexistent/directory')
        assert isinstance(result, dict)
        assert len(result) == 0

    def test_excluded_file_patterns(self, test_directory):
        """Test that files matching excluded patterns are not included."""
        # Create a batcher that excludes files with "data" in the name
        batcher = FileBatcher(exclude_patterns=['*data*'])
        result = batcher.collect_files(test_directory)
        
        # Check that files with "data" in the name are excluded
        all_files = [item for sublist in result.values() for item in sublist]
        assert not any('data' in file.lower() for file in all_files)

    def test_file_type_mapping(self):
        """Test that files are mapped to the correct type."""
        # Create a batcher with custom mapping
        batcher = FileBatcher(file_type_map={
            'code': ['.py', '.js', '.java'],
            'data': ['.json', '.csv'],
            'doc': ['.md', '.txt']
        })
        
        # Check the extension to type mapping
        assert batcher.ext_to_type['.py'] == 'code'
        assert batcher.ext_to_type['.js'] == 'code'
        assert batcher.ext_to_type['.json'] == 'data'
        assert batcher.ext_to_type['.md'] == 'doc'

    def test_unknown_extension(self, test_directory):
        """Test handling of unknown file extensions."""
        # Create a file with an unknown extension
        unknown_file = os.path.join(test_directory, 'unknown.xyz')
        with open(unknown_file, 'w') as f:
            f.write('Unknown file type')
        
        batcher = FileBatcher()
        result = batcher.collect_files(test_directory)
        
        # Check that the file with unknown extension is categorized as 'other'
        assert 'other' in result
        assert any(file.endswith('unknown.xyz') for file in result['other'])
