"""
Tests for the file batcher tool.

This module contains unit tests for the FileBatcher class and related
functions in the src.tools.batching.file_batcher module.
"""

import os
import tempfile
from pathlib import Path
import unittest
from typing import Dict, List, Set, Any
from unittest import mock

from src.tools.batching.file_batcher import FileBatcher, collect_and_batch_files


class TestFileBatcher(unittest.TestCase):
    """Test cases for FileBatcher class."""

    def setUp(self):
        """Set up test environment before each test."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_dir = Path(self.temp_dir.name)
        
        # Default batcher for tests
        self.batcher = FileBatcher()
        
        # Create test directory structure
        self._create_test_files()
        
    def tearDown(self):
        """Clean up test environment after each test."""
        self.temp_dir.cleanup()
        
    def _create_test_files(self):
        """Create a test directory structure with various file types."""
        # Create directories
        (self.root_dir / "src").mkdir()
        (self.root_dir / "src" / "module1").mkdir()
        (self.root_dir / "src" / "module2").mkdir()
        (self.root_dir / "docs").mkdir()
        (self.root_dir / "config").mkdir()
        (self.root_dir / "__pycache__").mkdir()  # Should be excluded
        
        # Create Python files
        self._create_empty_file("src/module1/file1.py")
        self._create_empty_file("src/module1/file2.py")
        self._create_empty_file("src/module2/file3.py")
        
        # Create documentation files
        self._create_empty_file("docs/readme.md")
        self._create_empty_file("docs/guide.pdf")
        
        # Create config files
        self._create_empty_file("config/settings.json")
        self._create_empty_file("config/config.yaml")
        self._create_empty_file("config/data.xml")
        
        # Create files that should be excluded
        self._create_empty_file("__pycache__/module.cpython-39.pyc")
        self._create_empty_file(".DS_Store")
        
    def _create_empty_file(self, relative_path: str):
        """Create an empty file at the specified path relative to root_dir."""
        file_path = self.root_dir / relative_path
        file_path.parent.mkdir(exist_ok=True)
        file_path.touch()
        
    def test_init_default_values(self):
        """Test that FileBatcher initializes with proper default values."""
        batcher = FileBatcher()
        
        # Check file type map contains expected types
        self.assertIn('python', batcher.file_type_map)
        self.assertIn('markdown', batcher.file_type_map)
        self.assertIn('json', batcher.file_type_map)
        
        # Check extension mapping
        self.assertEqual(batcher.ext_to_type['.py'], 'python')
        self.assertEqual(batcher.ext_to_type['.md'], 'markdown')
        self.assertEqual(batcher.ext_to_type['.json'], 'json')
        
        # Check default exclude patterns
        self.assertIn('__pycache__', batcher.exclude_patterns)
        self.assertIn('.git', batcher.exclude_patterns)
        
    def test_init_custom_values(self):
        """Test that FileBatcher initializes with custom values."""
        file_type_map = {'custom_type': ['.xyz', '.abc']}
        exclude_patterns = ['exclude_me', 'pattern_*']
        
        batcher = FileBatcher(file_type_map, exclude_patterns)
        
        # Check custom file type map
        self.assertEqual(batcher.file_type_map, file_type_map)
        
        # Check custom extension mapping
        self.assertEqual(batcher.ext_to_type['.xyz'], 'custom_type')
        self.assertEqual(batcher.ext_to_type['.abc'], 'custom_type')
        
        # Check custom exclude patterns
        self.assertEqual(batcher.exclude_patterns, exclude_patterns)
        
    def test_collect_files(self):
        """Test collecting files and batching by type."""
        batches = self.batcher.collect_files(self.root_dir)
        
        # Check that files were batched by type
        self.assertIn('python', batches)
        self.assertIn('markdown', batches)
        self.assertIn('json', batches)
        self.assertIn('yaml', batches)
        self.assertIn('pdf', batches)
        self.assertIn('xml', batches)
        
        # Check that the correct number of files was found for each type
        self.assertEqual(len(batches['python']), 3)
        self.assertEqual(len(batches['markdown']), 1)
        self.assertEqual(len(batches['json']), 1)
        self.assertEqual(len(batches['yaml']), 1)
        self.assertEqual(len(batches['pdf']), 1)
        self.assertEqual(len(batches['xml']), 1)
        
        # Check that excluded files were not included
        all_files = [str(path) for files in batches.values() for path in files]
        self.assertNotIn('__pycache__', ''.join(all_files))
        self.assertNotIn('.DS_Store', all_files)
        
    def test_collect_files_with_string_path(self):
        """Test collecting files with a string path instead of Path object."""
        batches = self.batcher.collect_files(str(self.root_dir))
        
        # Check that files were collected correctly
        self.assertIn('python', batches)
        self.assertEqual(len(batches['python']), 3)
        
    def test_get_stats(self):
        """Test getting statistics about batched files."""
        # Create test batch data
        batches = {
            'python': [Path('file1.py'), Path('file2.py')],
            'markdown': [Path('file3.md')],
            'json': [Path('file4.json'), Path('file5.json')]
        }
        
        stats = self.batcher.get_stats(batches)
        
        # Check statistics
        self.assertEqual(stats['total'], 5)
        self.assertEqual(stats['by_type']['python'], 2)
        self.assertEqual(stats['by_type']['markdown'], 1)
        self.assertEqual(stats['by_type']['json'], 2)
        
    def test_filter_batch(self):
        """Test filtering a batch to include only specific file types."""
        # Create test batch data
        batches = {
            'python': [Path('file1.py'), Path('file2.py')],
            'markdown': [Path('file3.md')],
            'json': [Path('file4.json'), Path('file5.json')]
        }
        
        # Filter to include only Python and JSON files
        filtered = self.batcher.filter_batch(batches, ['python', 'json'])
        
        # Check filtered batch
        self.assertIn('python', filtered)
        self.assertIn('json', filtered)
        self.assertNotIn('markdown', filtered)
        
        # Filter with None should return the original batch
        unfiltered = self.batcher.filter_batch(batches, None)
        self.assertEqual(unfiltered, batches)
        
    def test_collect_and_batch_files(self):
        """Test the convenience function for collecting and batching files."""
        # Define custom file type map and exclude patterns
        file_type_map = {
            'py': ['.py'],
            'doc': ['.md', '.pdf']
        }
        exclude_patterns = ['__pycache__']
        allowed_types = ['py']
        
        # Call the convenience function
        batches = collect_and_batch_files(
            self.root_dir,
            file_type_map,
            exclude_patterns,
            allowed_types
        )
        
        # Check that only Python files were included
        self.assertIn('py', batches)
        self.assertEqual(len(batches['py']), 3)
        self.assertEqual(len(batches), 1)  # Only one type should be present
        
    def test_collect_and_batch_files_no_filtering(self):
        """Test the convenience function without filtering by allowed types."""
        # Define custom file type map and exclude patterns
        file_type_map = {
            'py': ['.py'],
            'doc': ['.md', '.pdf']
        }
        exclude_patterns = ['__pycache__']
        
        # Call the convenience function without allowed_types
        batches = collect_and_batch_files(
            self.root_dir,
            file_type_map,
            exclude_patterns
        )
        
        # Check that all defined types were included
        self.assertIn('py', batches)
        self.assertIn('doc', batches)
        self.assertEqual(len(batches['py']), 3)
        self.assertEqual(len(batches['doc']), 2)  # md and pdf files


if __name__ == '__main__':
    unittest.main()
