#!/usr/bin/env python3
"""
Unit tests for FireCrawl import functionality.
Tests validation and import logic without requiring database or GPU.
"""

import unittest
from unittest.mock import Mock, MagicMock, patch
import json
import tempfile
from pathlib import Path

from import_firecrawl_output import FireCrawlImporter, ImportStats


class TestFireCrawlImporter(unittest.TestCase):
    """Test FireCrawl importer functionality."""
    
    def setUp(self):
        """Set up test importer with mocked database."""
        self.mock_db = MagicMock()
        
        with patch.object(FireCrawlImporter, '_init_database', return_value=self.mock_db):
            with patch.object(FireCrawlImporter, '_init_embedder'):
                self.importer = FireCrawlImporter({})
                # Mock the embedder to prevent GPU initialization
                self.importer.embedder = MagicMock()
    
    def test_validate_content_valid(self):
        """Test validation of valid content."""
        data = {
            'content': 'This is a test document with enough words to pass validation. ' * 10,
            'url': 'https://example.com/page',
            'title': 'Test Page'
        }
        
        self.assertTrue(self.importer.validate_content(data))
    
    def test_validate_content_too_short(self):
        """Test validation rejects short content."""
        data = {
            'content': 'Too short',
            'url': 'https://example.com'
        }
        
        self.assertFalse(self.importer.validate_content(data))
    
    def test_validate_content_missing_url(self):
        """Test validation rejects missing URL."""
        data = {
            'content': 'This is valid content ' * 20
        }
        
        self.assertFalse(self.importer.validate_content(data))
    
    def test_validate_content_navigation_heavy(self):
        """Test validation rejects navigation-heavy content."""
        data = {
            'content': '[Home] [About] [Contact] [Products] [Services] ' * 50,
            'url': 'https://example.com'
        }
        
        self.assertFalse(self.importer.validate_content(data))
    
    def test_import_directory_with_valid_files(self):
        """Test importing a directory with valid JSON files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test JSON files
            test_data = {
                'url': 'https://example.com/test',
                'title': 'Test Page',
                'content': 'This is test content that should pass validation. ' * 20
            }
            
            json_path = Path(tmpdir) / 'test.json'
            with open(json_path, 'w') as f:
                json.dump(test_data, f)
            
            # Mock the collection insert
            mock_collection = MagicMock()
            self.mock_db.collection.return_value = mock_collection
            
            # Mock the embedder to prevent GPU issues
            with patch.object(self.importer, '_generate_embeddings'):
                # Import directory
                stats = self.importer.import_directory(Path(tmpdir), 'test_source')
            
            # Check stats
            self.assertEqual(stats.total_files, 1)
            self.assertEqual(stats.imported, 1)
            self.assertEqual(stats.skipped, 0)
            self.assertEqual(stats.failed, 0)
    
    def test_import_directory_with_invalid_files(self):
        """Test importing a directory with invalid content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test JSON with invalid content
            test_data = {
                'url': 'https://example.com',
                'content': 'Too short'
            }
            
            json_path = Path(tmpdir) / 'invalid.json'
            with open(json_path, 'w') as f:
                json.dump(test_data, f)
            
            # Import directory
            stats = self.importer.import_directory(Path(tmpdir), 'test_source')
            
            # Check stats
            self.assertEqual(stats.total_files, 1)
            self.assertEqual(stats.imported, 0)
            self.assertEqual(stats.skipped, 1)
    
    def test_import_array_format(self):
        """Test importing JSON array format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create JSON array
            test_data = [
                {
                    'url': 'https://example.com/page1',
                    'content': 'Valid content for page 1. ' * 20
                },
                {
                    'url': 'https://example.com/page2',
                    'content': 'Valid content for page 2. ' * 20
                }
            ]
            
            json_path = Path(tmpdir) / 'array.json'
            with open(json_path, 'w') as f:
                json.dump(test_data, f)
            
            # Mock the collection
            mock_collection = MagicMock()
            self.mock_db.collection.return_value = mock_collection
            
            # Mock embeddings to prevent GPU issues
            with patch.object(self.importer, '_generate_embeddings'):
                # Import directory
                stats = self.importer.import_directory(Path(tmpdir), 'test_source')
            
            # Should import both documents
            self.assertEqual(stats.imported, 2)


if __name__ == "__main__":
    unittest.main()