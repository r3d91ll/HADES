"""
Tests for base adapter class.

This module tests the BaseAdapter abstract class functionality.
"""

import unittest
from pathlib import Path
from typing import Dict, Any, Optional
from unittest.mock import patch

from src.docproc.adapters.base import BaseAdapter


class ConcreteAdapter(BaseAdapter):
    """Concrete implementation of BaseAdapter for testing."""
    
    def process(self, file_path: Path, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a file."""
        options = options or {}
        return {
            'content': f"Processed content from {file_path}",
            'metadata': {
                'format': 'test',
                'file_path': str(file_path),
                'options': options
            }
        }
    
    def process_text(self, text: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process text content."""
        options = options or {}
        return {
            'content': f"Processed text: {text[:20]}...",
            'metadata': {
                'format': 'test',
                'options': options
            }
        }
    
    def extract_entities(self, content: Any) -> list:
        """Extract entities from content."""
        return [{'type': 'test_entity', 'content': str(content)[:30]}]
    
    def extract_metadata(self, content: Any) -> Dict[str, Any]:
        """Extract metadata from content."""
        return {'format': 'test', 'length': len(str(content))}


class TestBaseAdapter(unittest.TestCase):
    """Test cases for BaseAdapter."""

    def setUp(self):
        """Set up the test environment."""
        self.adapter = ConcreteAdapter()
    
    def test_instantiate_abstract_class(self):
        """Test that BaseAdapter cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            BaseAdapter()
    
    def test_concrete_adapter_instantiation(self):
        """Test that a concrete subclass can be instantiated."""
        adapter = ConcreteAdapter()
        self.assertIsInstance(adapter, BaseAdapter)
        self.assertIsInstance(adapter, ConcreteAdapter)
    
    def test_required_methods(self):
        """Test that the required methods are implemented."""
        methods = [
            'process',
            'process_text',
            'extract_entities',
            'extract_metadata'
        ]
        
        for method in methods:
            self.assertTrue(hasattr(self.adapter, method))
            self.assertTrue(callable(getattr(self.adapter, method)))
    
    def test_process_with_path_str(self):
        """Test process method with string path."""
        result = self.adapter.process("/test/path.txt")
        
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)
        self.assertIn('content', result)
        self.assertIn('metadata', result)
        self.assertEqual(result['metadata']['file_path'], "/test/path.txt")
    
    def test_process_with_path_object(self):
        """Test process method with Path object."""
        path = Path("/test/path.txt")
        result = self.adapter.process(path)
        
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)
        self.assertIn('content', result)
        self.assertIn('metadata', result)
        self.assertEqual(result['metadata']['file_path'], str(path))
    
    def test_process_with_options(self):
        """Test process method with options."""
        options = {'option1': True, 'option2': 'value'}
        result = self.adapter.process("/test/path.txt", options)
        
        self.assertIsNotNone(result)
        self.assertIn('metadata', result)
        self.assertIn('options', result['metadata'])
        self.assertEqual(result['metadata']['options'], options)
    
    def test_process_text_basic(self):
        """Test process_text method basic functionality."""
        text = "This is test content."
        result = self.adapter.process_text(text)
        
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)
        self.assertIn('content', result)
        self.assertIn('metadata', result)
        self.assertIn(text[:20], result['content'])
    
    def test_process_text_with_options(self):
        """Test process_text method with options."""
        text = "This is test content."
        options = {'option1': True, 'option2': 'value'}
        result = self.adapter.process_text(text, options)
        
        self.assertIsNotNone(result)
        self.assertIn('metadata', result)
        self.assertIn('options', result['metadata'])
        self.assertEqual(result['metadata']['options'], options)
    
    def test_extract_entities_implementation(self):
        """Test extract_entities method implementation."""
        content = "Test content for entity extraction"
        entities = self.adapter.extract_entities(content)
        
        self.assertIsNotNone(entities)
        self.assertIsInstance(entities, list)
        self.assertTrue(len(entities) > 0)
        self.assertEqual(entities[0]['type'], 'test_entity')
    
    def test_extract_metadata_implementation(self):
        """Test extract_metadata method implementation."""
        content = "Test content for metadata extraction"
        metadata = self.adapter.extract_metadata(content)
        
        self.assertIsNotNone(metadata)
        self.assertIsInstance(metadata, dict)
        self.assertEqual(metadata['format'], 'test')
        self.assertEqual(metadata['length'], len(content))


if __name__ == '__main__':
    unittest.main()
