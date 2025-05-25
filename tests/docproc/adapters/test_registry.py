"""
Tests for adapter registry module.

This module tests the functionality in src.docproc.adapters.registry.
"""

import unittest
from typing import Dict, Any, Optional, List
from pathlib import Path

from src.docproc.adapters.base import BaseAdapter
from src.docproc.adapters.registry import (
    register_adapter,
    get_adapter_for_format
)

# Add functions that might be missing from the registry
def get_registered_formats() -> List[str]:
    """Get all registered format names."""
    # This is a test helper function that mimics the functionality
    # if not present in the actual registry module
    from src.docproc.adapters.registry import _ADAPTER_REGISTRY
    return list(_ADAPTER_REGISTRY.keys())

def clear_registry() -> None:
    """Clear the adapter registry."""
    # This is a test helper function that mimics the functionality
    # if not present in the actual registry module
    from src.docproc.adapters.registry import _ADAPTER_REGISTRY
    _ADAPTER_REGISTRY.clear()


class MockAdapter(BaseAdapter):
    """Mock adapter for testing registry."""
    
    def process(self, file_path: Path, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a file."""
        return {
            'content': f"Mock processed {file_path}",
            'metadata': {'format': 'mock'}
        }
    
    def process_text(self, text: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process text content."""
        return {
            'content': f"Mock processed text",
            'metadata': {'format': 'mock'}
        }
    
    def extract_entities(self, content: Any) -> list:
        """Extract entities from content."""
        return [{'type': 'mock_entity'}]
    
    def extract_metadata(self, content: Any) -> Dict[str, Any]:
        """Extract metadata from content."""
        return {'format': 'mock'}


class AnotherMockAdapter(BaseAdapter):
    """Another mock adapter for testing registry."""
    
    def process(self, file_path: Path, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a file."""
        return {
            'content': f"Another mock processed {file_path}",
            'metadata': {'format': 'another_mock'}
        }
    
    def process_text(self, text: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process text content."""
        return {
            'content': f"Another mock processed text",
            'metadata': {'format': 'another_mock'}
        }
    
    def extract_entities(self, content: Any) -> list:
        """Extract entities from content."""
        return [{'type': 'another_mock_entity'}]
    
    def extract_metadata(self, content: Any) -> Dict[str, Any]:
        """Extract metadata from content."""
        return {'format': 'another_mock'}
    
    # convert_to_markdown and convert_to_text methods have been removed
    # as they are no longer part of the BaseAdapter interface


class TestRegistry(unittest.TestCase):
    """Test cases for adapter registry."""

    def setUp(self) -> None:
        """Set up the test environment."""
        # Clear the registry before each test
        clear_registry()
        
        # Create mock adapters
        self.mock_adapter = MockAdapter()
        self.another_mock_adapter = AnotherMockAdapter()
    
    def tearDown(self) -> None:
        """Clean up after the test."""
        # Clear the registry after each test
        clear_registry()
    
    def test_register_adapter(self) -> None:
        """Test registering an adapter."""
        # Register the mock adapter for a format - using the class instead of instance
        register_adapter('mock_format', MockAdapter)
        
        # Get the registered formats
        formats = get_registered_formats()
        
        # Check if the format was registered
        self.assertIn('mock_format', formats)
        
        # Get the adapter for the format
        adapter = get_adapter_for_format('mock_format')
        
        # Check if the correct adapter was returned
        self.assertIsNotNone(adapter)
        self.assertIsInstance(adapter, MockAdapter)
    
    def test_register_multiple_adapters(self) -> None:
        """Test registering multiple adapters."""
        # Register multiple adapters - using classes instead of instances
        register_adapter('mock_format', MockAdapter)
        register_adapter('another_format', AnotherMockAdapter)
        
        # Get the registered formats
        formats = get_registered_formats()
        
        # Check if both formats were registered
        self.assertIn('mock_format', formats)
        self.assertIn('another_format', formats)
        
        # Get the adapters
        mock_adapter = get_adapter_for_format('mock_format')
        another_adapter = get_adapter_for_format('another_format')
        
        # Check if the correct adapters were returned
        self.assertIsInstance(mock_adapter, MockAdapter)
        self.assertIsInstance(another_adapter, AnotherMockAdapter)
    
    def test_register_same_format_twice(self) -> None:
        """Test registering the same format twice."""
        # Register an adapter
        register_adapter('test_format', MockAdapter)
        
        # Register another adapter for the same format
        register_adapter('test_format', AnotherMockAdapter)
        
        # Get the adapter
        adapter = get_adapter_for_format('test_format')
        
        # Check if the most recently registered adapter was returned
        self.assertIsInstance(adapter, AnotherMockAdapter)
    
    def test_get_adapter_for_unknown_format(self) -> None:
        """Test getting an adapter for an unknown format."""
        # Try to get an adapter for an unknown format
        with self.assertRaises(ValueError):
            get_adapter_for_format('unknown_format')
    
    def test_get_registered_formats(self) -> None:
        """Test getting registered formats."""
        # Register multiple adapters
        register_adapter('format1', MockAdapter)
        register_adapter('format2', AnotherMockAdapter)
        register_adapter('format3', MockAdapter)
        
        # Get the registered formats
        formats = get_registered_formats()
        
        # Check if all formats were returned
        self.assertEqual(len(formats), 3)
        self.assertIn('format1', formats)
        self.assertIn('format2', formats)
        self.assertIn('format3', formats)
    
    def test_clear_registry(self) -> None:
        """Test clearing the registry."""
        # Register multiple adapters
        register_adapter('format1', MockAdapter)
        register_adapter('format2', AnotherMockAdapter)
        
        # Clear the registry
        clear_registry()
        
        # Get the registered formats
        formats = get_registered_formats()
        
        # Check if the registry is empty
        self.assertEqual(len(formats), 0)
        
        # Try to get an adapter
        with self.assertRaises(ValueError):
            get_adapter_for_format('format1')
    
    def test_case_insensitive_format_names(self) -> None:
        """Test that format names are case-insensitive."""
        # Register an adapter with lowercase format
        register_adapter('test_format', MockAdapter)
        
        # Get the adapter with different case
        adapter1 = get_adapter_for_format('TEST_FORMAT')
        adapter2 = get_adapter_for_format('Test_Format')
        
        # Check if the adapter was returned
        self.assertIsNotNone(adapter1)
        self.assertIsNotNone(adapter2)
        self.assertIsInstance(adapter1, MockAdapter)
        self.assertIsInstance(adapter2, MockAdapter)
    
    def test_format_name_validation(self) -> None:
        """Test validation of format names."""
        # Try to register with empty format name
        with self.assertRaises(ValueError):
            register_adapter('', MockAdapter)
        
        # Try to register with None format name
        with self.assertRaises(ValueError):
            register_adapter(None, self.mock_adapter)  # type: ignore
    
    def test_adapter_validation(self) -> None:
        """Test validation of adapters."""
        # Try to register with None adapter
        with self.assertRaises(ValueError):
            register_adapter('test_format', None)  # type: ignore
        
        # Try to register with a non-BaseAdapter object
        with self.assertRaises(ValueError):
            register_adapter('test_format', "not an adapter")  # type: ignore


if __name__ == '__main__':
    unittest.main()
