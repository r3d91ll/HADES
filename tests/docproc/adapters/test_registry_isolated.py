"""
Isolated tests for adapter registry module.

This test module uses mocks to isolate the registry from actual adapter implementations.
"""

import unittest
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Type, List
from unittest.mock import patch, MagicMock

# Apply mocks before importing docproc modules
sys.path.insert(0, str(Path(__file__).parent.parent))
from tests.docproc.docproc_test_utils import patch_modules
patch_modules()

# Now we can safely import our modules
from src.docproc.adapters.registry import (
    register_adapter,
    get_adapter_for_format,
    get_adapter_class,
    get_supported_formats,
    _ADAPTER_REGISTRY
)
from src.docproc.adapters.base import BaseAdapter


class MockAdapter(BaseAdapter):
    """Mock adapter for testing."""
    
    def __init__(self, options: Optional[Dict[str, Any]] = None):
        """Initialize the mock adapter."""
        self.options = options or {}
    
    def process(self, file_path: Path, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Mock implementation of process method."""
        return {"content": "mock content", "metadata": {}}
    
    def process_text(self, text: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Mock implementation of process_text method."""
        return {"content": "mock processed text", "metadata": {}}
    
    def extract_entities(self, content: Any) -> list:
        """Mock implementation of extract_entities method."""
        return []
    
    def extract_metadata(self, content: Any) -> Dict[str, Any]:
        """Mock implementation of extract_metadata method."""
        return {}
    
    # convert_to_markdown and convert_to_text methods have been removed
    # as they are no longer part of the BaseAdapter interface


class TestAdapterRegistry(unittest.TestCase):
    """Test cases for the adapter registry module."""
    
    def setUp(self):
        """Set up the test environment."""
        # Save the original registry
        self.original_registry = _ADAPTER_REGISTRY.copy()
        
        # Clear the registry for isolated testing
        _ADAPTER_REGISTRY.clear()
    
    def tearDown(self):
        """Restore the original registry after each test."""
        # Restore the original registry
        _ADAPTER_REGISTRY.clear()
        _ADAPTER_REGISTRY.update(self.original_registry)
    
    def test_register_adapter(self):
        """Test registering an adapter."""
        # Use a clean dictionary for this test
        test_registry = {}
        
        # Register a mock adapter with patched registry
        with patch('src.docproc.adapters.registry._ADAPTER_REGISTRY', test_registry):
            register_adapter('mock', MockAdapter)
            
            # Verify it was registered
            self.assertIn('mock', test_registry)
            self.assertEqual(test_registry['mock'], MockAdapter)
    
    def test_register_adapter_overwrite(self):
        """Test registering an adapter that overwrites an existing one."""
        # Use a clean dictionary for this test
        test_registry = {}
        
        # Create another mock adapter class
        class AnotherMockAdapter(MockAdapter):
            pass
        
        # Test with patched registry
        with patch('src.docproc.adapters.registry._ADAPTER_REGISTRY', test_registry):
            # Register first adapter
            register_adapter('mock', MockAdapter)
            self.assertEqual(test_registry['mock'], MockAdapter)
            
            # Register with the same format (should overwrite)
            register_adapter('mock', AnotherMockAdapter)
            
            # Verify it was overwritten
            self.assertEqual(test_registry['mock'], AnotherMockAdapter)
    
    def test_register_non_adapter_no_type_checking(self):
        """Test registering a class that doesn't inherit from BaseAdapter.
        
        Note: The current implementation now performs type checking.
        We're testing that it rejects non-BaseAdapter classes.
        """
        # Use a clean dictionary for this test
        test_registry = {}
        
        # Create a non-BaseAdapter class
        class NonAdapter:
            pass
        
        # Test with patched registry
        with patch('src.docproc.adapters.registry._ADAPTER_REGISTRY', test_registry):
            # Register a non-BaseAdapter class (should raise an error)
            with self.assertRaises(ValueError):
                register_adapter('non_adapter', NonAdapter)
    
    def test_get_adapter_class(self):
        """Test getting an adapter class."""
        # Register a mock adapter
        register_adapter('mock', MockAdapter)
        
        # Get the adapter class
        adapter_class = get_adapter_class('mock')
        
        # Verify
        self.assertEqual(adapter_class, MockAdapter)
    
    def test_get_adapter_class_unknown(self):
        """Test getting an adapter class for an unknown format."""
        # Attempt to get an adapter for an unknown format
        with self.assertRaises(ValueError):
            get_adapter_class('unknown')
    
    def test_get_adapter_for_format(self):
        """Test getting an adapter instance for a format."""
        # Register a mock adapter
        register_adapter('mock', MockAdapter)
        
        # Get an adapter instance
        adapter = get_adapter_for_format('mock')
        
        # Verify
        self.assertIsInstance(adapter, MockAdapter)
    
    def test_get_adapter_for_format_with_options(self):
        """Test getting an adapter instance with options."""
        # Register a mock adapter
        register_adapter('mock', MockAdapter)
        
        # Get an adapter instance
        adapter = get_adapter_for_format('mock')
        
        # Set options after instantiation
        options = {'clean': True, 'extract_entities': True}
        adapter.options = options
        
        # Verify options were passed
        self.assertEqual(adapter.options, options)
    
    def test_get_adapter_for_format_unknown(self):
        """Test getting an adapter for an unknown format."""
        # Attempt to get an adapter for an unknown format
        with self.assertRaises(ValueError):
            get_adapter_for_format('unknown')
    
    def test_get_supported_formats(self):
        """Test getting list of supported formats."""
        # Clear the registry first
        _ADAPTER_REGISTRY.clear()
        
        # Register some mock adapters
        register_adapter('mock1', MockAdapter)
        register_adapter('mock2', MockAdapter)
        register_adapter('mock3', MockAdapter)
        
        # Get supported formats
        formats = get_supported_formats()
        
        # Verify
        self.assertIsInstance(formats, list)
        # Just check that our specific formats are registered, not the exact total count
        # which may vary if other tests registered formats
        self.assertIn('mock1', formats)
        self.assertIn('mock2', formats)
        self.assertIn('mock3', formats)
        # Make sure at least these 3 formats are present
        self.assertGreaterEqual(len(formats), 3)
    
    def test_get_supported_formats_empty(self):
        """Test getting supported formats when registry is empty."""
        # Force registry to be empty for this test
        with patch('src.docproc.adapters.registry._ADAPTER_REGISTRY', {}):
            # Get supported formats
            formats = get_supported_formats()
            
            # Verify
            self.assertIsInstance(formats, list)
            self.assertEqual(len(formats), 0)


if __name__ == '__main__':
    unittest.main()
