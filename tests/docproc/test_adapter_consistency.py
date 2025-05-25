"""
Integration tests for adapter consistency.

These tests ensure that all adapters produce consistent output formats to ensure
compatibility with downstream components like chunkers.
"""

import unittest
import sys
import tempfile
import os
from pathlib import Path
from typing import Dict, Any, List, Type, Optional

# Apply mocks before importing docproc modules
sys.path.insert(0, str(Path(__file__).parent))
from tests.docproc.docproc_test_utils import patch_modules
patch_modules()

# Now we can safely import our modules
from src.docproc.adapters.base import BaseAdapter
from src.docproc.adapters.python_adapter import PythonAdapter
from src.docproc.adapters.docling_adapter import DoclingAdapter
from src.docproc.core import process_document, process_text


class AdapterConsistencyTest(unittest.TestCase):
    """Test case to ensure all adapters produce consistent output."""
    
    def setUp(self):
        """Set up test environment with sample files for each adapter."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_dir_path = Path(self.temp_dir.name)
        
        # Create sample files for each adapter
        self.sample_files = {
            'text': self._create_text_file(),
            'python': self._create_python_file(),
        }
        
        # Create sample text content for each adapter
        self.sample_text = {
            'text': "This is plain text content.",
            'python': 'def test_function():\n    return "Hello, World!"',
        }
        
        # List of adapter classes to test
        self.adapter_classes = [
            DoclingAdapter,
            PythonAdapter,
        ]
        
        # Note: Both adapters should implement the BaseAdapter interface.
    
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    def _create_text_file(self) -> Path:
        """Create a sample text file."""
        file_path = self.temp_dir_path / "sample.txt"
        with open(file_path, "w") as f:
            f.write("This is a sample text file.\nIt has multiple lines.\n")
        return file_path
    
    def _create_python_file(self) -> Path:
        """Create a sample Python file."""
        file_path = self.temp_dir_path / "sample.py"
        with open(file_path, "w") as f:
            f.write("""#!/usr/bin/env python3
\"\"\"
Sample Python module.
\"\"\"

class SampleClass:
    \"\"\"A sample class for testing.\"\"\"
    
    def __init__(self, name):
        \"\"\"Initialize with a name.\"\"\"
        self.name = name
    
    def greet(self):
        \"\"\"Return a greeting.\"\"\"
        return f"Hello, {self.name}!"

def sample_function(value):
    \"\"\"A sample function that returns twice the input.\"\"\"
    return value * 2

if __name__ == "__main__":
    sample = SampleClass("World")
    print(sample.greet())
    print(sample_function(21))
""")
        return file_path
    
    def _verify_adapter_output_structure(self, result: Dict[str, Any], adapter_name: str):
        """Verify that an adapter's output has the correct structure."""
        # Result should be a dictionary
        self.assertIsInstance(result, dict, f"{adapter_name} output is not a dictionary")
        
        # Result should have 'content' and 'metadata' keys
        self.assertIn('content', result, f"{adapter_name} output missing 'content' key")
        self.assertIn('metadata', result, f"{adapter_name} output missing 'metadata' key")
        
        # Content should be a string
        self.assertIsInstance(result['content'], str, 
                              f"{adapter_name} 'content' is not a string")
        
        # Metadata should be a dictionary
        self.assertIsInstance(result['metadata'], dict, 
                              f"{adapter_name} 'metadata' is not a dictionary")
        
        # Metadata should have at least 'format' key
        self.assertIn('format', result['metadata'], 
                      f"{adapter_name} metadata missing 'format' key")
    
    def test_adapter_instance_consistency(self):
        """Test that all adapter instances produce consistent output structure."""
        for adapter_class in self.adapter_classes:
            adapter_name = adapter_class.__name__
            adapter = adapter_class()
            
            # Get the format from the adapter name
            format_name = adapter_name.lower().replace('adapter', '')
            
            # Skip if we don't have sample text for this format
            if format_name not in self.sample_text:
                continue
                
            # Test process_text method
            sample_text = self.sample_text[format_name]
            result = adapter.process_text(sample_text)
            
            # Verify output structure
            self._verify_adapter_output_structure(result, adapter_name)
            
            # Format in metadata should match the adapter's format
            self.assertEqual(result['metadata']['format'], format_name,
                             f"{adapter_name} metadata format doesn't match expected '{format_name}'")
    
    def test_adapter_file_consistency(self):
        """Test that all adapters process files with consistent output structure."""
        for adapter_class in self.adapter_classes:
            adapter_name = adapter_class.__name__
            adapter = adapter_class()
            
            # Get the format from the adapter name
            format_name = adapter_name.lower().replace('adapter', '')
            
            # Skip if we don't have a sample file for this format
            if format_name not in self.sample_files:
                continue
                
            # Test process method
            sample_file = self.sample_files[format_name]
            result = adapter.process(sample_file)
            
            # Verify output structure
            self._verify_adapter_output_structure(result, adapter_name)
            
            # Format in metadata should match the adapter's format
            self.assertEqual(result['metadata']['format'], format_name,
                             f"{adapter_name} metadata format doesn't match expected '{format_name}'")
            
            # File path should be included in metadata
            self.assertIn('file_path', result['metadata'],
                          f"{adapter_name} metadata missing 'file_path' key")
            self.assertEqual(result['metadata']['file_path'], str(sample_file),
                             f"{adapter_name} metadata file_path doesn't match")
    
    def test_core_process_document_consistency(self):
        """Test that the core process_document function maintains consistent output."""
        for format_name, sample_file in self.sample_files.items():
            # Process the document
            result = process_document(sample_file)
            
            # Verify output structure
            self._verify_adapter_output_structure(result, f"process_document({format_name})")
            
            # Format in metadata should match the file format
            self.assertEqual(result['metadata']['format'], format_name,
                             f"process_document({format_name}) metadata format doesn't match")
            
            # File path should be included in metadata
            self.assertIn('file_path', result['metadata'],
                          f"process_document({format_name}) metadata missing 'file_path' key")
            self.assertEqual(result['metadata']['file_path'], str(sample_file),
                             f"process_document({format_name}) metadata file_path doesn't match")
    
    def test_core_process_text_consistency(self):
        """Test that the core process_text function maintains consistent output."""
        for format_name, sample_text in self.sample_text.items():
            # Process the text
            result = process_text(sample_text, format_name)
            
            # Verify output structure
            self._verify_adapter_output_structure(result, f"process_text({format_name})")
            
            # Format in metadata should match the specified format
            self.assertEqual(result['metadata']['format'], format_name,
                             f"process_text({format_name}) metadata format doesn't match")
    
    # Tests for markdown_conversion_consistency and text_conversion_consistency have been removed
    # as the convert_to_markdown and convert_to_text methods are no longer part of the BaseAdapter interface
    
    def test_entity_extraction_consistency(self):
        """Test that all adapters consistently extract entities."""
        for adapter_class in self.adapter_classes:
            adapter_name = adapter_class.__name__
            adapter = adapter_class()
            
            # Get the format from the adapter name
            format_name = adapter_name.lower().replace('adapter', '')
            
            # Skip if we don't have sample text for this format
            if format_name not in self.sample_text:
                continue
                
            # Get sample text and process it
            sample_text = self.sample_text[format_name]
            result = adapter.process_text(sample_text)
            
            # Extract entities
            entities = adapter.extract_entities(result['content'])
            
            # Entities should be a list
            self.assertIsInstance(entities, list, 
                                 f"{adapter_name} extract_entities didn't return a list")


if __name__ == '__main__':
    unittest.main()
