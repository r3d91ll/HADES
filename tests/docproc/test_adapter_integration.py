#!/usr/bin/env python
"""
Integration test for document processing adapters.

This test verifies that the three supported document formats (PDF, Markdown, Python)
are processed correctly by their specific adapters and that the output is valid
for the chunking module.
"""

import json
import os
import sys
import time
from pathlib import Path
import unittest

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(project_root))

# Import from the document processing module
from src.docproc.utils.format_detector import detect_format_from_path
from src.docproc.adapters import get_adapter_for_format
from src.docproc.adapters.python_adapter import PythonAdapter
from src.docproc.adapters.docling_adapter import DoclingAdapter
from src.docproc.serializers import save_to_json_file


class AdapterIntegrationTest(unittest.TestCase):
    """Test suite for document processing adapter integration."""
    
    def setUp(self):
        """Set up test environment."""
        self.data_dir = project_root / "data"
        self.output_dir = project_root / "test-output/adapter-integration"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Test files
        self.pdf_file = self.data_dir / "PathRAG_paper.pdf"
        self.markdown_file = self.data_dir / "docproc.md"
        self.python_file = self.data_dir / "file_batcher.py"
        
        # Required fields for chunking
        self.required_fields = ["id", "content", "metadata", "format"]
        
    def test_pdf_processing(self):
        """Test that PDF documents are processed correctly."""
        self._test_file_processing(self.pdf_file, "pdf", DoclingAdapter)
        
    def test_markdown_processing(self):
        """Test that Markdown documents are processed correctly."""
        self._test_file_processing(self.markdown_file, "markdown", DoclingAdapter)
        
    def test_python_processing(self):
        """Test that Python files are processed correctly."""
        self._test_file_processing(self.python_file, "python", PythonAdapter)
        
    def _test_file_processing(self, file_path, expected_format, expected_adapter_class):
        """Helper method to test processing of a specific file type."""
        # Verify the file exists
        self.assertTrue(file_path.exists(), f"Test file {file_path} not found")
        
        # Verify format detection works correctly
        detected_format = detect_format_from_path(file_path)
        self.assertEqual(detected_format, expected_format, 
                         f"Format detection failed for {file_path}")
        
        # Get the correct adapter for this format
        # Note: We're bypassing the registry to ensure we get the correct adapter
        adapter = expected_adapter_class()
        
        # Process the file
        result = adapter.process(file_path)
        
        # Verify the result contains required fields for chunking
        for field in self.required_fields:
            self.assertIn(field, result, f"Missing required field '{field}' in {expected_format} output")
            
        # Verify non-empty content
        self.assertIsNotNone(result.get("content"), f"Content is empty for {expected_format}")
        self.assertGreater(len(result.get("content", "")), 0, f"Content is empty for {expected_format}")
        
        # Verify metadata is a dictionary
        self.assertIsInstance(result.get("metadata", {}), dict, 
                              f"Metadata is not a dictionary for {expected_format}")
        
        # Verify the top-level format field matches the source document format
        self.assertEqual(result.get("format"), expected_format, 
                       f"Format field should match source document format: {expected_format}")
        
        # Verify format in metadata matches content format
        format_in_metadata = result.get("metadata", {}).get("format")
        if expected_format == "python":
            self.assertEqual(format_in_metadata, "python", 
                          f"metadata.format should be 'python' for Python files")
        else:
            self.assertEqual(format_in_metadata, "markdown", 
                          f"metadata.format should be 'markdown' for non-Python files")
        
        # Verify content_format field is present and correct
        self.assertIn("content_format", result, f"Missing content_format field in {expected_format} output")
        
        # Verify raw_content field is NOT present (should be removed)
        self.assertNotIn("raw_content", result, f"Redundant raw_content field found in {expected_format} output")
        
        # Verify content_type field is present at top level
        self.assertIn("content_type", result, f"Missing content_type field in {expected_format} output")
        
        # Verify content_type is also in metadata
        self.assertIn("content_type", result.get("metadata", {}), 
                     f"Missing content_type in metadata for {expected_format} output")
        
        # Verify format-specific content_format and content_type
        if expected_format == "python":
            self.assertEqual(result.get("content_format"), "python", 
                             f"Python files should have content_format='python'")
            self.assertEqual(result.get("content_type"), "code", 
                             f"Python files should have content_type='code'")
            self.assertEqual(result.get("metadata", {}).get("content_type"), "code", 
                             f"Python files should have metadata.content_type='code'")
        else:
            self.assertEqual(result.get("content_format"), "markdown", 
                             f"Non-Python files should have content_format='markdown'")
            self.assertEqual(result.get("content_type"), "text", 
                             f"Non-Python files should have content_type='text'")
            self.assertEqual(result.get("metadata", {}).get("content_type"), "text", 
                             f"Non-Python files should have metadata.content_type='text'")
        
        # Verify entities are present (all our adapters should extract some entities)
        self.assertIn("entities", result, f"Missing entities in {expected_format} output")
        
        # Save the results for inspection
        output_file = self.output_dir / f"{file_path.stem}_{expected_format}_test.json"
        save_to_json_file(result, output_file)
        print(f"Saved {expected_format} test results to {output_file}")
        
        # Return the result for additional verification
        return result
        
    def test_python_adapter_directly(self):
        """Test the Python adapter directly with a Python file."""
        # Create a Python adapter instance
        adapter = PythonAdapter()
        
        # Process the Python file
        result = adapter.process(self.python_file)
        
        # Verify Python-specific output
        self.assertIn("function_count", result.get("metadata", {}), 
                      "Python metadata should include function_count")
        self.assertIn("class_count", result.get("metadata", {}), 
                      "Python metadata should include class_count")
        self.assertIn("import_count", result.get("metadata", {}), 
                      "Python metadata should include import_count")
        
        # Save the results for inspection
        output_file = self.output_dir / "python_direct_adapter_test.json"
        save_to_json_file(result, output_file)
        
        return result
        
    def test_end_to_end_chunking_readiness(self):
        """Test that all document types produce output that is ready for chunking."""
        files = [
            (self.pdf_file, "pdf", DoclingAdapter),
            (self.markdown_file, "markdown", DoclingAdapter),
            (self.python_file, "python", PythonAdapter)
        ]
        
        for file_path, expected_format, adapter_class in files:
            # Process the file
            result = self._test_file_processing(file_path, expected_format, adapter_class)
            
            # Verify key requirements for chunking
            self.assertIsInstance(result.get("content", ""), str, 
                                  f"Content must be a string for chunking: {expected_format}")
            
            # Verify the id is a unique string
            self.assertIsInstance(result.get("id", ""), str, 
                                  f"ID must be a string for chunking: {expected_format}")
            self.assertGreater(len(result.get("id", "")), 0, 
                             f"ID must not be empty for chunking: {expected_format}")
            
            # Metadata should contain basic information
            metadata = result.get("metadata", {})
            self.assertIsInstance(metadata, dict, 
                                  f"Metadata must be a dictionary for chunking: {expected_format}")
            
            # Entities should be a list
            self.assertIsInstance(result.get("entities", []), list, 
                                  f"Entities must be a list for chunking: {expected_format}")
            
        print("\n✅ All document types produce chunking-ready output")


if __name__ == "__main__":
    unittest.main()
