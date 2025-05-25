"""
Integration test for the chunking pipeline components.

This test runs both code chunking (AST-based) and text chunking (Chonky-based)
end-to-end, verifying proper integration of all components.
"""

import os
import sys
import json
import pytest
from pathlib import Path
from typing import Dict, List, Any

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingest.pre_processor.python_pre_processor import PythonPreProcessor
from src.ingest.pre_processor.markdown_pre_processor import MarkdownPreProcessor
from src.ingest.chunking import chunk_code, chunk_text
from src.ingest.pre_processor.manager import PreprocessorManager
from src.config.chunker_config import get_chunker_for_language, get_chunker_config


def create_sample_files(tmp_path):
    """Create sample Python and Markdown files for testing."""
    # Create sample Python file
    py_path = tmp_path / "sample.py"
    py_path.write_text('''
import os
import sys
from typing import List, Dict, Any

class TestClass:
    """A test class for chunking."""
    
    def __init__(self, name):
        self.name = name
        self.value = 42
        
    def get_value(self) -> int:
        """Return the value."""
        return self.value
        
def main():
    """Main function."""
    test = TestClass("test")
    print(f"Value: {test.get_value()}")
    
if __name__ == "__main__":
    main()
''')

    # Create sample Markdown file
    md_path = tmp_path / "sample.md"
    md_path.write_text('''
# Sample Document

This is a sample document for testing chunking.

## Section 1

This is the first section with some content.
It has multiple lines that should be grouped together.

## Section 2

This is the second section with more content.
- Item 1
- Item 2
- Item 3

## Conclusion

This is the conclusion of the document.
''')
    
    return py_path, md_path


class TestChunkIntegration:
    """Integration tests for the chunking components."""
    
    @pytest.fixture
    def sample_files(self, tmp_path):
        """Create sample files for testing."""
        return create_sample_files(tmp_path)
    
    def test_language_detection(self, sample_files):
        """Test that language is correctly detected and mapped to chunker."""
        py_path, md_path = sample_files
        
        # Python file should use AST chunker
        py_chunker = get_chunker_for_language("python")
        assert py_chunker == "ast", "Python should use AST chunker"
        
        # Markdown file should use Chonky chunker
        md_chunker = get_chunker_for_language("markdown")
        assert md_chunker == "chonky", "Markdown should use Chonky chunker"
    
    def test_code_chunking_pipeline(self, sample_files):
        """Test the full code chunking pipeline."""
        py_path, _ = sample_files
        
        # Step 1: Preprocess
        preprocessor = PythonPreProcessor(create_symbol_table=True)
        preprocessed = preprocessor.process_file(str(py_path))
        assert preprocessed is not None, "Preprocessing failed"
        
        # Step 2: Chunk
        chunks = chunk_code(preprocessed)
        assert len(chunks) > 0, "Chunking produced no chunks"
        
        # Verify chunks have required properties
        for chunk in chunks:
            assert "id" in chunk, "Chunk missing ID"
            assert "content" in chunk, "Chunk missing content"
            assert "symbol_type" in chunk, "Chunk missing symbol_type"
            assert "type" in chunk, "Chunk missing type"
    
    def test_text_chunking_pipeline(self, sample_files):
        """Test the full text chunking pipeline."""
        _, md_path = sample_files
        
        # Step 1: Preprocess
        preprocessor = MarkdownPreProcessor()
        preprocessed = preprocessor.process_file(str(md_path))
        assert preprocessed is not None, "Preprocessing failed"
        
        # Step 2: Chunk
        chunks = chunk_text(preprocessed)
        assert len(chunks) > 0, "Chunking produced no chunks"
        
        # Verify chunks have required properties
        for chunk in chunks:
            assert "id" in chunk, "Chunk missing ID"
            assert "content" in chunk, "Chunk missing content"
            assert "symbol_type" in chunk, "Chunk missing symbol_type"
            assert "type" in chunk, "Chunk missing type"
    
    def test_preprocessor_manager_integration(self, sample_files):
        """Test integration with PreprocessorManager."""
        py_path, md_path = sample_files
        
        # Create preprocessor manager
        manager = PreprocessorManager()
        
        # Process files
        py_result = {"path": str(py_path), "type": "python"}
        md_result = {"path": str(md_path), "type": "markdown"}
        
        # Create entities and relationships containers
        entities = []
        relationships = []
        
        # Test code file processing
        py_preprocessor = PythonPreProcessor(create_symbol_table=True)
        py_data = py_preprocessor.process_file(str(py_path))
        assert py_data is not None, "Python preprocessing failed"
        
        # Process code file
        manager._process_code_file(py_data, entities, relationships)
        
        # Verify entities and relationships were created
        assert len(entities) > 0, "No entities created for Python file"
        assert len(relationships) > 0, "No relationships created for Python file"
        
        # Reset for text file
        entities_before = len(entities)
        relationships_before = len(relationships)
        
        # Test text file processing
        md_preprocessor = MarkdownPreProcessor()
        md_data = md_preprocessor.process_file(str(md_path))
        assert md_data is not None, "Markdown preprocessing failed"
        
        # Process text file
        manager._process_text_file(md_data, entities, relationships)
        
        # Verify additional entities and relationships were created
        assert len(entities) > entities_before, "No entities created for Markdown file"
        assert len(relationships) > relationships_before, "No relationships created for Markdown file"


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
