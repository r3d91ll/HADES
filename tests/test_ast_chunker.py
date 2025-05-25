"""
Integration test for AST-based code chunking.

This module tests the end-to-end code chunking pipeline, processing
a sample Python file through the AST-based chunker.
"""

import os
import sys
import json
import pytest
from pathlib import Path
from pprint import pprint
from typing import Dict, List, Any

from src.ingest.pre_processor.python_pre_processor import PythonPreProcessor
from src.ingest.chunking import chunk_code
from src.ingest.pre_processor.manager import PreprocessorManager


@pytest.fixture
def sample_python_file():
    """Return the path to a sample Python file for testing."""
    return str(Path(__file__).parent / "sample_code" / "test_chunking.py")


def test_ast_chunking_pipeline(sample_python_file):
    """Test the complete AST chunking pipeline."""
    # Step 1: Pre-process the Python file
    preprocessor = PythonPreProcessor(create_symbol_table=True)
    preprocessed = preprocessor.process_file(sample_python_file)
    
    assert preprocessed is not None, "Failed to pre-process Python file"
    assert "functions" in preprocessed, "No functions extracted in preprocessing"
    assert "classes" in preprocessed, "No classes extracted in preprocessing"
    
    # Step 2: Apply chunking
    chunks = chunk_code(preprocessed)
    assert len(chunks) > 0, "No chunks were created"
    
    # Verify chunk properties
    for chunk in chunks:
        assert "id" in chunk, "Chunk is missing ID"
        assert "content" in chunk, "Chunk is missing content"
        assert "symbol_type" in chunk, "Chunk is missing symbol_type"
    
    # Step 3: Create entities and relationships
    entities = []
    relationships = []
    
    manager = PreprocessorManager()
    manager._process_code_file(preprocessed, entities, relationships)
    
    assert len(entities) > 0, "No entities were created"
    assert len(relationships) > 0, "No relationships were created"
    
    # Check entity structure
    for entity in entities:
        assert "id" in entity, "Entity is missing ID"
        assert "type" in entity, "Entity is missing type"
        
    # Check relationship structure
    for rel in relationships:
        assert "from" in rel, "Relationship is missing source"
        assert "to" in rel, "Relationship is missing target"
        assert "type" in rel, "Relationship is missing type"


def test_ast_chunking_results(sample_python_file):
    """Test the results of AST chunking for expected patterns."""
    # Pre-process and chunk
    preprocessor = PythonPreProcessor(create_symbol_table=True)
    preprocessed = preprocessor.process_file(sample_python_file)
    chunks = chunk_code(preprocessed)
    
    # Check chunk types
    symbol_types = [chunk.get("symbol_type") for chunk in chunks]
    
    # The test file should have at least these types of chunks
    assert "class" in symbol_types, "No class chunks found"
    assert "function" in symbol_types, "No function chunks found"
    
    # Ensure chunks have proper boundaries
    for chunk in chunks:
        if "line_start" in chunk and "line_end" in chunk:
            assert chunk["line_start"] <= chunk["line_end"], f"Invalid line range: {chunk['line_start']}-{chunk['line_end']}"


if __name__ == "__main__":
    # This allows the file to be run directly for debugging
    pytest.main(["-xvs", __file__])
