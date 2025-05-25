"""
Tests for chunking various document types with Chonky.

This module tests the chunking functionality across different document types:
- PDF documents
- Plain text files
- CSV, XML, JSON, YAML, and TOML documents

The tests verify that the chunker can properly handle different document formats
while preserving content and boundaries.
"""

import sys
import os
import pytest
from pathlib import Path
from typing import Dict, List, Any, Union
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.chunking.text_chunkers.chonky_chunker import chunk_text
from src.docproc.adapters.docling_adapter import DoclingAdapter


# Create test fixtures for different document types
@pytest.fixture
def pdf_document():
    """Create a mock PDF document for testing."""
    return {
        "id": "test_pdf_001",
        "path": "/test/document.pdf",
        "content": """This is a simulated PDF document content.
        
It contains multiple paragraphs with various formatting.

Section 1: Introduction
This section introduces the key concepts.

Section 2: Methodology
This section describes the methods used.

Section 3: Results
This section presents the findings.

Section 4: Discussion
This section interprets the results.

Section 5: Conclusion
This section summarizes the main points.""",
        "type": "pdf"
    }


@pytest.fixture
def txt_document():
    """Create a plain text document for testing."""
    return {
        "id": "test_txt_001",
        "path": "/test/document.txt",
        "content": """This is a plain text document.
        
It has simple formatting with line breaks.

It contains several paragraphs.

Each paragraph should be properly identified and chunked.""",
        "type": "text"
    }


@pytest.fixture
def csv_document():
    """Create a CSV document for testing."""
    return {
        "id": "test_csv_001",
        "path": "/test/document.csv",
        "content": """id,name,value,description
1,Item 1,10.5,"This is item 1"
2,Item 2,20.3,"This is item 2"
3,Item 3,30.7,"This is item 3"
4,Item 4,40.2,"This is item 4"
5,Item 5,50.9,"This is item 5"
""",
        "type": "csv"
    }


@pytest.fixture
def json_document():
    """Create a JSON document for testing."""
    return {
        "id": "test_json_001",
        "path": "/test/document.json",
        "content": """{
  "items": [
    {
      "id": 1,
      "name": "Item 1",
      "properties": {
        "value": 10.5,
        "description": "This is item 1"
      }
    },
    {
      "id": 2,
      "name": "Item 2",
      "properties": {
        "value": 20.3,
        "description": "This is item 2"
      }
    }
  ],
  "metadata": {
    "version": "1.0",
    "author": "Test Author"
  }
}""",
        "type": "json"
    }


@pytest.fixture
def xml_document():
    """Create an XML document for testing."""
    return {
        "id": "test_xml_001",
        "path": "/test/document.xml",
        "content": """<?xml version="1.0" encoding="UTF-8"?>
<root>
  <items>
    <item id="1">
      <name>Item 1</name>
      <value>10.5</value>
      <description>This is item 1</description>
    </item>
    <item id="2">
      <name>Item 2</name>
      <value>20.3</value>
      <description>This is item 2</description>
    </item>
  </items>
  <metadata>
    <version>1.0</version>
    <author>Test Author</author>
  </metadata>
</root>""",
        "type": "xml"
    }


@pytest.fixture
def yaml_document():
    """Create a YAML document for testing."""
    return {
        "id": "test_yaml_001",
        "path": "/test/document.yaml",
        "content": """items:
  - id: 1
    name: Item 1
    properties:
      value: 10.5
      description: This is item 1
  - id: 2
    name: Item 2
    properties:
      value: 20.3
      description: This is item 2
metadata:
  version: 1.0
  author: Test Author""",
        "type": "yaml"
    }


@pytest.fixture
def toml_document():
    """Create a TOML document for testing."""
    return {
        "id": "test_toml_001",
        "path": "/test/document.toml",
        "content": """# This is a TOML document

title = "TOML Example"

[owner]
name = "Test Author"
organization = "Test Org"

[database]
server = "192.168.1.1"
ports = [ 8000, 8001, 8002 ]
connection_max = 5000
enabled = true

[servers]

[servers.alpha]
ip = "10.0.0.1"
role = "frontend"

[servers.beta]
ip = "10.0.0.2"
role = "backend"
""",
        "type": "toml"
    }


# Test chunking for different document types
@patch('src.chunking.text_chunkers.chonky_chunker._check_model_engine_availability')
@patch('src.chunking.text_chunkers.chonky_chunker.get_tokenizer')
@patch('src.chunking.text_chunkers.chonky_chunker.ensure_model_engine')
def test_pdf_chunking(mock_ensure_engine, mock_get_tokenizer, mock_check_availability, pdf_document):
    """Test that PDF documents can be properly chunked."""
    # Mock the model engine availability check
    mock_check_availability.return_value = False
    
    # Create a mock tokenizer
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = {"input_ids": [0] * 10}  # Simulate 10 tokens
    mock_get_tokenizer.return_value = mock_tokenizer
    
    # Process the document
    chunks = chunk_text(pdf_document, max_tokens=128, output_format="python")
    
    # Verify that chunks were created
    assert len(chunks) > 0, "No chunks were generated for PDF document"
    
    # Check that each chunk has the required fields
    for chunk in chunks:
        assert "id" in chunk, "Chunk is missing id field"
        assert "content" in chunk, "Chunk is missing content field"
        assert "parent" in chunk, "Chunk is missing parent field"
        assert "type" in chunk, "Chunk is missing type field"
        assert chunk["type"] == "pdf", "Chunk type should be 'pdf'"
        
        # Verify content is not empty
        assert chunk["content"].strip(), "Chunk content should not be empty"
        
        # Verify content is a subset of the original document
        assert chunk["content"] in pdf_document["content"], "Chunk content should be from the original document"


@patch('src.chunking.text_chunkers.chonky_chunker._check_model_engine_availability')
@patch('src.chunking.text_chunkers.chonky_chunker.get_tokenizer')
@patch('src.chunking.text_chunkers.chonky_chunker.ensure_model_engine')
def test_txt_chunking(mock_ensure_engine, mock_get_tokenizer, mock_check_availability, txt_document):
    """Test that plain text documents can be properly chunked."""
    # Mock the model engine availability check
    mock_check_availability.return_value = False
    
    # Create a mock tokenizer
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = {"input_ids": [0] * 10}  # Simulate 10 tokens
    mock_get_tokenizer.return_value = mock_tokenizer
    
    # Process the document
    chunks = chunk_text(txt_document, max_tokens=128, output_format="python")
    
    # Verify that chunks were created
    assert len(chunks) > 0, "No chunks were generated for text document"
    
    # Check that each chunk has the required fields
    for chunk in chunks:
        assert "id" in chunk, "Chunk is missing id field"
        assert "content" in chunk, "Chunk is missing content field"
        assert "parent" in chunk, "Chunk is missing parent field"
        assert "type" in chunk, "Chunk is missing type field"
        assert chunk["type"] == "text", "Chunk type should be 'text'"
        
        # Verify content is not empty
        assert chunk["content"].strip(), "Chunk content should not be empty"
        
        # Verify content is a subset of the original document
        assert chunk["content"] in txt_document["content"], "Chunk content should be from the original document"


@patch('src.chunking.text_chunkers.chonky_chunker._check_model_engine_availability')
@patch('src.chunking.text_chunkers.chonky_chunker.get_tokenizer')
@patch('src.chunking.text_chunkers.chonky_chunker.ensure_model_engine')
def test_structured_document_chunking(mock_ensure_engine, mock_get_tokenizer, mock_check_availability, 
                                     csv_document, json_document, xml_document, yaml_document, toml_document):
    """Test that structured documents (CSV, JSON, XML, YAML, TOML) can be properly chunked."""
    # Mock the model engine availability check
    mock_check_availability.return_value = False
    
    # Create a mock tokenizer
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = {"input_ids": [0] * 10}  # Simulate 10 tokens
    mock_get_tokenizer.return_value = mock_tokenizer
    
    # Test each document type
    for doc in [csv_document, json_document, xml_document, yaml_document, toml_document]:
        # Process the document
        chunks = chunk_text(doc, max_tokens=128, output_format="python")
        
        # Verify that chunks were created
        assert len(chunks) > 0, f"No chunks were generated for {doc['type']} document"
        
        # Check that each chunk has the required fields
        for chunk in chunks:
            assert "id" in chunk, "Chunk is missing id field"
            assert "content" in chunk, "Chunk is missing content field"
            assert "parent" in chunk, "Chunk is missing parent field"
            assert "type" in chunk, "Chunk is missing type field"
            assert chunk["type"] == doc["type"], f"Chunk type should be '{doc['type']}'"
            
            # Verify content is not empty
            assert chunk["content"].strip(), "Chunk content should not be empty"
            
            # Verify content is a subset of the original document
            assert chunk["content"] in doc["content"], "Chunk content should be from the original document"


@patch('src.chunking.text_chunkers.chonky_chunker._check_model_engine_availability')
@patch('src.chunking.text_chunkers.chonky_chunker.get_tokenizer')
@patch('src.chunking.text_chunkers.chonky_chunker.ensure_model_engine')
def test_docling_integration(mock_ensure_engine, mock_get_tokenizer, mock_check_availability, 
                            pdf_document, monkeypatch):
    """Test integration with Docling for PDF processing."""
    # Skip if Docling is not available
    docling_adapter = pytest.importorskip("src.docproc.adapters.docling_adapter")
    
    # Mock Docling adapter to avoid actual PDF processing
    mock_adapter = MagicMock()
    mock_adapter.process.return_value = pdf_document
    monkeypatch.setattr(docling_adapter, "DoclingAdapter", lambda: mock_adapter)
    
    # Mock the model engine availability check
    mock_check_availability.return_value = False
    
    # Create a mock tokenizer
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = {"input_ids": [0] * 10}  # Simulate 10 tokens
    mock_get_tokenizer.return_value = mock_tokenizer
    
    # Process the document
    chunks = chunk_text(pdf_document, max_tokens=128, output_format="python")
    
    # Verify that chunks were created
    assert len(chunks) > 0, "No chunks were generated for PDF document"
    
    # Check that each chunk has the required fields
    for chunk in chunks:
        assert "id" in chunk, "Chunk is missing id field"
        assert "content" in chunk, "Chunk is missing content field"
        assert "parent" in chunk, "Chunk is missing parent field"
        assert "type" in chunk, "Chunk is missing type field"
        assert chunk["type"] == "pdf", "Chunk type should be 'pdf'"


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
