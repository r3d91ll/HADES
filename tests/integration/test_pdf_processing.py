"""
Integration test for PDF processing pipeline.

This test validates the full pipeline functionality with real-world PDF files.
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.docproc.adapters.docling_adapter import DoclingAdapter
from src.chunking.text_chunkers.chonky_chunker import chunk_document
from src.chunking.text_chunkers.chonky_batch import chunk_document_batch


def test_single_pdf_processing():
    """Test processing a single PDF file."""
    # Path to the test PDF
    pdf_path = Path("docs/PathRAG_paper.pdf")
    assert pdf_path.exists(), f"Test PDF not found at {pdf_path}"
    
    # Initialize the DoclingAdapter
    adapter = DoclingAdapter()
    
    # Process the PDF
    start_time = time.time()
    processed_doc = adapter.process(pdf_path)
    processing_time = time.time() - start_time
    
    # Validate the processed document
    assert processed_doc is not None
    assert "id" in processed_doc
    assert "content" in processed_doc
    assert "metadata" in processed_doc
    assert len(processed_doc["content"]) > 0
    
    print(f"\nProcessed {pdf_path.name} in {processing_time:.2f} seconds")
    print(f"Document ID: {processed_doc['id']}")
    print(f"Content length: {len(processed_doc['content'])} characters")
    print(f"Metadata: {processed_doc['metadata']}")
    
    # Chunk the document
    start_time = time.time()
    chunked_doc = chunk_document(processed_doc)
    chunking_time = time.time() - start_time
    
    # Validate the chunked document
    assert chunked_doc is not None
    assert "chunks" in chunked_doc
    assert len(chunked_doc["chunks"]) > 0
    
    print(f"Chunked document in {chunking_time:.2f} seconds")
    print(f"Number of chunks: {len(chunked_doc['chunks'])}")
    print(f"Average chunk size: {sum(len(c['content']) for c in chunked_doc['chunks']) / len(chunked_doc['chunks']):.0f} characters")
    
    return chunked_doc


def test_serial_pdf_processing():
    """Test processing multiple PDF files in sequence."""
    # Paths to the test PDFs
    pdf_paths = [
        Path("docs/PathRAG_paper.pdf"),
        Path("test-data/ISNE_paper.pdf")
    ]
    
    for pdf_path in pdf_paths:
        assert pdf_path.exists(), f"Test PDF not found at {pdf_path}"
    
    # Initialize the DoclingAdapter
    adapter = DoclingAdapter()
    
    # Process each PDF in sequence
    results = []
    total_start_time = time.time()
    
    for pdf_path in pdf_paths:
        start_time = time.time()
        processed_doc = adapter.process(pdf_path)
        processing_time = time.time() - start_time
        
        # Validate the processed document
        assert processed_doc is not None
        assert "id" in processed_doc
        assert "content" in processed_doc
        assert "metadata" in processed_doc
        assert len(processed_doc["content"]) > 0
        
        print(f"\nProcessed {pdf_path.name} in {processing_time:.2f} seconds")
        print(f"Document ID: {processed_doc['id']}")
        print(f"Content length: {len(processed_doc['content'])} characters")
        
        # Chunk the document
        start_time = time.time()
        chunked_doc = chunk_document(processed_doc)
        chunking_time = time.time() - start_time
        
        # Validate the chunked document
        assert chunked_doc is not None
        assert "chunks" in chunked_doc
        assert len(chunked_doc["chunks"]) > 0
        
        print(f"Chunked document in {chunking_time:.2f} seconds")
        print(f"Number of chunks: {len(chunked_doc['chunks'])}")
        
        results.append(chunked_doc)
    
    total_time = time.time() - total_start_time
    print(f"\nTotal processing time for {len(pdf_paths)} PDFs in sequence: {total_time:.2f} seconds")
    
    return results


def test_batch_pdf_processing():
    """Test processing multiple PDF files as a batch."""
    # Paths to the test PDFs
    pdf_paths = [
        Path("docs/PathRAG_paper.pdf"),
        Path("test-data/ISNE_paper.pdf")
    ]
    
    for pdf_path in pdf_paths:
        assert pdf_path.exists(), f"Test PDF not found at {pdf_path}"
    
    # Initialize the DoclingAdapter
    adapter = DoclingAdapter()
    
    # Process each PDF and collect the results
    processed_docs = []
    total_start_time = time.time()
    
    for pdf_path in pdf_paths:
        processed_doc = adapter.process(pdf_path)
        processed_docs.append(processed_doc)
    
    processing_time = time.time() - total_start_time
    print(f"\nProcessed {len(pdf_paths)} PDFs in {processing_time:.2f} seconds")
    
    # Batch chunk the documents
    start_time = time.time()
    chunked_docs = chunk_document_batch(processed_docs)
    chunking_time = time.time() - start_time
    
    # Validate the chunked documents
    assert chunked_docs is not None
    assert len(chunked_docs) == len(pdf_paths)
    
    for i, chunked_doc in enumerate(chunked_docs):
        assert "chunks" in chunked_doc
        assert len(chunked_doc["chunks"]) > 0
        print(f"\nDocument {i+1} ({pdf_paths[i].name}):")
        print(f"Number of chunks: {len(chunked_doc['chunks'])}")
        print(f"Average chunk size: {sum(len(c['content']) for c in chunked_doc['chunks']) / len(chunked_doc['chunks']):.0f} characters")
    
    print(f"Batch chunked {len(pdf_paths)} documents in {chunking_time:.2f} seconds")
    
    return chunked_docs


def compare_results(serial_results, batch_results):
    """Compare the results from serial and batch processing."""
    assert len(serial_results) == len(batch_results)
    
    print("\nComparing serial vs. batch processing results:")
    
    for i in range(len(serial_results)):
        serial_doc = serial_results[i]
        batch_doc = batch_results[i]
        
        # Compare number of chunks
        serial_chunks = len(serial_doc["chunks"])
        batch_chunks = len(batch_doc["chunks"])
        print(f"\nDocument {i+1}:")
        print(f"Serial chunks: {serial_chunks}, Batch chunks: {batch_chunks}")
        
        # Compare chunk content (first few chunks)
        for j in range(min(3, min(serial_chunks, batch_chunks))):
            serial_content = serial_doc["chunks"][j]["content"]
            batch_content = batch_doc["chunks"][j]["content"]
            content_match = serial_content == batch_content
            print(f"Chunk {j+1} content match: {content_match}")
            if not content_match:
                print(f"Serial length: {len(serial_content)}, Batch length: {len(batch_content)}")


if __name__ == "__main__":
    print("Testing single PDF processing...")
    single_result = test_single_pdf_processing()
    
    print("\n" + "="*80 + "\n")
    
    print("Testing serial PDF processing...")
    serial_results = test_serial_pdf_processing()
    
    print("\n" + "="*80 + "\n")
    
    print("Testing batch PDF processing...")
    batch_results = test_batch_pdf_processing()
    
    print("\n" + "="*80 + "\n")
    
    print("Comparing serial and batch processing results...")
    compare_results(serial_results, batch_results)
