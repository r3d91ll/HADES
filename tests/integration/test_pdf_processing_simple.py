"""
Integration test for PDF processing and chunking.

This test validates that we can process PDF files correctly through the document
processor and chunker, and writes the chunked JSON output to the test-output directory.

Since the pipeline is still in active development, this test focuses on the implemented
parts: file discovery, document processing, and entity extraction (chunking).
"""

import sys
import json
import time
import os
import datetime
from pathlib import Path
from typing import Dict, Any, List, Union, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.docproc.adapters.docling_adapter import DoclingAdapter
from src.chunking.text_chunkers.chonky_chunker import chunk_document
from src.chunking.text_chunkers.chonky_batch import chunk_document_batch


def ensure_output_dir() -> Path:
    """Ensure the test-output directory exists.
    
    Returns:
        Path to the test-output directory
    """
    output_dir = Path("test-output")
    output_dir.mkdir(exist_ok=True)
    return output_dir


def convert_to_dict(data: Any) -> Dict[str, Any]:
    """Convert Pydantic models to dictionaries.
    
    Args:
        data: Data that might be a Pydantic model
        
    Returns:
        Dictionary representation of the data
    """
    if hasattr(data, "model_dump"):
        return data.model_dump()
    elif hasattr(data, "dict"):
        # For older Pydantic versions
        return data.dict()
    elif isinstance(data, dict):
        # Process nested dictionaries
        result = {}
        for key, value in data.items():
            if hasattr(value, "model_dump") or hasattr(value, "dict"):
                result[key] = convert_to_dict(value)
            elif isinstance(value, list):
                result[key] = [convert_to_dict(item) if hasattr(item, "model_dump") or hasattr(item, "dict") else item for item in value]
            elif hasattr(value, "isoformat"):  # Handle datetime objects
                result[key] = value.isoformat()
            else:
                result[key] = value
        return result
    elif isinstance(data, list):
        return [convert_to_dict(item) if hasattr(item, "model_dump") or hasattr(item, "dict") else item for item in data]
    elif hasattr(data, "isoformat"):  # Handle datetime objects
        return data.isoformat()
    else:
        return data


class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder that can handle datetime objects."""
    
    def default(self, obj):
        if isinstance(obj, (datetime.datetime, datetime.date, datetime.time)):
            return obj.isoformat()
        return super().default(obj)


def write_json_output(data: Any, filename: str) -> Path:
    """Write data as formatted JSON to the test-output directory.
    
    Args:
        data: The data to write as JSON (can be a Pydantic model)
        filename: The filename to write to
        
    Returns:
        Path to the written file
    """
    output_dir = ensure_output_dir()
    output_path = output_dir / filename
    
    # Convert Pydantic models to dictionaries for JSON serialization
    dict_data = convert_to_dict(data)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dict_data, f, indent=2, ensure_ascii=False, cls=DateTimeEncoder)
        
    print(f"Wrote output to {output_path}")
    return output_path


def test_single_pdf_processing(pdf_path: Path) -> Dict[str, Any]:
    """Test processing a single PDF file.
    
    Args:
        pdf_path: Path to the PDF file to process
        
    Returns:
        Dictionary with processing results and statistics
    """
    print(f"\n{'='*80}\nTesting single PDF processing: {pdf_path.name}\n{'='*80}")
    
    # Initialize the DoclingAdapter
    adapter = DoclingAdapter()
    
    # Process the PDF
    start_time = time.time()
    processed_doc = adapter.process(pdf_path)
    processing_time = time.time() - start_time
    
    # Print document info
    print(f"Processed {pdf_path.name} in {processing_time:.2f} seconds")
    print(f"Document ID: {processed_doc['id']}")
    print(f"Content length: {len(processed_doc['content'])} characters")
    print(f"Metadata: {processed_doc.get('metadata', {})}")
    
    # Chunk the document
    start_time = time.time()
    chunked_doc = chunk_document(processed_doc)
    chunking_time = time.time() - start_time
    
    # Convert to dict if it's a Pydantic model
    chunked_doc_dict = chunked_doc
    if hasattr(chunked_doc, "model_dump"):
        chunked_doc_dict = chunked_doc.model_dump()
    elif hasattr(chunked_doc, "dict"):
        # For older Pydantic versions
        chunked_doc_dict = chunked_doc.dict()
    
    # Print chunking info
    chunk_count = len(chunked_doc_dict.get("chunks", []))
    avg_chunk_size = 0
    if chunk_count > 0:
        avg_chunk_size = sum(len(c.get("content", "")) for c in chunked_doc_dict.get("chunks", [])) / chunk_count
    
    print(f"Chunked document in {chunking_time:.2f} seconds")
    print(f"Number of chunks: {chunk_count}")
    print(f"Average chunk size: {avg_chunk_size:.0f} characters")
    
    # Print first few chunks
    if chunk_count > 0:
        print("\nSample chunks:")
        for i, chunk in enumerate(chunked_doc_dict.get("chunks", [])[:3]):
            print(f"\nChunk {i+1}:")
            print(f"  ID: {chunk.get('id', 'N/A')}")
            print(f"  Type: {chunk.get('symbol_type', 'N/A')}")
            content = chunk.get("content", "")
            print(f"  Content length: {len(content)} characters")
            print(f"  Content preview: {content[:100]}...")
    
    # Write chunked document to a JSON file
    output_filename = f"single_{pdf_path.stem}_chunked.json"
    output_path = write_json_output(chunked_doc, output_filename)
    
    return {
        "document": processed_doc,
        "chunked_document": chunked_doc,
        "processing_time": processing_time,
        "chunking_time": chunking_time,
        "chunk_count": chunk_count,
        "avg_chunk_size": avg_chunk_size,
        "output_path": str(output_path)
    }


def test_batch_pdf_processing(pdf_paths: List[Path]) -> Dict[str, Any]:
    """Test processing multiple PDF files as a batch.
    
    Args:
        pdf_paths: List of paths to PDF files to process
        
    Returns:
        Dictionary with processing results and statistics
    """
    print(f"\n{'='*80}\nTesting batch PDF processing: {len(pdf_paths)} files\n{'='*80}")
    
    # Initialize the DoclingAdapter
    adapter = DoclingAdapter()
    
    # Process each PDF and collect the results
    processed_docs = []
    total_start_time = time.time()
    
    for pdf_path in pdf_paths:
        print(f"\nProcessing {pdf_path.name}...")
        processed_doc = adapter.process(pdf_path)
        processed_docs.append(processed_doc)
        print(f"Document ID: {processed_doc['id']}")
        print(f"Content length: {len(processed_doc['content'])} characters")
    
    processing_time = time.time() - total_start_time
    print(f"\nProcessed {len(pdf_paths)} PDFs in {processing_time:.2f} seconds")
    
    # Batch chunk the documents
    start_time = time.time()
    chunked_docs = chunk_document_batch(processed_docs)
    chunking_time = time.time() - start_time
    
    # Convert Pydantic models to dictionaries if needed
    chunked_docs_dict = []
    for doc in chunked_docs:
        chunked_docs_dict.append(convert_to_dict(doc))
    
    # Print chunking info
    total_chunks = sum(len(doc.get("chunks", [])) for doc in chunked_docs_dict)
    avg_chunks_per_doc = total_chunks / len(chunked_docs_dict) if chunked_docs_dict else 0
    
    print(f"Batch chunked {len(chunked_docs_dict)} documents in {chunking_time:.2f} seconds")
    print(f"Total chunks: {total_chunks}")
    print(f"Average chunks per document: {avg_chunks_per_doc:.1f}")
    
    # Print info for each document and write to JSON files
    output_paths = []
    for i, chunked_doc in enumerate(chunked_docs_dict):
        pdf_name = pdf_paths[i].name if i < len(pdf_paths) else f"Document {i+1}"
        pdf_stem = pdf_paths[i].stem if i < len(pdf_paths) else f"document_{i+1}"
        chunk_count = len(chunked_doc.get("chunks", []))
        avg_chunk_size = 0
        if chunk_count > 0:
            avg_chunk_size = sum(len(c.get("content", "")) for c in chunked_doc.get("chunks", [])) / chunk_count
        
        print(f"\nDocument {i+1} ({pdf_name}):")
        print(f"  Number of chunks: {chunk_count}")
        print(f"  Average chunk size: {avg_chunk_size:.0f} characters")
        
        # Write chunked document to a JSON file
        output_filename = f"batch_{pdf_stem}_chunked.json"
        output_path = write_json_output(chunked_doc, output_filename)
        output_paths.append(str(output_path))
    
    # Also write the batch results summary
    batch_summary = {
        "total_documents": len(chunked_docs_dict),
        "total_chunks": total_chunks,
        "avg_chunks_per_doc": avg_chunks_per_doc,
        "processing_time": processing_time,
        "chunking_time": chunking_time,
        "document_ids": [doc.get("id") for doc in chunked_docs_dict]
    }
    summary_path = write_json_output(batch_summary, "batch_processing_summary.json")
    
    return {
        "processed_docs": processed_docs,
        "chunked_docs": chunked_docs,
        "processing_time": processing_time,
        "chunking_time": chunking_time,
        "total_chunks": total_chunks,
        "avg_chunks_per_doc": avg_chunks_per_doc,
        "output_paths": output_paths,
        "summary_path": str(summary_path)
    }


def run_tests():
    """Run all PDF processing tests."""
    # Define paths to test PDFs
    pdf_paths = [
        Path("docs/PathRAG_paper.pdf"),
        Path("test-data/ISNE_paper.pdf")
    ]
    
    # Validate PDF files exist
    valid_paths = []
    for pdf_path in pdf_paths:
        if pdf_path.exists():
            valid_paths.append(pdf_path)
        else:
            print(f"WARNING: Test PDF not found at {pdf_path}")
    
    if not valid_paths:
        print("ERROR: No valid PDF files found for testing")
        return
    
    # Test single PDF processing for each file
    single_results = {}
    for pdf_path in valid_paths:
        single_results[pdf_path.name] = test_single_pdf_processing(pdf_path)
    
    # Test batch PDF processing
    batch_results = test_batch_pdf_processing(valid_paths)
    
    # Compare single vs. batch processing
    print("\n" + "=" * 80)
    print("Comparing single vs. batch processing")
    print("=" * 80)
    
    total_single_time = sum(r["processing_time"] + r["chunking_time"] for r in single_results.values())
    total_batch_time = batch_results["processing_time"] + batch_results["chunking_time"]
    time_diff = total_single_time - total_batch_time
    speedup = total_single_time / total_batch_time if total_batch_time > 0 else 0
    
    print(f"Total time for single processing: {total_single_time:.2f} seconds")
    print(f"Total time for batch processing: {total_batch_time:.2f} seconds")
    print(f"Time difference: {time_diff:.2f} seconds")
    print(f"Speedup: {speedup:.2f}x")
    
    # Compare chunk counts
    print("\nChunk count comparison:")
    valid_paths = [p for p in pdf_paths if p.name in single_results]
    
    # Get chunk counts for both methods
    single_chunk_counts = {}
    batch_chunk_counts = {}
    
    # Extract chunk counts from batch results
    chunked_docs = batch_results["chunked_docs"]
    # Convert to dict if needed
    if not isinstance(chunked_docs, list):
        chunked_docs = [chunked_docs]
    
    for i, doc in enumerate(chunked_docs):
        if i < len(valid_paths):
            # Convert Pydantic model to dict if needed
            doc_dict = doc
            if hasattr(doc, "model_dump"):
                doc_dict = doc.model_dump()
            elif hasattr(doc, "dict"):
                doc_dict = doc.dict()
                
            batch_chunk_counts[valid_paths[i].name] = len(doc_dict.get("chunks", []))
    
    for path_name, result in single_results.items():
        single_chunk_counts[path_name] = result["chunk_count"]
    
    # Print comparison
    for path_name in sorted(single_chunk_counts.keys()):
        single_count = single_chunk_counts.get(path_name, 0)
        batch_count = batch_chunk_counts.get(path_name, 0)
        diff = single_count - batch_count
        print(f"  {path_name}: Single: {single_count}, Batch: {batch_count}, Difference: {diff}")
    
    # Summary
    print(f"\n{'='*80}\nSummary\n{'='*80}")
    print(f"Successfully processed {len(valid_paths)} PDF files")
    print(f"Total chunks created: {sum(r['chunk_count'] for r in single_results.values())} (single) / {batch_results['total_chunks']} (batch)")
    print("Test completed successfully!")


if __name__ == "__main__":
    run_tests()
