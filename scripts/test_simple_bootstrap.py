#!/usr/bin/env python3
"""
Simple Bootstrap Test Script

This script tests just the document processing and chunking stages
to examine the JSON output structure without training complexity.
"""

import sys
import logging
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.orchestration.pipelines.stages.document_processor import DocumentProcessorStage
from src.orchestration.pipelines.stages.chunking import ChunkingStage

def setup_logging():
    """Set up logging for simple bootstrap test."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('simple_bootstrap_test.log')
        ]
    )

def main():
    """Run simple bootstrap test to examine JSON structures."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Define paths
    corpus_dir = project_root / "test-data"
    output_dir = project_root / "test-output" / "simple-bootstrap-test" / datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "docproc").mkdir(exist_ok=True)
    (output_dir / "chunking").mkdir(exist_ok=True)
    
    logger.info(f"Starting simple bootstrap test")
    logger.info(f"Corpus directory: {corpus_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    try:
        # Initialize pipeline components
        doc_processor = DocumentProcessorStage()
        chunker = ChunkingStage()
        
        # Process all PDF files
        pdf_files = list(corpus_dir.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        all_documents = []
        all_chunks = []
        
        for i, pdf_file in enumerate(pdf_files):
            logger.info(f"Processing {i+1}/{len(pdf_files)}: {pdf_file.name}")
            
            try:
                # Stage 1: Document Processing
                documents = doc_processor.run([str(pdf_file)])
                logger.info(f"Document processing produced {len(documents)} documents")
                
                # Save document processing output
                doc_output_file = output_dir / "docproc" / f"{pdf_file.stem}_docproc.json"
                with open(doc_output_file, 'w') as f:
                    json.dump(documents, f, indent=2, default=str)
                logger.info(f"Saved document processing output to: {doc_output_file}")
                
                if documents:
                    all_documents.extend(documents)
                    
                    # Stage 2: Chunking
                    chunks = chunker.run(documents)
                    logger.info(f"Chunking produced {len(chunks)} chunks")
                    
                    # Convert chunks to dictionaries for JSON serialization
                    chunks_dict = []
                    for chunk in chunks:
                        chunk_dict = {
                            'id': chunk.id,
                            'text': chunk.text,
                            'source_document': pdf_file.name,
                            'metadata': chunk.metadata if hasattr(chunk, 'metadata') else {}
                        }
                        chunks_dict.append(chunk_dict)
                    
                    all_chunks.extend(chunks_dict)
                    
                    # Save chunking output
                    chunk_output_file = output_dir / "chunking" / f"{pdf_file.stem}_chunks.json"
                    with open(chunk_output_file, 'w') as f:
                        json.dump(chunks_dict, f, indent=2, default=str)
                    logger.info(f"Saved chunking output to: {chunk_output_file}")
                    
            except Exception as e:
                logger.error(f"Failed to process {pdf_file.name}: {e}")
                continue
        
        # Save combined outputs
        combined_docs_file = output_dir / "all_documents.json"
        with open(combined_docs_file, 'w') as f:
            json.dump(all_documents, f, indent=2, default=str)
        
        combined_chunks_file = output_dir / "all_chunks.json"
        with open(combined_chunks_file, 'w') as f:
            json.dump(all_chunks, f, indent=2, default=str)
        
        # Print analysis
        print(f"\n=== Simple Bootstrap Test Results ===")
        print(f"Output directory: {output_dir}")
        print(f"Total documents processed: {len(all_documents)}")
        print(f"Total chunks created: {len(all_chunks)}")
        
        # Examine document structure
        if all_documents:
            sample_doc = all_documents[0]
            print(f"\n=== Sample Document Structure ===")
            print(f"Document keys: {list(sample_doc.keys())}")
            print(f"Document ID: {sample_doc.get('id', 'N/A')}")
            print(f"Document file name: {sample_doc.get('file_name', 'N/A')}")
            
        # Examine chunk structure
        if all_chunks:
            sample_chunk = all_chunks[0]
            print(f"\n=== Sample Chunk Structure ===")
            print(f"Chunk keys: {list(sample_chunk.keys())}")
            print(f"Chunk ID: {sample_chunk.get('id', 'N/A')}")
            print(f"Chunk text preview: {sample_chunk.get('text', '')[:100]}...")
            print(f"Chunk source: {sample_chunk.get('source_document', 'N/A')}")
            
        print(f"\n=== JSON Files Created ===")
        for subdir in ['docproc', 'chunking']:
            subdir_path = output_dir / subdir
            if subdir_path.exists():
                files = list(subdir_path.glob("*.json"))
                print(f"{subdir}/: {len(files)} files")
                for file in files:
                    print(f"  - {file.name} ({file.stat().st_size} bytes)")
        
        print(f"\nCombined files:")
        print(f"  - all_documents.json ({combined_docs_file.stat().st_size} bytes)")
        print(f"  - all_chunks.json ({combined_chunks_file.stat().st_size} bytes)")
        
        return True
        
    except Exception as e:
        logger.error(f"Simple bootstrap test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)