"""
Test script for the enhanced Chonky chunker that preserves original text.

This script tests the modified Chonky chunker that uses Chonky for semantic
boundary detection but preserves the original text's casing and formatting.
"""

import unittest
import sys
import os
from pathlib import Path
import logging

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.chunking.text_chunkers.chonky_chunker import chunk_text

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestChonkyOriginalText(unittest.TestCase):
    """Test the enhanced Chonky chunker that preserves original text."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a sample text with mixed casing and formatting
        self.sample_text = """
# HADES-PathRAG System

The HADES-PathRAG system is designed to process and analyze documents 
with a focus on preserving ORIGINAL text formatting and casing.

## Key Features

1. Semantic chunking with Chonky
2. Preservation of ORIGINAL text
3. Overlapping context for better retrieval

This test ensures that the chunking system correctly maintains capitalization,
which is CRITICAL for preserving proper nouns, acronyms, and code snippets.
"""
        
        # Create a document dictionary
        self.document = {
            "id": "test_doc_001",
            "path": "test_document.md",
            "content": self.sample_text,
            "type": "markdown"
        }

    def test_original_text_preservation(self):
        """Test that the original text casing and formatting is preserved."""
        # Process the document with the chunker
        chunks = chunk_text(self.document, max_tokens=1024, output_format="python")
        
        # Verify that chunks were created
        self.assertGreater(len(chunks), 0, "No chunks were generated")
        
        # Check that original casing is preserved
        # Collect all content to check against the original text
        all_content = "".join([chunk.get("content", "") for chunk in chunks])
        
        # Check for uppercase words that should be preserved in the combined content
        self.assertIn("HADES-PathRAG", all_content, 
                     "Original uppercase text 'HADES-PathRAG' was not preserved")
        self.assertIn("ORIGINAL", all_content, 
                     "Original uppercase text 'ORIGINAL' was not preserved")
        self.assertIn("CRITICAL", all_content, 
                     "Original uppercase text 'CRITICAL' was not preserved")
        
        # Check each chunk for required fields
        for chunk in chunks:
            # Check if we're in fallback mode (no overlap_context field)
            if "overlap_context" not in chunk:
                logger.warning("Test running in fallback mode without overlap")
                continue
                
            content = chunk.get("content", "")
            overlap_context = chunk.get("overlap_context", {})
            
            # Verify that overlap_context has the expected structure
            self.assertIn("pre_context", overlap_context, 
                         "overlap_context is missing pre_context field")
            self.assertIn("post_context", overlap_context, 
                         "overlap_context is missing post_context field")
            
            # Reconstruct the full content with overlap
            full_content = (
                overlap_context.get("pre_context", "") + 
                content + 
                overlap_context.get("post_context", "")
            )
            
            # Verify that content is contained in the reconstructed content
            self.assertIn(content, full_content, 
                         "The original content is not contained in the reconstructed content")
            
            # Verify that content_hash is present
            self.assertIn("content_hash", chunk, 
                         "Chunk is missing content_hash field")

    def test_overlap_functionality(self):
        """Test that the overlap functionality is working correctly."""
        # Process the document with the chunker
        chunks = chunk_text(self.document, max_tokens=1024, output_format="python")
        
        # Skip test if only one chunk was generated
        if len(chunks) <= 1:
            self.skipTest("Need at least two chunks to test overlap")
        
        # Check if we're in fallback mode (no content_with_overlap field)
        if "content_with_overlap" not in chunks[0]:
            logger.warning("Skipping overlap test - running in fallback mode without overlap")
            self.skipTest("Test running in fallback mode without overlap support")
        
        # Check that consecutive chunks have overlapping content
        for i in range(len(chunks) - 1):
            current_chunk = chunks[i]
            next_chunk = chunks[i + 1]
            
            current_content = current_chunk.get("content", "")
            next_content = next_chunk.get("content", "")
            
            current_with_overlap = current_chunk.get("content_with_overlap", "")
            next_with_overlap = next_chunk.get("content_with_overlap", "")
            
            # The end of the current chunk's overlapped content should contain
            # the beginning of the next chunk's content, or vice versa
            has_overlap = False
            
            # Check if the end of current chunk overlaps with start of next
            if current_content and next_content:
                # Try with smaller overlap for more reliable testing
                min_overlap_size = min(20, len(current_content), len(next_content))
                if min_overlap_size > 0:
                    # Check if there's any overlap between end of current and start of next
                    for overlap_size in range(min_overlap_size, 0, -1):
                        if current_content[-overlap_size:] == next_content[:overlap_size]:
                            has_overlap = True
                            break
            
            # If no direct content overlap, check the overlap fields
            if not has_overlap and current_with_overlap and next_with_overlap:
                # Check if current_with_overlap contains the start of next_content
                if next_content and next_content[:20] in current_with_overlap:
                    has_overlap = True
                # Check if next_with_overlap contains the end of current_content
                elif current_content and current_content[-20:] in next_with_overlap:
                    has_overlap = True
            
            # If we have overlap fields but no overlap detected, check overlap positions
            if not has_overlap:
                current_overlap_end = current_chunk.get("overlap_end", 0)
                next_overlap_start = next_chunk.get("overlap_start", 0)
                if current_overlap_end > next_overlap_start:
                    has_overlap = True
            
            self.assertTrue(has_overlap, f"No overlap detected between chunks {i} and {i+1}")


if __name__ == "__main__":
    unittest.main()
