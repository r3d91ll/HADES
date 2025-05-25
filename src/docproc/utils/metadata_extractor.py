"""
Metadata extraction utilities for document processing.

This module provides functions to extract metadata from various document types,
focusing on academic PDFs and markdown documents.
"""

import re
from typing import Dict, Any, List, Optional, Union, cast, Tuple
from pathlib import Path


def extract_metadata(content: str, source_path: str, format_type: str, source_url: str = "") -> Dict[str, Any]:
    """
    Extract metadata from document content based on document format.
    
    Args:
        content: Document content as text
        source_path: Path to the source document
        format_type: Document format (e.g., pdf, markdown)
        source_url: Optional URL where the content was sourced from
        
    Returns:
        Dictionary with extracted metadata, using "UNK" for unknown values
    """
    if format_type.lower() == "pdf":
        return extract_academic_pdf_metadata(content, source_path)
    else:
        # Default minimal metadata for other formats (including markdown and python)
        return {
            "title": Path(source_path).stem,
            "authors": [],
            "date_published": "UNK",
            "publisher": "UNK",
            "source": source_path,
            "doc_type": format_type
        }


def extract_academic_pdf_metadata(content: str, source_path: str) -> Dict[str, Any]:
    """
    Extract metadata from academic PDF content.
    
    Args:
        content: Document content as text
        source_path: Path to the source document
        
    Returns:
        Dictionary with extracted metadata
    """
    metadata = {
        "doc_type": "academic_pdf",
        "title": "UNK",
        "authors": ["UNK"],
        "date_published": "UNK",
        "publisher": "UNK",
        "source": source_path
    }
    
    lines = content.strip().split("\n")
    non_empty_lines = [l for l in lines if l.strip()]
    
    # Extract title
    # Look for lines starting with ## or the first line of content
    for line in non_empty_lines:
        if line.startswith("##") and len(line) > 3:
            metadata["title"] = line.strip("# ").strip()
            break
    if metadata["title"] == "UNK" and non_empty_lines:
        metadata["title"] = non_empty_lines[0].strip()
    
    # Extract authors
    # Look for author lines between title and abstract
    title_index = -1
    abstract_index = -1
    # Handle metadata["title"] which could be a string or a sequence
    title_value = metadata["title"]
    # Convert to string for searching if it's not already
    title_str = title_value if isinstance(title_value, str) else (
        title_value[0] if isinstance(title_value, (list, tuple)) and title_value else "UNK"
    )
    
    for i, line in enumerate(non_empty_lines):
        if title_str in line:
            title_index = i
        if "abstract" in line.lower() and i > title_index:
            abstract_index = i
            break
    
    if title_index >= 0 and abstract_index > title_index:
        author_section = non_empty_lines[title_index+1:abstract_index]
        potential_authors = []
        for line in author_section:
            # Common patterns in author sections: affiliations, emails, commas between names
            line_lower = line.lower()
            if "university" in line_lower or "@" in line_lower or "," in line_lower:
                potential_authors.append(line.strip())
        if potential_authors:
            metadata["authors"] = potential_authors
    
    # Extract publication date
    # Look for year patterns in the content
    date_patterns = [
        r'\b(19|20)\d{2}\b',  # Year like 2023
        r'\b(19|20)\d{2}[-/](0[1-9]|1[0-2])\b'  # Year-month
    ]
    
    for pattern in date_patterns:
        for line in non_empty_lines[:30]:  # Check only the first 30 lines
            match = re.search(pattern, line)
            if match:
                metadata["date_published"] = match.group(0)
                break
        if metadata["date_published"] != "UNK":
            break
    
    # Extract publisher
    # Look for common academic publishers
    publishers = ["arXiv", "IEEE", "ACM", "Springer", "Elsevier", "Nature", "Science"]
    for publisher in publishers:
        if publisher.lower() in content.lower():
            metadata["publisher"] = publisher
            break
            
    return metadata
