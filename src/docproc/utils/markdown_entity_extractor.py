"""
Markdown entity extraction utilities.

This module provides functions to extract structured entities from markdown content,
such as headings, code blocks, tables, and links.
"""

import re
from typing import Dict, Any, List

# Regex patterns for markdown elements
HEADING_PATTERN = re.compile(r'^(#{1,6})\s+(.*?)(?:\s+#+)?$', re.MULTILINE)
CODE_BLOCK_PATTERN = re.compile(r'```([a-z]*)\n(.*?)\n```', re.DOTALL)
TABLE_PATTERN = re.compile(r'\|.*\|\n\|(?:-+\|)+\n(\|.*\|\n)+', re.MULTILINE)
LINK_PATTERN = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')


def extract_markdown_entities(content: str) -> List[Dict[str, Any]]:
    """
    Extract structured entities from markdown content.
    
    Args:
        content: The markdown content as a string
        
    Returns:
        List of extracted entities with their metadata
    """
    entities = []
    
    # Extract headings
    for match in HEADING_PATTERN.finditer(content):
        level = len(match.group(1))  # Number of # characters
        heading_text = match.group(2).strip()
        start_pos = match.start()
        
        entities.append({
            "type": f"heading_{level}",
            "name": heading_text,
            "value": heading_text,
            "level": level,
            "start_pos": start_pos,
            "confidence": 1.0
        })
    
    # Extract code blocks
    for match in CODE_BLOCK_PATTERN.finditer(content):
        language = match.group(1) or "text"
        code_content = match.group(2)
        start_pos = match.start()
        
        entities.append({
            "type": "code_block",
            "name": f"code_{language}",
            "value": code_content,
            "language": language,
            "start_pos": start_pos,
            "confidence": 1.0
        })
    
    # Extract tables
    for match in TABLE_PATTERN.finditer(content):
        table_content = match.group(0)
        start_pos = match.start()
        
        entities.append({
            "type": "table",
            "name": "table",
            "value": table_content,
            "start_pos": start_pos,
            "confidence": 1.0
        })
    
    # Extract links
    for match in LINK_PATTERN.finditer(content):
        link_text = match.group(1)
        link_url = match.group(2)
        start_pos = match.start()
        
        entities.append({
            "type": "link",
            "name": link_text,
            "value": link_url,
            "url": link_url,
            "text": link_text,
            "start_pos": start_pos,
            "confidence": 1.0
        })
    
    return entities


def extract_markdown_metadata(content: str, file_path: str) -> Dict[str, Any]:
    """
    Extract metadata from markdown content.
    
    Args:
        content: The markdown content as a string
        file_path: Path to the markdown file
        
    Returns:
        Dictionary of extracted metadata
    """
    metadata: Dict[str, Any] = {
        "doc_type": "markdown",
        "source": file_path
    }
    
    # Extract title from first heading
    title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
    if title_match:
        metadata["title"] = title_match.group(1).strip()
    
    # Look for author information
    author_patterns = [
        r'(?:Author|Authors):\s*(.+?)(?:\n|$)',  # Author: Name
        r'By\s+(.+?)(?:\n|$)',                    # By Name
        r'\*\s*(.+?)\s*\*\s*$'                   # *Name* (italics at end of line)
    ]
    
    authors: List[str] = []
    for pattern in author_patterns:
        author_match = re.search(pattern, content, re.MULTILINE | re.IGNORECASE)
        if author_match:
            author_text = author_match.group(1).strip()
            # Split authors if comma-separated
            if ',' in author_text:
                authors.extend([a.strip() for a in author_text.split(',')])
            else:
                authors.append(author_text)
    
    if authors:
        metadata["authors"] = authors
    
    # Look for date information
    date_patterns = [
        r'(?:Date|Published|Updated):\s*(.+?)(?:\n|$)',  # Date: YYYY-MM-DD
        r'(?:\d{1,2}\s+\w+\s+\d{4})',                    # 15 January 2023
        r'(?:\d{4}-\d{2}-\d{2})'                         # 2023-01-15
    ]
    
    for pattern in date_patterns:
        date_match = re.search(pattern, content, re.MULTILINE | re.IGNORECASE)
        if date_match:
            metadata["date_published"] = date_match.group(0).strip()
            break
    
    return metadata
