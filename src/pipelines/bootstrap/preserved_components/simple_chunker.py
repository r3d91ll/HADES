#!/usr/bin/env python3
"""
Simple Chunker for ISNE - No Embeddings Required

AST-based chunking for Python code and simple text chunking for other files.
No dependency on embedding models or semantic chunkers.
"""

import ast
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import re

logger = logging.getLogger(__name__)


class SimpleChunker:
    """
    Simple chunker that doesn't require any embedding models.
    - Python code → AST-based chunking
    - Text/Markdown → Line/paragraph-based chunking
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Text chunking configuration
        self.max_chunk_size = self.config.get('max_chunk_size', 1000)
        self.overlap_size = self.config.get('overlap_size', 100)
        
        # Python chunking configuration
        self.python_chunk_functions = self.config.get('python_chunk_functions', True)
        self.python_chunk_classes = self.config.get('python_chunk_classes', True)
        
        logger.info("Initialized SimpleChunker (no embeddings required)")
    
    def chunk_content(self, content: str, file_type: str, source_file: str = "") -> List[Dict[str, Any]]:
        """
        Chunk content based on file type.
        
        Args:
            content: Raw content to chunk
            file_type: 'python' or 'text'/'markdown'
            source_file: Original file path for metadata
            
        Returns:
            List of chunk dictionaries with content and metadata
        """
        logger.debug(f"Chunking {source_file} with file_type='{file_type}'")
        
        if file_type == 'python':
            logger.debug(f"Using AST chunking for {source_file}")
            return self._chunk_python_code(content, source_file)
        else:
            logger.debug(f"Using simple text chunking for {source_file}")
            return self._chunk_text_simple(content, source_file, file_type)
    
    def _chunk_python_code(self, code: str, source_file: str) -> List[Dict[str, Any]]:
        """
        Chunk Python code using AST analysis.
        Extracts functions, classes, and top-level statements as separate chunks.
        """
        chunks: list[dict[str, Any]] = []
        
        try:
            tree = ast.parse(code)
            
            # Track processed lines to avoid duplication
            processed_lines = set()
            
            # Extract classes and functions
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
                        start_line = node.lineno
                        end_line = node.end_lineno or start_line
                        
                        # Skip if already processed
                        if any(line in processed_lines for line in range(start_line, end_line + 1)):
                            continue
                        
                        # Extract source lines
                        code_lines = code.split('\n')
                        if start_line <= len(code_lines):
                            chunk_content = '\n'.join(code_lines[start_line-1:end_line])
                            
                            chunk_type = 'function' if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else 'class'
                            
                            chunks.append({
                                'content': chunk_content.strip(),
                                'chunk_type': chunk_type,
                                'chunk_index': len(chunks),
                                'start_offset': start_line,
                                'end_offset': end_line,
                                'metadata': {
                                    'source_file': source_file,
                                    'name': node.name,
                                    'chunking_method': 'ast'
                                }
                            })
                            
                            # Mark lines as processed
                            for line_num in range(start_line, end_line + 1):
                                processed_lines.add(line_num)
            
            # Extract remaining top-level code (imports, constants, etc.)
            code_lines = code.split('\n')
            unprocessed_content = []
            
            for i, line in enumerate(code_lines, 1):
                if i not in processed_lines and line.strip():
                    unprocessed_content.append(line)
            
            if unprocessed_content:
                chunks.append({
                    'content': '\n'.join(unprocessed_content).strip(),
                    'chunk_type': 'module_level',
                    'chunk_index': len(chunks),
                    'start_offset': 0,
                    'end_offset': len(unprocessed_content),
                    'metadata': {
                        'source_file': source_file,
                        'name': 'module_imports_and_constants',
                        'chunking_method': 'ast'
                    }
                })
            
            logger.debug(f"AST chunking extracted {len(chunks)} chunks from {source_file}")
            
        except SyntaxError as e:
            logger.warning(f"Failed to parse Python code in {source_file}: {e}")
            # Fallback to text chunking for malformed Python
            return self._chunk_text_simple(code, source_file, 'python')
        
        return chunks
    
    def _chunk_text_simple(self, text: str, source_file: str, file_type: str) -> List[Dict[str, Any]]:
        """
        Simple text chunking based on character count and paragraph boundaries.
        No embeddings required.
        """
        chunks: list[dict[str, Any]] = []
        
        # Split by double newlines (paragraphs) or markdown sections
        if file_type == 'markdown':
            # Split on markdown headers
            sections = re.split(r'\n(?=#)', text)
        else:
            # Split on double newlines (paragraphs)
            sections = re.split(r'\n\s*\n', text)
        
        current_chunk: list[str] = []
        current_size = 0
        
        for section in sections:
            section = section.strip()
            if not section:
                continue
            
            section_size = len(section)
            
            # If adding this section would exceed max size, save current chunk
            if current_chunk and current_size + section_size > self.max_chunk_size:
                chunk_content = '\n\n'.join(current_chunk)
                chunks.append({
                    'content': chunk_content,
                    'chunk_type': 'text',
                    'chunk_index': len(chunks),
                    'start_offset': len(text) - len(chunk_content),
                    'end_offset': len(text),
                    'metadata': {
                        'source_file': source_file,
                        'chunking_method': 'simple_text'
                    }
                })
                
                # Start new chunk with overlap if configured
                if self.overlap_size > 0 and current_chunk:
                    # Keep last part of previous chunk as overlap
                    overlap_text = chunk_content[-self.overlap_size:]
                    current_chunk = [overlap_text, section]
                    current_size = len(overlap_text) + section_size
                else:
                    current_chunk = [section]
                    current_size = section_size
            else:
                current_chunk.append(section)
                current_size += section_size
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_content = '\n\n'.join(current_chunk)
            chunks.append({
                'content': chunk_content,
                'chunk_type': 'text',
                'chunk_index': len(chunks),
                'start_offset': len(text) - len(chunk_content),
                'end_offset': len(text),
                'metadata': {
                    'source_file': source_file,
                    'chunking_method': 'simple_text'
                }
            })
        
        # If no chunks were created, create a single chunk with all content
        if not chunks and text.strip():
            chunks.append({
                'content': text.strip(),
                'chunk_type': 'text',
                'chunk_index': 0,
                'start_offset': 0,
                'end_offset': len(text),
                'metadata': {
                    'source_file': source_file,
                    'chunking_method': 'simple_text'
                }
            })
        
        logger.debug(f"Simple text chunking created {len(chunks)} chunks from {source_file}")
        return chunks