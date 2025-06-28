"""
Chunk processor for bootstrap pipeline.
"""

from typing import Dict, List, Any, Optional, Iterator, Tuple
import logging
import hashlib
from datetime import datetime

logger = logging.getLogger(__name__)


class ChunkProcessor:
    """
    Processes document chunks and creates chunk-level nodes.
    
    This processor handles chunk nodes and maintains relationships
    between chunks and their source documents.
    """
    
    def __init__(self, maintain_hierarchy: bool = True):
        """
        Initialize processor.
        
        Args:
            maintain_hierarchy: Whether to maintain document-chunk hierarchy
        """
        self.maintain_hierarchy = maintain_hierarchy
        logger.info("Initialized ChunkProcessor")
        
    def process_chunks(self, 
                      chunks: List[Dict[str, Any]], 
                      document_node: Optional[Dict[str, Any]] = None) -> Tuple[List[Dict[str, Any]], List[Tuple[str, str]]]:
        """
        Process chunks into nodes with optional document relationship.
        
        Args:
            chunks: List of chunk dictionaries
            document_node: Optional parent document node
            
        Returns:
            Tuple of (chunk_nodes, document_chunk_relationships)
        """
        chunk_nodes = []
        relationships = []
        
        for idx, chunk in enumerate(chunks):
            # Create chunk node
            chunk_node = self._create_chunk_node(chunk, idx)
            chunk_nodes.append(chunk_node)
            
            # Create document-chunk relationship if maintaining hierarchy
            if self.maintain_hierarchy and document_node:
                relationships.append((
                    document_node['node_id'],
                    chunk_node['node_id']
                ))
                
        return chunk_nodes, relationships
        
    def _create_chunk_node(self, chunk: Dict[str, Any], index: int) -> Dict[str, Any]:
        """Create a node from a chunk."""
        # Generate node ID
        content_hash = hashlib.md5(
            chunk.get('content', '').encode()
        ).hexdigest()[:8]
        
        source_id = chunk.get('document_id', 'unknown')
        node_id = f"chunk_{source_id}_{index}_{content_hash}"
        
        # Build chunk node
        node = {
            'node_id': node_id,
            'node_type': 'chunk',
            'chunk_index': index,
            'content': chunk.get('content', ''),
            'start_pos': chunk.get('start', 0),
            'end_pos': chunk.get('end', 0),
            'source_file_id': chunk.get('document_id'),
            'metadata': chunk.get('metadata', {})
        }
        
        # Add embedding if present
        if 'embedding' in chunk:
            node['embedding'] = chunk['embedding']
            
        # Copy over additional fields
        for key in ['file_path', 'file_name', 'directory']:
            if key in chunk:
                node[key] = chunk[key]
                
        # Add timestamp
        node['processed_at'] = datetime.utcnow().isoformat()
        
        return node
        
    def create_sequential_relationships(self, 
                                      chunk_nodes: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
        """
        Create sequential relationships between consecutive chunks.
        
        Args:
            chunk_nodes: List of chunk nodes
            
        Returns:
            List of (from_id, to_id) tuples
        """
        relationships = []
        
        # Group chunks by source document
        chunks_by_source: Dict[str, List[Dict[str, Any]]] = {}
        for chunk in chunk_nodes:
            source = chunk.get('source_file_id', 'unknown')
            if source not in chunks_by_source:
                chunks_by_source[source] = []
            chunks_by_source[source].append(chunk)
            
        # Create sequential relationships within each document
        for source, chunks in chunks_by_source.items():
            # Sort by chunk index
            sorted_chunks = sorted(chunks, key=lambda c: c.get('chunk_index', 0))
            
            # Create relationships between consecutive chunks
            for i in range(len(sorted_chunks) - 1):
                relationships.append((
                    sorted_chunks[i]['node_id'],
                    sorted_chunks[i + 1]['node_id']
                ))
                
        return relationships
        
    def merge_overlapping_chunks(self, 
                               chunks: List[Dict[str, Any]], 
                               overlap_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Merge chunks with significant overlap.
        
        Args:
            chunks: List of chunk dictionaries
            overlap_threshold: Minimum overlap ratio to merge
            
        Returns:
            List of merged chunks
        """
        if not chunks:
            return []
            
        # Sort chunks by start position
        sorted_chunks = sorted(chunks, key=lambda c: (
            c.get('source_file_id', ''),
            c.get('start_pos', 0)
        ))
        
        merged = []
        current = sorted_chunks[0].copy()
        
        for chunk in sorted_chunks[1:]:
            # Check if same source
            if chunk.get('source_file_id') != current.get('source_file_id'):
                merged.append(current)
                current = chunk.copy()
                continue
                
            # Calculate overlap
            overlap_start = max(
                current.get('start_pos', 0),
                chunk.get('start_pos', 0)
            )
            overlap_end = min(
                current.get('end_pos', 0),
                chunk.get('end_pos', 0)
            )
            
            if overlap_end > overlap_start:
                overlap_size = overlap_end - overlap_start
                chunk_size = chunk.get('end_pos', 0) - chunk.get('start_pos', 0)
                
                if chunk_size > 0 and overlap_size / chunk_size >= overlap_threshold:
                    # Merge chunks
                    current['end_pos'] = max(
                        current.get('end_pos', 0),
                        chunk.get('end_pos', 0)
                    )
                    current['content'] = self._merge_content(
                        current.get('content', ''),
                        chunk.get('content', ''),
                        overlap_size
                    )
                else:
                    merged.append(current)
                    current = chunk.copy()
            else:
                merged.append(current)
                current = chunk.copy()
                
        merged.append(current)
        return merged
        
    def _merge_content(self, content1: str, content2: str, overlap_size: int) -> str:
        """Merge two content strings with overlap."""
        # Simple merge - could be improved with better overlap detection
        if overlap_size > 0 and content1[-overlap_size:] == content2[:overlap_size]:
            return content1 + content2[overlap_size:]
        return content1 + " " + content2