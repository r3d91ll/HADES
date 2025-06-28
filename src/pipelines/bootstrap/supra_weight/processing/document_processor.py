"""
Document processor for bootstrap pipeline.
"""

from typing import Dict, List, Any, Optional, Iterator
from pathlib import Path
import logging
import hashlib
from datetime import datetime

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Processes documents into nodes for the graph.
    
    This processor creates file-level nodes with metadata needed
    for relationship detection.
    """
    
    def __init__(self, 
                 base_path: Optional[Path] = None,
                 include_content: bool = True,
                 max_content_size: int = 1_000_000):
        """
        Initialize processor.
        
        Args:
            base_path: Base path for relative path calculation
            include_content: Whether to include file content in nodes
            max_content_size: Maximum content size to include
        """
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self.include_content = include_content
        self.max_content_size = max_content_size
        
        logger.info(f"Initialized DocumentProcessor with base_path={self.base_path}")
        
    def process_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """
        Process a single file into a node.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Node dictionary or None if processing fails
        """
        try:
            # Get file metadata
            stat = file_path.stat()
            
            # Calculate relative path
            try:
                relative_path = file_path.relative_to(self.base_path)
            except ValueError:
                relative_path = file_path
                
            # Create node ID
            node_id = self._generate_node_id(str(file_path))
            
            # Build node
            node = {
                'node_id': node_id,
                'node_type': 'file',
                'file_path': str(file_path),
                'relative_path': str(relative_path),
                'file_name': file_path.name,
                'directory': str(file_path.parent),
                'extension': file_path.suffix.lower(),
                'size_bytes': stat.st_size,
                'modified_at': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'created_at': datetime.fromtimestamp(stat.st_ctime).isoformat()
            }
            
            # Add content if requested
            if self.include_content and stat.st_size <= self.max_content_size:
                try:
                    content = file_path.read_text(encoding='utf-8')
                    node['content'] = content
                    node['content_hash'] = hashlib.sha256(content.encode()).hexdigest()
                except Exception as e:
                    logger.warning(f"Could not read content from {file_path}: {e}")
                    
            return node
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return None
            
    def process_directory(self, 
                         directory_path: Path,
                         extensions: Optional[List[str]] = None,
                         recursive: bool = True) -> Iterator[Dict[str, Any]]:
        """
        Process all files in a directory.
        
        Args:
            directory_path: Path to directory
            extensions: Optional list of extensions to include (e.g., ['.py', '.js'])
            recursive: Whether to process subdirectories
            
        Yields:
            Node dictionaries
        """
        pattern = "**/*" if recursive else "*"
        
        for file_path in directory_path.glob(pattern):
            if not file_path.is_file():
                continue
                
            # Check extension filter
            if extensions and file_path.suffix.lower() not in extensions:
                continue
                
            node = self.process_file(file_path)
            if node:
                yield node
                
    def process_nodes(self, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process existing nodes (e.g., from PathRAG).
        
        This method enriches existing nodes with additional metadata
        needed for relationship detection.
        
        Args:
            nodes: List of existing nodes
            
        Returns:
            Enriched nodes
        """
        processed = []
        
        for node in nodes:
            # Copy node
            enriched = node.copy()
            
            # Ensure node_id exists
            if 'node_id' not in enriched:
                if '_key' in enriched:
                    enriched['node_id'] = enriched['_key']
                else:
                    enriched['node_id'] = self._generate_node_id(
                        str(enriched.get('file_path', '')) + 
                        str(enriched.get('chunk_index', ''))
                    )
                    
            # Add node type if missing
            if 'node_type' not in enriched:
                if 'chunk_index' in enriched:
                    enriched['node_type'] = 'chunk'
                else:
                    enriched['node_type'] = 'file'
                    
            # Extract directory from file path if missing
            if 'directory' not in enriched and 'file_path' in enriched:
                enriched['directory'] = str(Path(enriched['file_path']).parent)
                
            # Add file name if missing
            if 'file_name' not in enriched and 'file_path' in enriched:
                enriched['file_name'] = Path(enriched['file_path']).name
                
            processed.append(enriched)
            
        return processed
        
    def _generate_node_id(self, content: str) -> str:
        """Generate a unique node ID."""
        return hashlib.md5(content.encode()).hexdigest()[:16]