"""
File batching utilities for preprocessing and ingestion.

This module provides a centralized system for file discovery and batching
by file type to enable parallel processing in pipelines.
"""

from collections import defaultdict
import os
import fnmatch
from pathlib import Path
from typing import Dict, List, Set, Optional, Union, TypedDict, Mapping, Any


class BatchStats(TypedDict):
    """Statistics about batched files."""
    total: int
    by_type: Dict[str, int]


class FileBatcher:
    """Centralized file discovery and batching system for document processing."""
    
    def __init__(self, 
                 file_type_map: Optional[Dict[str, List[str]]] = None,
                 exclude_patterns: Optional[List[str]] = None):
        """
        Initialize with file type mappings and exclusion patterns.
        
        Args:
            file_type_map: Dictionary mapping file types to extensions
                           e.g., {'python': ['.py'], 'markdown': ['.md']}
            exclude_patterns: List of glob patterns to exclude
        """
        self.file_type_map = file_type_map or {
            'python': ['.py'],
            'javascript': ['.js', '.jsx', '.ts', '.tsx'],
            'java': ['.java'],
            'cpp': ['.cpp', '.hpp', '.cc', '.h'],
            'markdown': ['.md', '.markdown'],
            'pdf': ['.pdf'],
            'json': ['.json'],
            'yaml': ['.yaml', '.yml'],
            'csv': ['.csv'],
            'text': ['.txt'],
            'html': ['.html', '.htm'],
            'xml': ['.xml'],
        }
        
        # Reverse mapping for quick lookup: extension -> type
        self.ext_to_type: Dict[str, str] = {}
        for file_type, extensions in self.file_type_map.items():
            for ext in extensions:
                self.ext_to_type[ext] = file_type
                
        self.exclude_patterns = exclude_patterns or [
            '__pycache__', 
            '.git',
            'node_modules',
            'venv',
            '.env',
            '.DS_Store'
        ]
        
    def collect_files(self, root_dir: Union[str, Path]) -> Dict[str, List[Path]]:
        """
        Scan directory once, returning files batched by type.
        
        Args:
            root_dir: Root directory to scan
            
        Returns:
            Dictionary mapping file types to lists of file paths
        """
        if isinstance(root_dir, str):
            root_dir = Path(root_dir)
            
        batches: Dict[str, List[Path]] = defaultdict(list)
        
        for dirpath, dirnames, filenames in os.walk(root_dir):
            # Exclude directories in-place
            dirnames[:] = [d for d in dirnames if not any(
                fnmatch.fnmatch(os.path.relpath(os.path.join(dirpath, d), root_dir), pattern) or pattern in d
                for pattern in self.exclude_patterns
            )]

            for fname in filenames:
                rel_file_path = os.path.relpath(os.path.join(dirpath, fname), root_dir)
                # Exclude files based on patterns
                if any(fnmatch.fnmatch(rel_file_path, pattern) or 
                       fnmatch.fnmatch(fname, pattern) or 
                       pattern in fname
                       for pattern in self.exclude_patterns):
                    continue
                    
                # Get file extension and determine type
                _, ext = os.path.splitext(fname)
                ext = ext.lower()
                file_type = self.ext_to_type.get(ext, 'other')
                file_path = Path(os.path.join(dirpath, fname))
                batches[file_type].append(file_path)
                
        return dict(batches)
    
    def get_stats(self, batches: Mapping[str, List[Any]]) -> BatchStats:
        """
        Get statistics about the batched files.
        
        Args:
            batches: Output from collect_files()
            
        Returns:
            Dictionary with counts per file type and total
        """
        type_counts = {file_type: len(files) for file_type, files in batches.items()}
        total = sum(len(files) for files in batches.values())
        
        return {
            'total': total,
            'by_type': type_counts
        }
    
    def filter_batch(self, 
                     batch: Dict[str, List[Path]], 
                     allowed_types: Optional[List[str]] = None) -> Dict[str, List[Path]]:
        """
        Filter a batch to only include specific file types.
        
        Args:
            batch: Batch dictionary from collect_files()
            allowed_types: List of file types to keep, or None to keep all
            
        Returns:
            Filtered batch dictionary
        """
        if allowed_types is None:
            return batch
            
        return {
            file_type: files 
            for file_type, files in batch.items() 
            if file_type in allowed_types
        }


def collect_and_batch_files(
    root_dir: Union[str, Path], 
    file_type_map: Optional[Dict[str, List[str]]] = None,
    exclude_patterns: Optional[List[str]] = None,
    allowed_types: Optional[List[str]] = None
) -> Dict[str, List[Path]]:
    """
    Convenience function to collect and batch files from a directory.
    
    Args:
        root_dir: Root directory to scan
        file_type_map: Optional mapping of file types to extensions
        exclude_patterns: Optional list of patterns to exclude
        allowed_types: Optional list of file types to include
        
    Returns:
        Dictionary mapping file types to lists of file paths
    """
    batcher = FileBatcher(file_type_map, exclude_patterns)
    batches = batcher.collect_files(root_dir)
    
    if allowed_types:
        return batcher.filter_batch(batches, allowed_types)
    return batches
