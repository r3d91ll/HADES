"""
Semantic Chunking Manager for Bootstrap Pipeline

Manages different chunking strategies based on file type to create semantically meaningful
chunks for Sequential-ISNE training.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

from src.storage.incremental.sequential_isne_types import FileType, Chunk, ChunkType
from src.types.components.contracts import ChunkingInput, ChunkingOutput, TextChunk
from src.components.chunking.factory import create_chunking_component
from src.pipelines.bootstrap.config import BootstrapConfig

logger = logging.getLogger(__name__)


class SemanticChunkingManager:
    """
    Manages semantic chunking strategies for different file types.
    
    Routes files to appropriate chunkers:
    - AST Chunker: Python/code files for semantic boundaries
    - Chonky Chunker: Documentation for transformer-based chunking
    - Text Chunker: Fallback for simple cases
    """
    
    def __init__(self, config: BootstrapConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize chunkers
        self._chunkers = {}
        self._initialize_chunkers()
        
        self.logger.info("Initialized SemanticChunkingManager")
    
    def _initialize_chunkers(self) -> None:
        """Initialize different chunking components."""
        try:
            # AST chunker for code files
            ast_config = {
                'max_chunk_size': self.config.chunk_size,
                'min_chunk_size': max(50, self.config.chunk_size // 4),
                'respect_boundaries': True,
                'include_imports': True,
                'include_docstrings': True
            }
            
            self._chunkers['ast'] = create_chunking_component('ast', ast_config)
            self.logger.info("✅ Initialized AST chunker for code files")
            
            # Chonky chunker for documentation  
            chonky_config = {
                'chunk_size': self.config.chunk_size,
                'chunk_overlap': self.config.chunk_overlap,
                'use_semantic_boundaries': True,
                'respect_sentence_boundaries': True,
                'respect_paragraph_boundaries': True
            }
            
            self._chunkers['chonky'] = create_chunking_component('chonky', chonky_config)
            self.logger.info("✅ Initialized Chonky chunker for documentation")
            
            # Text chunker as fallback
            text_config = {
                'chunk_size': self.config.chunk_size,
                'chunk_overlap': self.config.chunk_overlap,
                'respect_word_boundaries': True
            }
            
            self._chunkers['text'] = create_chunking_component('text', text_config)
            self.logger.info("✅ Initialized Text chunker as fallback")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize chunkers: {e}")
            raise RuntimeError(f"Chunking initialization failed: {e}")
    
    def chunk_file_content(
        self, 
        file_path: str, 
        content: str, 
        file_type: FileType
    ) -> List[Chunk]:
        """
        Create semantic chunks from file content using appropriate chunker.
        
        Args:
            file_path: Path to the file
            content: File content
            file_type: Type of file (CODE, DOCUMENTATION, CONFIG)
            
        Returns:
            List of semantic chunks
        """
        try:
            # Select chunking strategy based on file type
            chunker_type = self._select_chunker_type(file_path, file_type)
            chunker = self._chunkers.get(chunker_type)
            
            if not chunker:
                self.logger.warning(f"No chunker available for type {chunker_type}, using fallback")
                chunker = self._chunkers['text']
                chunker_type = 'text'
            
            self.logger.debug(f"Using {chunker_type} chunker for {file_path}")
            
            # Prepare chunking input
            chunking_input = ChunkingInput(
                text=content,
                document_id=file_path,
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                metadata={
                    'file_path': file_path,
                    'file_type': file_type.value,
                    'chunker_type': chunker_type
                }
            )
            
            # Perform chunking
            chunking_output = chunker.chunk(chunking_input)
            
            # Convert HADES chunks to Sequential-ISNE chunks
            sequential_chunks = self._convert_to_sequential_chunks(
                chunking_output.chunks,
                file_path,
                file_type,
                chunker_type
            )
            
            self.logger.debug(f"Created {len(sequential_chunks)} semantic chunks for {file_path}")
            return sequential_chunks
            
        except Exception as e:
            self.logger.error(f"Failed to chunk file {file_path}: {e}")
            # Fallback to simple chunking
            return self._create_fallback_chunks(file_path, content, file_type)
    
    def _select_chunker_type(self, file_path: str, file_type: FileType) -> str:
        """Select appropriate chunker based on file type and extension."""
        path = Path(file_path)
        extension = path.suffix.lower()
        
        if file_type == FileType.CODE:
            # Use AST chunker for Python files
            if extension in {'.py', '.pyx', '.pyi'}:
                return 'ast'
            else:
                # For other code files, use text chunker with code-aware settings
                return 'text'
        
        elif file_type == FileType.DOCUMENTATION:
            # Use Chonky for documentation (semantic chunking)
            if extension in {'.md', '.rst', '.txt', '.tex'}:
                return 'chonky'
            else:
                return 'text'
        
        elif file_type == FileType.CONFIG:
            # Use text chunker for config files
            return 'text'
        
        else:
            # Default fallback
            return 'text'
    
    def _convert_to_sequential_chunks(
        self,
        hades_chunks: List[TextChunk],
        file_path: str,
        file_type: FileType,
        chunker_type: str
    ) -> List[Chunk]:
        """Convert HADES TextChunk objects to Sequential-ISNE Chunk objects."""
        sequential_chunks = []
        
        for i, hades_chunk in enumerate(hades_chunks):
            # Determine chunk type based on content and file type
            chunk_type = self._determine_semantic_chunk_type(
                hades_chunk.text, 
                file_type,
                chunker_type
            )
            
            # Create Sequential-ISNE chunk
            sequential_chunk = Chunk(
                source_file_collection=self._get_collection_for_file_type(file_type),
                source_file_id=file_path,  # Will be updated with actual ID later
                content=hades_chunk.text,  # HADES TextChunk uses 'text' not 'content'
                content_hash=self._compute_content_hash(hades_chunk.text),
                start_pos=getattr(hades_chunk, 'start_index', i * self.config.chunk_size),
                end_pos=getattr(hades_chunk, 'end_index', (i + 1) * self.config.chunk_size),
                chunk_index=i,
                chunk_type=chunk_type,
                metadata={
                    'file_path': file_path,
                    'file_type': file_type.value,
                    'chunker_type': chunker_type,
                    'semantic_boundaries': getattr(hades_chunk, 'boundaries', {}),
                    'chunk_quality_score': getattr(hades_chunk, 'quality_score', 1.0),
                    'original_chunk_metadata': getattr(hades_chunk, 'metadata', {})
                }
            )
            
            sequential_chunks.append(sequential_chunk)
        
        return sequential_chunks
    
    def _determine_semantic_chunk_type(
        self, 
        content: str, 
        file_type: FileType, 
        chunker_type: str
    ) -> ChunkType:
        """Determine chunk type based on content analysis and semantic context."""
        content_lower = content.lower().strip()
        
        if file_type == FileType.CODE:
            if chunker_type == 'ast':
                # AST chunker provides semantic context
                if any(keyword in content_lower for keyword in ['def ', 'class ', 'async def ']):
                    return ChunkType.CODE
                elif content_lower.startswith('#') or '"""' in content or "'''" in content:
                    return ChunkType.COMMENT
                else:
                    return ChunkType.CODE
            else:
                # Fallback analysis for non-Python code
                return ChunkType.CODE
        
        elif file_type == FileType.DOCUMENTATION:
            if chunker_type == 'chonky':
                # Chonky provides semantic understanding
                if content_lower.startswith('#') or content.startswith('<!--'):
                    return ChunkType.COMMENT
                elif any(marker in content_lower for marker in ['```', '~~~', '    ']):
                    return ChunkType.CODE  # Code blocks in documentation
                else:
                    return ChunkType.TEXT
            else:
                return ChunkType.TEXT
        
        elif file_type == FileType.CONFIG:
            return ChunkType.CONFIG
        
        else:
            return ChunkType.TEXT
    
    def _get_collection_for_file_type(self, file_type: FileType) -> str:
        """Get the collection name for a file type."""
        mapping = {
            FileType.CODE: "code_files",
            FileType.DOCUMENTATION: "documentation_files", 
            FileType.CONFIG: "config_files"
        }
        return mapping.get(file_type, "unknown_files")
    
    def _compute_content_hash(self, content: str) -> str:
        """Compute SHA256 hash of content."""
        import hashlib
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def _create_fallback_chunks(
        self, 
        file_path: str, 
        content: str, 
        file_type: FileType
    ) -> List[Chunk]:
        """Create simple chunks as fallback when semantic chunking fails."""
        chunks = []
        chunk_size = self.config.chunk_size
        chunk_overlap = self.config.chunk_overlap
        
        start = 0
        chunk_index = 0
        
        while start < len(content):
            end = min(start + chunk_size, len(content))
            chunk_content = content[start:end]
            
            # Skip empty chunks
            if not chunk_content.strip():
                start = end
                continue
            
            # Create fallback chunk
            chunk = Chunk(
                source_file_collection=self._get_collection_for_file_type(file_type),
                source_file_id=file_path,
                content=chunk_content,
                content_hash=self._compute_content_hash(chunk_content),
                start_pos=start,
                end_pos=end,
                chunk_index=chunk_index,
                chunk_type=ChunkType.TEXT,
                metadata={
                    'file_path': file_path,
                    'file_type': file_type.value,
                    'chunker_type': 'fallback',
                    'fallback_reason': 'semantic_chunking_failed'
                }
            )
            
            chunks.append(chunk)
            
            # Move to next chunk with overlap
            start = max(start + 1, end - chunk_overlap)
            chunk_index += 1
            
            # Prevent infinite loops
            if start >= len(content):
                break
        
        self.logger.warning(f"Used fallback chunking for {file_path}: {len(chunks)} chunks created")
        return chunks
    
    def get_chunking_stats(self) -> Dict[str, Any]:
        """Get statistics about chunking performance."""
        return {
            'available_chunkers': list(self._chunkers.keys()),
            'chunker_status': {
                name: chunker is not None 
                for name, chunker in self._chunkers.items()
            },
            'config': {
                'chunk_size': self.config.chunk_size,
                'chunk_overlap': self.config.chunk_overlap
            }
        }