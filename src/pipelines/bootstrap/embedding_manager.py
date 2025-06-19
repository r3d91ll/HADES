"""
Embedding Manager for Bootstrap Pipeline

Manages embedding generation for chunks in the Bootstrap Pipeline with support for
CodeBERT embeddings for both code and text modalities, dual storage (content + vectors),
and proper linking in the Sequential-ISNE schema.
"""

import logging
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from src.storage.incremental.sequential_isne_types import Chunk, Embedding, SourceType, EmbeddingType, FileType
from src.types.components.contracts import EmbeddingInput, EmbeddingOutput, DocumentChunk, ChunkEmbedding
from src.components.embedding.factory import create_embedding_component
from src.pipelines.bootstrap.config import BootstrapConfig

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """
    Manages embedding generation for Bootstrap Pipeline chunks.
    
    Features:
    - CodeBERT embeddings for code and text modalities
    - Batch processing for performance
    - Dual storage: content preservation + vector generation
    - Sequential-ISNE schema compliance
    """
    
    def __init__(self, config: BootstrapConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize embedding components
        self._embedders = {}
        self._initialize_embedders()
        
        # Statistics tracking
        self._stats = {
            'total_chunks_processed': 0,
            'total_embeddings_generated': 0,
            'code_embeddings': 0,
            'text_embeddings': 0,
            'processing_errors': 0,
            'batch_count': 0
        }
        
        self.logger.info("Initialized EmbeddingManager for Bootstrap Pipeline")
    
    def _initialize_embedders(self) -> None:
        """Initialize embedding components for different modalities."""
        try:
            # For now, use CPU embedder with CodeBERT-like model for both code and text
            # This can be enhanced later with specialized models per modality
            embedder_config = {
                'model_name': 'microsoft/codebert-base',  # CodeBERT for unified code/text
                'batch_size': self.config.embedding_batch_size if hasattr(self.config, 'embedding_batch_size') else 32,
                'max_length': 512,
                'device': 'cpu',  # Start with CPU, can upgrade to GPU later
                'normalize_embeddings': True
            }
            
            # Create unified embedder for both code and text
            self._embedders['unified'] = create_embedding_component('cpu', embedder_config)
            self.logger.info("✅ Initialized CodeBERT embedder for unified code/text processing")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize embedders: {e}")
            raise RuntimeError(f"Embedding initialization failed: {e}")
    
    def generate_embeddings_for_chunks(
        self,
        chunks: List[Chunk],
        file_path: str,
        file_type: FileType
    ) -> List[Tuple[Chunk, Embedding]]:
        """
        Generate embeddings for chunks and create embedding objects.
        
        Args:
            chunks: List of Sequential-ISNE chunks
            file_path: Source file path
            file_type: Type of source file (CODE, DOCUMENTATION, CONFIG)
            
        Returns:
            List of (chunk, embedding) tuples with preserved content and generated vectors
        """
        if not chunks:
            return []
        
        try:
            self.logger.debug(f"Generating embeddings for {len(chunks)} chunks from {file_path}")
            
            # Convert Sequential-ISNE chunks to HADES embedding input format
            document_chunks = self._convert_chunks_to_document_chunks(chunks, file_path, file_type)
            
            # Prepare embedding input
            embedding_input = EmbeddingInput(
                chunks=document_chunks,
                model_name='microsoft/codebert-base',
                batch_size=self.config.embedding_batch_size if hasattr(self.config, 'embedding_batch_size') else 32,
                metadata={
                    'file_path': file_path,
                    'file_type': file_type.value if hasattr(file_type, 'value') else str(file_type),
                    'chunk_count': len(chunks)
                }
            )
            
            # Generate embeddings using unified embedder
            embedding_output = self._embedders['unified'].embed(embedding_input)
            
            # Convert back to Sequential-ISNE format with linking
            chunk_embedding_pairs = self._create_chunk_embedding_pairs(
                chunks, embedding_output, file_path, file_type
            )
            
            # Update statistics
            self._update_stats(chunks, file_type, len(embedding_output.embeddings))
            
            self.logger.debug(f"Generated {len(chunk_embedding_pairs)} embedding pairs for {file_path}")
            return chunk_embedding_pairs
            
        except Exception as e:
            self.logger.error(f"Failed to generate embeddings for {file_path}: {e}")
            self._stats['processing_errors'] += 1
            return []
    
    def _convert_chunks_to_document_chunks(
        self,
        chunks: List[Chunk],
        file_path: str,
        file_type: FileType
    ) -> List[DocumentChunk]:
        """Convert Sequential-ISNE chunks to HADES DocumentChunk format."""
        document_chunks = []
        
        for i, chunk in enumerate(chunks):
            # Create HADES DocumentChunk from Sequential-ISNE Chunk
            doc_chunk = DocumentChunk(
                id=f"{file_path}:chunk_{i}",
                content=chunk.content,  # Preserve original content
                document_id=file_path,
                chunk_index=chunk.chunk_index,
                start_position=chunk.start_pos,
                end_position=chunk.end_pos,
                chunk_size=len(chunk.content),
                metadata={
                    'chunk_type': chunk.chunk_type.value if hasattr(chunk.chunk_type, 'value') else str(chunk.chunk_type),
                    'file_type': file_type.value if hasattr(file_type, 'value') else str(file_type),
                    'source_file_collection': chunk.source_file_collection,
                    'content_hash': chunk.content_hash
                }
            )
            document_chunks.append(doc_chunk)
        
        return document_chunks
    
    def _create_chunk_embedding_pairs(
        self,
        original_chunks: List[Chunk],
        embedding_output: EmbeddingOutput,
        file_path: str,
        file_type: FileType
    ) -> List[Tuple[Chunk, Embedding]]:
        """Create linked pairs of chunks (with content) and embeddings (with vectors)."""
        pairs = []
        
        for i, (chunk, chunk_embedding) in enumerate(zip(original_chunks, embedding_output.embeddings)):
            # Generate unique embedding ID
            embedding_id = self._generate_embedding_id(chunk.content_hash, chunk_embedding.model_name)
            
            # Update chunk with embedding reference
            updated_chunk = chunk.model_copy(update={'embedding_id': embedding_id})
            
            # Create Sequential-ISNE Embedding object
            embedding = Embedding(
                source_type=SourceType.CHUNK,
                source_collection='chunks',
                source_id=chunk.content_hash,  # Link back to chunk
                embedding_type=EmbeddingType.ORIGINAL,  # All embeddings start as original
                vector=chunk_embedding.embedding,
                model_name=chunk_embedding.model_name,
                model_version='1.0.0',
                embedding_dim=len(chunk_embedding.embedding),
                confidence=chunk_embedding.confidence or 1.0,
                created_at=datetime.utcnow(),
                metadata={
                    'file_path': file_path,
                    'file_type': file_type.value if hasattr(file_type, 'value') else str(file_type),
                    'chunk_index': chunk.chunk_index,
                    'chunk_type': chunk.chunk_type.value if hasattr(chunk.chunk_type, 'value') else str(chunk.chunk_type),
                    'embedding_id': embedding_id
                }
            )
            
            pairs.append((updated_chunk, embedding))
        
        return pairs
    
    def _generate_embedding_id(self, content_hash: str, model_name: str) -> str:
        """Generate unique embedding ID based on content and model."""
        combined = f"{content_hash}:{model_name}"
        return hashlib.sha256(combined.encode('utf-8')).hexdigest()[:16]
    
    def _update_stats(self, chunks: List[Chunk], file_type: FileType, embedding_count: int) -> None:
        """Update embedding generation statistics."""
        self._stats['total_chunks_processed'] += len(chunks)
        self._stats['total_embeddings_generated'] += embedding_count
        self._stats['batch_count'] += 1
        
        if file_type == FileType.CODE:
            self._stats['code_embeddings'] += embedding_count
        else:
            self._stats['text_embeddings'] += embedding_count
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get embedding generation statistics."""
        return {
            **self._stats,
            'success_rate': (
                self._stats['total_embeddings_generated'] / 
                max(self._stats['total_chunks_processed'], 1) * 100
            ),
            'code_embedding_ratio': (
                self._stats['code_embeddings'] / 
                max(self._stats['total_embeddings_generated'], 1) * 100
            ),
            'available_embedders': list(self._embedders.keys())
        }