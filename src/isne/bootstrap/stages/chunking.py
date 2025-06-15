"""
Chunking Stage for ISNE Bootstrap Pipeline

Handles chunking of processed documents into semantic chunks
suitable for embedding and graph construction.
"""

import logging
import traceback
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from src.components.chunking.factory import create_chunking_component
from src.types.components.contracts import ChunkingInput, DocumentChunk
from .base import BaseBootstrapStage

logger = logging.getLogger(__name__)


@dataclass
class ChunkingResult:
    """Result of chunking stage."""
    success: bool
    chunks: List[DocumentChunk]
    stats: Dict[str, Any]
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None


class ChunkingStage(BaseBootstrapStage):
    """Chunking stage for bootstrap pipeline."""
    
    def __init__(self):
        """Initialize chunking stage."""
        super().__init__("chunking")
        
    def execute(self, documents: List[Any], config: Any) -> ChunkingResult:
        """
        Execute chunking stage.
        
        Args:
            documents: List of processed documents from document processing stage
            config: ChunkingConfig object
            
        Returns:
            ChunkingResult with chunks and stats
        """
        logger.info(f"Starting chunking stage with {len(documents)} documents")
        
        try:
            # Create chunker
            chunker = create_chunking_component(config.chunker_type)
            
            all_chunks = []
            stats = {
                "input_documents": len(documents),
                "output_chunks": 0,
                "total_characters": 0,
                "document_details": [],
                "chunker_type": config.chunker_type,
                "chunk_strategy": config.strategy,
                "chunk_size": config.chunk_size,
                "chunk_overlap": config.chunk_overlap
            }
            
            # Process each document
            for doc in documents:
                try:
                    logger.info(f"Chunking document {doc.id}...")
                    
                    # Create chunking input
                    chunking_input = ChunkingInput(
                        text=doc.content,
                        document_id=doc.id,
                        chunking_strategy=config.strategy,
                        chunk_size=config.chunk_size,
                        chunk_overlap=config.chunk_overlap,
                        processing_options={
                            "preserve_structure": config.preserve_structure,
                            "extract_metadata": True,
                            **config.options
                        },
                        metadata={
                            "bootstrap_stage": "chunking",
                            "document_id": doc.id,
                            "source_file": doc.metadata.get("source_file", "unknown"),
                            "chunking_strategy": config.strategy
                        }
                    )
                    
                    # Process chunks for this document
                    doc_output = chunker.chunk(chunking_input)
                    
                    # Convert TextChunks to DocumentChunks for consistency
                    doc_chunks = []
                    for i, text_chunk in enumerate(doc_output.chunks):
                        doc_chunk = DocumentChunk(
                            id=f"{doc.id}_chunk_{i}",
                            content=text_chunk.text,
                            document_id=doc.id,
                            chunk_index=i,
                            chunk_size=len(text_chunk.text),
                            metadata={
                                **text_chunk.metadata,
                                "source_document": doc.id,
                                "chunk_strategy": config.strategy,
                                "chunk_size_config": config.chunk_size,
                                "chunk_overlap_config": config.chunk_overlap,
                                "document_metadata": doc.metadata
                            }
                        )
                        doc_chunks.append(doc_chunk)
                        all_chunks.append(doc_chunk)
                    
                    # Collect stats
                    doc_chars = sum(len(chunk.content) for chunk in doc_chunks)
                    stats["output_chunks"] += len(doc_chunks)
                    stats["total_characters"] += doc_chars
                    stats["document_details"].append({
                        "document_id": doc.id,
                        "chunks_created": len(doc_chunks),
                        "characters": doc_chars,
                        "avg_chunk_size": doc_chars / len(doc_chunks) if doc_chunks else 0,
                        "source_file": doc.metadata.get("source_file", "unknown")
                    })
                    
                    logger.info(f"  ✓ {doc.id}: {len(doc_chunks)} chunks, {doc_chars} characters")
                    
                except Exception as e:
                    error_msg = f"Failed to chunk document {doc.id}: {e}"
                    logger.error(error_msg)
                    stats["document_details"].append({
                        "document_id": doc.id,
                        "error": str(e),
                        "source_file": getattr(doc, 'metadata', {}).get("source_file", "unknown")
                    })
                    continue
            
            # Check if any chunks were created
            if not all_chunks:
                error_msg = "No chunks were successfully created"
                logger.error(error_msg)
                return ChunkingResult(
                    success=False,
                    chunks=[],
                    stats=stats,
                    error_message=error_msg
                )
            
            # Calculate additional stats
            stats["avg_chunk_size"] = stats["total_characters"] / stats["output_chunks"]
            stats["chunks_per_document"] = stats["output_chunks"] / stats["input_documents"]
            
            logger.info(f"Chunking completed: {len(all_chunks)} chunks from {len(documents)} documents")
            logger.info(f"  Average chunk size: {stats['avg_chunk_size']:.1f} characters")
            logger.info(f"  Average chunks per document: {stats['chunks_per_document']:.1f}")
            
            return ChunkingResult(
                success=True,
                chunks=all_chunks,
                stats=stats
            )
            
        except Exception as e:
            error_msg = f"Chunking stage failed: {e}"
            error_traceback = traceback.format_exc()
            logger.error(error_msg)
            logger.debug(error_traceback)
            
            return ChunkingResult(
                success=False,
                chunks=[],
                stats={},
                error_message=error_msg,
                error_traceback=error_traceback
            )
    
    def validate_inputs(self, documents: List[Any], config: Any) -> List[str]:
        """
        Validate inputs for chunking stage.
        
        Args:
            documents: List of processed documents
            config: ChunkingConfig object
            
        Returns:
            List of validation error messages
        """
        errors = []
        
        if not documents:
            errors.append("No documents provided for chunking")
            return errors
        
        # Check document structure
        for i, doc in enumerate(documents):
            if not hasattr(doc, 'content'):
                errors.append(f"Document {i} missing 'content' attribute")
            elif not isinstance(doc.content, str):
                errors.append(f"Document {i} content is not a string")
            elif not doc.content.strip():
                errors.append(f"Document {i} has empty content")
            
            if not hasattr(doc, 'id'):
                errors.append(f"Document {i} missing 'id' attribute")
        
        # Validate chunking config
        if config.chunk_size <= 0:
            errors.append("chunk_size must be positive")
        
        if config.chunk_overlap < 0:
            errors.append("chunk_overlap cannot be negative")
        
        if config.chunk_overlap >= config.chunk_size:
            errors.append("chunk_overlap must be less than chunk_size")
        
        # Validate chunker type
        valid_chunkers = ["core", "cpu", "text", "code", "chonky"]
        if config.chunker_type not in valid_chunkers:
            errors.append(f"Invalid chunker type: {config.chunker_type}. "
                         f"Valid options: {valid_chunkers}")
        
        # Validate strategy
        valid_strategies = ["semantic", "sliding_window", "paragraph", "sentence"]
        if config.strategy not in valid_strategies:
            errors.append(f"Invalid chunking strategy: {config.strategy}. "
                         f"Valid options: {valid_strategies}")
        
        return errors
    
    def get_expected_outputs(self) -> List[str]:
        """Get list of expected output keys."""
        return ["chunks", "stats"]
    
    def estimate_duration(self, input_size: int) -> float:
        """
        Estimate stage duration based on input size.
        
        Args:
            input_size: Number of documents
            
        Returns:
            Estimated duration in seconds
        """
        # Rough estimate: 1 second per document + base overhead
        return max(30, input_size * 1)
    
    def get_resource_requirements(self, input_size: int) -> Dict[str, Any]:
        """
        Get estimated resource requirements.
        
        Args:
            input_size: Number of documents
            
        Returns:
            Dictionary with resource estimates
        """
        return {
            "memory_mb": max(200, input_size * 5),  # 5MB per document estimate
            "cpu_cores": 1,
            "disk_mb": input_size * 2,  # 2MB per document for chunks
            "network_required": False
        }