"""
Chunking stage for the unified HADES-PathRAG pipeline.

This module provides a concrete implementation of the PipelineStage
abstract base class for document chunking operations, including
segmentation of documents into semantically meaningful chunks.

This is the consolidated version that integrates with the orchestration system.
"""

import logging
from typing import Dict, Any, List, Tuple, Optional, Union
import uuid

from .base import PipelineStage, PipelineStageError
from ..schema import DocumentSchema, ChunkSchema, ValidationResult, ValidationIssue, ValidationSeverity

logger = logging.getLogger(__name__)

# Try to import chunking components
try:
    from src.chunking.registry import get_chunker
    CHUNKING_AVAILABLE = True
except ImportError:
    CHUNKING_AVAILABLE = False
    logger.warning("Chunking module not available")


class ChunkingStage(PipelineStage):
    """Pipeline stage for document chunking in the unified system.
    
    This stage handles the segmentation of documents into semantically
    meaningful chunks, preparing them for embedding generation.
    
    Supports both sequential and parallel processing modes, and integrates
    with the centralized chunking system.
    """
    
    def __init__(
        self,
        name: str = "chunking",
        config: Optional[Dict[str, Any]] = None,
        enable_parallel: bool = False,
        worker_pool=None,
        queue_manager=None
    ):
        """Initialize chunking stage.
        
        Args:
            name: Name of the stage
            config: Configuration dictionary with chunking options
            enable_parallel: Whether to enable parallel processing
            worker_pool: Worker pool for parallel execution
            queue_manager: Queue manager for task distribution
        """
        super().__init__(name, config or {}, enable_parallel, worker_pool, queue_manager)
        
        # Configure chunking parameters
        self.chunk_size = self.config.get("chunk_size", 500)
        self.chunk_overlap = self.config.get("chunk_overlap", 50)
        self.chunking_strategy = self.config.get("chunking_strategy", "paragraph")
        self.min_chunk_size = self.config.get("min_chunk_size", 50)
        self.max_chunk_size = self.config.get("max_chunk_size", 1000)
        self.chunker_type = self.config.get("chunker_type", "cpu")  # cpu, chonky, etc.
        
        # Initialize chunker if available
        if CHUNKING_AVAILABLE:
            try:
                self.chunker = get_chunker(self.chunker_type, self.config)
            except Exception as e:
                self.logger.warning(f"Could not initialize chunker {self.chunker_type}: {str(e)}")
                self.chunker = None
        else:
            self.chunker = None
    
    def run(self, input_data: List[DocumentSchema]) -> List[DocumentSchema]:
        """Process documents by chunking them into semantic segments.
        
        This method divides each document into chunks according to the configured
        chunking strategy, size, and overlap parameters.
        
        Args:
            input_data: List of DocumentSchema objects to chunk
            
        Returns:
            List of DocumentSchema objects with added chunks
            
        Raises:
            PipelineStageError: If chunking fails
        """
        if not input_data:
            raise PipelineStageError(
                self.name,
                "No documents provided for chunking"
            )
        
        chunked_documents = []
        for document in input_data:
            try:
                chunked_doc = self._process_single_document(document)
                if chunked_doc:
                    chunked_documents.append(chunked_doc)
                    
            except Exception as e:
                self.logger.error(f"Error chunking document {document.file_name}: {str(e)}", exc_info=True)
                # Continue processing other documents
        
        self.logger.info(f"Chunked {len(chunked_documents)} documents successfully")
        return chunked_documents
    
    def _process_single_document(self, document: DocumentSchema) -> Optional[DocumentSchema]:
        """Process a single document for chunking.
        
        Args:
            document: Document schema object to chunk
            
        Returns:
            DocumentSchema with chunks or None if processing failed
        """
        self.logger.info(f"Chunking document: {document.file_name}")
        
        # Get document text from appropriate location
        document_text = self._extract_document_text(document)
        
        if not document_text:
            self.logger.warning(f"Document has no text content: {document.file_name}")
            return None
        
        # Chunk the document based on the selected strategy
        if self.chunker and CHUNKING_AVAILABLE:
            # Use the centralized chunking system
            chunks = self._chunk_with_registry(document_text, document)
        else:
            # Fall back to local chunking implementation
            chunks = self._chunk_document_local(document_text, document)
        
        # Add chunks to the document
        document_copy = document.copy(deep=True)
        document_copy.chunks = chunks
        
        # Validate chunks
        validation_result = self._validate_chunks(document_copy)
        if validation_result.has_errors:
            self.logger.warning(
                f"Chunk validation failed for {document.file_name}: "
                f"{len(validation_result.get_issues_by_severity(ValidationSeverity.ERROR))} errors"
            )
        
        self.logger.info(f"Created {len(chunks)} chunks for document: {document.file_name}")
        return document_copy
    
    def _chunk_with_registry(self, text: str, document: DocumentSchema) -> List[ChunkSchema]:
        """Chunk document using the centralized chunking registry.
        
        Args:
            text: Document text content
            document: Document schema object for context
            
        Returns:
            List of ChunkSchema objects
        """
        try:
            # Prepare document for chunking
            chunk_input = {
                "text": text,
                "document_id": document.file_id,
                "file_path": document.file_path,
                "metadata": document.metadata
            }
            
            # Chunk using the registered chunker
            chunk_results = self.chunker.chunk(chunk_input)
            
            # Convert to ChunkSchema objects
            chunks = []
            for i, chunk_result in enumerate(chunk_results):
                chunk = ChunkSchema(
                    id=f"{document.file_id}_chunk_{i}",
                    text=chunk_result.get("text", ""),
                    metadata={
                        "index": i,
                        "document_id": document.file_id,
                        "document_name": document.file_name,
                        "chunking_strategy": self.chunking_strategy,
                        "chunker_type": self.chunker_type,
                        **chunk_result.get("metadata", {})
                    }
                )
                chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            self.logger.warning(f"Registry chunking failed, falling back to local: {str(e)}")
            return self._chunk_document_local(text, document)
    
    def validate(self, data: Union[List[DocumentSchema], Any]) -> Tuple[bool, List[str]]:
        """Validate input or output data for this stage.
        
        Args:
            data: Data to validate (either input or output DocumentSchema objects)
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        # Validate that input is a list of DocumentSchema objects
        if not isinstance(data, list):
            return False, ["Input must be a list of DocumentSchema objects"]
        
        if not all(isinstance(doc, DocumentSchema) for doc in data):
            return False, ["All items in the list must be DocumentSchema objects"]
        
        # For output validation, check that all documents have chunks
        output_validation = all(hasattr(doc, 'chunks') and len(doc.chunks) > 0 for doc in data)
        
        if not output_validation:
            return False, ["One or more documents have no chunks after processing"]
        
        return True, []
    
    def _extract_document_text(self, document: DocumentSchema) -> str:
        """Extract text content from a document.
        
        Args:
            document: Document schema object
            
        Returns:
            Document text content
        """
        # Check various possible locations for the document content
        if "content" in document.metadata:
            return document.metadata["content"]
        elif "text" in document.metadata:
            return document.metadata["text"]
        
        # If not in metadata, check if we need to load from file
        try:
            with open(document.file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            self.logger.warning(f"Could not read document content from file: {str(e)}")
            
        return ""
    
    def _chunk_document_local(self, text: str, document: DocumentSchema) -> List[ChunkSchema]:
        """Local fallback chunking implementation.
        
        Args:
            text: Document text content
            document: Document schema object for context
            
        Returns:
            List of ChunkSchema objects
        """
        chunks = []
        
        # Select chunking strategy based on configuration
        if self.chunking_strategy == "paragraph":
            text_chunks = self._chunk_by_paragraph(text)
        elif self.chunking_strategy == "fixed_size":
            text_chunks = self._chunk_by_fixed_size(text)
        elif self.chunking_strategy == "sentence":
            text_chunks = self._chunk_by_sentence(text)
        else:
            # Default to paragraph chunking
            text_chunks = self._chunk_by_paragraph(text)
        
        # Create chunk schema objects
        for i, chunk_text in enumerate(text_chunks):
            chunk = ChunkSchema(
                id=f"{document.file_id}_chunk_{i}",
                text=chunk_text,
                metadata={
                    "index": i,
                    "document_id": document.file_id,
                    "document_name": document.file_name,
                    "chunking_strategy": self.chunking_strategy,
                    "chunker_type": "local_fallback"
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_by_paragraph(self, text: str) -> List[str]:
        """Chunk text by paragraphs."""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) > self.max_chunk_size and current_chunk:
                chunks.append(current_chunk)
                current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return self._combine_short_chunks(chunks)
    
    def _chunk_by_fixed_size(self, text: str) -> List[str]:
        """Chunk text by fixed size with overlap."""
        chunks = []
        text_length = len(text)
        
        start = 0
        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            
            if end < text_length:
                search_start = max(start, end - int(self.chunk_size * 0.2))
                last_space = text.rfind(' ', search_start, end)
                
                if last_space != -1:
                    end = last_space + 1
            
            chunks.append(text[start:end].strip())
            start = min(end, start + self.chunk_size - self.chunk_overlap)
        
        return chunks
    
    def _chunk_by_sentence(self, text: str) -> List[str]:
        """Chunk text by sentences, combining into appropriate sized chunks."""
        sentences = []
        for paragraph in text.split("\n\n"):
            for sentence in paragraph.replace(". ", ".|").replace("! ", "!|").replace("? ", "?|").split("|"):
                if sentence.strip():
                    sentences.append(sentence.strip())
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > self.max_chunk_size and current_chunk:
                chunks.append(current_chunk)
                current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _combine_short_chunks(self, chunks: List[str]) -> List[str]:
        """Combine short chunks to avoid very small chunks."""
        if not chunks:
            return []
        
        combined_chunks = []
        current_chunk = chunks[0]
        
        for i in range(1, len(chunks)):
            if len(current_chunk) < self.min_chunk_size and len(current_chunk) + len(chunks[i]) <= self.max_chunk_size:
                current_chunk += "\n\n" + chunks[i]
            elif len(current_chunk) + len(chunks[i]) <= self.max_chunk_size:
                current_chunk += "\n\n" + chunks[i]
            else:
                combined_chunks.append(current_chunk)
                current_chunk = chunks[i]
        
        if current_chunk:
            combined_chunks.append(current_chunk)
        
        return combined_chunks
    
    def _validate_chunks(self, document: DocumentSchema) -> ValidationResult:
        """Validate chunks for a document.
        
        Args:
            document: Document with chunks to validate
            
        Returns:
            ValidationResult object
        """
        issues = []
        
        # Check if any chunks were created
        if not document.chunks:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message="No chunks were created for document",
                location=document.file_path
            ))
            return ValidationResult(
                is_valid=False,
                issues=issues,
                stage_name=self.name
            )
        
        # Check for very small or large chunks
        for i, chunk in enumerate(document.chunks):
            if len(chunk.text) < self.min_chunk_size:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"Chunk {i} is smaller than the minimum size",
                    location=f"{document.file_path}:chunk_{i}",
                    context={"chunk_size": len(chunk.text), "min_size": self.min_chunk_size}
                ))
            
            if len(chunk.text) > self.max_chunk_size:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"Chunk {i} exceeds the maximum size",
                    location=f"{document.file_path}:chunk_{i}",
                    context={"chunk_size": len(chunk.text), "max_size": self.max_chunk_size}
                ))
        
        # Return validation result
        return ValidationResult(
            is_valid=not any(issue.severity == ValidationSeverity.ERROR for issue in issues),
            issues=issues,
            stage_name=self.name
        )