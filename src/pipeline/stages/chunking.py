"""
Chunking stage for the HADES-PathRAG pipeline.

This module provides a concrete implementation of the PipelineStage
abstract base class for document chunking operations, including
segmentation of documents into semantically meaningful chunks.
"""

import logging
from typing import Dict, Any, List, Tuple, Optional, Union
import uuid

from src.pipeline.stages import PipelineStage, PipelineStageError
from src.pipeline.schema import DocumentSchema, ChunkSchema, ValidationResult, ValidationIssue, ValidationSeverity

logger = logging.getLogger(__name__)


class ChunkingStage(PipelineStage):
    """Pipeline stage for document chunking.
    
    This stage handles the segmentation of documents into semantically
    meaningful chunks, preparing them for embedding generation.
    """
    
    def __init__(
        self,
        name: str = "chunking",
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize chunking stage.
        
        Args:
            name: Name of the stage
            config: Configuration dictionary with chunking options
        """
        super().__init__(name, config or {})
        
        # Configure chunking parameters
        self.chunk_size = self.config.get("chunk_size", 500)
        self.chunk_overlap = self.config.get("chunk_overlap", 50)
        self.chunking_strategy = self.config.get("chunking_strategy", "paragraph")
        self.min_chunk_size = self.config.get("min_chunk_size", 50)
        self.max_chunk_size = self.config.get("max_chunk_size", 1000)
    
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
                self.logger.info(f"Chunking document: {document.file_name}")
                
                # Get document text from appropriate location
                document_text = self._extract_document_text(document)
                
                if not document_text:
                    self.logger.warning(f"Document has no text content: {document.file_name}")
                    continue
                
                # Chunk the document based on the selected strategy
                chunks = self._chunk_document(document_text, document)
                
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
                
                chunked_documents.append(document_copy)
                self.logger.info(f"Created {len(chunks)} chunks for document: {document.file_name}")
                
            except Exception as e:
                self.logger.error(f"Error chunking document {document.file_name}: {str(e)}", exc_info=True)
                # Continue processing other documents
        
        self.logger.info(f"Chunked {len(chunked_documents)} documents successfully")
        return chunked_documents
    
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
        # For now, we assume the document content is in the metadata
        # In a real implementation, this would extract from appropriate location
        if "text" in document.metadata:
            return document.metadata["text"]
        
        # If not in metadata, check if we need to load from file
        try:
            with open(document.file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            self.logger.warning(f"Could not read document content from file: {str(e)}")
            
        return ""
    
    def _chunk_document(self, text: str, document: DocumentSchema) -> List[ChunkSchema]:
        """Chunk document text into semantic segments.
        
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
                    "chunking_strategy": self.chunking_strategy
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_by_paragraph(self, text: str) -> List[str]:
        """Chunk text by paragraphs.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        # Split by double newlines (paragraphs)
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        
        # Process each paragraph
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed max chunk size, 
            # store the current chunk and start a new one
            if len(current_chunk) + len(paragraph) > self.max_chunk_size and current_chunk:
                chunks.append(current_chunk)
                current_chunk = paragraph
            # Otherwise, add the paragraph to the current chunk
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(current_chunk)
        
        # Handle very short chunks by combining them
        return self._combine_short_chunks(chunks)
    
    def _chunk_by_fixed_size(self, text: str) -> List[str]:
        """Chunk text by fixed size with overlap.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        chunks = []
        text_length = len(text)
        
        # Use a sliding window approach with the configured chunk size and overlap
        start = 0
        while start < text_length:
            # Calculate end position (not exceeding text length)
            end = min(start + self.chunk_size, text_length)
            
            # Try to find a good break point (space or punctuation) near the end
            if end < text_length:
                # Look for a space within the last 20% of the chunk
                search_start = max(start, end - int(self.chunk_size * 0.2))
                last_space = text.rfind(' ', search_start, end)
                
                if last_space != -1:
                    end = last_space + 1  # Include the space
            
            # Add the chunk
            chunks.append(text[start:end].strip())
            
            # Move the start position (accounting for overlap)
            start = min(end, start + self.chunk_size - self.chunk_overlap)
        
        return chunks
    
    def _chunk_by_sentence(self, text: str) -> List[str]:
        """Chunk text by sentences, combining into appropriate sized chunks.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        # Simple sentence splitting (this could be more sophisticated)
        sentences = []
        for paragraph in text.split("\n\n"):
            # Split by common sentence terminators
            for sentence in paragraph.replace(". ", ".|").replace("! ", "!|").replace("? ", "?|").split("|"):
                if sentence.strip():
                    sentences.append(sentence.strip())
        
        # Combine sentences into chunks
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence would exceed max chunk size, 
            # store the current chunk and start a new one
            if len(current_chunk) + len(sentence) > self.max_chunk_size and current_chunk:
                chunks.append(current_chunk)
                current_chunk = sentence
            # Otherwise, add the sentence to the current chunk
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _combine_short_chunks(self, chunks: List[str]) -> List[str]:
        """Combine short chunks to avoid very small chunks.
        
        Args:
            chunks: List of text chunks
            
        Returns:
            List of combined chunks
        """
        if not chunks:
            return []
        
        combined_chunks = []
        current_chunk = chunks[0]
        
        for i in range(1, len(chunks)):
            # If the current chunk is too short and adding the next chunk wouldn't make it too long
            if len(current_chunk) < self.min_chunk_size and len(current_chunk) + len(chunks[i]) <= self.max_chunk_size:
                current_chunk += "\n\n" + chunks[i]
            # If adding the next chunk would still be under the max size
            elif len(current_chunk) + len(chunks[i]) <= self.max_chunk_size:
                current_chunk += "\n\n" + chunks[i]
            else:
                # Add the current chunk and start a new one
                combined_chunks.append(current_chunk)
                current_chunk = chunks[i]
        
        # Add the last chunk
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
        
        # Check for very small chunks
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
