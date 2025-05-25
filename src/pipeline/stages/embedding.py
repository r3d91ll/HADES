"""
Embedding stage for the HADES-PathRAG pipeline.

This module provides a concrete implementation of the PipelineStage
abstract base class for generating vector embeddings for document chunks.
"""

import logging
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np

from src.pipeline.stages import PipelineStage, PipelineStageError
from src.pipeline.schema import DocumentSchema, ChunkSchema, ValidationResult, ValidationIssue, ValidationSeverity
from src.embedding.registry import get_embedding_model

logger = logging.getLogger(__name__)


class EmbeddingStage(PipelineStage):
    """Pipeline stage for generating embeddings.
    
    This stage handles the generation of vector embeddings for document chunks,
    using the specified embedding model from the embedding registry.
    """
    
    def __init__(
        self,
        name: str = "embedding",
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize embedding stage.
        
        Args:
            name: Name of the stage
            config: Configuration dictionary with embedding options
        """
        super().__init__(name, config or {})
        
        # Configure embedding parameters
        self.model_name = self.config.get("model_name", "default")
        self.batch_size = self.config.get("batch_size", 32)
        self.normalize_embeddings = self.config.get("normalize_embeddings", True)
        
        # Load the embedding model
        try:
            self.embedding_model = get_embedding_model(self.model_name)
            self.embedding_dim = self.embedding_model.embedding_dimension
            self.logger.info(f"Loaded embedding model: {self.model_name} with dimension {self.embedding_dim}")
        except Exception as e:
            raise PipelineStageError(
                self.name,
                f"Failed to load embedding model '{self.model_name}': {str(e)}",
                original_error=e
            )
    
    def run(self, input_data: List[DocumentSchema]) -> List[DocumentSchema]:
        """Generate embeddings for document chunks.
        
        This method processes each document's chunks to generate vector
        embeddings using the configured embedding model.
        
        Args:
            input_data: List of DocumentSchema objects with chunks
            
        Returns:
            List of DocumentSchema objects with embeddings added to chunks
            
        Raises:
            PipelineStageError: If embedding generation fails
        """
        if not input_data:
            raise PipelineStageError(
                self.name,
                "No documents provided for embedding generation"
            )
        
        embedded_documents = []
        total_chunks = 0
        
        for document in input_data:
            try:
                if not document.chunks:
                    self.logger.warning(f"Document has no chunks: {document.file_name}")
                    continue
                
                self.logger.info(f"Generating embeddings for document: {document.file_name} ({len(document.chunks)} chunks)")
                
                # Create batches of chunks for efficient processing
                chunk_batches = self._create_batches(document.chunks, self.batch_size)
                
                # Process each batch
                updated_chunks = []
                
                for batch in chunk_batches:
                    # Extract texts for embedding
                    texts = [chunk.text for chunk in batch]
                    
                    # Generate embeddings
                    embeddings = self.embedding_model.embed_texts(texts)
                    
                    # Normalize if configured
                    if self.normalize_embeddings:
                        embeddings = self._normalize_embeddings(embeddings)
                    
                    # Update chunks with embeddings
                    for i, chunk in enumerate(batch):
                        chunk_copy = chunk.copy(deep=True)
                        chunk_copy.embedding = embeddings[i].tolist()
                        updated_chunks.append(chunk_copy)
                
                # Create updated document with embedded chunks
                document_copy = document.copy(deep=True)
                document_copy.chunks = updated_chunks
                
                # Validate embeddings
                validation_result = self._validate_embeddings(document_copy)
                if validation_result.has_errors:
                    self.logger.warning(
                        f"Embedding validation failed for {document.file_name}: "
                        f"{len(validation_result.get_issues_by_severity(ValidationSeverity.ERROR))} errors"
                    )
                
                embedded_documents.append(document_copy)
                total_chunks += len(updated_chunks)
                
            except Exception as e:
                self.logger.error(f"Error generating embeddings for document {document.file_name}: {str(e)}", exc_info=True)
                # Continue processing other documents
        
        self.logger.info(f"Generated embeddings for {total_chunks} chunks across {len(embedded_documents)} documents")
        return embedded_documents
    
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
        
        # Validate that documents have chunks
        if not all(hasattr(doc, 'chunks') and doc.chunks for doc in data):
            return False, ["One or more documents have no chunks"]
        
        # For output validation, check that all chunks have embeddings
        output_validation = all(
            all(chunk.embedding is not None for chunk in doc.chunks)
            for doc in data if doc.chunks
        )
        
        if not output_validation:
            return False, ["One or more chunks have no embeddings after processing"]
        
        return True, []
    
    def _create_batches(self, items: List, batch_size: int) -> List[List]:
        """Create batches from a list of items.
        
        Args:
            items: List of items to batch
            batch_size: Size of each batch
            
        Returns:
            List of batches
        """
        return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
    
    def _normalize_embeddings(self, embeddings: List[List[float]]) -> List[List[float]]:
        """Normalize embeddings to unit length.
        
        Args:
            embeddings: List of embedding vectors
            
        Returns:
            Normalized embedding vectors
        """
        # Convert to numpy array for efficient normalization
        embeddings_array = np.array(embeddings)
        
        # Calculate L2 norm (Euclidean norm) for each embedding
        norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        
        # Avoid division by zero
        norms[norms == 0] = 1.0
        
        # Normalize
        normalized_embeddings = embeddings_array / norms
        
        return normalized_embeddings
    
    def _validate_embeddings(self, document: DocumentSchema) -> ValidationResult:
        """Validate embeddings for a document's chunks.
        
        Args:
            document: Document with embedded chunks to validate
            
        Returns:
            ValidationResult object
        """
        issues = []
        
        # Check if any chunks were embedded
        if not document.chunks:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message="No chunks available for embedding validation",
                location=document.file_path
            ))
            return ValidationResult(
                is_valid=False,
                issues=issues,
                stage_name=self.name
            )
        
        # Check for missing embeddings
        for i, chunk in enumerate(document.chunks):
            if chunk.embedding is None:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message=f"Chunk {i} has no embedding",
                    location=f"{document.file_path}:chunk_{i}"
                ))
                continue
            
            # Check embedding dimension
            if len(chunk.embedding) != self.embedding_dim:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message=f"Chunk {i} has incorrect embedding dimension",
                    location=f"{document.file_path}:chunk_{i}",
                    context={
                        "expected_dim": self.embedding_dim,
                        "actual_dim": len(chunk.embedding)
                    }
                ))
            
            # Check for NaN or infinite values
            if any(not np.isfinite(v) for v in chunk.embedding):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message=f"Chunk {i} has NaN or infinite values in embedding",
                    location=f"{document.file_path}:chunk_{i}"
                ))
            
            # Check for zero vectors
            if all(abs(v) < 1e-6 for v in chunk.embedding):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"Chunk {i} has a zero or near-zero embedding vector",
                    location=f"{document.file_path}:chunk_{i}"
                ))
            
            # Check for proper normalization if enabled
            if self.normalize_embeddings:
                norm = np.linalg.norm(chunk.embedding)
                if abs(norm - 1.0) > 1e-3:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Chunk {i} embedding is not properly normalized",
                        location=f"{document.file_path}:chunk_{i}",
                        context={"norm": norm}
                    ))
        
        # Return validation result
        return ValidationResult(
            is_valid=not any(issue.severity == ValidationSeverity.ERROR for issue in issues),
            issues=issues,
            stage_name=self.name
        )
