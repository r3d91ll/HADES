"""
Embedding stage for the unified HADES pipeline.

This module provides a concrete implementation of the PipelineStage
abstract base class for generating vector embeddings for document chunks.

This is the consolidated version that integrates with the orchestration system.
"""

import logging
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np

from .base import PipelineStage, PipelineStageError
from src.orchestration.pipelines.schema import DocumentSchema, ChunkSchema, ValidationResult, ValidationIssue, ValidationSeverity

logger = logging.getLogger(__name__)

# Try to import embedding components
try:
    from src.embedding.registry import get_adapter_by_name
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False
    logger.warning("Embedding module not available")


class EmbeddingStage(PipelineStage):
    """Pipeline stage for generating embeddings in the unified system.
    
    This stage handles the generation of vector embeddings for document chunks,
    using the specified embedding model from the embedding registry.
    
    Supports both sequential and parallel processing modes.
    """
    
    def __init__(
        self,
        name: str = "embedding",
        config: Optional[Dict[str, Any]] = None,
        enable_parallel: bool = False,
        worker_pool: Optional[Any] = None,
        queue_manager: Optional[Any] = None
    ):
        """Initialize embedding stage.
        
        Args:
            name: Name of the stage
            config: Configuration dictionary with embedding options
            enable_parallel: Whether to enable parallel processing
            worker_pool: Worker pool for parallel execution
            queue_manager: Queue manager for task distribution
        """
        super().__init__(name, config or {}, enable_parallel, worker_pool, queue_manager)
        
        # Configure embedding parameters
        self.adapter_name = self.config.get("adapter_name", "cpu")
        self.batch_size = self.config.get("batch_size", 32)
        self.normalize_embeddings = self.config.get("normalize_embeddings", True)
        
        # Load the embedding adapter
        self.embedding_adapter: Optional[Any] = None
        if EMBEDDING_AVAILABLE:
            try:
                self.embedding_adapter = get_adapter_by_name(self.adapter_name)()
                # Try to get embedding dimension if available
                if hasattr(self.embedding_adapter, 'embedding_dimension'):
                    self.embedding_dim = self.embedding_adapter.embedding_dimension
                else:
                    self.embedding_dim = self.config.get("embedding_dim", 768)  # Default dimension
                
                self.logger.info(f"Loaded embedding adapter: {self.adapter_name} with dimension {self.embedding_dim}")
            except Exception as e:
                self.logger.warning(f"Failed to load embedding adapter '{self.adapter_name}': {str(e)}")
                self.embedding_adapter = None
                self.embedding_dim = self.config.get("embedding_dim", 768)
        else:
            self.embedding_dim = self.config.get("embedding_dim", 768)
    
    def run(self, input_data: List[DocumentSchema]) -> List[DocumentSchema]:
        """Generate embeddings for document chunks.
        
        This method processes each document's chunks to generate vector
        embeddings using the configured embedding adapter.
        
        Args:
            input_data: List of DocumentSchema objects with chunks
            
        Returns:
            List of DocumentSchema objects with embeddings added to chunks
            
        Raises:
            PipelineStageError: If embedding generation fails
        """
        if not EMBEDDING_AVAILABLE or not self.embedding_adapter:
            raise PipelineStageError(
                self.name,
                "Embedding module not available"
            )
        
        if not input_data:
            raise PipelineStageError(
                self.name,
                "No documents provided for embedding generation"
            )
        
        embedded_documents = []
        total_chunks = 0
        
        for document in input_data:
            try:
                embedded_doc = self._process_single_document(document)
                if embedded_doc:
                    embedded_documents.append(embedded_doc)
                    total_chunks += len(embedded_doc.chunks) if embedded_doc.chunks else 0
                    
            except Exception as e:
                self.logger.error(f"Error generating embeddings for document {document.file_name}: {str(e)}", exc_info=True)
                # Continue processing other documents
        
        self.logger.info(f"Generated embeddings for {total_chunks} chunks across {len(embedded_documents)} documents")
        return embedded_documents
    
    def _process_single_document(self, document: DocumentSchema) -> Optional[DocumentSchema]:
        """Process a single document for embedding generation.
        
        Args:
            document: Document schema object with chunks
            
        Returns:
            DocumentSchema with embeddings or None if processing failed
        """
        if not document.chunks:
            self.logger.warning(f"Document has no chunks: {document.file_name}")
            return None
        
        self.logger.info(f"Generating embeddings for document: {document.file_name} ({len(document.chunks)} chunks)")
        
        try:
            # Create batches of chunks for efficient processing
            chunk_batches = self._create_batches(document.chunks, self.batch_size)
            
            # Process each batch
            updated_chunks = []
            
            for batch in chunk_batches:
                # Extract texts for embedding
                texts = [chunk.text for chunk in batch]
                
                # Generate embeddings using the adapter
                try:
                    if self.embedding_adapter is None:
                        raise RuntimeError("Embedding adapter not initialized")
                    embeddings = self.embedding_adapter.embed_batch(texts)
                    
                    # Normalize if configured
                    if self.normalize_embeddings:
                        embeddings = self._normalize_embeddings(embeddings)
                    
                    # Update chunks with embeddings
                    for i, chunk in enumerate(batch):
                        chunk_copy = chunk.copy(deep=True)
                        chunk_copy.embedding = embeddings[i].tolist() if isinstance(embeddings[i], np.ndarray) else embeddings[i]
                        updated_chunks.append(chunk_copy)
                        
                except Exception as e:
                    self.logger.error(f"Failed to generate embeddings for batch: {str(e)}")
                    # Add chunks without embeddings to maintain document structure
                    for chunk in batch:
                        chunk_copy = chunk.copy(deep=True)
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
            
            return document_copy
            
        except Exception as e:
            self.logger.error(f"Failed to process document {document.file_name}: {str(e)}")
            return None
    
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
    
    def _normalize_embeddings(self, embeddings: Union[List[List[float]], np.ndarray]) -> np.ndarray:
        """Normalize embeddings to unit length.
        
        Args:
            embeddings: List or array of embedding vectors
            
        Returns:
            Normalized embedding vectors as numpy array
        """
        # Convert to numpy array for efficient normalization
        if not isinstance(embeddings, np.ndarray):
            embeddings_array = np.array(embeddings)
        else:
            embeddings_array = embeddings
        
        # Calculate L2 norm (Euclidean norm) for each embedding
        norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        
        # Avoid division by zero
        norms[norms == 0] = 1.0
        
        # Normalize
        normalized_embeddings = embeddings_array / norms
        
        return np.asarray(normalized_embeddings)
    
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
                    severity=ValidationSeverity.WARNING,
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
            try:
                if any(not np.isfinite(v) for v in chunk.embedding):
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message=f"Chunk {i} has NaN or infinite values in embedding",
                        location=f"{document.file_path}:chunk_{i}"
                    ))
            except (TypeError, ValueError):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message=f"Chunk {i} has invalid embedding data type",
                    location=f"{document.file_path}:chunk_{i}"
                ))
                continue
            
            # Check for zero vectors
            try:
                if all(abs(v) < 1e-6 for v in chunk.embedding):
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Chunk {i} has a zero or near-zero embedding vector",
                        location=f"{document.file_path}:chunk_{i}"
                    ))
            except (TypeError, ValueError):
                pass  # Already handled above
            
            # Check for proper normalization if enabled
            if self.normalize_embeddings:
                try:
                    norm = np.linalg.norm(chunk.embedding)
                    if abs(norm - 1.0) > 1e-3:
                        issues.append(ValidationIssue(
                            severity=ValidationSeverity.INFO,
                            message=f"Chunk {i} embedding is not properly normalized",
                            location=f"{document.file_path}:chunk_{i}",
                            context={"norm": norm}
                        ))
                except (TypeError, ValueError):
                    pass  # Already handled above
        
        # Return validation result
        return ValidationResult(
            is_valid=not any(issue.severity == ValidationSeverity.ERROR for issue in issues),
            issues=issues,
            stage_name=self.name
        )