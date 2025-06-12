"""
ISNE (Inductive Shallow Node Embedding) stage for the unified HADES pipeline.

This module provides a concrete implementation of the PipelineStage
abstract base class for enhancing embeddings using graph-based relationships.

This is the consolidated version that integrates with the orchestration system.
"""

import logging
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
import uuid
from pathlib import Path
from datetime import datetime

from .base import PipelineStage, PipelineStageError
from src.orchestration.pipelines.schema import DocumentSchema, ChunkSchema, Relationship, ValidationResult, ValidationIssue, ValidationSeverity

logger = logging.getLogger(__name__)

# Try to import ISNE components
try:
    from src.isne.pipeline.isne_pipeline import ISNEPipeline
    ISNE_AVAILABLE = True
except ImportError:
    ISNE_AVAILABLE = False
    logger.warning("ISNE module not available")

# Try to import validation components
try:
    from src.validation.embedding_validator import validate_embeddings_after_isne
    VALIDATION_AVAILABLE = True
except ImportError:
    VALIDATION_AVAILABLE = False
    logger.warning("Validation module not available")


class ISNEStage(PipelineStage):
    """Pipeline stage for ISNE processing in the unified system.
    
    This stage handles the enhancement of embeddings using graph-based 
    relationships through the ISNE (Inductive Shallow Node Embedding) model.
    It builds a graph from document chunks, identifies relationships,
    and generates improved embeddings.
    
    Supports both sequential and parallel processing modes.
    """
    
    def __init__(
        self,
        name: str = "isne",
        config: Optional[Dict[str, Any]] = None,
        enable_parallel: bool = False,
        worker_pool: Optional[Any] = None,
        queue_manager: Optional[Any] = None
    ):
        """Initialize ISNE stage.
        
        Args:
            name: Name of the stage
            config: Configuration dictionary with ISNE options
            enable_parallel: Whether to enable parallel processing
            worker_pool: Worker pool for parallel execution
            queue_manager: Queue manager for task distribution
        """
        super().__init__(name, config or {}, enable_parallel, worker_pool, queue_manager)
        
        # Configure ISNE parameters
        self.model_path = self.config.get("model_path")
        self.training_config = self.config.get("training_config", {})
        self.relationship_threshold = self.config.get("relationship_threshold", 0.7)
        self.max_relationships = self.config.get("max_relationships", 10)
        self.generate_relationships = self.config.get("generate_relationships", True)
        self.validate_results = self.config.get("validate_results", True)
        
        # Initialize ISNE pipeline
        self.isne_pipeline: Optional[ISNEPipeline] = None
        if ISNE_AVAILABLE:
            try:
                self.isne_pipeline = ISNEPipeline(
                    model_path=self.model_path,
                    validate=self.validate_results,
                    alert_threshold=self.config.get("alert_threshold", "medium"),
                    device=self.config.get("device"),
                    alert_dir=self.config.get("alert_dir", "./alerts")
                )
                self.logger.info("Initialized ISNE pipeline")
            except Exception as e:
                self.logger.warning(f"Failed to initialize ISNE pipeline: {str(e)}")
                self.isne_pipeline = None
    
    def run(self, input_data: List[DocumentSchema]) -> List[DocumentSchema]:
        """Enhance embeddings using ISNE.
        
        This method processes document chunks to identify relationships,
        build a graph, and generate improved embeddings using the ISNE model.
        
        Args:
            input_data: List of DocumentSchema objects with embedded chunks
            
        Returns:
            List of DocumentSchema objects with ISNE-enhanced embeddings
            
        Raises:
            PipelineStageError: If ISNE processing fails
        """
        if not ISNE_AVAILABLE or not self.isne_pipeline:
            raise PipelineStageError(
                self.name,
                "ISNE module not available"
            )
        
        if not input_data:
            raise PipelineStageError(
                self.name,
                "No documents provided for ISNE processing"
            )
        
        # Validate input embeddings
        all_chunks = []
        for document in input_data:
            if not document.chunks:
                self.logger.warning(f"Document has no chunks: {document.file_name}")
                continue
                
            if not all(chunk.embedding is not None for chunk in document.chunks):
                raise PipelineStageError(
                    self.name,
                    f"Document {document.file_name} has chunks without embeddings"
                )
            
            all_chunks.extend(document.chunks)
        
        self.logger.info(f"Processing {len(all_chunks)} chunks from {len(input_data)} documents")
        
        try:
            # Process documents with ISNE pipeline
            enhanced_documents = []
            
            for document in input_data:
                enhanced_doc = self._process_single_document(document)
                if enhanced_doc:
                    enhanced_documents.append(enhanced_doc)
            
            self.logger.info(f"ISNE processing completed for {len(enhanced_documents)} documents")
            return enhanced_documents
            
        except Exception as e:
            raise PipelineStageError(
                self.name,
                f"ISNE processing failed: {str(e)}",
                original_error=e
            )
    
    def _process_single_document(self, document: DocumentSchema) -> Optional[DocumentSchema]:
        """Process a single document for ISNE enhancement.
        
        Args:
            document: Document schema object with embedded chunks
            
        Returns:
            DocumentSchema with ISNE embeddings or None if processing failed
        """
        if not document.chunks:
            return None
        
        self.logger.info(f"ISNE processing document: {document.file_name} ({len(document.chunks)} chunks)")
        
        try:
            # Generate relationships if enabled
            if self.generate_relationships:
                self._generate_relationships_for_document(document)
            
            # Prepare document for ISNE processing
            document_data = self._prepare_document_for_isne(document)
            
            # Process with ISNE pipeline
            if self.isne_pipeline is not None:
                enhanced_docs, _ = self.isne_pipeline.process_documents([document_data], True, "./temp_isne_output")
                # Get the first document from the enhanced results
                enhanced_data = enhanced_docs[0] if enhanced_docs else {}
            else:
                raise RuntimeError("ISNE pipeline is not available")
            
            # Update document with ISNE embeddings
            document_copy = document.copy(deep=True)
            self._update_document_with_isne_embeddings(document_copy, enhanced_data)
            
            # Validate results
            validation_result = self._validate_isne_embeddings(document_copy)
            if validation_result.has_errors:
                self.logger.warning(
                    f"ISNE validation failed for {document.file_name}: "
                    f"{len(validation_result.get_issues_by_severity(ValidationSeverity.ERROR))} errors"
                )
            
            return document_copy
            
        except Exception as e:
            self.logger.error(f"Failed to process document {document.file_name}: {str(e)}")
            return document  # Return original document on failure
    
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
        
        # For input validation, check that all chunks have embeddings
        if not all(
            all(chunk.embedding is not None for chunk in doc.chunks)
            for doc in data if doc.chunks
        ):
            return False, ["One or more chunks have no embeddings"]
        
        # For output validation, check that all chunks have ISNE embeddings
        output_validation = all(
            all(chunk.isne_embedding is not None for chunk in doc.chunks)
            for doc in data if doc.chunks
        )
        
        if not output_validation:
            return False, ["One or more chunks have no ISNE embeddings after processing"]
        
        return True, []
    
    def _generate_relationships_for_document(self, document: DocumentSchema) -> None:
        """Generate relationships between chunks in a single document.
        
        Args:
            document: Document to process for relationships
        """
        if not document.chunks or len(document.chunks) < 2:
            return
        
        chunks = document.chunks
        
        # Create embedding matrix for efficient calculation
        embeddings = np.array([chunk.embedding for chunk in chunks])
        
        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # Avoid division by zero
        normalized_embeddings = embeddings / norms
        
        # Calculate pairwise cosine similarities
        similarities = np.dot(normalized_embeddings, normalized_embeddings.T)
        
        # Generate relationships based on similarities
        for i, chunk in enumerate(chunks):
            # Get top relationships excluding self
            similarities[i, i] = 0  # Exclude self-similarity
            
            # Get indices of top similar chunks  
            top_indices = np.argsort(similarities[i])[::-1][:self.max_relationships].tolist()
            
            # Filter by threshold
            top_indices = [idx for idx in top_indices if similarities[i, idx] >= self.relationship_threshold]
            
            # Create relationships
            for idx in top_indices:
                target_chunk = chunks[idx]
                relationship = Relationship(
                    source=chunk.id,
                    target=target_chunk.id,
                    type="similarity",
                    weight=float(similarities[i, idx]),
                    metadata={
                        "source_document": document.file_id,
                        "target_document": document.file_id,
                        "relationship_type": "intra_document"
                    }
                )
                
                # Add relationship to chunk (relationships is always a list according to schema)
                chunk.relationships.append(relationship)
    
    def _prepare_document_for_isne(self, document: DocumentSchema) -> Dict[str, Any]:
        """Prepare document in the format expected by ISNE pipeline.
        
        Args:
            document: Document to prepare
            
        Returns:
            Document dictionary in ISNE format
        """
        doc_dict = {
            "id": document.file_id,
            "file_path": document.file_path,
            "chunks": []
        }
        
        # Process chunks
        for chunk in document.chunks:
            chunk_dict = {
                "id": chunk.id,
                "text": chunk.text,
                "embedding": chunk.embedding,
                "relationships": []
            }
            
            # Process relationships
            for rel in chunk.relationships:
                rel_dict = {
                    "source_id": rel.source,
                    "target_id": rel.target,
                    "weight": rel.weight,
                    "type": rel.type
                }
                chunk_dict["relationships"].append(rel_dict)  # type: ignore[attr-defined]
            
            doc_dict["chunks"].append(chunk_dict)
        
        return doc_dict
    
    def _update_document_with_isne_embeddings(self, document: DocumentSchema, enhanced_data: Dict[str, Any]) -> None:
        """Update document chunks with ISNE embeddings.
        
        Args:
            document: Document to update
            enhanced_data: Enhanced data from ISNE pipeline
        """
        # Create a mapping from chunk IDs to enhanced embeddings
        enhanced_embeddings = {}
        
        if "chunks" in enhanced_data:
            for chunk_data in enhanced_data["chunks"]:
                chunk_id = chunk_data.get("id")
                isne_embedding = chunk_data.get("isne_embedding")
                if chunk_id and isne_embedding:
                    enhanced_embeddings[chunk_id] = isne_embedding
        
        # Update document chunks with ISNE embeddings
        for chunk in document.chunks:
            if chunk.id in enhanced_embeddings:
                chunk.isne_embedding = enhanced_embeddings[chunk.id]
            else:
                # If no enhanced embedding available, use original embedding as fallback
                self.logger.warning(f"No ISNE embedding for chunk {chunk.id}, using original")
                chunk.isne_embedding = chunk.embedding
    
    def _validate_isne_embeddings(self, document: DocumentSchema) -> ValidationResult:
        """Validate ISNE embeddings for a document's chunks.
        
        Args:
            document: Document with ISNE-embedded chunks to validate
            
        Returns:
            ValidationResult object
        """
        issues = []
        
        # Check if any chunks were processed
        if not document.chunks:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message="No chunks available for ISNE validation",
                location=document.file_path
            ))
            return ValidationResult(
                is_valid=False,
                issues=issues,
                stage_name=self.name
            )
        
        # Check for missing ISNE embeddings
        for i, chunk in enumerate(document.chunks):
            if chunk.isne_embedding is None:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message=f"Chunk {i} has no ISNE embedding",
                    location=f"{document.file_path}:chunk_{i}"
                ))
                continue
            
            # Check embedding dimension matches original
            if chunk.embedding is not None and len(chunk.isne_embedding) != len(chunk.embedding):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message=f"Chunk {i} ISNE embedding dimension mismatch",
                    location=f"{document.file_path}:chunk_{i}",
                    context={
                        "original_dim": len(chunk.embedding),
                        "isne_dim": len(chunk.isne_embedding)
                    }
                ))
            
            # Check for NaN or infinite values
            try:
                if any(not np.isfinite(v) for v in chunk.isne_embedding):
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message=f"Chunk {i} has NaN or infinite values in ISNE embedding",
                        location=f"{document.file_path}:chunk_{i}"
                    ))
            except (TypeError, ValueError):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message=f"Chunk {i} has invalid ISNE embedding data type",
                    location=f"{document.file_path}:chunk_{i}"
                ))
                continue
            
            # Check for zero vectors
            try:
                if all(abs(v) < 1e-6 for v in chunk.isne_embedding):
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Chunk {i} has a zero or near-zero ISNE embedding vector",
                        location=f"{document.file_path}:chunk_{i}"
                    ))
            except (TypeError, ValueError):
                pass  # Already handled above
            
            # Check for relationships (info level)
            if not chunk.relationships:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    message=f"Chunk {i} has no relationships",
                    location=f"{document.file_path}:chunk_{i}"
                ))
        
        # Return validation result
        return ValidationResult(
            is_valid=not any(issue.severity == ValidationSeverity.ERROR for issue in issues),
            issues=issues,
            stage_name=self.name
        )