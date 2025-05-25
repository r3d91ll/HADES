"""
ISNE (Inductive Shallow Node Embedding) stage for the HADES-PathRAG pipeline.

This module provides a concrete implementation of the PipelineStage
abstract base class for enhancing embeddings using graph-based relationships.
"""

import logging
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
import uuid
import torch
from pathlib import Path
from datetime import datetime

from src.pipeline.stages import PipelineStage, PipelineStageError
from src.pipeline.schema import DocumentSchema, ChunkSchema, Relationship, ValidationResult, ValidationIssue, ValidationSeverity
from src.isne.trainer.training_orchestrator import ISNETrainingOrchestrator
from src.validation.embedding_validator import validate_embeddings_after_isne

logger = logging.getLogger(__name__)


class ISNEStage(PipelineStage):
    """Pipeline stage for ISNE processing.
    
    This stage handles the enhancement of embeddings using graph-based 
    relationships through the ISNE (Inductive Shallow Node Embedding) model.
    It builds a graph from document chunks, identifies relationships,
    and generates improved embeddings.
    """
    
    def __init__(
        self,
        name: str = "isne",
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize ISNE stage.
        
        Args:
            name: Name of the stage
            config: Configuration dictionary with ISNE options
        """
        super().__init__(name, config or {})
        
        # Configure ISNE parameters
        self.model_path = self.config.get("model_path")
        self.training_config = self.config.get("training_config", {})
        self.relationship_threshold = self.config.get("relationship_threshold", 0.7)
        self.max_relationships = self.config.get("max_relationships", 10)
        self.generate_relationships = self.config.get("generate_relationships", True)
        self.validate_results = self.config.get("validate_results", True)
        
        # Load or initialize ISNE model
        try:
            if self.model_path:
                self.logger.info(f"Loading ISNE model from {self.model_path}")
                self.isne_model = ISNETrainingOrchestrator.load_model(self.model_path)
            else:
                self.logger.info("Initializing new ISNE model")
                self.isne_model = None  # Will be created during processing
        except Exception as e:
            raise PipelineStageError(
                self.name,
                f"Failed to initialize ISNE model: {str(e)}",
                original_error=e
            )
    
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
        if not input_data:
            raise PipelineStageError(
                self.name,
                "No documents provided for ISNE processing"
            )
        
        # Extract all chunks across documents
        all_chunks = []
        chunk_map = {}  # Map chunk IDs to chunks
        
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
            
            # Build map of chunk IDs to chunks
            for chunk in document.chunks:
                chunk_map[chunk.id] = chunk
        
        self.logger.info(f"Processing {len(all_chunks)} chunks from {len(input_data)} documents")
        
        # Generate relationships between chunks if enabled
        if self.generate_relationships:
            self._generate_relationships(all_chunks, chunk_map)
            self.logger.info("Generated relationships between chunks")
        
        # Create training data for ISNE
        documents_dict_list = self._prepare_documents_for_isne(input_data)
        
        # Store pre-validation data for later comparison
        pre_validation = {
            "document_count": len(input_data),
            "chunk_count": len(all_chunks),
            "relationship_counts": self._count_relationships(all_chunks),
            "timestamp": datetime.now().isoformat()
        }
        
        # Train or update ISNE model and get enhanced embeddings
        try:
            self.logger.info("Training ISNE model and generating enhanced embeddings")
            # Initialize the orchestrator with documents and proper config override
            isne_orchestrator = ISNETrainingOrchestrator(
                documents=documents_dict_list,
                config_override={"isne": self.training_config},
                model_output_dir=str(Path(self.model_path).parent) if self.model_path else None
            )
            
            # Train the model and get the training metrics
            training_metrics = isne_orchestrator.train()
            
            # Extract enhanced embeddings from the trainer
            if hasattr(isne_orchestrator, 'trainer') and isne_orchestrator.trainer:
                # Get the model and device
                model = isne_orchestrator.trainer.model
                device = isne_orchestrator.device
                
                # Process each document and add ISNE embeddings
                for document in input_data:
                    for chunk in document.chunks:
                        # Find the chunk in the document graph
                        chunk_id = chunk.id
                        # If we have a model, generate the ISNE embedding
                        if model and hasattr(chunk, 'embedding') and chunk.embedding is not None:
                            # Convert embedding to tensor and send to device
                            embedding_tensor = torch.tensor(chunk.embedding, device=device)
                            # Generate ISNE embedding
                            with torch.no_grad():
                                isne_embedding = model(embedding_tensor.unsqueeze(0))
                                chunk.isne_embedding = isne_embedding.squeeze().cpu().numpy().tolist()
            
            # The model is automatically saved by the orchestrator if model_output_dir is provided
            if self.model_path:
                self.logger.info(f"ISNE model saved to directory: {Path(self.model_path).parent}")
                
            # Return the training metrics for debugging
            return {"training_metrics": training_metrics}
            
        except Exception as e:
            raise PipelineStageError(
                self.name,
                f"ISNE processing failed: {str(e)}",
                original_error=e
            )
        
        # Validate results if enabled
        if self.validate_results:
            try:
                validation_results = validate_embeddings_after_isne(documents_dict_list, pre_validation)
                self.logger.info(f"ISNE validation: {validation_results['valid_documents']} valid documents")
                
                # Log any validation issues
                if validation_results.get("missing_relationships", 0) > 0:
                    self.logger.warning(f"ISNE validation: {validation_results['missing_relationships']} missing relationships")
                
                if validation_results.get("invalid_embeddings", 0) > 0:
                    self.logger.warning(f"ISNE validation: {validation_results['invalid_embeddings']} invalid embeddings")
                
            except Exception as e:
                self.logger.error(f"ISNE validation failed: {str(e)}", exc_info=True)
        
        # Final validation
        processed_documents = []
        for document in input_data:
            validation_result = self._validate_isne_embeddings(document)
            if validation_result.has_errors:
                self.logger.warning(
                    f"ISNE validation failed for {document.file_name}: "
                    f"{len(validation_result.get_issues_by_severity(ValidationSeverity.ERROR))} errors"
                )
            processed_documents.append(document)
        
        self.logger.info(f"ISNE processing completed for {len(processed_documents)} documents")
        return processed_documents
    
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
    
    def _generate_relationships(self, chunks: List[ChunkSchema], chunk_map: Dict[str, ChunkSchema]) -> None:
        """Generate relationships between chunks based on embedding similarity.
        
        Args:
            chunks: List of chunks to process
            chunk_map: Map of chunk IDs to chunks
        """
        if not chunks:
            return
        
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
            top_indices = np.argsort(similarities[i])[::-1][:self.max_relationships]
            
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
                        "source_document": chunk.metadata.get("document_id", ""),
                        "target_document": target_chunk.metadata.get("document_id", "")
                    }
                )
                
                # Add relationship to chunk
                chunk.relationships.append(relationship)
    
    def _prepare_documents_for_isne(self, documents: List[DocumentSchema]) -> List[Dict[str, Any]]:
        """Prepare documents in the format expected by ISNETrainingOrchestrator.
        
        Args:
            documents: List of documents to prepare
            
        Returns:
            List of document dictionaries in ISNE format
        """
        isne_documents = []
        
        for document in documents:
            if not document.chunks:
                continue
                
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
                    chunk_dict["relationships"].append(rel_dict)
                
                doc_dict["chunks"].append(chunk_dict)
            
            isne_documents.append(doc_dict)
        
        return isne_documents
    
    def _count_relationships(self, chunks: List[ChunkSchema]) -> Dict[str, int]:
        """Count relationships by type.
        
        Args:
            chunks: List of chunks to analyze
            
        Returns:
            Dictionary of relationship counts by type
        """
        counts = {"total": 0}
        
        for chunk in chunks:
            for rel in chunk.relationships:
                counts["total"] += 1
                rel_type = rel.type
                counts[rel_type] = counts.get(rel_type, 0) + 1
        
        return counts
    
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
            if any(not np.isfinite(v) for v in chunk.isne_embedding):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message=f"Chunk {i} has NaN or infinite values in ISNE embedding",
                    location=f"{document.file_path}:chunk_{i}"
                ))
            
            # Check for zero vectors
            if all(abs(v) < 1e-6 for v in chunk.isne_embedding):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"Chunk {i} has a zero or near-zero ISNE embedding vector",
                    location=f"{document.file_path}:chunk_{i}"
                ))
            
            # Check for relationships
            if not chunk.relationships:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"Chunk {i} has no relationships",
                    location=f"{document.file_path}:chunk_{i}"
                ))
        
        # Return validation result
        return ValidationResult(
            is_valid=not any(issue.severity == ValidationSeverity.ERROR for issue in issues),
            issues=issues,
            stage_name=self.name
        )
