"""
ModernBERT output loader for the ISNE pipeline.

This module provides a specialized loader for reading documents from the ModernBERT
pipeline JSON outputs, extracting embedded chunks and their relationships.

ONLY TO BE USED FOR TROUBLESHOOTING THE ISNE MODULE IN ISOLATION
"""

import json
import uuid
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Set, Iterator, Tuple, cast
import logging
from datetime import datetime

from src.isne.types.models import IngestDocument, DocumentRelation, RelationType, EmbeddingVector
from src.isne.loaders.base_loader import BaseLoader, LoaderConfig, LoaderResult

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModernBERTLoader(BaseLoader):
    """
    Loader for documents from ModernBERT pipeline outputs.
    
    This loader specializes in extracting documents, chunks, and their embeddings
    from the ModernBERT pipeline JSON outputs, establishing relationships based
    on document structure and content similarity.
    """
    
    def __init__(
        self,
        config: Optional[LoaderConfig] = None,
        embedding_field: str = "embedding",
        content_field: str = "content",
        chunk_field: str = "chunks",
        metadata_field: str = "metadata",
        id_field: str = "id",
        similarity_threshold: float = 0.7,
        extract_sequential_relations: bool = True,
        extract_similarity_relations: bool = True
    ) -> None:
        """
        Initialize the ModernBERT loader.
        
        Args:
            config: Loader configuration
            embedding_field: Field name containing embeddings
            content_field: Field name for chunk content
            chunk_field: Field name for the chunks array
            metadata_field: Field name for document metadata
            id_field: Field name for document/chunk IDs
            similarity_threshold: Threshold for creating similarity relationships
            extract_sequential_relations: Whether to create sequential relationships
            extract_similarity_relations: Whether to create similarity relationships
        """
        super().__init__(config)
        
        self.embedding_field = embedding_field
        self.content_field = content_field
        self.chunk_field = chunk_field
        self.metadata_field = metadata_field
        self.id_field = id_field
        self.similarity_threshold = similarity_threshold
        self.extract_sequential_relations = extract_sequential_relations
        self.extract_similarity_relations = extract_similarity_relations
    
    def load(self, source: Union[str, Path]) -> LoaderResult:
        """
        Load documents and chunks from a ModernBERT pipeline output file.
        
        Args:
            source: Path to the JSON file to load
            
        Returns:
            LoaderResult containing loaded documents, chunks, and relationships
        """
        source_path = Path(source)
        
        if not source_path.exists():
            raise ValueError(f"Source file does not exist: {source}")
        
        if not source_path.is_file():
            raise ValueError(f"Source is not a file: {source}")
        
        logger.info(f"Loading ModernBERT output from: {source_path}")
        
        # Load JSON data
        with open(source_path, 'r', encoding=self.config.encoding) as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON file: {e}")
        
        # Process the document
        documents: List[IngestDocument] = []
        relations: List[DocumentRelation] = []
        errors: List[Dict[str, Any]] = []
        
        try:
            # Process parent document
            doc_id = data.get(self.id_field, f"doc_{uuid.uuid4().hex[:8]}")
            source_path_str = str(data.get("source", source_path))
            
            # Extract metadata
            metadata = data.get(self.metadata_field, {})
            if not metadata and "format" in data:
                # Create basic metadata if not present
                metadata = {
                    "format": data.get("format", "unknown"),
                    "content_type": data.get("content_type", "text"),
                    "source": source_path_str
                }
            
            # Process chunks
            chunks = data.get(self.chunk_field, [])
            chunk_documents: List[IngestDocument] = []
            
            for i, chunk in enumerate(chunks):
                chunk_id = chunk.get(self.id_field, f"{doc_id}_chunk_{i}")
                chunk_content = chunk.get(self.content_field, "")
                
                # Skip empty chunks
                if not chunk_content.strip():
                    logger.warning(f"Skipping empty chunk: {chunk_id}")
                    continue
                
                # Extract embedding if present
                embedding = None
                if self.embedding_field in chunk and chunk[self.embedding_field]:
                    embedding_data = chunk[self.embedding_field]
                    if isinstance(embedding_data, list):
                        embedding = cast(EmbeddingVector, embedding_data)
                
                # Create chunk document
                chunk_doc = IngestDocument(
                    id=chunk_id,
                    content=chunk_content,
                    source=source_path_str,
                    document_type="chunk",
                    embedding=embedding,
                    metadata={
                        "parent_id": doc_id,
                        "index": i,
                        "chunk_type": chunk.get("type", "text"),
                        "source": source_path_str
                    }
                )
                
                # Add overlap context if present
                if "overlap_context" in chunk:
                    chunk_doc.metadata["overlap_context"] = chunk["overlap_context"]
                
                chunk_documents.append(chunk_doc)
            
            # Create parent document
            parent_doc = IngestDocument(
                id=doc_id,
                content=data.get("content", ""),  # Get content if available
                source=source_path_str,
                document_type=data.get("format", "document"),
                title=data.get("title"),
                author=data.get("author"),
                created_at=datetime.now(),
                updated_at=datetime.now(),
                metadata=metadata,
                # Use average of chunk embeddings as parent embedding if no embedding provided
                embedding=data.get("embedding", self._calculate_average_embedding(chunks) if chunks else None),
                embedding_model=data.get("embedding_model", "modernbert_aggregated") if chunks else None,
                # Store original chunk data for reference
                chunks=chunks if chunks else []
            )
            
            # Add parent document
            documents.append(parent_doc)
            
            # Add chunk documents
            documents.extend(chunk_documents)
            
            # Create parent-child relationships
            for chunk_doc in chunk_documents:
                relation = DocumentRelation(
                    source_id=doc_id,
                    target_id=chunk_doc.id,
                    relation_type=RelationType.CONTAINS,
                    weight=1.0,
                    metadata={
                        "source_type": parent_doc.document_type,
                        "target_type": chunk_doc.document_type
                    }
                )
                relations.append(relation)
            
            # Create sequential relationships between chunks
            if self.extract_sequential_relations and len(chunk_documents) > 1:
                for i in range(len(chunk_documents) - 1):
                    relation = DocumentRelation(
                        source_id=chunk_documents[i].id,
                        target_id=chunk_documents[i+1].id,
                        relation_type=RelationType.FOLLOWS,
                        weight=1.0,
                        metadata={
                            "sequential_index": i
                        }
                    )
                    relations.append(relation)
            
            # Create similarity relationships if enabled
            if self.extract_similarity_relations and len(chunk_documents) > 1:
                # Only process chunks with embeddings
                embeddable_chunks = [c for c in chunk_documents if c.embedding is not None]
                
                # Calculate similarity and create relationships
                similarity_relations = self._create_similarity_relations(embeddable_chunks)
                relations.extend(similarity_relations)
                
            logger.info(f"Processed document with {len(chunk_documents)} chunks and {len(relations)} relationships")
                
        except Exception as e:
            error_msg = f"Error processing document: {e}"
            logger.error(error_msg)
            errors.append({
                "source": str(source_path),
                "error": error_msg,
                "type": type(e).__name__
            })
        
        # Create dataset
        dataset_name = f"modernbert_{source_path.stem}"
        dataset = self.create_dataset(
            name=dataset_name,
            documents=documents,
            relations=relations,
            description=f"ModernBERT output from {source_path}",
            metadata={
                "source_path": str(source_path),
                "document_count": len(documents),
                "chunk_count": len(documents) - 1 if documents else 0,  # Subtract parent doc
                "relation_count": len(relations),
                "error_count": len(errors)
            }
        )
        
        return LoaderResult(
            documents=documents,
            relations=relations,
            dataset=dataset,
            errors=errors
        )
    
    def _calculate_average_embedding(self, chunks: List[Dict[str, Any]]) -> Optional[EmbeddingVector]:
        """
        Calculate the average embedding from a list of chunks.
        
        Args:
            chunks: List of chunks with embeddings
            
        Returns:
            Average embedding vector or None if no valid embeddings
        """
        # Collect all valid embeddings
        embeddings = []
        for chunk in chunks:
            if self.embedding_field in chunk and chunk[self.embedding_field]:
                embedding = chunk[self.embedding_field]
                if isinstance(embedding, list) or isinstance(embedding, np.ndarray):
                    embeddings.append(embedding)
        
        # Return None if no valid embeddings
        if not embeddings:
            return None
        
        # Convert all to numpy arrays for calculation
        numpy_embeddings = []
        for emb in embeddings:
            if isinstance(emb, list):
                numpy_embeddings.append(np.array(emb, dtype=np.float32))
            else:
                numpy_embeddings.append(emb)
        
        # Calculate average
        if numpy_embeddings:
            avg_embedding = np.mean(numpy_embeddings, axis=0)
            return avg_embedding.tolist()
        
        return None
    
    def _create_similarity_relations(self, chunks: List[IngestDocument]) -> List[DocumentRelation]:
        """
        Create similarity relationships between chunks with embeddings.
        
        Args:
            chunks: List of chunks with embeddings
            
        Returns:
            List of similarity relationships
        """
        relations: List[DocumentRelation] = []
        
        # Skip if insufficient chunks with embeddings
        if len(chunks) < 2:
            return relations
        
        # Calculate cosine similarity between all pairs
        for i in range(len(chunks)):
            for j in range(i+1, len(chunks)):
                # Skip if either chunk has no embedding
                if not chunks[i].embedding or not chunks[j].embedding:
                    continue
                
                # Calculate cosine similarity
                embedding_i = np.array(chunks[i].embedding)
                embedding_j = np.array(chunks[j].embedding)
                
                # Normalize embeddings
                norm_i = np.linalg.norm(embedding_i)
                norm_j = np.linalg.norm(embedding_j)
                
                if norm_i == 0 or norm_j == 0:
                    continue
                
                embedding_i = embedding_i / norm_i
                embedding_j = embedding_j / norm_j
                
                similarity = float(np.dot(embedding_i, embedding_j))
                
                # Create relationship if similarity exceeds threshold
                if similarity >= self.similarity_threshold:
                    relation = DocumentRelation(
                        source_id=chunks[i].id,
                        target_id=chunks[j].id,
                        relation_type=RelationType.SIMILAR_TO,
                        weight=similarity,
                        metadata={
                            "similarity_score": similarity
                        }
                    )
                    relations.append(relation)
        
        logger.info(f"Created {len(relations)} similarity relationships")
        return relations
    
    def load_batch(self, sources: List[Union[str, Path]]) -> List[LoaderResult]:
        """
        Load multiple ModernBERT output files.
        
        Args:
            sources: List of paths to JSON files
            
        Returns:
            List of LoaderResults
        """
        results: List[LoaderResult] = []
        
        for source in sources:
            try:
                result = self.load(source)
                results.append(result)
            except Exception as e:
                logger.error(f"Error loading {source}: {e}")
                # Create empty result with error
                error_result = LoaderResult(
                    documents=[],
                    relations=[],
                    errors=[{
                        "source": str(source),
                        "error": str(e),
                        "type": type(e).__name__
                    }],
                    metadata={
                        "source_path": str(source),
                        "error": str(e)
                    }
                )
                results.append(error_result)
        
        return results
