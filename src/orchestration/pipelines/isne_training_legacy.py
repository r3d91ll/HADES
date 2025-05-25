"""ISNE Training Pipeline for Text Documents (LEGACY VERSION).

## MIGRATION NOTICE
# TODO: This file needs to be refactored into smaller components following the new architecture.
# The TextTrainingPipeline class should be broken down into smaller, more focused classes.
# mypy: ignore-errors

This module provides a specialized implementation of the document transformation pipeline
focused on training the ISNE (Inductive Shallow Node Embedding) model from a directory of
text documents. It processes all PDF files found in a specified directory and creates
a training dataset for the ISNE model.

The pipeline follows these stages:
1. Document Collection: Find all PDF files in the specified directory
2. Document Processing: Transform source documents into normalized markdown text
3. Chunking: Transform normalized text into semantic chunks
4. Embedding: Enrich chunks with vector embeddings
5. Graph Construction: Build a document graph from the embeddings
6. ISNE Training: Train the ISNE model on the constructed graph
7. Model Saving: Save the trained model for later use

Usage:
    python -m src.pipelines.ingest.orchestrator.isne_training_text --input_dir /path/to/pdfs --output_dir /path/to/output
"""

import json
import logging
import sys
import os
import time
import asyncio
import glob
import random
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple, cast

import torch
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch_geometric
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import document processing functions
from src.docproc.core import process_document

# Import chunking functions
from src.chunking.text_chunkers.chonky_chunker import chunk_text

# Import embedding module
try:
    from src.embedding.processors import add_embeddings_to_document
    from src.embedding.base import get_adapter, EmbeddingAdapter
    from src.config.embedding_config import get_adapter_config, load_config
    EMBEDDING_AVAILABLE = True
except ImportError:
    logger.warning("Embedding module not available or incomplete - embeddings will be skipped")
    EMBEDDING_AVAILABLE = False

# Import ISNE module
try:
    from src.isne.models.isne_model import ISNEModel
    from src.isne.types.models import IngestDocument, DocumentRelation, RelationType, DocumentType
    from src.isne.loaders.modernbert_loader import ModernBERTLoader
    ISNE_AVAILABLE = True
except ImportError:
    logger.warning("ISNE module not available or incomplete - ISNE enhancement will be skipped")
    ISNE_AVAILABLE = False

# Explicitly import and register adapters to ensure they're available
try:
    from src.embedding.adapters.modernbert_adapter import ModernBERTEmbeddingAdapter
    from src.embedding.base import register_adapter
    register_adapter("modernbert", ModernBERTEmbeddingAdapter)
except ImportError:
    logger.warning("ModernBERT adapter not available")


class TextTrainingPipeline:
    """
    Pipeline for training the ISNE model using text documents.
    
    This pipeline processes PDF documents, extracts text chunks, computes embeddings,
    builds a document graph, and trains the ISNE model on this graph structure.
    """
    
    def __init__(
        self,
        input_dir: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        model_output_dir: Optional[Union[str, Path]] = None,
        chunking_options: Optional[Dict[str, Any]] = None,
        embedding_options: Optional[Dict[str, Any]] = None,
        training_options: Optional[Dict[str, Any]] = None,
        save_intermediate_results: bool = True
    ):
        """
        Initialize the ISNE training pipeline.
        
        Args:
            input_dir: Directory containing PDF files for training
            output_dir: Directory to save intermediate results
            model_output_dir: Directory to save trained models
            chunking_options: Options for document chunking
            embedding_options: Options for embedding computation
            training_options: Options for ISNE model training
            save_intermediate_results: Whether to save intermediate results
        """
        # Set directories
        if input_dir:
            self.input_dir = Path(input_dir) if isinstance(input_dir, str) else input_dir
        else:
            self.input_dir = None
            
        if output_dir:
            self.output_dir = Path(output_dir) if isinstance(output_dir, str) else output_dir
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.output_dir = None
            
        if model_output_dir:
            self.model_output_dir = Path(model_output_dir) if isinstance(model_output_dir, str) else model_output_dir
            self.model_output_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.model_output_dir = self.output_dir
            
        # Default chunking options
        self.chunking_options = chunking_options or {
            "max_tokens": 1024,
            "output_format": "json",
            "doc_type": "academic_pdf"
        }
        
        # Load embedding configuration
        if EMBEDDING_AVAILABLE:
            try:
                embedding_config = load_config()
                default_adapter = embedding_config.get("default_adapter", "modernbert")
                default_options = get_adapter_config(default_adapter)
                
                if embedding_options:
                    self.embedding_options = dict(default_options)
                    self.embedding_options.update(embedding_options)
                else:
                    self.embedding_options = default_options
                    
                logger.info(f"Using embedding adapter: {self.embedding_options.get('adapter_name', default_adapter)} "
                           f"on device: {self.embedding_options.get('device', 'cpu')}")
            except Exception as e:
                logger.warning(f"Error loading embedding configuration: {e}")
                # Fallback to defaults
                self.embedding_options = embedding_options or {
                    "adapter_name": "modernbert",
                    "model_name": "answerdotai/ModernBERT-base",
                    "max_length": 8192,
                    "pooling_strategy": "cls",
                    "batch_size": 8,
                    "device": "cpu",
                    "normalize_embeddings": True
                }
        else:
            self.embedding_options = embedding_options or {}
            logger.warning("Embedding module not available - using default options")
        
        # Default training options
        self.training_options = training_options or {
            "batch_size": 32,
            "learning_rate": 0.001,
            "epochs": 50,
            "weight_decay": 1e-5,
            "hidden_dim": 256,
            "dropout": 0.1,
            "num_layers": 2,
            "num_heads": 8,
            "device": "cpu",
            "early_stopping_patience": 10,
            "checkpoint_interval": 5
        }
        
        # Whether to save intermediate results
        self.save_intermediate_results = save_intermediate_results
        
        # Setup embedding adapter
        self.embedding_adapter = None
        if EMBEDDING_AVAILABLE:
            try:
                adapter_name = self.embedding_options.get("adapter_name", "modernbert")
                adapter_options = {
                    k: v for k, v in self.embedding_options.items()
                    if k not in ["adapter_name"]
                }
                self.embedding_adapter = get_adapter(adapter_name, **adapter_options)
                logger.info(f"Initialized embedding adapter: {adapter_name}")
            except Exception as e:
                logger.error(f"Failed to initialize embedding adapter: {e}")
        
        # Validation set for early stopping
        self.validation_fraction = self.training_options.get("validation_fraction", 0.2)
        
        # Initialize tracking variables
        self.document_count = 0
        self.chunk_count = 0
        self.edge_count = 0
        self.training_data = None
        
    async def process_document(self, file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """
        Process a document through the document processing module.
        
        This stage transforms the source document (PDF, etc.) into a normalized
        text format (markdown) with extracted metadata.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Normalized document dictionary or None if processing failed
        """
        path_obj = Path(file_path) if isinstance(file_path, str) else file_path
        
        logger.info(f"Processing document: {path_obj}")
        start_time = time.time()
        
        try:
            # Process document using docproc - this is a synchronous function, not async
            doc_result = process_document(path_obj)
            
            # Ensure the document has an ID
            if 'id' not in doc_result:
                # Generate a unique ID based on filename and hash
                file_hash = hashlib.md5(path_obj.name.encode()).hexdigest()[:8]
                doc_id = f"pdf_{file_hash}_{path_obj.stem.replace(' ', '_')}"
                doc_result['id'] = doc_id
            
            # Add path info to result
            doc_result['path'] = str(path_obj)
            
            # Save processed result to output dir if specified
            if self.output_dir:
                output_path = self.output_dir / f"{doc_result['id']}_processed.json"
                with open(output_path, 'w') as f:
                    json.dump(doc_result, f, indent=2)
                logger.info(f"Saved processed document to {output_path}")
            
            processing_time = time.time() - start_time
            logger.info(f"Document processing completed in {processing_time:.2f} seconds")
            
            return doc_result
        
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            logger.warning(f"Failed to process document: {file_path}")
            return None
            
    async def chunk_document(self, doc_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Transform a processed document into chunks.
        
        This method transforms the document representation by splitting the content
        into semantic chunks and reorganizing the data structure to be chunk-centric.
        The original document content is removed to eliminate redundancy, while preserving
        all document metadata.
        
        Args:
            doc_result: Document processing result from process_document()
            
        Returns:
            Transformed document with chunks, or None if chunking failed
        """
        if not doc_result:
            logger.error("No document provided for chunking")
            return None
            
        try:
            # Get document properties
            doc_id = doc_result.get('id', f"doc_{os.urandom(4).hex()}")
            doc_path = doc_result.get('path', 'unknown')
            
            logger.info(f"Chunking document {doc_id} with path {doc_path}")
            
            # Setup chunking options
            options = self.chunking_options.copy()
            
            # Set document-specific options
            options["doc_id"] = doc_id
            options["path"] = doc_path
            
            # Prepare for chunking - extract content string
            doc_content = doc_result.get("content", "")
            if not doc_content:
                logger.warning(f"Document {doc_id} has no content")
                return None
                
            logger.info(f"Content length: {len(doc_content)} characters")
            
            # Create a shallow copy of doc_result and remove the full content
            pipeline_result = {k: v for k, v in doc_result.items() if k != 'content'}
            
            # Process through chunking - using the same approach as the prototype pipeline
            chunks_result = chunk_text(
                content=doc_content,
                doc_id=doc_id,
                path=doc_path,
                doc_type=options.get("doc_type", "academic_pdf"),
                max_tokens=options.get("max_tokens", 1024),
                output_format=options.get("output_format", "json")
            )
            
            # Extract chunks from the chunking result
            if isinstance(chunks_result, dict) and "chunks" in chunks_result:
                chunks = chunks_result.get("chunks", [])
            else:
                # Handle direct list of chunks
                chunks = chunks_result if isinstance(chunks_result, list) else []
                
            logger.info(f"Created {len(chunks)} chunks from document {doc_id}")
            
            # Add chunks array to pipeline result
            pipeline_result['chunks'] = []
            
            # Add properly formatted chunks to pipeline result
            for idx, chunk in enumerate(chunks):
                # Handle different chunk formats
                if isinstance(chunk, dict):
                    chunk_content = chunk.get('text', chunk.get('content', ''))
                    chunk_metadata = chunk.get('metadata', {})
                else:
                    # If chunk is just a string
                    chunk_content = str(chunk)
                    chunk_metadata = {}
                    
                chunk_id = f"{doc_id}_chunk_{idx}"
                pipeline_result['chunks'].append({
                    'id': chunk_id,
                    'content': chunk_content,
                    'metadata': {
                        'seq_num': idx,
                        'token_count': chunk_metadata.get('token_count', 0),
                        'start_position': chunk_metadata.get('start_idx', 0),
                        'end_position': chunk_metadata.get('end_idx', len(chunk_content)),
                        **{k: v for k, v in chunk_metadata.items() if k not in ['token_count', 'start_idx', 'end_idx']}
                    }
                })
            
            # Save intermediate result if requested
            if self.save_intermediate_results and self.output_dir:
                output_path = self.output_dir / f"{doc_id}_chunked.json"
                with open(output_path, 'w') as f:
                    json.dump(pipeline_result, f, indent=2)
                logger.info(f"Saved chunked document to {output_path}")
            
            # Update statistics
            self.document_count += 1
            self.chunk_count += len(pipeline_result['chunks'])
            
            return pipeline_result
        except Exception as e:
            logger.error(f"Error chunking document: {e}")
            return None
    
    async def add_embeddings(self, pipeline_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Add embeddings to document chunks.
        
        This continues the document transformation by adding embeddings to each chunk,
        further enriching the document representation for downstream tasks.
        
        Args:
            pipeline_result: Transformed document from chunk_document()
            
        Returns:
            Further transformed document with embeddings, or original if embedding failed
        """
        if not pipeline_result or "chunks" not in pipeline_result:
            logger.error("No valid chunks provided for embedding")
            return pipeline_result
        
        if not self.embedding_adapter:
            logger.warning("No embedding adapter configured - skipping embedding generation")
            return pipeline_result
        
        doc_id = pipeline_result.get('id', 'Unknown')
        logger.info(f"Generating embeddings for document: {doc_id}")
        start_time = time.time()
        
        try:
            # Extract chunks from the pipeline result
            chunks = pipeline_result.get("chunks", [])
            num_chunks = len(chunks)
            
            if num_chunks == 0:
                logger.warning(f"Document {doc_id} has no chunks to embed")
                return pipeline_result
                
            # Extract text content for embedding
            chunk_texts: List[str] = [chunk.get("content", "") for chunk in chunks]
            
            # Log stats about the chunks being processed
            total_tokens = sum(len(text.split()) for text in chunk_texts)
            avg_tokens = total_tokens / num_chunks if num_chunks > 0 else 0
            logger.info(f"Generating embeddings for {num_chunks} chunks with avg {avg_tokens:.1f} tokens per chunk")
            
            # Generate embeddings using the configured adapter
            embeddings = await self.embedding_adapter.embed(chunk_texts)
            
            # Validate that we received the expected number of embeddings
            if len(embeddings) != num_chunks:
                logger.warning(f"Expected {num_chunks} embeddings but received {len(embeddings)}")
            
            # Update chunks with embeddings, ensuring proper conversion
            embedding_dim = None
            for i, embedding in enumerate(embeddings):
                if i < len(chunks):
                    # Handle different embedding types (numpy arrays, lists, etc.)
                    if hasattr(embedding, "tolist"):
                        embedding_list = embedding.tolist()
                    else:
                        embedding_list = list(embedding)
                    
                    # Store the first dimension for metadata
                    if embedding_dim is None and embedding_list:
                        embedding_dim = len(embedding_list)
                    
                    # Add embedding to the chunk
                    chunks[i]["embedding"] = embedding_list
            
            # Update the pipeline result with the enhanced chunks
            pipeline_result["chunks"] = chunks
            
            # Add embedding metadata to the processing metadata
            if "processing_metadata" not in pipeline_result:
                pipeline_result["processing_metadata"] = {}
                
            # Add detailed embedding metadata
            pipeline_result["processing_metadata"]["embedding"] = {
                "adapter": getattr(self.embedding_adapter, "name", "unknown"),
                "embedding_dimensions": embedding_dim,
                "chunks_embedded": len(embeddings),
                "timestamp": time.time(),
                "processing_time_sec": time.time() - start_time
            }
            
            embedding_time = time.time() - start_time
            logger.info(f"Embedding generation completed in {embedding_time:.2f} seconds")
            
            # Save embedded result if output directory is specified
            if self.output_dir:
                output_path = self.output_dir / f"{doc_id}_embedded.json"
                with open(output_path, 'w') as f:
                    json.dump(pipeline_result, f, indent=2)
                logger.info(f"Embedded document saved to {output_path}")
            
            return pipeline_result
            
        except Exception as e:
            logger.error(f"Error adding embeddings: {e}")
            return pipeline_result
            
    def build_document_graph(self, documents: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build a graph structure from document chunks for ISNE training.
        
        This method extracts embeddings from all chunks and creates edges between
        chunks based on document structure and similarity.
        
        Args:
            documents: List of processed documents with embeddings
            
        Returns:
            Tuple of (node_features, edge_index, edge_attr) for ISNE training
        """
        if not documents:
            logger.warning("No documents provided for graph construction")
            return torch.tensor([]), torch.tensor([[0, 0]]).t(), torch.tensor([])
            
        logger.info(f"Building document graph from {len(documents)} documents")
        
        # Collect all chunks and their embeddings
        all_chunks = []
        all_embeddings = []
        chunk_to_doc = {}
        doc_to_chunks = {}
        
        for doc_idx, doc in enumerate(documents):
            doc_id = doc.get('id', f'doc_{doc_idx}')
            chunks = doc.get('chunks', [])
            
            # Skip documents without chunks or embeddings
            has_embeddings = any('embedding' in chunk for chunk in chunks)
            if not chunks or not has_embeddings:
                continue
                
            # Store doc_id -> chunk_indices mapping
            doc_to_chunks[doc_id] = []
            
            # Store all chunks with their global index
            for chunk_idx, chunk in enumerate(chunks):
                if 'embedding' not in chunk:
                    continue
                    
                global_idx = len(all_chunks)
                all_chunks.append(chunk)
                all_embeddings.append(chunk['embedding'])
                
                # Store mappings
                chunk_to_doc[global_idx] = doc_id
                doc_to_chunks[doc_id].append(global_idx)
        
        # Convert embeddings to tensor
        if not all_embeddings:
            logger.warning("No valid embeddings found in documents")
            return torch.tensor([]), torch.tensor([[0, 0]]).t(), torch.tensor([])
            
        node_features = torch.tensor(all_embeddings)
        
        # Create edges
        edges = []
        edge_attributes = []
        
        # 1. Sequential edges within documents
        for doc_id, chunk_indices in doc_to_chunks.items():
            for i in range(len(chunk_indices) - 1):
                src = chunk_indices[i]
                dst = chunk_indices[i + 1]
                
                # Add bidirectional edges with weight 1.0 (strongest connection)
                edges.append((src, dst))
                edge_attributes.append(1.0)  # Sequential chunks have strongest relation
                
                edges.append((dst, src))
                edge_attributes.append(1.0)
        
        # 2. Add similarity-based edges between documents
        # This is a simplified approach - in production, you'd want more sophisticated relations
        similarity_threshold = 0.7
        max_cross_doc_edges = 1000  # Limit total number of cross-document edges to avoid explosion
        
        # Convert edges to tensor format
        if not edges:
            # Create a minimal edge set if none was created
            if len(all_chunks) > 1:
                edges = [(0, 1), (1, 0)]
                edge_attributes = [0.5, 0.5]
            else:
                # Self-loop if only one node
                edges = [(0, 0)]
                edge_attributes = [1.0]
                
        edge_index = torch.tensor(edges).t()  # Transpose to PyTorch Geometric format
        edge_attr = torch.tensor(edge_attributes).reshape(-1, 1)  # Shape: [num_edges, 1]
        
        self.edge_count = len(edges)
        logger.info(f"Created graph with {len(all_chunks)} nodes and {len(edges)} edges")
        
        return node_features, edge_index, edge_attr
    
    async def train_isne_model(self, documents: List[Dict[str, Any]]) -> Optional[ISNEModel]:
        """
        Train the ISNE model on a collection of documents.
        
        Args:
            documents: List of processed documents with embeddings
            
        Returns:
            Trained ISNE model or None if training failed
        """
        if not ISNE_AVAILABLE:
            logger.error("ISNE module not available - cannot train model")
            return None
            
        # Build document graph
        node_features, edge_index, edge_attr = self.build_document_graph(documents)
        
        if node_features.size(0) == 0 or edge_index.size(1) == 0:
            logger.error("Failed to create valid graph structure")
            return None
            
        # Get device
        device = self.training_options.get('device', 'cpu')
        device = torch.device(device if torch.cuda.is_available() and 'cuda' in device else 'cpu')
        
        # Move data to device
        node_features = node_features.to(device)
        edge_index = edge_index.to(device)
        edge_attr = edge_attr.to(device)
        
        # Create ISNE model
        embedding_dim = node_features.size(1)
        hidden_dim = self.training_options.get('hidden_dim', 256)
        
        model = ISNEModel(
            in_features=embedding_dim,
            hidden_features=hidden_dim,
            out_features=embedding_dim,  # Output same dimension as input for easier comparison
            num_layers=self.training_options.get('num_layers', 2),
            num_heads=self.training_options.get('num_heads', 8),
            dropout=self.training_options.get('dropout', 0.1),
        ).to(device)
        
        # Training parameters
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.training_options.get('learning_rate', 0.001),
            weight_decay=self.training_options.get('weight_decay', 1e-5)
        )
        
        scheduler = StepLR(
            optimizer,
            step_size=self.training_options.get('scheduler_step_size', 10),
            gamma=self.training_options.get('scheduler_gamma', 0.5)
        )
        
        # Training loop
        epochs = self.training_options.get('epochs', 50)
        patience = self.training_options.get('early_stopping_patience', 10)
        checkpoint_interval = self.training_options.get('checkpoint_interval', 5)
        
        best_loss = float('inf')
        no_improve_epochs = 0
        
        logger.info(f"Starting ISNE model training for {epochs} epochs on {device}")
        
        try:
            model.train()
            for epoch in range(epochs):
                # Forward pass
                optimizer.zero_grad()
                enhanced_embeddings = model(node_features, edge_index, edge_attr)
                
                # Compute loss - combination of reconstruction loss and structural preservation
                recon_loss = F.mse_loss(enhanced_embeddings, node_features)  # Reconstruction
                
                # Compute structure preservation loss
                # This encourages connected nodes to have similar embeddings
                src, dst = edge_index
                src_emb = enhanced_embeddings[src]
                dst_emb = enhanced_embeddings[dst]
                weights = edge_attr.view(-1)
                
                # Weighted cosine similarity (structure preservation)
                src_norm = F.normalize(src_emb, p=2, dim=1)
                dst_norm = F.normalize(dst_emb, p=2, dim=1)
                cosine_sim = (src_norm * dst_norm).sum(dim=1)
                struct_loss = -torch.mean(weights * cosine_sim)  # Negative because we want high similarity
                
                # Combined loss
                loss = recon_loss + struct_loss
                
                # Backward pass
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                # Log progress
                if (epoch + 1) % 5 == 0 or epoch == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, "
                               f"Recon: {recon_loss.item():.4f}, Struct: {struct_loss.item():.4f}")
                    
                # Save checkpoint if requested
                if (epoch + 1) % checkpoint_interval == 0 and self.model_output_dir:
                    checkpoint_path = self.model_output_dir / f"isne_model_epoch_{epoch+1}.pt"
                    torch.save(model.state_dict(), checkpoint_path)
                    logger.info(f"Checkpoint saved to {checkpoint_path}")
                    
                # Early stopping
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    no_improve_epochs = 0
                    
                    # Save best model
                    if self.model_output_dir:
                        best_model_path = self.model_output_dir / "isne_model_best.pt"
                        torch.save(model.state_dict(), best_model_path)
                        logger.info(f"New best model saved to {best_model_path}")
                else:
                    no_improve_epochs += 1
                    
                if no_improve_epochs >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1} due to no improvement for {patience} epochs")
                    break
                    
            # Training complete
            logger.info(f"ISNE model training completed with final loss: {best_loss:.4f}")
            
            # Save final model
            if self.model_output_dir:
                final_model_path = self.model_output_dir / "isne_model_final.pt"
                torch.save(model.state_dict(), final_model_path)
                logger.info(f"Final model saved to {final_model_path}")
                
            return model
            
        except Exception as e:
            logger.error(f"Error during ISNE model training: {e}")
            return None
    
    async def process_document_full(self, pdf_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """
        Process a document through the complete pipeline: processing, chunking, and embedding.
        
        Args:
            pdf_path: Path to the document file
            
        Returns:
            Transformed document with embeddings or None if processing failed
        """
        # Process document
        doc_result = await self.process_document(pdf_path)
        if not doc_result:
            return None
            
        # Chunk document
        chunked_result = await self.chunk_document(doc_result)
        if not chunked_result:
            return None
            
        # Add embeddings
        embedded_result = await self.add_embeddings(chunked_result)
        
        return embedded_result
        
    async def run_training(self, pdf_paths: List[Union[str, Path]]) -> Optional[ISNEModel]:
        """
        Run the complete training pipeline on a list of PDF files.
        
        This method:
        1. Processes each document to get chunks and embeddings
        2. Builds a document graph from all processed documents
        3. Trains the ISNE model on this graph
        4. Saves the trained model
        
        Args:
            pdf_paths: List of paths to PDF files for training
            
        Returns:
            Trained ISNE model or None if training failed
        """
        if not ISNE_AVAILABLE:
            logger.error("ISNE module not available - cannot train model")
            return None
            
        if not pdf_paths:
            logger.error("No PDF files provided for training")
            return None
            
        logger.info(f"Starting ISNE training pipeline with {len(pdf_paths)} documents")
        
        # Process all documents
        processed_documents = []
        success_count = 0
        
        for pdf_path in pdf_paths:
            logger.info(f"Processing document {success_count+1}/{len(pdf_paths)}: {pdf_path}")
            result = await self.process_document_full(pdf_path)
            
            if result:
                processed_documents.append(result)
                success_count += 1
            else:
                logger.warning(f"Failed to process document: {pdf_path}")
                
        logger.info(f"Successfully processed {success_count}/{len(pdf_paths)} documents")
        
        if not processed_documents:
            logger.error("No documents were successfully processed - aborting training")
            return None
            
        # Save the combined dataset if requested
        if self.save_intermediate_results and self.output_dir:
            dataset_path = self.output_dir / "training_dataset.json"
            with open(dataset_path, 'w') as f:
                json.dump(processed_documents, f, indent=2)
            logger.info(f"Saved combined dataset to {dataset_path}")
            
        # Train the ISNE model
        logger.info("Starting ISNE model training")
        model = await self.train_isne_model(processed_documents)
        
        if model:
            logger.info("ISNE model training completed successfully")
        else:
            logger.error("ISNE model training failed")
            
        return model


async def run_training_pipeline(input_dir: Union[str, Path], 
                        output_dir: Union[str, Path] = "./test-output/isne-training",
                        model_output_dir: Optional[Union[str, Path]] = None,
                        chunking_options: Optional[Dict[str, Any]] = None,
                        embedding_options: Optional[Dict[str, Any]] = None,
                        training_options: Optional[Dict[str, Any]] = None,
                        file_pattern: str = "*.pdf",
                        save_intermediate_results: bool = True) -> None:
    """
    Run the ISNE training pipeline on all PDF files in the specified directory.
    
    Args:
        input_dir: Directory containing PDF files for training
        output_dir: Directory to save intermediate results
        model_output_dir: Directory to save trained models (defaults to output_dir/models)
        chunking_options: Options for document chunking
        embedding_options: Options for embedding computation
        training_options: Options for ISNE model training
        file_pattern: Pattern to match input files (default: *.pdf)
        save_intermediate_results: Whether to save intermediate results
    """
    # Setup paths
    input_path = Path(input_dir) if isinstance(input_dir, str) else input_dir
    output_path = Path(output_dir) if isinstance(output_dir, str) else output_dir
    
    if not model_output_dir:
        model_output_dir = output_path / "models"
    
    # Ensure directories exist
    output_path.mkdir(parents=True, exist_ok=True)
    Path(model_output_dir).mkdir(parents=True, exist_ok=True)
    
    # Find all PDF files
    pdf_files = list(input_path.glob(file_pattern))
    
    if not pdf_files:
        logger.error(f"No files matching pattern '{file_pattern}' found in {input_path}")
        return
        
    logger.info(f"Found {len(pdf_files)} files matching pattern '{file_pattern}' in {input_path}")
    
    # Initialize pipeline
    pipeline = TextTrainingPipeline(
        input_dir=input_path,
        output_dir=output_path,
        model_output_dir=model_output_dir,
        chunking_options=chunking_options,
        embedding_options=embedding_options,
        training_options=training_options,
        save_intermediate_results=save_intermediate_results
    )
    
    # Run training
    logger.info("Starting training pipeline")
    model = await pipeline.run_training(pdf_files)
    
    if model:
        logger.info("Training completed successfully")
    else:
        logger.error("Training failed")


# Command-line entry point
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ISNE Training Pipeline for Text Documents")
    
    # Required arguments
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing PDF files for training")
    
    # Optional arguments
    parser.add_argument("--output_dir", type=str, default="./test-output/isne-training",
                        help="Directory to save intermediate results")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Directory to save trained models (defaults to output_dir/models)")
    parser.add_argument("--file_pattern", type=str, default="*.pdf",
                        help="File pattern to match input files (default: *.pdf)")
    parser.add_argument("--save_intermediates", action="store_true", default=True,
                        help="Save intermediate results from each stage")
    
    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Learning rate for training")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="Hidden dimension for ISNE model")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device for training (cpu or cuda)")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup training options
    training_options = {
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "hidden_dim": args.hidden_dim,
        "device": args.device
    }
    
    # Run training
    asyncio.run(run_training_pipeline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_output_dir=args.model_dir,
        training_options=training_options,
        file_pattern=args.file_pattern,
        save_intermediate_results=args.save_intermediates
    ))

