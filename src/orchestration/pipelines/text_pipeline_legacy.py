"""Text Processing Transformation Pipeline Prototype (LEGACY VERSION).

## MIGRATION NOTICE
# TODO: This file needs to be refactored into smaller components following the new architecture.
# The TextPipeline class should be broken down into smaller, more focused classes.
# mypy: ignore-errors

This module provides a prototype implementation of the document transformation pipeline,
focusing on the integration of document processing, chunking, and embedding components.
It serves as a reference implementation for future refactoring of the main ingestor.

The pipeline follows these transformation stages:
1. Document Processing: Transform source document into normalized text
2. Chunking: Transform normalized text into semantic chunks
3. Embedding: Enrich chunks with vector embeddings
4. ISNE Enhancement: Apply ISNE to improve embeddings with graph structure (optional)
5. Storage: Persist transformed representation to the database (future)

This pipeline is designed to be modality-agnostic, handling documents regardless of their
original format (PDF, DOCX, etc.) after they have been processed by the docproc module.

The pipeline comes in two modes:
- Inference mode: Process documents for use in retrieval and QA
- Training mode: Process documents and train ISNE model for embedding enhancement

Usage:
    from src.pipelines.ingest.orchestrator.text_pipeline_prototype import TextPipeline
    processor = TextPipeline()
    result = await processor.process_document('/path/to/document.pdf')
"""

import json
import logging
import sys
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple, cast

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Explicitly import and register adapters to ensure they're available
try:
    from src.embedding.adapters.modernbert_adapter import ModernBERTEmbeddingAdapter
    from src.embedding.base import register_adapter
    # Re-register the adapter to ensure it's in the registry
    register_adapter("modernbert", ModernBERTEmbeddingAdapter)
    logger.info("Successfully registered ModernBERTEmbeddingAdapter")
except Exception as e:
    logger.error(f"Error registering ModernBERTEmbeddingAdapter: {e}")

# Import document processing functions
from src.docproc.core import process_document

# Import chunking functions
from src.chunking.text_chunkers.chonky_chunker import chunk_text

# Try to import embedding module if available
try:
    from src.embedding.processors import add_embeddings_to_document
    from src.embedding.base import get_adapter, EmbeddingAdapter
    from src.config.embedding_config import get_adapter_config, load_config
    EMBEDDING_AVAILABLE = True
except ImportError:
    logger.warning("Embedding module not available or incomplete - embeddings will be skipped")
    EMBEDDING_AVAILABLE = False

# Try to import ISNE module if available
try:
    import torch
    from src.isne.models.isne_model import ISNEModel
    from src.isne.types.models import IngestDocument, DocumentRelation, RelationType
    from src.isne.loaders.modernbert_loader import ModernBERTLoader
    ISNE_AVAILABLE = True
except ImportError:
    logger.warning("ISNE module not available or incomplete - ISNE enhancement will be skipped")
    ISNE_AVAILABLE = False


class TextPipeline:
    """Pipeline for processing documents through the HADES-PathRAG system.
    
    This class implements a complete transformation pipeline that:
    1. Processes documents using the docproc module (normalizes to text)
    2. Transforms document into semantic chunks using the chunking module
    3. Enriches chunks with vector embeddings (if embedding module available)
    4. Can enhance embeddings with ISNE (if model available)
    5. Can save the results at each stage for analysis
    
    The pipeline is designed as a true transformation pipeline, where each stage
    transforms the document representation rather than simply adding to it.
    """
    
    def __init__(
            self,
            output_dir: Optional[Union[str, Path]] = None,
            chunking_options: Optional[Dict[str, Any]] = None,
            embedding_options: Optional[Dict[str, Any]] = None,
            isne_options: Optional[Dict[str, Any]] = None,
            save_intermediate_results: bool = False,
            mode: str = "inference"
        ):
        """Initialize the transformation pipeline.
        
        Args:
            output_dir: Directory to save output files (None = don't save)
            chunking_options: Options for the chunking stage
            embedding_options: Options for the embedding stage
            isne_options: Options for the ISNE model and training
            save_intermediate_results: Whether to save results from each stage
            mode: Pipeline operation mode ('inference' or 'training')
        """
        # Set pipeline mode
        self.mode = mode.lower()
        if self.mode not in ["inference", "training"]:
            logger.warning(f"Invalid mode '{mode}', falling back to 'inference'")
            self.mode = "inference"
        
        # Set output directory
        if output_dir:
            self.output_dir = Path(output_dir) if isinstance(output_dir, str) else output_dir
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.output_dir = None
            
        # Default chunking options
        self.chunking_options = chunking_options or {
            "max_tokens": 1024,
            "output_format": "json",
            "doc_type": "academic_text"
        }
        
        # Load embedding configuration from config system
        if EMBEDDING_AVAILABLE:
            try:
                embedding_config = load_config()
                default_adapter = embedding_config.get("default_adapter", "modernbert")
                # Get default adapter configuration
                default_options = get_adapter_config(default_adapter)
                
                # If user provided options, merge them with defaults
                if embedding_options:
                    self.embedding_options = dict(default_options)
                    self.embedding_options.update(embedding_options)
                else:
                    self.embedding_options = default_options
                    
                logger.info(f"Using embedding adapter: {self.embedding_options.get('adapter_name', default_adapter)} "
                           f"on device: {self.embedding_options.get('device', 'cpu')}")
            except Exception as e:
                logger.warning(f"Error loading embedding configuration: {e}")
                # Fallback to defaults if config loading fails
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
            # Fallback if embedding module isn't available
            self.embedding_options = embedding_options or {}
            logger.warning("Embedding module not available - using default options")
        
        # Whether to save intermediate results
        self.save_intermediate_results = save_intermediate_results
        
        # Setup embedding adapter if available
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
                
        # Setup ISNE options
        self.isne_options = isne_options or {}
        self.use_isne = self.isne_options.get("use_isne", False)
        
        # Initialize training statistics (only used in training mode)
        if self.mode == "training":
            self.document_count = 0
            self.chunk_count = 0
            self.processed_documents = []
        
        # Setup ISNE model if available and in inference mode
        self.isne_model = None
        if ISNE_AVAILABLE and self.use_isne and self.mode == "inference":
            self._initialize_isne_model()
    
    def _initialize_isne_model(self):
        """Initialize the ISNE model from a saved model file.
        
        This method loads a pre-trained ISNE model for inference.
        It is only used in inference mode, not in training mode.
        """
        try:
            # Load ISNE model from path if provided
            model_path = self.isne_options.get("model_path")
            if model_path:
                model_path = Path(model_path) if isinstance(model_path, str) else model_path
                if model_path.exists():
                    # Configure model dimensions based on our embedding configuration
                    embedding_dim = self.embedding_options.get("embedding_dim", 768)
                    hidden_dim = self.isne_options.get("hidden_dim", 256)
                    output_dim = self.isne_options.get("output_dim", embedding_dim)  # Default to same as input
                    
                    # Get device configuration
                    device = self.isne_options.get("device", "cpu")
                    
                    # Create model with proper dimensions
                    self.isne_model = ISNEModel(
                        in_features=embedding_dim,
                        hidden_features=hidden_dim,
                        out_features=output_dim,
                        num_layers=self.isne_options.get("num_layers", 2),
                        num_heads=self.isne_options.get("num_heads", 8),
                        dropout=self.isne_options.get("dropout", 0.1),
                    ).to(device)
                    
                    # Load saved weights
                    state_dict = torch.load(model_path, map_location=device)
                    self.isne_model.load_state_dict(state_dict)
                    self.isne_model.eval()  # Set to inference mode
                    
                    logger.info(f"Loaded ISNE model from {model_path} on {device}")
                else:
                    logger.warning(f"ISNE model path not found: {model_path} - ISNE enhancement disabled")
                    self.use_isne = False
            else:
                logger.info("No ISNE model path provided - ISNE enhancement disabled")
                self.use_isne = False
        except Exception as e:
            logger.error(f"Failed to initialize ISNE model: {e}")
            self.use_isne = False
    
    async def process_document(self, file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """Process a document through the document processing module.
        
        This stage transforms the source document (PDF, DOCX, etc.) into a normalized
        text format with extracted metadata.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Normalized document dictionary or None if processing failed
        """
        path_obj = Path(file_path) if isinstance(file_path, str) else file_path
        
        logger.info(f"STAGE 1: Document Processing - {path_obj.name}")
        start_time = time.time()
        
        try:
            # Process document using docproc
            doc_result = process_document(path_obj)
            
            # Save intermediate result if requested
            if self.save_intermediate_results and self.output_dir:
                output_path = self.output_dir / f"docproc_{path_obj.stem}_output.json"
                with open(output_path, 'w') as f:
                    json.dump(doc_result, f, indent=2)
                logger.info(f"Document processing output saved to: {output_path}")
            
            processing_time = time.time() - start_time
            logger.info(f"Document processing completed in {processing_time:.2f} seconds")
            
            # In training mode, track document count
            if self.mode == "training":
                self.document_count += 1
            
            return doc_result
            
        except Exception as e:
            logger.error(f"Error processing {path_obj.name}: {e}")
            return None
    
    async def chunk_document(self, doc_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Transform a processed document into chunks.
        
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
        
        logger.info(f"STAGE 2: Document Chunking - {doc_result.get('id', 'Unknown')}")
        start_time = time.time()
        
        try:
            # Get document properties
            doc_id = doc_result.get('id', f"doc_{os.urandom(4).hex()}")
            doc_path = doc_result.get('source', 'unknown')
            
            # Setup chunking options
            options = self.chunking_options.copy()
            
            # Set document-specific options
            options["doc_id"] = doc_id
            options["path"] = doc_path
            
            # Prepare for chunking - extract content string
            doc_content = doc_result.get("content", "")
            
            # Process through chunking - note that chunk_text expects content as string, not dict
            chunks_result = chunk_text(
                content=doc_content,
                doc_id=doc_id,
                path=doc_path,
                doc_type=options.get("doc_type", "academic_text"),
                max_tokens=options.get("max_tokens", 1024),
                output_format=options.get("output_format", "json")
            )
            
            # Extract chunks from the chunking result
            if isinstance(chunks_result, dict) and "chunks" in chunks_result:
                chunks = chunks_result.get("chunks", [])
            else:
                # Direct list of chunks
                chunks = chunks_result if isinstance(chunks_result, list) else []
            
            # Create a shallow copy of doc_result and remove the full content
            # This reduces redundancy while preserving metadata
            pipeline_result = {k: v for k, v in doc_result.items() if k != 'content'}
            
            # Add metadata about chunking if not already present
            if "processing_metadata" not in pipeline_result:
                pipeline_result["processing_metadata"] = {}
                
            # Add chunking metadata
            pipeline_result["processing_metadata"]["chunking"] = {
                "strategy": "semantic",
                "chunk_count": len(chunks),
                "max_tokens": options.get("max_tokens", 1024),
                "timestamp": time.time(),
                "processing_time_sec": time.time() - start_time
            }
            
            # Add chunks array
            pipeline_result["chunks"] = []
            
            # Process each chunk and add to result
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}_chunk_{i}"
                
                # Extract content - handle different chunk formats
                if isinstance(chunk, dict):
                    content = chunk.get("text", chunk.get("content", ""))
                    metadata = chunk.get("metadata", {})
                else:
                    content = str(chunk)
                    metadata = {}
                
                # Structured chunk representation
                structured_chunk = {
                    "id": chunk_id,
                    "content": content,
                    "metadata": {
                        "seq_num": i,
                        "doc_id": doc_id,
                        "token_count": metadata.get("token_count", len(content.split())),
                    }
                }
                
                # Add position info if available
                if "start_idx" in metadata:
                    structured_chunk["metadata"]["start_position"] = metadata["start_idx"]
                if "end_idx" in metadata:
                    structured_chunk["metadata"]["end_position"] = metadata["end_idx"]
                
                # Add chunk to result
                pipeline_result["chunks"].append(structured_chunk)
            
            # Save intermediate result if requested
            if self.save_intermediate_results and self.output_dir:
                output_path = self.output_dir / f"{doc_id}_chunked.json"
                with open(output_path, 'w') as f:
                    json.dump(pipeline_result, f, indent=2)
                logger.info(f"Chunked document saved to: {output_path}")
            
            # In training mode, track chunk count
            if self.mode == "training":
                self.chunk_count += len(pipeline_result["chunks"])
            
            # Log stats
            chunking_time = time.time() - start_time
            logger.info(f"Document chunking completed in {chunking_time:.2f} seconds with {len(pipeline_result['chunks'])} chunks")
            
            return pipeline_result
            
        except Exception as e:
            logger.error(f"Error chunking document: {e}")
            return None
    
    async def add_embeddings(self, pipeline_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Add embeddings to document chunks.
        
        This continues the document transformation by adding embeddings to each chunk,
        further enriching the document representation for downstream tasks.
        
        ModernBERT adapter is used by default to generate embeddings with support for longer
        contexts (up to 8K tokens), making it suitable for academic papers with complex
        semantic structure. 
        
        Args:
            pipeline_result: Transformed document from chunk_document()
            
        Returns:
            Further transformed document with embeddings, or original if embedding failed
        """
        if not pipeline_result or "chunks" not in pipeline_result:
            logger.error("No valid chunks provided for embedding")
            return pipeline_result
        
        if not EMBEDDING_AVAILABLE:
            logger.warning("Embedding module not available - skipping embedding generation")
            return pipeline_result
        
        if not self.embedding_adapter:
            logger.warning("No embedding adapter configured - skipping embedding generation")
            return pipeline_result
        
        doc_id = pipeline_result.get('id', 'Unknown')
        logger.info(f"STAGE 3: Embedding Generation - {doc_id}")
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
            
            # Generate embeddings using the configured adapter (ModernBERT by default)
            adapter_name = self.embedding_options.get("adapter_name", "modernbert")
            pooling = self.embedding_options.get("pooling_strategy", "cls")
            logger.info(f"Using {adapter_name} adapter with {pooling} pooling strategy")
            
            # Use embed method directly from our adapter for type safety
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
                "adapter": adapter_name,
                "model": self.embedding_options.get("model_name", "answerdotai/ModernBERT-base"),
                "pooling_strategy": pooling,
                "embedding_dimensions": embedding_dim,
                "chunks_embedded": len(embeddings),
                "timestamp": time.time(),
                "processing_time_sec": time.time() - start_time
            }
            
            embedding_time = time.time() - start_time
            logger.info(f"Embedding generation completed in {embedding_time:.2f} seconds")
            
            # Save intermediate result if requested
            if self.save_intermediate_results and self.output_dir:
                output_path = self.output_dir / f"{doc_id}_embedded.json"
                with open(output_path, 'w') as f:
                    json.dump(pipeline_result, f, indent=2)
                logger.info(f"Document with embeddings saved to: {output_path}")
                
            # In training mode, store processed document for later training
            if self.mode == "training":
                self.processed_documents.append(pipeline_result)
            
            return pipeline_result
            
        except Exception as e:
            logger.error(f"Error adding embeddings: {e}")
            return pipeline_result
    
    async def enhance_with_isne(self, pipeline_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance embeddings using the ISNE model.
        
        This continues the document transformation by applying ISNE to enhance embeddings
        with graph structural information. This captures semantic relationships between chunks
        and improves embedding quality for downstream retrieval.
        
        Args:
            pipeline_result: Pipeline result with embeddings from add_embeddings()
            
        Returns:
            Enhanced document with ISNE-improved embeddings
        """
        # Only applicable in inference mode with a loaded ISNE model
        if self.mode != "inference" or not self.use_isne or not self.isne_model:
            return pipeline_result
            
        if not pipeline_result or "chunks" not in pipeline_result:
            logger.error("No valid chunks provided for ISNE enhancement")
            return pipeline_result
            
        doc_id = pipeline_result.get('id', 'Unknown')
        logger.info(f"STAGE 4: ISNE Enhancement - {doc_id}")
        start_time = time.time()
        
        try:
            # Extract embeddings from the chunks
            chunks = pipeline_result.get("chunks", [])
            if not chunks:
                logger.warning(f"Document {doc_id} has no chunks to enhance")
                return pipeline_result
                
            # Get embeddings from chunks
            chunk_embeddings = []
            for chunk in chunks:
                embedding = chunk.get("embedding")
                if embedding is None:
                    logger.warning("Chunk without embedding found - ISNE enhancement requires embeddings")
                    return pipeline_result
                chunk_embeddings.append(embedding)
                
            # Convert embeddings to tensor format
            device = self.isne_options.get("device", "cpu")
            x = torch.tensor(chunk_embeddings, dtype=torch.float32).to(device)
            
            # Apply ISNE model to enhance embeddings
            with torch.no_grad():
                enhanced_embeddings = self.isne_model(x)
                
            # Convert enhanced embeddings back to list format
            enhanced_embeddings_list = enhanced_embeddings.cpu().numpy().tolist()
            
            # Update chunks with enhanced embeddings
            for i, enhanced_embedding in enumerate(enhanced_embeddings_list):
                if i < len(chunks):
                    # Keep original embedding for reference/comparison
                    chunks[i]["original_embedding"] = chunks[i]["embedding"]
                    chunks[i]["embedding"] = enhanced_embedding
            
            # Update the pipeline result with enhanced chunks
            pipeline_result["chunks"] = chunks
            
            # Add ISNE metadata to the processing metadata
            if "processing_metadata" not in pipeline_result:
                pipeline_result["processing_metadata"] = {}
                
            # Add ISNE metadata
            pipeline_result["processing_metadata"]["isne"] = {
                "model_path": str(self.isne_options.get("model_path", "unknown")),
                "timestamp": time.time(),
                "processing_time_sec": time.time() - start_time
            }
            
            enhancement_time = time.time() - start_time
            logger.info(f"ISNE enhancement completed in {enhancement_time:.2f} seconds")
            
            # Save intermediate result if requested
            if self.save_intermediate_results and self.output_dir:
                output_path = self.output_dir / f"{doc_id}_isne_enhanced.json"
                with open(output_path, 'w') as f:
                    json.dump(pipeline_result, f, indent=2)
                logger.info(f"ISNE enhanced document saved to: {output_path}")
                
            return pipeline_result
            
        except Exception as e:
            logger.error(f"Error enhancing with ISNE: {e}")
            return pipeline_result
    
    def build_document_graph(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build a graph structure from document chunks for ISNE training.
        
        This method extracts embeddings from all chunks and creates edges between
        chunks based on document structure and similarity.
        
        Returns:
            Tuple of (node_features, edge_index, edge_attr) for ISNE training
        """
        # Only applicable in training mode
        if self.mode != "training" or not self.processed_documents:
            logger.error("Cannot build document graph - either not in training mode or no documents processed")
            return torch.tensor([]), torch.tensor([[0, 0]]).t(), torch.tensor([])
            
        logger.info(f"Building document graph from {len(self.processed_documents)} documents with {self.chunk_count} total chunks")
        start_time = time.time()
        
        try:
            # Collect all chunks from all documents
            all_chunks = []
            doc_id_to_index = {}
            
            # First pass: collect all chunks and map document IDs to indices
            for doc_idx, doc in enumerate(self.processed_documents):
                doc_id = doc.get('id', f"doc_{doc_idx}")
                doc_id_to_index[doc_id] = doc_idx
                
                # Add all chunks from this document
                chunks = doc.get("chunks", [])
                all_chunks.extend(chunks)
            
            # Extract embeddings from all chunks
            node_features = []
            chunk_to_doc = []  # Map each chunk to its document
            
            for chunk in all_chunks:
                embedding = chunk.get("embedding")
                if embedding is None:
                    logger.warning("Chunk without embedding found - skipping")
                    continue
                    
                # Add embedding as node feature
                node_features.append(embedding)
                
                # Record which document this chunk belongs to
                doc_id = chunk.get("metadata", {}).get("doc_id")
                if doc_id in doc_id_to_index:
                    chunk_to_doc.append(doc_id_to_index[doc_id])
            
            # Convert node features to tensor
            node_features_tensor = torch.tensor(node_features, dtype=torch.float32)
            num_nodes = node_features_tensor.size(0)
            
            # Create edges between chunks
            edge_list = []
            edge_attr_list = []
            
            # 1. Intra-document edges: connect chunks from the same document sequentially
            doc_chunks = {}  # Group chunks by document
            for i, doc_idx in enumerate(chunk_to_doc):
                if doc_idx not in doc_chunks:
                    doc_chunks[doc_idx] = []
                doc_chunks[doc_idx].append(i)
            
            # Connect sequential chunks within each document
            for doc_idx, chunk_indices in doc_chunks.items():
                for i in range(len(chunk_indices)-1):
                    source = chunk_indices[i]
                    target = chunk_indices[i+1]
                    
                    # Add edges in both directions
                    edge_list.append([source, target])
                    edge_attr_list.append(1.0)  # Sequential relationship weight
                    
                    edge_list.append([target, source])
                    edge_attr_list.append(1.0)
            
            # 2. Inter-document similarity edges: connect similar chunks across documents
            # This is optional and can be controlled by a parameter
            if self.isne_options.get("use_similarity_edges", True) and num_nodes > 1:
                # Compute pairwise cosine similarity
                normalized_features = torch.nn.functional.normalize(node_features_tensor, p=2, dim=1)
                similarity_matrix = torch.mm(normalized_features, normalized_features.t())
                
                # Set threshold for creating edges
                similarity_threshold = self.isne_options.get("similarity_threshold", 0.8)
                
                # Find pairs above threshold, excluding self-connections
                for i in range(num_nodes):
                    for j in range(i+1, num_nodes):  # Only upper triangle to avoid duplicates
                        # Skip if from same document (already connected sequentially)
                        if chunk_to_doc[i] == chunk_to_doc[j]:
                            continue
                            
                        similarity = similarity_matrix[i, j].item()
                        if similarity > similarity_threshold:
                            # Add bidirectional edges
                            edge_list.append([i, j])
                            edge_attr_list.append(similarity)  # Use similarity as edge weight
                            
                            edge_list.append([j, i])
                            edge_attr_list.append(similarity)
            
            # Convert edge list to COO format tensor
            if edge_list:
                edge_index = torch.tensor(edge_list, dtype=torch.long).t()
                edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32)
            else:
                # Create a dummy self-loop if no edges exist
                edge_index = torch.tensor([[0, 0]], dtype=torch.long).t()
                edge_attr = torch.tensor([1.0], dtype=torch.float32)
            
            # Log graph statistics
            graph_build_time = time.time() - start_time
            logger.info(f"Built document graph with {num_nodes} nodes and {len(edge_list)} edges in {graph_build_time:.2f} seconds")
            
            return node_features_tensor, edge_index, edge_attr
            
        except Exception as e:
            logger.error(f"Error building document graph: {e}")
            return torch.tensor([]), torch.tensor([[0, 0]]).t(), torch.tensor([])
    
    async def train_isne_model(self) -> Optional[Dict[str, Any]]:
        """
        Train the ISNE model on the document graph.
        
        This method builds a document graph from all processed documents,
        then trains an ISNE model to learn graph-aware embeddings.
        
        Returns:
            Training results dictionary or None if training failed
        """
        # Only applicable in training mode
        if self.mode != "training":
            logger.error("Cannot train ISNE model - not in training mode")
            return None
            
        if not ISNE_AVAILABLE:
            logger.error("ISNE module not available - cannot train model")
            return None
            
        if not self.processed_documents:
            logger.error("No documents processed - cannot train ISNE model")
            return None
            
        logger.info(f"STAGE 5: ISNE Model Training - {len(self.processed_documents)} documents, {self.chunk_count} chunks")
        start_time = time.time()
        
        try:
            # Build document graph
            logger.info("Building document graph for ISNE training...")
            node_features, edge_index, edge_attr = self.build_document_graph()
            
            if node_features.size(0) == 0:
                logger.error("Empty document graph - cannot train ISNE model")
                return None
                
            # Set device for training
            device = self.isne_options.get("device", "cpu")
            logger.info(f"Training ISNE model on {device}")
            
            # Configure model dimensions
            embedding_dim = node_features.size(1)
            hidden_dim = self.isne_options.get("hidden_dim", 256)
            output_dim = self.isne_options.get("output_dim", embedding_dim)  # Default to same as input
            
            # Create ISNE model
            model = ISNEModel(
                in_features=embedding_dim,
                hidden_features=hidden_dim,
                out_features=output_dim,
                num_layers=self.isne_options.get("num_layers", 2),
                num_heads=self.isne_options.get("num_heads", 8),
                dropout=self.isne_options.get("dropout", 0.1),
            ).to(device)
            
            # Move data to device
            node_features = node_features.to(device)
            edge_index = edge_index.to(device)
            edge_attr = edge_attr.to(device)
            
            # Configure optimizer
            learning_rate = self.isne_options.get("learning_rate", 0.001)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            
            # Configure learning rate scheduler
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, 
                step_size=self.isne_options.get("lr_step_size", 10),
                gamma=self.isne_options.get("lr_decay", 0.5)
            )
            
            # Training loop
            num_epochs = self.isne_options.get("num_epochs", 100)
            train_losses = []
            best_loss = float('inf')
            patience = self.isne_options.get("patience", 10)
            patience_counter = 0
            best_model_state = None
            
            logger.info(f"Starting ISNE training for {num_epochs} epochs")
            for epoch in range(num_epochs):
                model.train()
                optimizer.zero_grad()
                
                # Forward pass
                enhanced_embeddings = model(node_features, edge_index, edge_attr)
                
                # Compute loss - reconstruction loss between original and enhanced embeddings
                loss = torch.nn.functional.mse_loss(enhanced_embeddings, node_features)
                
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                # Record loss
                current_loss = loss.item()
                train_losses.append(current_loss)
                
                # Log progress
                if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == num_epochs - 1:
                    logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {current_loss:.6f}")
                
                # Early stopping
                if current_loss < best_loss:
                    best_loss = current_loss
                    patience_counter = 0
                    # Save best model state
                    best_model_state = model.state_dict()
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"Early stopping at epoch {epoch+1} due to no improvement")
                        break
            
            # Restore best model
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            
            # Save trained model if output directory is provided
            if self.output_dir:
                model_filename = f"isne_model_{time.strftime('%Y%m%d_%H%M%S')}.pt"
                model_path = self.output_dir / model_filename
                torch.save(model.state_dict(), model_path)
                logger.info(f"Saved trained ISNE model to {model_path}")
            
            # Create training results
            training_time = time.time() - start_time
            training_results = {
                "training_info": {
                    "num_documents": len(self.processed_documents),
                    "num_chunks": self.chunk_count,
                    "num_epochs": epoch + 1,  # Actual number of epochs trained
                    "final_loss": train_losses[-1],
                    "best_loss": best_loss,
                    "training_time_sec": training_time,
                    "timestamp": time.time(),
                    "embedding_dim": embedding_dim,
                    "hidden_dim": hidden_dim,
                    "output_dim": output_dim,
                    "graph_info": {
                        "num_nodes": node_features.size(0),
                        "num_edges": edge_index.size(1),
                    }
                },
                "loss_history": train_losses,
                "model_path": str(model_path) if self.output_dir else None,
            }
            
            # Save training results
            if self.output_dir:
                results_path = self.output_dir / "isne_training_results.json"
                with open(results_path, 'w') as f:
                    json.dump(training_results, f, indent=2)
                logger.info(f"Saved training results to {results_path}")
            
            logger.info(f"ISNE training completed in {training_time:.2f} seconds")
            return training_results
            
        except Exception as e:
            logger.error(f"Error training ISNE model: {e}")
            return None
    
    async def process_document_full(self, file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """Process a document through the complete transformation pipeline.
        
        This method orchestrates the complete document transformation pipeline:
        1. Document Processing: Converts the document to normalized text
        2. Chunking: Transforms the document into chunks
        3. Embedding: Enriches chunks with vector embeddings
        4. ISNE Enhancement: Enhances embeddings with graph structure (inference mode only)
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Transformed document with chunks and enhanced embeddings
        """
        # Stage 1: Document Processing
        doc_result = await self.process_document(file_path)
        if not doc_result:
            return None
            
        # Stage 2: Chunking
        chunked_result = await self.chunk_document(doc_result)
        if not chunked_result:
            return None
            
        # Stage 3: Embedding
        embedded_result = await self.add_embeddings(chunked_result)
        if not embedded_result:
            return None
            
        # Stage 4: ISNE Enhancement (inference mode only)
        if self.mode == "inference" and self.use_isne and self.isne_model:
            enhanced_result = await self.enhance_with_isne(embedded_result)
            return enhanced_result
        else:
            return embedded_result

# Main run functions for the pipeline
async def run_inference_pipeline(
    document_paths: List[Union[str, Path]], 
    output_dir: Union[str, Path] = "./test-output/inference",
    chunking_options: Optional[Dict[str, Any]] = None,
    embedding_options: Optional[Dict[str, Any]] = None,
    isne_options: Optional[Dict[str, Any]] = None,
    save_intermediate_results: bool = True
) -> List[Dict[str, Any]]:
    """
    Run the pipeline in inference mode on multiple documents.
    
    Args:
        document_paths: List of paths to documents
        output_dir: Directory to save output files
        chunking_options: Options for the chunking stage
        embedding_options: Options for the embedding stage
        isne_options: Options for the ISNE enhancement stage
        save_intermediate_results: Whether to save intermediate results
        
    Returns:
        List of processed documents with embeddings and ISNE enhancement
    """
    logger.info(f"Running inference pipeline on {len(document_paths)} documents")
    
    # Initialize pipeline with inference mode
    pipeline = TextPipeline(
        output_dir=output_dir,
        chunking_options=chunking_options,
        embedding_options=embedding_options,
        isne_options=isne_options,
        save_intermediate_results=save_intermediate_results,
        mode="inference"
    )
    
    # Process documents one by one
    results = []
    for i, doc_path in enumerate(document_paths):
        logger.info(f"Processing document {i+1}/{len(document_paths)}: {doc_path}")
        doc_result = await pipeline.process_document_full(doc_path)
        if doc_result:
            results.append(doc_result)
    
    logger.info(f"Successfully processed {len(results)}/{len(document_paths)} documents")
    return results

async def run_training_pipeline(
    document_paths: List[Union[str, Path]], 
    output_dir: Union[str, Path] = "./test-output/training",
    chunking_options: Optional[Dict[str, Any]] = None,
    embedding_options: Optional[Dict[str, Any]] = None,
    isne_options: Optional[Dict[str, Any]] = None,
    save_intermediate_results: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Run the pipeline in training mode on multiple documents.
    
    This function processes all documents, builds a document graph,
    and trains an ISNE model on the graph.
    
    Args:
        document_paths: List of paths to documents
        output_dir: Directory to save output files
        chunking_options: Options for the chunking stage
        embedding_options: Options for the embedding stage
        isne_options: Options for the ISNE training
        save_intermediate_results: Whether to save intermediate results
        
    Returns:
        Training results dictionary or None if training failed
    """
    logger.info(f"Running training pipeline on {len(document_paths)} documents")
    
    # Initialize pipeline with training mode
    pipeline = TextPipeline(
        output_dir=output_dir,
        chunking_options=chunking_options,
        embedding_options=embedding_options,
        isne_options=isne_options,
        save_intermediate_results=save_intermediate_results,
        mode="training"
    )
    
    # Process all documents first
    successful_docs = 0
    for i, doc_path in enumerate(document_paths):
        logger.info(f"Processing document {i+1}/{len(document_paths)}: {doc_path}")
        doc_result = await pipeline.process_document_full(doc_path)
        if doc_result:
            successful_docs += 1
    
    logger.info(f"Successfully processed {successful_docs}/{len(document_paths)} documents")
    
    # Train ISNE model
    if successful_docs > 0:
        logger.info("Starting ISNE model training...")
        training_results = await pipeline.train_isne_model()
        return training_results
    else:
        logger.error("No documents were successfully processed - cannot train ISNE model")
        return None

# Command-line entry point
if __name__ == "__main__":
    import argparse
    import asyncio
    import glob
    
    parser = argparse.ArgumentParser(description="Text Processing Pipeline")
    parser.add_argument("--mode", type=str, choices=["inference", "training"], default="inference",
                       help="Pipeline operation mode")
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Directory containing documents to process")
    parser.add_argument("--output_dir", type=str, default="./test-output",
                       help="Directory to save output files")
    parser.add_argument("--file_pattern", type=str, default="*.pdf",
                       help="File pattern to match documents in input directory")
    parser.add_argument("--max_tokens", type=int, default=1024,
                       help="Maximum tokens per chunk")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to use for embeddings and ISNE (cpu, cuda, cuda:0, etc.)")
    parser.add_argument("--embedding_model", type=str, default="answerdotai/ModernBERT-base",
                       help="Embedding model to use")
    parser.add_argument("--isne_model_path", type=str, default=None,
                       help="Path to pre-trained ISNE model for inference mode")
    parser.add_argument("--save_intermediate", action="store_true",
                       help="Save intermediate results from each pipeline stage")
    parser.add_argument("--num_epochs", type=int, default=100,
                       help="Number of epochs for ISNE training")
    parser.add_argument("--hidden_dim", type=int, default=256,
                       help="Hidden dimension for ISNE model")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                       help="Learning rate for ISNE training")
    
    args = parser.parse_args()
    
    # Configure options from command line arguments
    chunking_options = {
        "max_tokens": args.max_tokens,
        "doc_type": "academic_text"
    }
    
    embedding_options = {
        "adapter_name": "modernbert",
        "model_name": args.embedding_model,
        "device": args.device,
        "pooling_strategy": "cls",
    }
    
    isne_options = {
        "use_isne": True,
        "device": args.device,
        "hidden_dim": args.hidden_dim,
        "model_path": args.isne_model_path,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
    }
    
    # Find files matching pattern in input directory
    input_path = Path(args.input_dir)
    file_pattern = str(input_path / args.file_pattern)
    document_paths = sorted(glob.glob(file_pattern))
    
    if not document_paths:
        print(f"No files matching pattern '{args.file_pattern}' found in {args.input_dir}")
        sys.exit(1)
    
    print(f"Found {len(document_paths)} documents to process")
    
    # Create output directory based on mode
    output_dir = Path(args.output_dir) / args.mode
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run appropriate pipeline based on mode
    if args.mode == "inference":
        asyncio.run(run_inference_pipeline(
            document_paths=document_paths,
            output_dir=output_dir,
            chunking_options=chunking_options,
            embedding_options=embedding_options,
            isne_options=isne_options,
            save_intermediate_results=args.save_intermediate
        ))
    else:  # training mode
        asyncio.run(run_training_pipeline(
            document_paths=document_paths,
            output_dir=output_dir,
            chunking_options=chunking_options,
            embedding_options=embedding_options,
            isne_options=isne_options,
            save_intermediate_results=args.save_intermediate
        ))
        print("ISNE training completed successfully")
