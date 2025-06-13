#!/usr/bin/env python3
"""
Complete ISNE Bootstrap Pipeline

This script implements the complete end-to-end ISNE bootstrap pipeline:
docproc → chunking → embedding → graph construction → ISNE training

Starting from raw documents in test-data/, this pipeline processes them through
the entire HADES component stack to create a trained ISNE model.

Usage:
    python scripts/complete_isne_bootstrap_pipeline.py --input-dir ./test-data --output-dir ./models/isne
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import traceback
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CompleteISNEBootstrapPipeline:
    """Complete ISNE Bootstrap Pipeline processing documents end-to-end."""
    
    def __init__(self, input_dir: Path, output_dir: Path):
        """Initialize the complete bootstrap pipeline."""
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = {
            "bootstrap_timestamp": datetime.now().isoformat(),
            "bootstrap_type": "complete_end_to_end_isne_training",
            "pipeline_stages": ["docproc", "chunking", "embedding", "graph_construction", "isne_training"],
            "stages": {},
            "model_info": {},
            "summary": {}
        }
        
        # Find input files
        self.input_files = []
        for pattern in ["*.pdf", "*.md", "*.py", "*.yaml", "*.txt"]:
            self.input_files.extend(list(input_dir.glob(pattern)))
        
        logger.info(f"Found {len(self.input_files)} input files: {[f.name for f in self.input_files]}")
    
    def save_stage_results(self, stage_name: str, stage_data: Dict[str, Any]) -> None:
        """Save results for a bootstrap stage."""
        self.results["stages"][stage_name] = stage_data
        
        # Save intermediate results
        results_file = self.output_dir / f"{self.timestamp}_complete_bootstrap_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
    
    def stage_1_document_processing(self) -> bool:
        """Stage 1: Process documents using the document processing component."""
        logger.info("\\n=== Complete Bootstrap Stage 1: Document Processing ===")
        
        try:
            # Import the document processing component system
            from src.components.docproc.factory import create_docproc_component
            from src.types.components.contracts import DocumentProcessingInput
            
            # Use core processor for reliable processing
            processor = create_docproc_component("core")
            
            documents = []
            processing_stats = {
                "files_processed": 0,
                "documents_generated": 0,
                "total_content_chars": 0,
                "file_details": []
            }
            
            for input_file in self.input_files:
                try:
                    logger.info(f"Processing {input_file.name}...")
                    
                    doc_input = DocumentProcessingInput(
                        file_path=str(input_file),
                        processing_options={
                            "extract_metadata": True,
                            "extract_sections": True,
                            "extract_entities": True
                        },
                        metadata={"bootstrap_file": input_file.name, "stage": "docproc"}
                    )
                    
                    output = processor.process(doc_input)
                    
                    file_docs = len(output.documents)
                    file_chars = sum(len(doc.content) for doc in output.documents)
                    
                    documents.extend(output.documents)
                    processing_stats["files_processed"] += 1
                    processing_stats["documents_generated"] += file_docs
                    processing_stats["total_content_chars"] += file_chars
                    processing_stats["file_details"].append({
                        "filename": input_file.name,
                        "documents": file_docs,
                        "characters": file_chars
                    })
                    
                    logger.info(f"  ✓ {input_file.name}: {file_docs} documents, {file_chars} characters")
                    
                except Exception as e:
                    logger.error(f"  ✗ Failed to process {input_file.name}: {e}")
                    # Continue with other files rather than failing completely
                    continue
            
            if not documents:
                logger.error("No documents were successfully processed")
                return False
            
            stage_results = {
                "stage_name": "document_processing",
                "pipeline_position": 1,
                "stats": processing_stats,
                "documents": documents  # Store full documents for next stage
            }
            
            self.save_stage_results("document_processing", stage_results)
            logger.info(f"Stage 1 complete: {len(documents)} documents from {processing_stats['files_processed']} files")
            return True
            
        except Exception as e:
            logger.error(f"Document processing stage failed: {e}")
            traceback.print_exc()
            return False
    
    def stage_2_chunking(self) -> bool:
        """Stage 2: Chunk documents using the chunking component."""
        logger.info("\\n=== Complete Bootstrap Stage 2: Document Chunking ===")
        
        try:
            # Get documents from previous stage
            doc_stage = self.results["stages"].get("document_processing")
            if not doc_stage:
                logger.error("Document processing stage must be completed first")
                return False
            
            documents = doc_stage["documents"]
            if not documents:
                logger.error("No documents available for chunking")
                return False
            
            # Import chunking components
            from src.components.chunking.factory import create_chunking_component
            from src.types.components.contracts import ChunkingInput, DocumentChunk
            
            # Use core chunker for reliable chunking
            chunker = create_chunking_component("core")
            
            all_chunks = []
            chunking_stats = {
                "input_documents": len(documents),
                "output_chunks": 0,
                "total_characters": 0,
                "document_details": []
            }
            
            for doc in documents:
                try:
                    logger.info(f"Chunking document {doc.id}...")
                    
                    # Create chunking input
                    chunking_input = ChunkingInput(
                        text=doc.content,
                        document_id=doc.id,
                        chunking_strategy="semantic",  # Best for ISNE training
                        chunk_size=512,  # Optimal for embedding models
                        chunk_overlap=50,
                        processing_options={
                            "preserve_structure": True,
                            "extract_metadata": True
                        },
                        metadata={
                            "bootstrap_stage": "chunking", 
                            "document_id": doc.id,
                            "source_file": doc.metadata.get("source_file", "unknown")
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
                                "chunk_strategy": "semantic"
                            }
                        )
                        doc_chunks.append(doc_chunk)
                        all_chunks.append(doc_chunk)
                    
                    doc_chars = sum(len(chunk.content) for chunk in doc_chunks)
                    chunking_stats["output_chunks"] += len(doc_chunks)
                    chunking_stats["total_characters"] += doc_chars
                    chunking_stats["document_details"].append({
                        "document_id": doc.id,
                        "chunks_created": len(doc_chunks),
                        "characters": doc_chars
                    })
                    
                    logger.info(f"  ✓ {doc.id}: {len(doc_chunks)} chunks, {doc_chars} characters")
                    
                except Exception as e:
                    logger.error(f"  ✗ Failed to chunk {doc.id}: {e}")
                    continue
            
            if not all_chunks:
                logger.error("No chunks were successfully created")
                return False
            
            stage_results = {
                "stage_name": "chunking",
                "pipeline_position": 2,
                "stats": chunking_stats,
                "chunks": all_chunks  # Store for next stage
            }
            
            self.save_stage_results("chunking", stage_results)
            logger.info(f"Stage 2 complete: {len(all_chunks)} chunks from {len(documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Chunking stage failed: {e}")
            traceback.print_exc()
            return False
    
    def stage_3_embedding(self) -> bool:
        """Stage 3: Generate embeddings using the embedding component."""
        logger.info("\\n=== Complete Bootstrap Stage 3: Embedding Generation ===")
        
        try:
            # Get chunks from previous stage
            chunk_stage = self.results["stages"].get("chunking")
            if not chunk_stage:
                logger.error("Chunking stage must be completed first")
                return False
            
            chunks = chunk_stage["chunks"]
            if not chunks:
                logger.error("No chunks available for embedding")
                return False
            
            # Import embedding components
            from src.components.embedding.factory import create_embedding_component
            from src.types.components.contracts import EmbeddingInput
            
            # Use CPU embedder for reliable processing
            embedder = create_embedding_component("cpu")
            
            logger.info(f"Generating embeddings for {len(chunks)} chunks...")
            
            # Create embedding input
            embedding_input = EmbeddingInput(
                chunks=chunks,
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                embedding_options={
                    "normalize": True,
                    "batch_size": 32
                },
                batch_size=32,
                metadata={"bootstrap_stage": "embedding", "pipeline": "complete_isne"}
            )
            
            # Generate embeddings
            output = embedder.embed(embedding_input)
            
            # Validate embeddings quality
            valid_embeddings = [emb for emb in output.embeddings if len(emb.embedding) > 0]
            
            embedding_stats = {
                "input_chunks": len(chunks),
                "output_embeddings": len(output.embeddings),
                "valid_embeddings": len(valid_embeddings),
                "embedding_dimension": valid_embeddings[0].embedding_dimension if valid_embeddings else 0,
                "processing_time": output.metadata.processing_time,
                "model_info": {
                    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                    "embedding_dimension": valid_embeddings[0].embedding_dimension if valid_embeddings else 0
                }
            }
            
            stage_results = {
                "stage_name": "embedding",
                "pipeline_position": 3,
                "stats": embedding_stats,
                "embeddings": output.embeddings  # Store for graph construction
            }
            
            self.save_stage_results("embedding", stage_results)
            logger.info(f"Stage 3 complete: {len(valid_embeddings)} embeddings generated")
            return True
            
        except Exception as e:
            logger.error(f"Embedding stage failed: {e}")
            traceback.print_exc()
            return False
    
    def stage_4_graph_construction(self) -> bool:
        """Stage 4: Construct graph from embeddings."""
        logger.info("\\n=== Complete Bootstrap Stage 4: Graph Construction ===")
        
        try:
            # Get embeddings from previous stage
            embedding_stage = self.results["stages"].get("embedding")
            if not embedding_stage:
                logger.error("Embedding stage must be completed first")
                return False
            
            embeddings = embedding_stage["embeddings"]
            if not embeddings:
                logger.error("No embeddings available for graph construction")
                return False
            
            logger.info(f"Constructing graph from {len(embeddings)} embeddings...")
            
            # Create node features and metadata
            node_features = []
            node_metadata = []
            chunk_stage = self.results["stages"]["chunking"]
            chunks = chunk_stage["chunks"]
            
            # Build mapping from chunk_id to chunk data
            chunk_map = {chunk.id: chunk for chunk in chunks}
            
            for i, emb_data in enumerate(embeddings):
                node_features.append(emb_data.embedding)
                
                # Get corresponding chunk data
                chunk_id = emb_data.chunk_id
                chunk_data = chunk_map.get(chunk_id, {})
                
                node_metadata.append({
                    "node_id": i,
                    "chunk_id": chunk_id,
                    "document_id": getattr(chunk_data, 'document_id', f'unknown_doc_{i}'),
                    "text": getattr(chunk_data, 'content', '')[:200],  # Truncate for storage
                    "embedding_model": "all-MiniLM-L6-v2",
                    "metadata": getattr(chunk_data, 'metadata', {})
                })
            
            # Convert to PyTorch tensors
            node_features_tensor = torch.tensor(node_features, dtype=torch.float)
            num_nodes = len(node_features)
            
            logger.info(f"Created {num_nodes} nodes with {node_features_tensor.size(1)}-dimensional features")
            
            # Create graph edges
            edge_index_src = []
            edge_index_dst = []
            
            # 1. Sequential edges (document flow)
            doc_chunks = {}
            for i, meta in enumerate(node_metadata):
                doc_id = meta["document_id"]
                if doc_id not in doc_chunks:
                    doc_chunks[doc_id] = []
                doc_chunks[doc_id].append(i)
            
            # Add sequential edges within each document
            for doc_id, chunk_indices in doc_chunks.items():
                chunk_indices.sort()  # Ensure proper order
                for i in range(len(chunk_indices) - 1):
                    src_idx = chunk_indices[i]
                    dst_idx = chunk_indices[i + 1]
                    # Add bidirectional edges
                    edge_index_src.extend([src_idx, dst_idx])
                    edge_index_dst.extend([dst_idx, src_idx])
            
            # 2. Similarity-based edges (using embedding similarity)
            import random
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np
            
            # Sample a subset for similarity calculation to avoid O(n²) complexity
            max_similarity_nodes = min(500, num_nodes)
            if num_nodes > max_similarity_nodes:
                similarity_indices = random.sample(range(num_nodes), max_similarity_nodes)
            else:
                similarity_indices = list(range(num_nodes))
            
            # Calculate similarities for sampled nodes
            sampled_features = node_features_tensor[similarity_indices].numpy()
            similarities = cosine_similarity(sampled_features)
            
            # Add high-similarity edges
            similarity_threshold = 0.8
            for i in range(len(similarity_indices)):
                for j in range(i + 1, len(similarity_indices)):
                    if similarities[i, j] > similarity_threshold:
                        actual_i = similarity_indices[i]
                        actual_j = similarity_indices[j]
                        edge_index_src.extend([actual_i, actual_j])
                        edge_index_dst.extend([actual_j, actual_i])
            
            # 3. Random edges for connectivity
            num_random_edges = min(200, num_nodes // 5)
            for _ in range(num_random_edges):
                src = random.randint(0, num_nodes - 1)
                dst = random.randint(0, num_nodes - 1)
                if src != dst:
                    edge_index_src.append(src)
                    edge_index_dst.append(dst)
            
            # Create final edge index tensor
            edge_index = torch.tensor([edge_index_src, edge_index_dst], dtype=torch.long)
            
            graph_stats = {
                "num_nodes": num_nodes,
                "num_edges": len(edge_index_src),
                "embedding_dimension": node_features_tensor.size(1),
                "documents_represented": len(doc_chunks),
                "edge_types": {
                    "sequential": len([i for doc_chunks_list in doc_chunks.values() for i in range(len(doc_chunks_list) - 1)]) * 2,
                    "similarity": len([i for i in range(len(similarity_indices)) for j in range(i + 1, len(similarity_indices)) if similarities[i, j] > similarity_threshold]) * 2,
                    "random": num_random_edges
                }
            }
            
            stage_results = {
                "stage_name": "graph_construction",
                "pipeline_position": 4,
                "stats": graph_stats,
                "node_features": node_features_tensor,
                "edge_index": edge_index,
                "node_metadata": node_metadata
            }
            
            self.save_stage_results("graph_construction", stage_results)
            logger.info(f"Stage 4 complete: Graph with {num_nodes} nodes, {len(edge_index_src)} edges")
            return True
            
        except Exception as e:
            logger.error(f"Graph construction stage failed: {e}")
            traceback.print_exc()
            return False
    
    def stage_5_isne_training(self) -> bool:
        """Stage 5: Train ISNE model on the constructed graph."""
        logger.info("\\n=== Complete Bootstrap Stage 5: ISNE Model Training ===")
        
        try:
            # Get graph data from previous stage
            graph_stage = self.results["stages"].get("graph_construction")
            if not graph_stage:
                logger.error("Graph construction stage must be completed first")
                return False
            
            node_features = graph_stage["node_features"]
            edge_index = graph_stage["edge_index"]
            
            if node_features is None or edge_index is None:
                logger.error("No graph data available for ISNE training")
                return False
            
            # Import ISNE trainer
            from src.isne.training.trainer import ISNETrainer
            
            # Get model configuration
            embedding_dim = node_features.size(1)
            hidden_dim = 256
            output_dim = 128
            
            logger.info(f"Training ISNE model: {embedding_dim} → {hidden_dim} → {output_dim}")
            
            # Create trainer
            trainer = ISNETrainer(
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                num_layers=3,  # Slightly deeper for better representation
                num_heads=8,
                dropout=0.1,
                learning_rate=0.001,
                weight_decay=1e-4,
                lambda_feat=1.0,      # Feature preservation
                lambda_struct=1.0,    # Structural preservation  
                lambda_contrast=0.5,  # Contrastive learning
                device="cpu"
            )
            
            # Prepare the model
            trainer.prepare_model()
            
            # Train the model
            logger.info("Starting ISNE model training...")
            training_results = trainer.train(
                features=node_features,
                edge_index=edge_index,
                epochs=50,  # More epochs for better convergence
                batch_size=64,
                num_hops=2,
                neighbor_size=15,
                eval_interval=10,
                early_stopping_patience=15,
                verbose=True
            )
            
            # Save the trained model
            model_path = self.output_dir / f"complete_isne_model_{self.timestamp}.pth"
            trainer.save_model(str(model_path))
            
            # Calculate model statistics
            total_params = sum(p.numel() for p in trainer.model.parameters()) if trainer.model else 0
            trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad) if trainer.model else 0
            
            training_stats = {
                "model_path": str(model_path),
                "training_epochs": training_results.get("epochs", 0),
                "final_losses": {
                    "total": training_results.get("total_loss", [])[-1] if training_results.get("total_loss") else 0.0,
                    "feature": training_results.get("feature_loss", [])[-1] if training_results.get("feature_loss") else 0.0,
                    "structural": training_results.get("structural_loss", [])[-1] if training_results.get("structural_loss") else 0.0,
                    "contrastive": training_results.get("contrastive_loss", [])[-1] if training_results.get("contrastive_loss") else 0.0
                },
                "model_architecture": {
                    "input_dim": embedding_dim,
                    "hidden_dim": hidden_dim,
                    "output_dim": output_dim,
                    "num_layers": 3,
                    "num_heads": 8,
                    "total_parameters": total_params,
                    "trainable_parameters": trainable_params
                },
                "training_config": {
                    "epochs": 50,
                    "batch_size": 64,
                    "learning_rate": 0.001,
                    "weight_decay": 1e-4,
                    "loss_weights": {
                        "feature": 1.0,
                        "structural": 1.0,
                        "contrastive": 0.5
                    }
                }
            }
            
            stage_results = {
                "stage_name": "isne_training",
                "pipeline_position": 5,
                "stats": training_stats
            }
            
            self.save_stage_results("isne_training", stage_results)
            self.results["model_info"] = training_stats
            
            logger.info(f"Stage 5 complete: ISNE model trained and saved to {model_path}")
            logger.info(f"Model: {total_params:,} parameters, Final loss: {training_stats['final_losses']['total']:.4f}")
            return True
            
        except Exception as e:
            logger.error(f"ISNE training stage failed: {e}")
            traceback.print_exc()
            return False
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate complete pipeline summary."""
        stages_completed = len(self.results["stages"])
        stages_successful = len([s for s in self.results["stages"].values() 
                               if s.get("stage_name") and "error" not in s])
        
        # Calculate pipeline metrics
        pipeline_metrics = {}
        if "document_processing" in self.results["stages"]:
            doc_stage = self.results["stages"]["document_processing"]
            pipeline_metrics["documents"] = doc_stage["stats"]["documents_generated"]
            pipeline_metrics["input_files"] = doc_stage["stats"]["files_processed"]
        
        if "chunking" in self.results["stages"]:
            chunk_stage = self.results["stages"]["chunking"]
            pipeline_metrics["chunks"] = chunk_stage["stats"]["output_chunks"]
        
        if "embedding" in self.results["stages"]:
            emb_stage = self.results["stages"]["embedding"]
            pipeline_metrics["embeddings"] = emb_stage["stats"]["valid_embeddings"]
            pipeline_metrics["embedding_dimension"] = emb_stage["stats"]["embedding_dimension"]
        
        if "graph_construction" in self.results["stages"]:
            graph_stage = self.results["stages"]["graph_construction"]
            pipeline_metrics["graph_nodes"] = graph_stage["stats"]["num_nodes"]
            pipeline_metrics["graph_edges"] = graph_stage["stats"]["num_edges"]
        
        if "isne_training" in self.results["stages"]:
            train_stage = self.results["stages"]["isne_training"]
            pipeline_metrics["model_parameters"] = train_stage["stats"]["model_architecture"]["total_parameters"]
            pipeline_metrics["training_epochs"] = train_stage["stats"]["training_epochs"]
        
        summary = {
            "total_stages_planned": 5,
            "stages_completed": stages_completed,
            "stages_successful": stages_successful,
            "overall_success": stages_successful == 5,
            "completion_rate": stages_completed / 5,
            "success_rate": stages_successful / stages_completed if stages_completed > 0 else 0,
            "pipeline_metrics": pipeline_metrics,
            "model_ready": self.results.get("model_info", {}).get("model_path") is not None,
            "bootstrap_duration": datetime.now().isoformat()
        }
        
        self.results["summary"] = summary
        return summary
    
    def run_complete_bootstrap(self) -> bool:
        """Run the complete end-to-end bootstrap pipeline."""
        logger.info("=== COMPLETE ISNE BOOTSTRAP PIPELINE ===")
        logger.info("Pipeline: docproc → chunking → embedding → graph construction → ISNE training")
        logger.info(f"Bootstrap timestamp: {self.timestamp}")
        logger.info(f"Input files: {[f.name for f in self.input_files]}")
        logger.info(f"Output directory: {self.output_dir}")
        
        try:
            # Stage 1: Document Processing
            if not self.stage_1_document_processing():
                logger.error("Complete bootstrap failed at document processing stage")
                return False
            
            # Stage 2: Chunking
            if not self.stage_2_chunking():
                logger.error("Complete bootstrap failed at chunking stage")
                return False
            
            # Stage 3: Embedding
            if not self.stage_3_embedding():
                logger.error("Complete bootstrap failed at embedding stage")
                return False
            
            # Stage 4: Graph Construction
            if not self.stage_4_graph_construction():
                logger.error("Complete bootstrap failed at graph construction stage")
                return False
            
            # Stage 5: ISNE Training
            if not self.stage_5_isne_training():
                logger.error("Complete bootstrap failed at ISNE training stage")
                return False
            
            # Generate final summary
            summary = self.generate_summary()
            
            # Save final results
            results_file = self.output_dir / f"{self.timestamp}_complete_bootstrap_results.json"
            with open(results_file, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            
            logger.info(f"\\n=== COMPLETE BOOTSTRAP SUMMARY ===")
            logger.info(f"Pipeline Success: {summary['overall_success']}")
            logger.info(f"Stages Completed: {summary['stages_completed']}/5")
            
            if "pipeline_metrics" in summary:
                metrics = summary["pipeline_metrics"]
                logger.info(f"\\nPipeline Flow:")
                logger.info(f"  {metrics.get('input_files', 0)} input files")
                logger.info(f"  → {metrics.get('documents', 0)} documents")
                logger.info(f"  → {metrics.get('chunks', 0)} chunks") 
                logger.info(f"  → {metrics.get('embeddings', 0)} embeddings ({metrics.get('embedding_dimension', 0)}D)")
                logger.info(f"  → Graph: {metrics.get('graph_nodes', 0)} nodes, {metrics.get('graph_edges', 0)} edges")
                logger.info(f"  → ISNE Model: {metrics.get('model_parameters', 0):,} parameters, {metrics.get('training_epochs', 0)} epochs")
            
            logger.info(f"\\nResults saved to: {results_file}")
            
            if summary["model_ready"]:
                model_path = self.results["model_info"]["model_path"]
                logger.info(f"✓ Complete ISNE model successfully trained: {model_path}")
                logger.info("✓ End-to-end bootstrap pipeline completed successfully!")
                logger.info("✓ ISNE model ready for integration with HADES RAG system")
            
            return summary["overall_success"]
            
        except Exception as e:
            logger.error(f"Complete bootstrap pipeline failed: {e}")
            traceback.print_exc()
            return False


def main():
    """Main function for complete bootstrap pipeline."""
    parser = argparse.ArgumentParser(
        description="Complete ISNE Bootstrap Pipeline - End-to-end training from documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument("--input-dir", default="./test-data",
                       help="Input directory containing documents to process")
    parser.add_argument("--output-dir", default="./models/isne",
                       help="Output directory for trained models and results")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Validate paths
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return False
    
    output_dir = Path(args.output_dir)
    
    # Run complete bootstrap
    bootstrap = CompleteISNEBootstrapPipeline(input_dir, output_dir)
    success = bootstrap.run_complete_bootstrap()
    
    if success:
        logger.info("✓ Complete ISNE bootstrap pipeline completed successfully!")
        return True
    else:
        logger.error("✗ Complete ISNE bootstrap pipeline failed!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)