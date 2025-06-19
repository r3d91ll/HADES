#!/usr/bin/env python3
"""
ISNE Bootstrap Pipeline

This script implements the complete ISNE bootstrap process that trains an ISNE model
from scratch using the pipeline: Document Processing → Chunking → Embedding → 
Graph Creation → ISNE Training → Model Saving.

This script must be run first to create a trained ISNE model before using the
graph enhancement components in the main pipeline.

Usage:
    python scripts/isne_bootstrap_pipeline.py -i ./test-data -o ./models/isne -c ./src/config/components/graph_enhancement/isne/training/config.yaml
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import traceback

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ISNEBootstrapPipeline:
    """ISNE Bootstrap Pipeline for training models from scratch."""
    
    def __init__(self, input_dir: Path, output_dir: Path, config_path: Optional[Path] = None):
        """Initialize the bootstrap pipeline."""
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.config_path = config_path
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = {
            "bootstrap_timestamp": datetime.now().isoformat(),
            "bootstrap_type": "isne_model_training",
            "stages": {},
            "model_info": {},
            "summary": {}
        }
        
        # Discovery of input files
        self.input_files = list(input_dir.glob("*.pdf"))
        logger.info(f"Found {len(self.input_files)} input files: {[f.name for f in self.input_files]}")
    
    def save_stage_results(self, stage_name: str, stage_data: Dict[str, Any]) -> None:
        """Save results for a bootstrap stage."""
        self.results["stages"][stage_name] = stage_data
        
        # Save intermediate results
        results_file = self.output_dir / f"{self.timestamp}_bootstrap_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
    
    def stage_1_document_processing(self) -> bool:
        """Stage 1: Process documents using the new component system."""
        logger.info("\\n=== Bootstrap Stage 1: Document Processing ===")
        
        try:
            # Import the new component system
            from src.components.docproc.factory import create_docproc_component
            from src.types.components.contracts import DocumentProcessingInput
            
            # Use core processor for bootstrap (most reliable)
            processor = create_docproc_component("core")
            
            documents = []
            for input_file in self.input_files:
                try:
                    doc_input = DocumentProcessingInput(
                        file_path=str(input_file),
                        processing_options={
                            "extract_metadata": True,
                            "extract_sections": True
                        },
                        metadata={"bootstrap_file": input_file.name}
                    )
                    
                    output = processor.process(doc_input)
                    documents.extend(output.documents)
                    logger.info(f"  ✓ Processed {input_file.name}: {len(output.documents)} documents")
                    
                except Exception as e:
                    logger.error(f"  ✗ Failed to process {input_file.name}: {e}")
                    return False
            
            stage_results = {
                "stage_name": "document_processing",
                "files_processed": len(self.input_files),
                "documents_generated": len(documents),
                "total_content_chars": sum(len(doc.content) for doc in documents),
                "documents": documents  # Store full documents for next stage
            }
            
            self.save_stage_results("document_processing", stage_results)
            logger.info(f"Stage 1 complete: {len(documents)} documents processed")
            return True
            
        except Exception as e:
            logger.error(f"Document processing stage failed: {e}")
            traceback.print_exc()
            return False
    
    def stage_2_chunking(self) -> bool:
        """Stage 2: Chunk documents for ISNE training."""
        logger.info("\\n=== Bootstrap Stage 2: Chunking ===")
        
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
            
            # Use core chunker for bootstrap
            chunker = create_chunking_component("core")
            
            # Process each document individually and combine chunks
            all_chunks = []
            total_chars = 0
            
            for doc in documents:
                try:
                    # Create chunking input for individual document
                    chunking_input = ChunkingInput(
                        text=doc.content,
                        document_id=doc.id,
                        chunking_strategy="semantic",  # Good for ISNE training
                        chunk_size=512,  # Optimal for embedding models
                        chunk_overlap=50,
                        processing_options={"preserve_structure": True},
                        metadata={"bootstrap_stage": "chunking", "document_id": doc.id}
                    )
                    
                    # Process chunks for this document
                    doc_output = chunker.chunk(chunking_input)
                    
                    # Convert TextChunks to DocumentChunks for consistency
                    for i, text_chunk in enumerate(doc_output.chunks):
                        doc_chunk = DocumentChunk(
                            id=f"{doc.id}_chunk_{i}",
                            content=text_chunk.text,
                            document_id=doc.id,
                            chunk_index=i,
                            chunk_size=len(text_chunk.text),
                            metadata=text_chunk.metadata
                        )
                        all_chunks.append(doc_chunk)
                    
                    total_chars += sum(len(chunk.text) for chunk in doc_output.chunks)
                    logger.info(f"  ✓ Chunked {doc.id}: {len(doc_output.chunks)} chunks")
                    
                except Exception as e:
                    logger.error(f"  ✗ Failed to chunk {doc.id}: {e}")
                    return False
            
            # Create mock output structure
            class ChunkingOutput:
                def __init__(self, chunks, total_characters):
                    self.chunks = chunks
                    self.total_characters = total_characters
            
            output = ChunkingOutput(all_chunks, total_chars)
            
            stage_results = {
                "stage_name": "chunking", 
                "input_documents": len(documents),
                "output_chunks": len(output.chunks),
                "total_characters": output.total_characters,
                "avg_chunk_size": output.total_characters / len(output.chunks) if output.chunks else 0,
                "chunks": output.chunks  # Store for next stage
            }
            
            self.save_stage_results("chunking", stage_results)
            logger.info(f"Stage 2 complete: {len(output.chunks)} chunks generated")
            return True
            
        except Exception as e:
            logger.error(f"Chunking stage failed: {e}")
            traceback.print_exc()
            return False
    
    def stage_3_embedding(self) -> bool:
        """Stage 3: Generate embeddings for ISNE training."""
        logger.info("\\n=== Bootstrap Stage 3: Embedding Generation ===")
        
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
            
            # Use CPU embedder for bootstrap (reliable)
            embedder = create_embedding_component("cpu")
            
            # Create embedding input
            embedding_input = EmbeddingInput(
                chunks=chunks,
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                embedding_options={"normalize": True},
                batch_size=16,  # Conservative for bootstrap
                metadata={"bootstrap_stage": "embedding"}
            )
            
            # Generate embeddings
            output = embedder.embed(embedding_input)
            
            # Validate embeddings quality
            valid_embeddings = [emb for emb in output.embeddings if len(emb.embedding) > 0]
            
            stage_results = {
                "stage_name": "embedding",
                "input_chunks": len(chunks),
                "output_embeddings": len(output.embeddings),
                "valid_embeddings": len(valid_embeddings),
                "embedding_dimension": valid_embeddings[0].embedding_dimension if valid_embeddings else 0,
                "processing_time": output.metadata.processing_time,
                "embeddings": output.embeddings  # Store for ISNE training
            }
            
            self.save_stage_results("embedding", stage_results)
            logger.info(f"Stage 3 complete: {len(valid_embeddings)} valid embeddings generated")
            return True
            
        except Exception as e:
            logger.error(f"Embedding stage failed: {e}")
            traceback.print_exc()
            return False
    
    def stage_4_graph_creation(self) -> bool:
        """Stage 4: Create graph structure for ISNE training."""
        logger.info("\\n=== Bootstrap Stage 4: Graph Creation ===")
        
        try:
            # Get embeddings from previous stage
            embedding_stage = self.results["stages"].get("embedding")
            if not embedding_stage:
                logger.error("Embedding stage must be completed first")
                return False
            
            embeddings = embedding_stage["embeddings"]
            if not embeddings:
                logger.error("No embeddings available for graph creation")
                return False
            
            # Import graph utilities
            from src.isne.utils.geometric_utils import create_graph_from_documents
            
            # Get documents and chunks for graph creation
            doc_stage = self.results["stages"]["document_processing"]
            chunk_stage = self.results["stages"]["chunking"]
            
            documents = doc_stage["documents"]
            chunks = chunk_stage["chunks"]
            
            # Create graph structure
            logger.info("Creating graph from documents and embeddings...")
            
            # Convert to format expected by create_graph_from_documents
            docs_with_embeddings = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                doc_data = {
                    "id": chunk.id,
                    "chunk_id": chunk.id,
                    "document_id": chunk.document_id,
                    "content": chunk.content,
                    "embedding": embedding.embedding,
                    "chunk_index": chunk.chunk_index,
                    "metadata": chunk.metadata
                }
                docs_with_embeddings.append(doc_data)
            
            # Call the function with correct signature
            graph_tuple = create_graph_from_documents(docs_with_embeddings)
            graph_data_obj, node_metadata, node_idx_map = graph_tuple
            
            # Convert to dict format expected by downstream code
            graph_data = {
                "graph_data": graph_data_obj,
                "nodes": node_metadata,
                "edges": [],  # Will be extracted from graph_data_obj if needed
                "node_idx_map": node_idx_map
            }
            
            stage_results = {
                "stage_name": "graph_creation",
                "input_embeddings": len(embeddings),
                "graph_nodes": len(graph_data.get("nodes", [])),
                "graph_edges": len(graph_data.get("edges", [])),
                "graph_connectivity": len(graph_data.get("edges", [])) / len(graph_data.get("nodes", [])) if graph_data.get("nodes") else 0,
                "graph_data": graph_data  # Store for ISNE training
            }
            
            self.save_stage_results("graph_creation", stage_results)
            logger.info(f"Stage 4 complete: Graph with {stage_results['graph_nodes']} nodes, {stage_results['graph_edges']} edges")
            return True
            
        except Exception as e:
            logger.error(f"Graph creation stage failed: {e}")
            traceback.print_exc()
            return False
    
    def stage_5_isne_training(self) -> bool:
        """Stage 5: Train ISNE model."""
        logger.info("\\n=== Bootstrap Stage 5: ISNE Model Training ===")
        
        try:
            # Get graph data from previous stage
            graph_stage = self.results["stages"].get("graph_creation")
            if not graph_stage:
                logger.error("Graph creation stage must be completed first")
                return False
            
            graph_data = graph_stage["graph_data"]
            if not graph_data:
                logger.error("No graph data available for training")
                return False
            
            # Import ISNE components
            from src.isne.models.isne_model import ISNEModel
            from src.isne.training.trainer import ISNETrainer
            
            # Get embedding dimension
            embedding_stage = self.results["stages"]["embedding"]
            embedding_dim = embedding_stage["embedding_dimension"]
            
            # Create ISNE model
            model = ISNEModel(
                in_features=embedding_dim,
                hidden_features=256,  # Good default for bootstrap
                out_features=128,     # Final embedding dimension
                num_layers=2,
                num_heads=8,
                dropout=0.1
            )
            
            # Create trainer
            trainer = ISNETrainer(
                model=model,
                learning_rate=0.001,
                weight_decay=1e-4,
                device="cpu"  # Safe default for bootstrap
            )
            
            # Train model
            logger.info("Starting ISNE model training...")
            training_results = trainer.train(
                graph_data=graph_data,
                num_epochs=50,  # Reasonable for bootstrap
                batch_size=32,
                validate_every=10
            )
            
            # Save trained model
            model_path = self.output_dir / f"isne_model_{self.timestamp}.pth"
            trainer.save_model(str(model_path))
            
            stage_results = {
                "stage_name": "isne_training",
                "model_path": str(model_path),
                "training_epochs": training_results.get("epochs_completed", 0),
                "final_loss": training_results.get("final_loss", 0.0),
                "training_time": training_results.get("training_time", 0.0),
                "model_parameters": sum(p.numel() for p in model.parameters()),
                "model_config": {
                    "in_features": embedding_dim,
                    "hidden_features": 256,
                    "out_features": 128,
                    "num_layers": 2,
                    "num_heads": 8
                }
            }
            
            self.save_stage_results("isne_training", stage_results)
            self.results["model_info"] = stage_results
            
            logger.info(f"Stage 5 complete: ISNE model trained and saved to {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"ISNE training stage failed: {e}")
            traceback.print_exc()
            return False
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate bootstrap summary."""
        stages_completed = len(self.results["stages"])
        stages_successful = len([s for s in self.results["stages"].values() 
                               if s.get("stage_name") and "error" not in s])
        
        summary = {
            "total_stages_planned": 5,
            "stages_completed": stages_completed,
            "stages_successful": stages_successful,
            "overall_success": stages_successful == 5,
            "completion_rate": stages_completed / 5,
            "success_rate": stages_successful / stages_completed if stages_completed > 0 else 0,
            "bootstrap_duration": datetime.now().isoformat(),
            "model_ready": self.results.get("model_info", {}).get("model_path") is not None
        }
        
        self.results["summary"] = summary
        return summary
    
    def run_bootstrap(self) -> bool:
        """Run the complete bootstrap pipeline."""
        logger.info("=== ISNE Bootstrap Pipeline ===")
        logger.info(f"Bootstrap timestamp: {self.timestamp}")
        logger.info(f"Input files: {[f.name for f in self.input_files]}")
        logger.info(f"Output directory: {self.output_dir}")
        
        try:
            # Stage 1: Document Processing
            if not self.stage_1_document_processing():
                logger.error("Bootstrap failed at document processing stage")
                return False
            
            # Stage 2: Chunking
            if not self.stage_2_chunking():
                logger.error("Bootstrap failed at chunking stage")
                return False
            
            # Stage 3: Embedding
            if not self.stage_3_embedding():
                logger.error("Bootstrap failed at embedding stage")
                return False
            
            # Stage 4: Graph Creation
            if not self.stage_4_graph_creation():
                logger.error("Bootstrap failed at graph creation stage")
                return False
            
            # Stage 5: ISNE Training
            if not self.stage_5_isne_training():
                logger.error("Bootstrap failed at ISNE training stage")
                return False
            
            # Generate final summary
            summary = self.generate_summary()
            
            # Save final results
            results_file = self.output_dir / f"{self.timestamp}_bootstrap_results.json"
            with open(results_file, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            
            logger.info(f"\\n=== Bootstrap Summary ===")
            logger.info(f"Stages completed: {summary['stages_completed']}/5")
            logger.info(f"Overall success: {summary['overall_success']}")
            logger.info(f"Model ready: {summary['model_ready']}")
            logger.info(f"Results saved to: {results_file}")
            
            if summary["model_ready"]:
                model_path = self.results["model_info"]["model_path"]
                logger.info(f"✓ ISNE model successfully trained and saved to: {model_path}")
                logger.info("✓ Bootstrap pipeline completed successfully!")
            
            return summary["overall_success"]
            
        except Exception as e:
            logger.error(f"Bootstrap pipeline failed: {e}")
            traceback.print_exc()
            return False


def main():
    """Main bootstrap function."""
    parser = argparse.ArgumentParser(
        description="ISNE Bootstrap Pipeline - Train ISNE models from scratch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument("-i", "--input-dir", required=True, 
                       help="Input directory containing documents to process")
    parser.add_argument("-o", "--output-dir", default="./models/isne",
                       help="Output directory for trained models and results")
    parser.add_argument("-c", "--config", 
                       help="Configuration file for training parameters")
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
    config_path = Path(args.config) if args.config else None
    
    # Run bootstrap
    bootstrap = ISNEBootstrapPipeline(input_dir, output_dir, config_path)
    success = bootstrap.run_bootstrap()
    
    if success:
        logger.info("✓ ISNE bootstrap completed successfully!")
        return True
    else:
        logger.error("✗ ISNE bootstrap failed!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)