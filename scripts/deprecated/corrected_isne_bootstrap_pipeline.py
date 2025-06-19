#!/usr/bin/env python3
"""
Corrected ISNE Bootstrap Pipeline

This script implements a corrected version of the ISNE bootstrap process that trains an ISNE model
from our existing embedding data. It uses the embedding test results we've already generated
to create a trained ISNE model.

Usage:
    python scripts/corrected_isne_bootstrap_pipeline.py --embedding-data ./test-out/embeddings/ --output-dir ./models/isne
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

class CorrectedISNEBootstrapPipeline:
    """Corrected ISNE Bootstrap Pipeline using existing embedding data."""
    
    def __init__(self, embedding_data_dir: Path, output_dir: Path):
        """Initialize the corrected bootstrap pipeline."""
        self.embedding_data_dir = embedding_data_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = {
            "bootstrap_timestamp": datetime.now().isoformat(),
            "bootstrap_type": "corrected_isne_model_training",
            "data_source": "existing_embedding_files",
            "stages": {},
            "model_info": {},
            "summary": {}
        }
        
        # Find embedding files
        self.embedding_files = list(embedding_data_dir.glob("*.json"))
        logger.info(f"Found {len(self.embedding_files)} embedding files")
    
    def save_stage_results(self, stage_name: str, stage_data: Dict[str, Any]) -> None:
        """Save results for a bootstrap stage."""
        self.results["stages"][stage_name] = stage_data
        
        # Save intermediate results
        results_file = self.output_dir / f"{self.timestamp}_bootstrap_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
    
    def stage_1_load_embeddings(self) -> bool:
        """Stage 1: Load embeddings from existing files."""
        logger.info("\\n=== Bootstrap Stage 1: Load Existing Embeddings ===")
        
        try:
            all_embeddings = []
            total_chunks = 0
            
            for embedding_file in self.embedding_files:
                try:
                    with open(embedding_file, 'r') as f:
                        embedding_data = json.load(f)
                    
                    # Extract embeddings from the file
                    if "embeddings" in embedding_data:
                        embeddings = embedding_data["embeddings"]
                        all_embeddings.extend(embeddings)
                        total_chunks += len(embeddings)
                        logger.info(f"  ✓ Loaded {len(embeddings)} embeddings from {embedding_file.name}")
                    
                except Exception as e:
                    logger.error(f"  ✗ Failed to load {embedding_file.name}: {e}")
                    continue
            
            if not all_embeddings:
                logger.error("No embeddings found in any files")
                return False
            
            # Get embedding dimension from first valid embedding
            embedding_dim = len(all_embeddings[0]["embedding"]) if all_embeddings else 0
            
            stage_results = {
                "stage_name": "load_embeddings",
                "files_processed": len(self.embedding_files),
                "total_embeddings": len(all_embeddings),
                "embedding_dimension": embedding_dim,
                "embeddings": all_embeddings  # Store for next stage
            }
            
            self.save_stage_results("load_embeddings", stage_results)
            logger.info(f"Stage 1 complete: {len(all_embeddings)} embeddings loaded")
            return True
            
        except Exception as e:
            logger.error(f"Load embeddings stage failed: {e}")
            traceback.print_exc()
            return False
    
    def stage_2_create_graph_data(self) -> bool:
        """Stage 2: Create PyTorch Geometric graph data."""
        logger.info("\\n=== Bootstrap Stage 2: Create Graph Data ===")
        
        try:
            # Get embeddings from previous stage
            embedding_stage = self.results["stages"].get("load_embeddings")
            if not embedding_stage:
                logger.error("Load embeddings stage must be completed first")
                return False
            
            embeddings = embedding_stage["embeddings"]
            if not embeddings:
                logger.error("No embeddings available for graph creation")
                return False
            
            # Create node features tensor
            node_features = []
            node_metadata = []
            
            for i, emb_data in enumerate(embeddings):
                node_features.append(emb_data["embedding"])
                node_metadata.append({
                    "chunk_id": emb_data.get("chunk_id", f"chunk_{i}"),
                    "text": emb_data.get("text", ""),
                    "metadata": emb_data.get("metadata", {})
                })
            
            # Convert to PyTorch tensor
            node_features_tensor = torch.tensor(node_features, dtype=torch.float)
            num_nodes = len(node_features)
            
            # Create simple sequential edges (each chunk connects to next)
            edge_index_src = []
            edge_index_dst = []
            
            for i in range(num_nodes - 1):
                edge_index_src.append(i)
                edge_index_dst.append(i + 1)
                # Add reverse edge for undirected graph
                edge_index_src.append(i + 1)
                edge_index_dst.append(i)
            
            # Add some random edges for better connectivity
            import random
            num_random_edges = min(100, num_nodes // 4)
            for _ in range(num_random_edges):
                src = random.randint(0, num_nodes - 1)
                dst = random.randint(0, num_nodes - 1)
                if src != dst:
                    edge_index_src.append(src)
                    edge_index_dst.append(dst)
            
            edge_index = torch.tensor([edge_index_src, edge_index_dst], dtype=torch.long)
            
            stage_results = {
                "stage_name": "create_graph_data",
                "num_nodes": num_nodes,
                "num_edges": len(edge_index_src),
                "node_features": node_features_tensor,
                "edge_index": edge_index,
                "node_metadata": node_metadata
            }
            
            self.save_stage_results("create_graph_data", stage_results)
            logger.info(f"Stage 2 complete: Graph with {num_nodes} nodes, {len(edge_index_src)} edges")
            return True
            
        except Exception as e:
            logger.error(f"Graph creation stage failed: {e}")
            traceback.print_exc()
            return False
    
    def stage_3_train_isne_model(self) -> bool:
        """Stage 3: Train ISNE model using corrected interface."""
        logger.info("\\n=== Bootstrap Stage 3: Train ISNE Model ===")
        
        try:
            # Get graph data from previous stage
            graph_stage = self.results["stages"].get("create_graph_data")
            if not graph_stage:
                logger.error("Graph creation stage must be completed first")
                return False
            
            node_features = graph_stage["node_features"]
            edge_index = graph_stage["edge_index"]
            
            if node_features is None or edge_index is None:
                logger.error("No graph data available for training")
                return False
            
            # Import ISNE trainer
            from src.isne.training.trainer import ISNETrainer
            
            # Get embedding dimension
            embedding_dim = node_features.size(1)
            logger.info(f"Training ISNE model with embedding dimension: {embedding_dim}")
            
            # Create trainer with corrected interface
            trainer = ISNETrainer(
                embedding_dim=embedding_dim,
                hidden_dim=256,
                output_dim=128,
                num_layers=2,
                num_heads=8,
                dropout=0.1,
                learning_rate=0.001,
                weight_decay=1e-4,
                device="cpu"  # Use CPU for bootstrap
            )
            
            # Prepare the model
            trainer.prepare_model()
            
            # Train the model
            logger.info("Starting ISNE model training...")
            training_results = trainer.train(
                features=node_features,
                edge_index=edge_index,
                epochs=20,  # Reduced for faster bootstrap
                batch_size=32,
                num_hops=1,
                neighbor_size=10,
                eval_interval=5,
                verbose=True
            )
            
            # Save the trained model
            model_path = self.output_dir / f"isne_model_{self.timestamp}.pth"
            trainer.save_model(str(model_path))
            
            stage_results = {
                "stage_name": "train_isne_model",
                "model_path": str(model_path),
                "training_epochs": training_results.get("epochs", 0),
                "final_loss": training_results.get("total_loss", [])[-1] if training_results.get("total_loss") else 0.0,
                "model_parameters": sum(p.numel() for p in trainer.model.parameters()) if trainer.model else 0,
                "model_config": {
                    "embedding_dim": embedding_dim,
                    "hidden_dim": 256,
                    "output_dim": 128,
                    "num_layers": 2,
                    "num_heads": 8
                }
            }
            
            self.save_stage_results("train_isne_model", stage_results)
            self.results["model_info"] = stage_results
            
            logger.info(f"Stage 3 complete: ISNE model trained and saved to {model_path}")
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
            "total_stages_planned": 3,
            "stages_completed": stages_completed,
            "stages_successful": stages_successful,
            "overall_success": stages_successful == 3,
            "completion_rate": stages_completed / 3,
            "success_rate": stages_successful / stages_completed if stages_completed > 0 else 0,
            "bootstrap_duration": datetime.now().isoformat(),
            "model_ready": self.results.get("model_info", {}).get("model_path") is not None
        }
        
        self.results["summary"] = summary
        return summary
    
    def run_bootstrap(self) -> bool:
        """Run the complete corrected bootstrap pipeline."""
        logger.info("=== Corrected ISNE Bootstrap Pipeline ===")
        logger.info(f"Bootstrap timestamp: {self.timestamp}")
        logger.info(f"Input embedding files: {len(self.embedding_files)}")
        logger.info(f"Output directory: {self.output_dir}")
        
        try:
            # Stage 1: Load embeddings
            if not self.stage_1_load_embeddings():
                logger.error("Bootstrap failed at load embeddings stage")
                return False
            
            # Stage 2: Create graph data
            if not self.stage_2_create_graph_data():
                logger.error("Bootstrap failed at graph creation stage")
                return False
            
            # Stage 3: Train ISNE model
            if not self.stage_3_train_isne_model():
                logger.error("Bootstrap failed at ISNE training stage")
                return False
            
            # Generate final summary
            summary = self.generate_summary()
            
            # Save final results
            results_file = self.output_dir / f"{self.timestamp}_bootstrap_results.json"
            with open(results_file, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            
            logger.info(f"\\n=== Bootstrap Summary ===")
            logger.info(f"Stages completed: {summary['stages_completed']}/3")
            logger.info(f"Overall success: {summary['overall_success']}")
            logger.info(f"Model ready: {summary['model_ready']}")
            logger.info(f"Results saved to: {results_file}")
            
            if summary["model_ready"]:
                model_path = self.results["model_info"]["model_path"]
                logger.info(f"✓ ISNE model successfully trained and saved to: {model_path}")
                logger.info("✓ Corrected bootstrap pipeline completed successfully!")
            
            return summary["overall_success"]
            
        except Exception as e:
            logger.error(f"Bootstrap pipeline failed: {e}")
            traceback.print_exc()
            return False


def main():
    """Main bootstrap function."""
    parser = argparse.ArgumentParser(
        description="Corrected ISNE Bootstrap Pipeline - Train ISNE models from existing embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument("--embedding-data", required=True, 
                       help="Directory containing embedding files")
    parser.add_argument("--output-dir", default="./models/isne",
                       help="Output directory for trained models and results")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Validate paths
    embedding_data_dir = Path(args.embedding_data)
    if not embedding_data_dir.exists():
        logger.error(f"Embedding data directory not found: {embedding_data_dir}")
        return False
    
    output_dir = Path(args.output_dir)
    
    # Run bootstrap
    bootstrap = CorrectedISNEBootstrapPipeline(embedding_data_dir, output_dir)
    success = bootstrap.run_bootstrap()
    
    if success:
        logger.info("✓ Corrected ISNE bootstrap completed successfully!")
        return True
    else:
        logger.error("✗ Corrected ISNE bootstrap failed!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)