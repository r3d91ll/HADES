#!/usr/bin/env python3
"""
GPU-Optimized ISNE Training from ArangoDB.

This script is optimized for maximum GPU utilization during ISNE training.
"""

import argparse
import logging
import sys
import time
import json
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.arango_client import ArangoClient
from src.isne.models.isne_model import ISNEModel
from src.isne.training.trainer import ISNETrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HighPerformanceArangoISNETrainer:
    """High-performance ISNE trainer optimized for GPU utilization."""
    
    def __init__(self, db_name: str):
        self.db_name = db_name
        self.arango_client = ArangoClient()
        
    def connect_to_database(self):
        """Connect to ArangoDB database."""
        success = self.arango_client.connect_to_database(self.db_name)
        if not success:
            raise Exception(f"Failed to connect to database: {self.db_name}")
        return self.arango_client._database
    
    def extract_graph_data_optimized(self, db) -> Dict[str, Any]:
        """Extract graph data from ArangoDB with optimized queries."""
        logger.info("Extracting graph data from ArangoDB (optimized)...")
        
        # Get all embeddings with chunk info in a single optimized query
        embeddings_query = """
        FOR e IN embeddings
            LET chunk = DOCUMENT(e.chunk_id)
            FILTER chunk != null
            RETURN {
                chunk_id: e.chunk_id,
                embedding: e.embedding,
                source_file: chunk.source_file_id,
                chunk_index: chunk.chunk_index,
                _key: chunk._key
            }
        """
        
        cursor = db.aql.execute(embeddings_query)
        embeddings_data = list(cursor)
        
        logger.info(f"Found {len(embeddings_data)} embeddings")
        
        # Create nodes from embeddings
        nodes = []
        chunk_to_node_id = {}
        chunk_key_to_full_id = {}
        
        for i, embedding_data in enumerate(embeddings_data):
            chunk_full_id = embedding_data['chunk_id']
            chunk_key = embedding_data['_key']
            
            nodes.append({
                'id': i,
                'chunk_id': chunk_full_id,
                'embedding': embedding_data['embedding'],
                'source_file': embedding_data['source_file'],
                'chunk_index': embedding_data['chunk_index']
            })
            chunk_to_node_id[chunk_full_id] = i
            chunk_key_to_full_id[chunk_key] = chunk_full_id
        
        # Get all edges in optimized batches
        edges = []
        edge_count = 0
        
        # Sequential edges (high volume, process efficiently)
        sequential_query = """
        FOR e IN sequential_edges
            RETURN {
                source: e._from,
                target: e._to,
                weight: NOTNULL(e.weight, 1.0),
                type: "sequential"
            }
        """
        
        cursor = db.aql.execute(sequential_query)
        for edge in cursor:
            source_id = chunk_to_node_id.get(edge['source'])
            target_id = chunk_to_node_id.get(edge['target'])
            
            if source_id is not None and target_id is not None:
                edges.append({
                    'source': source_id,
                    'target': target_id,
                    'weight': edge['weight'],
                    'type': edge['type']
                })
                edge_count += 1
        
        logger.info(f"Added {edge_count} sequential edges")
        
        # Similarity edges (high quality, preserve all)
        similarity_query = """
        FOR e IN similarity_edges
            RETURN {
                source: e._from,
                target: e._to,
                weight: NOTNULL(e.similarity, e.weight, 0.8),
                type: "similarity"
            }
        """
        
        similarity_count = 0
        cursor = db.aql.execute(similarity_query)
        for edge in cursor:
            source_id = chunk_to_node_id.get(edge['source'])
            target_id = chunk_to_node_id.get(edge['target'])
            
            if source_id is not None and target_id is not None:
                edges.append({
                    'source': source_id,
                    'target': target_id,
                    'weight': edge['weight'],
                    'type': edge['type']
                })
                similarity_count += 1
        
        logger.info(f"Added {similarity_count} similarity edges")
        
        # Cross-modal edges (connect different file types)
        cross_modal_query = """
        FOR e IN cross_modal_edges
            RETURN {
                source: e._from,
                target: e._to,
                weight: NOTNULL(e.confidence, e.weight, 0.5),
                type: "cross_modal"
            }
        """
        
        cross_modal_count = 0
        cursor = db.aql.execute(cross_modal_query)
        for edge in cursor:
            # For file-level edges, we need to connect their chunks
            source_chunks = [node for node in nodes if node['source_file'] == edge['source']]
            target_chunks = [node for node in nodes if node['source_file'] == edge['target']]
            
            # Create edges between chunks (sample to avoid explosion)
            for source_chunk in source_chunks[:5]:  # Limit to first 5 chunks per file
                for target_chunk in target_chunks[:5]:
                    edges.append({
                        'source': source_chunk['id'],
                        'target': target_chunk['id'],
                        'weight': edge['weight'],
                        'type': edge['type']
                    })
                    cross_modal_count += 1
        
        logger.info(f"Added {cross_modal_count} cross-modal edges")
        
        logger.info(f"Created optimized graph with {len(nodes)} nodes and {len(edges)} edges")
        
        return {
            'nodes': nodes,
            'edges': edges,
            'metadata': {
                'db_name': self.db_name,
                'extraction_timestamp': datetime.now().isoformat(),
                'num_embeddings': len(embeddings_data),
                'edge_types': {
                    'sequential': edge_count,
                    'similarity': similarity_count,
                    'cross_modal': cross_modal_count
                }
            }
        }
    
    def create_optimized_data_loaders(
        self,
        edge_index: torch.Tensor,
        batch_size: int,
        device: torch.device
    ) -> List[torch.Tensor]:
        """Create optimized data loaders for GPU training."""
        
        num_edges = edge_index.shape[1]
        logger.info(f"Creating optimized batches for {num_edges} edges with batch size {batch_size}")
        
        # Create edge batches
        edge_batches = []
        for i in range(0, num_edges, batch_size):
            end_idx = min(i + batch_size, num_edges)
            batch_edges = edge_index[:, i:end_idx].to(device)
            edge_batches.append(batch_edges)
        
        logger.info(f"Created {len(edge_batches)} edge batches")
        return edge_batches
    
    def train_isne_model_optimized(
        self,
        graph_data: Dict[str, Any],
        output_dir: Path,
        epochs: int = 100,
        learning_rate: float = 0.001,
        batch_size: int = 2048,  # Much larger batch size
        hidden_dim: int = 512,   # Larger model for GPU utilization
        num_layers: int = 6,     # Deeper model
        device: str = "auto"
    ) -> Dict[str, Any]:
        """Train ISNE model with optimized GPU utilization."""
        
        nodes = graph_data['nodes']
        edges = graph_data['edges']
        
        logger.info(f"Training optimized ISNE model on {len(nodes)} nodes and {len(edges)} edges")
        
        # Set device and enable optimizations
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)
        
        if device.type == "cuda":
            # Enable GPU optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            logger.info(f"GPU optimizations enabled on {torch.cuda.get_device_name()}")
            logger.info(f"GPU memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
        
        # Prepare node features with memory pinning for faster GPU transfer
        node_features = torch.tensor([node['embedding'] for node in nodes], dtype=torch.float32)
        if device.type == "cuda":
            node_features = node_features.pin_memory()
        
        # Prepare edges efficiently
        edge_list = [(edge['source'], edge['target']) for edge in edges]
        edge_weights = [edge['weight'] for edge in edges]
        
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_weights = torch.tensor(edge_weights, dtype=torch.float32)
            
            if device.type == "cuda":
                edge_index = edge_index.pin_memory()
                edge_weights = edge_weights.pin_memory()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_weights = torch.empty(0, dtype=torch.float32)
        
        # Initialize optimized trainer
        input_dim = node_features.shape[1]
        num_nodes = len(nodes)
        
        logger.info(f"Initializing large model: {input_dim}D → {hidden_dim}D, {num_layers} layers")
        
        trainer = ISNETrainer(
            embedding_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            learning_rate=learning_rate,
            device=device
        )
        
        # Initialize model
        trainer._initialize_model(num_nodes)
        
        model_params = sum(p.numel() for p in trainer.model.parameters())
        model_size_mb = model_params * 4 / 1024 / 1024  # Float32 in MB
        
        logger.info(f"Model initialized with {model_params:,} parameters ({model_size_mb:.1f} MB)")
        
        # Move data to device with non-blocking transfer
        node_features = node_features.to(device, non_blocking=True)
        edge_index = edge_index.to(device, non_blocking=True)
        edge_weights = edge_weights.to(device, non_blocking=True)
        
        # Create optimized data loaders
        edge_batches = self.create_optimized_data_loaders(edge_index, batch_size, device)
        
        # Training with GPU optimization
        logger.info(f"Starting optimized training for {epochs} epochs...")
        logger.info(f"Batch size: {batch_size}, Edge batches: {len(edge_batches)}")
        
        start_time = time.time()
        
        # Use optimized training method
        training_results = trainer.train_isne_simple(
            edge_index=edge_index,
            epochs=epochs,
            batch_size=batch_size,
            verbose=True
        )
        
        training_time = time.time() - start_time
        
        # Calculate throughput metrics
        total_computations = epochs * len(edges)
        computations_per_second = total_computations / training_time if training_time > 0 else 0
        
        # Save model with optimizations
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / f"isne_model_gpu_optimized_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
        
        # Create comprehensive save data
        save_data = {
            'model_state_dict': trainer.model.state_dict(),
            'model_config': {
                'num_nodes': num_nodes,
                'embedding_dim': input_dim,
                'hidden_dim': hidden_dim,
                'num_layers': num_layers,
                'learning_rate': learning_rate,
                'device': str(device)
            },
            'optimization_config': {
                'batch_size': batch_size,
                'gpu_optimized': device.type == "cuda",
                'model_parameters': model_params,
                'model_size_mb': model_size_mb
            },
            'training_config': {
                'epochs': epochs,
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'device': str(device),
                'training_time_seconds': training_time,
                'computations_per_second': computations_per_second
            },
            'graph_metadata': graph_data['metadata'],
            'training_results': training_results
        }
        
        torch.save(save_data, model_path)
        
        # Save performance metrics
        perf_path = output_dir / f"performance_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(perf_path, 'w') as f:
            performance_data = {
                'training_time_seconds': training_time,
                'epochs_completed': epochs,
                'total_computations': total_computations,
                'computations_per_second': computations_per_second,
                'final_loss': training_results.get('final_loss', 0.0),
                'best_loss': training_results.get('best_loss', 0.0),
                'model_parameters': model_params,
                'model_size_mb': model_size_mb,
                'batch_size': batch_size,
                'gpu_optimized': device.type == "cuda",
                'graph_stats': {
                    'num_nodes': len(nodes),
                    'num_edges': len(edges),
                    'embedding_dim': input_dim
                }
            }
            json.dump(performance_data, f, indent=2)
        
        results = {
            'success': True,
            'model_path': str(model_path),
            'performance_path': str(perf_path),
            'training_time_seconds': training_time,
            'final_loss': training_results.get('final_loss', 0.0),
            'best_loss': training_results.get('best_loss', 0.0),
            'epochs_completed': epochs,
            'model_parameters': model_params,
            'model_size_mb': model_size_mb,
            'computations_per_second': computations_per_second,
            'gpu_optimized': device.type == "cuda",
            'graph_stats': {
                'num_nodes': len(nodes),
                'num_edges': len(edges),
                'embedding_dim': input_dim
            }
        }
        
        logger.info("Optimized training completed successfully!")
        logger.info(f"Model saved to: {model_path}")
        logger.info(f"Training time: {training_time:.2f} seconds")
        logger.info(f"Computations/second: {computations_per_second:,.0f}")
        logger.info(f"Final loss: {results['final_loss']:.6f}")
        logger.info(f"Best loss: {results['best_loss']:.6f}")
        logger.info(f"Model size: {model_size_mb:.1f} MB")
        
        return results


def main():
    parser = argparse.ArgumentParser(description="GPU-Optimized ISNE training from ArangoDB")
    parser.add_argument(
        '--db-name',
        type=str,
        default='isne_training_database',
        help='ArangoDB database name'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./output/isne_training_optimized',
        help='Output directory for trained model'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Learning rate'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=2048,
        help='Batch size (optimized for GPU)'
    )
    parser.add_argument(
        '--hidden-dim',
        type=int,
        default=512,
        help='Hidden dimension (larger for GPU utilization)'
    )
    parser.add_argument(
        '--num-layers',
        type=int,
        default=6,
        help='Number of model layers'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='Device to use for training'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize optimized trainer
        trainer = HighPerformanceArangoISNETrainer(args.db_name)
        
        # Connect to database
        db = trainer.connect_to_database()
        
        # Extract graph data with optimizations
        graph_data = trainer.extract_graph_data_optimized(db)
        
        # Train model with GPU optimizations
        results = trainer.train_isne_model_optimized(
            graph_data=graph_data,
            output_dir=Path(args.output_dir),
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            device=args.device
        )
        
        # Print comprehensive summary
        print("\n" + "="*70)
        print("GPU-OPTIMIZED ISNE TRAINING COMPLETED")
        print("="*70)
        print(f"Database: {args.db_name}")
        print(f"Model saved to: {results['model_path']}")
        print(f"Training time: {results['training_time_seconds']:.2f} seconds")
        print(f"Final loss: {results['final_loss']:.6f}")
        print(f"Best loss: {results['best_loss']:.6f}")
        print(f"Epochs completed: {results['epochs_completed']}")
        print(f"Model parameters: {results['model_parameters']:,}")
        print(f"Model size: {results['model_size_mb']:.1f} MB")
        print(f"GPU optimized: {results['gpu_optimized']}")
        print(f"Computations/second: {results['computations_per_second']:,.0f}")
        print(f"\nGraph statistics:")
        print(f"  Nodes: {results['graph_stats']['num_nodes']:,}")
        print(f"  Edges: {results['graph_stats']['num_edges']:,}")
        print(f"  Embedding dimension: {results['graph_stats']['embedding_dim']}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Optimized training failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())