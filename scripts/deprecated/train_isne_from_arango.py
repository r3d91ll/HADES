#!/usr/bin/env python3
"""
Train ISNE model using data from ArangoDB.

This script reads the bootstrapped graph data from ArangoDB and trains
an ISNE model on it.
"""

import argparse
import logging
import sys
import time
import json
import torch
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


class ArangoISNETrainer:
    """Train ISNE model from ArangoDB data."""
    
    def __init__(self, db_name: str):
        self.db_name = db_name
        self.arango_client = ArangoClient()
        
    def connect_to_database(self):
        """Connect to ArangoDB database."""
        success = self.arango_client.connect_to_database(self.db_name)
        if not success:
            raise Exception(f"Failed to connect to database: {self.db_name}")
        return self.arango_client._database
    
    def extract_graph_data(self, db) -> Dict[str, Any]:
        """Extract graph data from ArangoDB."""
        logger.info("Extracting graph data from ArangoDB...")
        
        # Get all embeddings with chunk info
        embeddings_query = """
        FOR e IN embeddings
            LET chunk = DOCUMENT(e.chunk_id)
            FILTER chunk != null
            RETURN {
                chunk_id: e.chunk_id,
                embedding: e.embedding,
                source_file: chunk.source_file_id,
                chunk_index: chunk.chunk_index
            }
        """
        
        cursor = db.aql.execute(embeddings_query)
        embeddings_data = list(cursor)
        
        logger.info(f"Found {len(embeddings_data)} embeddings")
        
        # Create nodes from embeddings
        nodes = []
        chunk_to_node_id = {}
        for i, embedding_data in enumerate(embeddings_data):
            nodes.append({
                'id': i,
                'chunk_id': embedding_data['chunk_id'],
                'embedding': embedding_data['embedding'],
                'source_file': embedding_data['source_file'],
                'chunk_index': embedding_data['chunk_index']
            })
            chunk_to_node_id[embedding_data['chunk_id']] = i
        
        # Get all edges
        edges = []
        
        # Sequential edges
        sequential_query = """
        FOR e IN sequential_edges
            RETURN {
                source: e._from,
                target: e._to,
                weight: e.weight,
                type: "sequential"
            }
        """
        
        cursor = db.aql.execute(sequential_query)
        sequential_edges = list(cursor)
        
        for edge in sequential_edges:
            source_id = chunk_to_node_id.get(edge['source'])
            target_id = chunk_to_node_id.get(edge['target'])
            
            if source_id is not None and target_id is not None:
                edges.append({
                    'source': source_id,
                    'target': target_id,
                    'weight': edge['weight'],
                    'type': edge['type']
                })
        
        # Similarity edges
        similarity_query = """
        FOR e IN similarity_edges
            RETURN {
                source: e._from,
                target: e._to,
                weight: e.weight,
                type: "similarity"
            }
        """
        
        cursor = db.aql.execute(similarity_query)
        similarity_edges = list(cursor)
        
        for edge in similarity_edges:
            source_id = chunk_to_node_id.get(edge['source'])
            target_id = chunk_to_node_id.get(edge['target'])
            
            if source_id is not None and target_id is not None:
                edges.append({
                    'source': source_id,
                    'target': target_id,
                    'weight': edge['weight'],
                    'type': edge['type']
                })
        
        # Directory edges (convert from file edges to chunk edges)
        directory_query = """
        FOR e IN directory_edges
            RETURN {
                source: e._from,
                target: e._to,
                weight: e.weight,
                type: "directory"
            }
        """
        
        cursor = db.aql.execute(directory_query)
        directory_edges = list(cursor)
        
        # For directory edges, we need to connect chunks from files in the same directory
        for edge in directory_edges:
            # Find chunks from source file
            source_chunks = [node for node in nodes if node['source_file'] == edge['source']]
            target_chunks = [node for node in nodes if node['source_file'] == edge['target']]
            
            # Create edges between chunks from different files in same directory
            for source_chunk in source_chunks:
                for target_chunk in target_chunks:
                    edges.append({
                        'source': source_chunk['id'],
                        'target': target_chunk['id'],
                        'weight': edge['weight'],
                        'type': edge['type']
                    })
        
        logger.info(f"Created graph with {len(nodes)} nodes and {len(edges)} edges")
        
        return {
            'nodes': nodes,
            'edges': edges,
            'metadata': {
                'db_name': self.db_name,
                'extraction_timestamp': datetime.now().isoformat(),
                'num_embeddings': len(embeddings_data)
            }
        }
    
    def train_isne_model(
        self,
        graph_data: Dict[str, Any],
        output_dir: Path,
        epochs: int = 50,
        learning_rate: float = 0.001,
        batch_size: int = 512,  # Increased for better GPU utilization
        device: str = "auto"
    ) -> Dict[str, Any]:
        """Train ISNE model on graph data."""
        
        nodes = graph_data['nodes']
        edges = graph_data['edges']
        
        logger.info(f"Training ISNE model on {len(nodes)} nodes and {len(edges)} edges")
        
        # Set device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)
        
        logger.info(f"Using device: {device}")
        
        # Prepare node features
        node_features = torch.tensor([node['embedding'] for node in nodes], dtype=torch.float32)
        
        # Prepare edges
        edge_index = []
        edge_weights = []
        
        for edge in edges:
            edge_index.append([edge['source'], edge['target']])
            edge_weights.append(edge['weight'])
        
        if edge_index:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_weights = torch.tensor(edge_weights, dtype=torch.float32)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_weights = torch.empty(0, dtype=torch.float32)
        
        # Initialize trainer
        input_dim = node_features.shape[1]
        num_nodes = len(nodes)
        
        trainer = ISNETrainer(
            embedding_dim=input_dim,
            learning_rate=learning_rate,
            device=device,
            hidden_dim=256,  # Increased hidden dimension for better GPU utilization
            num_layers=4     # More layers for deeper computation
        )
        
        # Initialize model
        trainer._initialize_model(num_nodes)
        
        logger.info(f"Model initialized with {sum(p.numel() for p in trainer.model.parameters()):,} parameters")
        
        # Move data to device
        node_features = node_features.to(device)
        edge_index = edge_index.to(device)
        edge_weights = edge_weights.to(device)
        
        # Train model
        logger.info(f"Starting training for {epochs} epochs...")
        start_time = time.time()
        
        training_results = trainer.train_isne_simple(
            edge_index=edge_index,
            epochs=epochs,
            batch_size=batch_size,
            verbose=True
        )
        
        training_time = time.time() - start_time
        
        # Save model
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / f"isne_model_arango_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
        
        # Create comprehensive save data
        save_data = {
            'model_state_dict': trainer.model.state_dict(),
            'model_config': {
                'num_nodes': num_nodes,
                'embedding_dim': input_dim,
                'learning_rate': learning_rate,
                'device': str(device)
            },
            'training_config': {
                'epochs': epochs,
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'device': str(device),
                'training_time_seconds': training_time
            },
            'graph_metadata': graph_data['metadata'],
            'training_results': training_results
        }
        
        torch.save(save_data, model_path)
        
        # Save training history
        history_path = output_dir / f"training_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(history_path, 'w') as f:
            history_data = {
                'loss_history': training_results.get('loss_history', []),
                'training_time': training_time,
                'epochs_completed': epochs,
                'final_loss': training_results.get('final_loss', 0.0),
                'best_loss': training_results.get('best_loss', 0.0)
            }
            json.dump(history_data, f, indent=2)
        
        results = {
            'success': True,
            'model_path': str(model_path),
            'history_path': str(history_path),
            'training_time_seconds': training_time,
            'final_loss': training_results.get('final_loss', 0.0),
            'best_loss': training_results.get('best_loss', 0.0),
            'epochs_completed': epochs,
            'model_parameters': sum(p.numel() for p in trainer.model.parameters()),
            'graph_stats': {
                'num_nodes': len(nodes),
                'num_edges': len(edges),
                'embedding_dim': input_dim
            }
        }
        
        logger.info("Training completed successfully!")
        logger.info(f"Model saved to: {model_path}")
        logger.info(f"Training time: {training_time:.2f} seconds")
        logger.info(f"Final loss: {results['final_loss']:.6f}")
        logger.info(f"Best loss: {results['best_loss']:.6f}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Train ISNE model from ArangoDB data")
    parser.add_argument(
        '--db-name',
        type=str,
        default='isne_training_database',
        help='ArangoDB database name'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./output/isne_training',
        help='Output directory for trained model'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
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
        default=512,
        help='Batch size (increased default for better GPU utilization)'
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
        # Initialize trainer
        trainer = ArangoISNETrainer(args.db_name)
        
        # Connect to database
        db = trainer.connect_to_database()
        
        # Extract graph data
        graph_data = trainer.extract_graph_data(db)
        
        # Train model
        results = trainer.train_isne_model(
            graph_data=graph_data,
            output_dir=Path(args.output_dir),
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            device=args.device
        )
        
        # Print summary
        print("\n" + "="*60)
        print("ISNE TRAINING COMPLETED")
        print("="*60)
        print(f"Database: {args.db_name}")
        print(f"Model saved to: {results['model_path']}")
        print(f"Training time: {results['training_time_seconds']:.2f} seconds")
        print(f"Final loss: {results['final_loss']:.6f}")
        print(f"Best loss: {results['best_loss']:.6f}")
        print(f"Epochs completed: {results['epochs_completed']}")
        print(f"Model parameters: {results['model_parameters']:,}")
        print(f"\nGraph statistics:")
        print(f"  Nodes: {results['graph_stats']['num_nodes']:,}")
        print(f"  Edges: {results['graph_stats']['num_edges']:,}")
        print(f"  Embedding dimension: {results['graph_stats']['embedding_dim']}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())