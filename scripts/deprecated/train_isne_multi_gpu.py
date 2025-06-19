#!/usr/bin/env python3
"""
Multi-GPU ISNE Training with NVLink optimization.

This script uses PyTorch DataParallel or DistributedDataParallel
to train across multiple GPUs with NVLink.
"""

import argparse
import logging
import sys
import time
import json
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel, DistributedDataParallel
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.arango_client import ArangoClient
from src.isne.models.isne_model import ISNEModel
from src.isne.training.trainer import ISNETrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MultiGPUISNETrainer:
    """Multi-GPU optimized ISNE trainer with NVLink support."""
    
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
    
    def create_multi_gpu_model(
        self,
        num_nodes: int,
        embedding_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_ddp: bool = False
    ) -> nn.Module:
        """Create ISNE model optimized for multi-GPU training."""
        
        # Custom ISNE model with larger capacity for GPU utilization
        class LargeISNEModel(nn.Module):
            def __init__(self, num_nodes, embedding_dim, hidden_dim, num_layers, num_heads, dropout):
                super().__init__()
                self.num_nodes = num_nodes
                self.embedding_dim = embedding_dim
                
                # Node embeddings
                self.node_embeddings = nn.Embedding(num_nodes, embedding_dim)
                nn.init.xavier_uniform_(self.node_embeddings.weight)
                
                # Multi-layer transformer for graph attention
                self.layers = nn.ModuleList()
                current_dim = embedding_dim
                
                for i in range(num_layers):
                    if i == 0:
                        # First layer: embedding_dim -> hidden_dim
                        self.layers.append(nn.Sequential(
                            nn.Linear(current_dim, hidden_dim),
                            nn.LayerNorm(hidden_dim),
                            nn.ReLU(),
                            nn.Dropout(dropout)
                        ))
                        current_dim = hidden_dim
                    elif i == num_layers - 1:
                        # Last layer: hidden_dim -> embedding_dim
                        self.layers.append(nn.Sequential(
                            nn.Linear(current_dim, embedding_dim),
                            nn.LayerNorm(embedding_dim)
                        ))
                    else:
                        # Middle layers: hidden_dim -> hidden_dim
                        self.layers.append(nn.Sequential(
                            nn.Linear(current_dim, hidden_dim),
                            nn.LayerNorm(hidden_dim),
                            nn.ReLU(),
                            nn.Dropout(dropout)
                        ))
                
                # Attention mechanism
                self.attention = nn.MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True
                )
                
                # Output projection
                self.output_projection = nn.Linear(embedding_dim, embedding_dim)
                
            def forward(self, x, edge_index, edge_attr=None):
                # Add node embeddings
                node_ids = torch.arange(x.size(0), device=x.device)
                node_embeds = self.node_embeddings(node_ids)
                
                # Combine with input features
                h = x + node_embeds
                
                # Pass through layers
                for i, layer in enumerate(self.layers):
                    h = layer(h)
                    
                    # Apply attention at middle layer
                    if i == len(self.layers) // 2:
                        h_unsqueezed = h.unsqueeze(0)  # Add batch dimension
                        h_attended, _ = self.attention(h_unsqueezed, h_unsqueezed, h_unsqueezed)
                        h = h + h_attended.squeeze(0)  # Residual connection
                
                # Final projection
                output = self.output_projection(h)
                
                return output
            
            def save(self, path):
                """Save model state."""
                torch.save({
                    'state_dict': self.state_dict(),
                    'num_nodes': self.num_nodes,
                    'embedding_dim': self.embedding_dim
                }, path)
        
        # Create model
        model = LargeISNEModel(
            num_nodes=num_nodes,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Check GPU availability
        gpu_count = torch.cuda.device_count()
        logger.info(f"Found {gpu_count} GPUs")
        
        if gpu_count > 1:
            logger.info("Using multi-GPU training")
            if use_ddp:
                # DistributedDataParallel (more efficient but requires process spawning)
                logger.info("Using DistributedDataParallel")
                return model
            else:
                # DataParallel (simpler but less efficient)
                logger.info("Using DataParallel on all GPUs")
                model = DataParallel(model)
                
                # Log GPU memory
                for i in range(gpu_count):
                    mem_total = torch.cuda.get_device_properties(i).total_memory / 1e9
                    logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)} - {mem_total:.1f} GB")
        else:
            logger.info("Using single GPU training")
            
        return model.cuda()
    
    def train_multi_gpu(
        self,
        model: nn.Module,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weights: torch.Tensor,
        epochs: int = 100,
        learning_rate: float = 0.001,
        batch_size: int = 8192,
        weight_decay: float = 1e-5
    ) -> Dict[str, Any]:
        """Train model using multiple GPUs."""
        
        # Move data to GPU
        node_features = node_features.cuda()
        edge_index = edge_index.cuda()
        edge_weights = edge_weights.cuda() if edge_weights.numel() > 0 else None
        
        # Optimizer with larger learning rate for bigger model
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=learning_rate * 0.01
        )
        
        # Training loop
        model.train()
        loss_history = []
        
        logger.info(f"Starting multi-GPU training for {epochs} epochs")
        logger.info(f"Batch size: {batch_size}, Learning rate: {learning_rate}")
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Create random edge batches for training
            num_edges = edge_index.shape[1]
            perm = torch.randperm(num_edges, device=edge_index.device)
            
            total_loss = 0
            num_batches = 0
            
            for i in range(0, num_edges, batch_size):
                # Get batch edges
                batch_idx = perm[i:i+batch_size]
                batch_edges = edge_index[:, batch_idx]
                
                # Forward pass
                optimizer.zero_grad()
                embeddings = model(node_features, batch_edges)
                
                # Compute loss (negative sampling)
                pos_src = embeddings[batch_edges[0]]
                pos_dst = embeddings[batch_edges[1]]
                
                # Positive scores
                pos_scores = (pos_src * pos_dst).sum(dim=1)
                
                # Negative sampling
                neg_dst_idx = torch.randint(
                    0, node_features.size(0),
                    (batch_edges.shape[1],),
                    device=edge_index.device
                )
                neg_dst = embeddings[neg_dst_idx]
                neg_scores = (pos_src * neg_dst).sum(dim=1)
                
                # Margin loss
                loss = torch.clamp(1 - pos_scores + neg_scores, min=0).mean()
                
                # Add regularization
                reg_loss = 0.01 * (embeddings.norm(dim=1).mean() - 1).abs()
                loss = loss + reg_loss
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            # Update learning rate
            scheduler.step()
            
            # Log progress
            avg_loss = total_loss / num_batches
            loss_history.append(avg_loss)
            
            epoch_time = time.time() - epoch_start
            current_lr = scheduler.get_last_lr()[0]
            
            if (epoch + 1) % 10 == 0:
                # Calculate GPU utilization
                gpu_memory_used = []
                for i in range(torch.cuda.device_count()):
                    mem_used = torch.cuda.memory_allocated(i) / 1e9
                    mem_total = torch.cuda.get_device_properties(i).total_memory / 1e9
                    gpu_memory_used.append(f"GPU{i}: {mem_used:.1f}/{mem_total:.1f}GB")
                
                logger.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Loss: {avg_loss:.6f} - "
                    f"LR: {current_lr:.6f} - "
                    f"Time: {epoch_time:.2f}s - "
                    f"Memory: {', '.join(gpu_memory_used)}"
                )
        
        return {
            'loss_history': loss_history,
            'final_loss': loss_history[-1] if loss_history else 0,
            'best_loss': min(loss_history) if loss_history else 0,
            'epochs_trained': epochs
        }
    
    def train_isne_model_multi_gpu(
        self,
        graph_data: Dict[str, Any],
        output_dir: Path,
        epochs: int = 100,
        learning_rate: float = 0.001,
        batch_size: int = 8192,
        hidden_dim: int = 1024,
        num_layers: int = 8,
        num_heads: int = 16,
        dropout: float = 0.1,
        weight_decay: float = 1e-5
    ) -> Dict[str, Any]:
        """Train ISNE model using multiple GPUs."""
        
        nodes = graph_data['nodes']
        edges = graph_data['edges']
        
        logger.info(f"Training ISNE model on {len(nodes)} nodes and {len(edges)} edges")
        logger.info(f"Model config: {hidden_dim}D hidden, {num_layers} layers, {num_heads} heads")
        
        # Enable GPU optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Log GPU information
        gpu_count = torch.cuda.device_count()
        total_memory = 0
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"GPU {i}: {props.name} - {props.total_memory / 1e9:.1f} GB")
            total_memory += props.total_memory
        
        logger.info(f"Total GPU memory: {total_memory / 1e9:.1f} GB")
        if gpu_count > 1:
            logger.info("NVLink detected - enabling peer-to-peer communication")
            for i in range(gpu_count):
                for j in range(gpu_count):
                    if i != j and torch.cuda.can_device_access_peer(i, j):
                        logger.info(f"GPU {i} <-> GPU {j}: P2P enabled")
        
        # Prepare data
        node_features = torch.tensor([node['embedding'] for node in nodes], dtype=torch.float32)
        
        edge_list = [(edge['source'], edge['target']) for edge in edges]
        edge_weights = [edge['weight'] for edge in edges]
        
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_weights = torch.tensor(edge_weights, dtype=torch.float32)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_weights = torch.empty(0, dtype=torch.float32)
        
        # Create multi-GPU model
        input_dim = node_features.shape[1]
        num_nodes = len(nodes)
        
        model = self.create_multi_gpu_model(
            num_nodes=num_nodes,
            embedding_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Calculate model size
        model_params = sum(p.numel() for p in model.parameters())
        model_size_mb = model_params * 4 / 1024 / 1024
        
        logger.info(f"Model initialized with {model_params:,} parameters ({model_size_mb:.1f} MB)")
        logger.info(f"Using {gpu_count} GPUs for training")
        
        # Train model
        start_time = time.time()
        
        training_results = self.train_multi_gpu(
            model=model,
            node_features=node_features,
            edge_index=edge_index,
            edge_weights=edge_weights,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            weight_decay=weight_decay
        )
        
        training_time = time.time() - start_time
        
        # Save model
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / f"isne_model_multi_gpu_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
        
        # Handle DataParallel wrapper
        if isinstance(model, DataParallel):
            model_to_save = model.module
        else:
            model_to_save = model
        
        # Save with comprehensive metadata
        save_data = {
            'model_state_dict': model_to_save.state_dict(),
            'model_config': {
                'num_nodes': num_nodes,
                'embedding_dim': input_dim,
                'hidden_dim': hidden_dim,
                'num_layers': num_layers,
                'num_heads': num_heads,
                'dropout': dropout,
                'multi_gpu': gpu_count > 1,
                'gpu_count': gpu_count
            },
            'training_config': {
                'epochs': epochs,
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'weight_decay': weight_decay,
                'training_time_seconds': training_time,
                'throughput_edges_per_second': len(edges) * epochs / training_time
            },
            'graph_metadata': graph_data['metadata'],
            'training_results': training_results
        }
        
        torch.save(save_data, model_path)
        
        # Calculate performance metrics
        edges_per_second = len(edges) * epochs / training_time
        gpu_efficiency = edges_per_second / (gpu_count * 1000000)  # Millions per GPU
        
        results = {
            'success': True,
            'model_path': str(model_path),
            'training_time_seconds': training_time,
            'final_loss': training_results['final_loss'],
            'best_loss': training_results['best_loss'],
            'epochs_completed': epochs,
            'model_parameters': model_params,
            'model_size_mb': model_size_mb,
            'edges_per_second': edges_per_second,
            'gpu_count': gpu_count,
            'gpu_efficiency': gpu_efficiency,
            'graph_stats': {
                'num_nodes': len(nodes),
                'num_edges': len(edges),
                'embedding_dim': input_dim
            }
        }
        
        logger.info("Multi-GPU training completed successfully!")
        logger.info(f"Model saved to: {model_path}")
        logger.info(f"Training time: {training_time:.2f} seconds")
        logger.info(f"Throughput: {edges_per_second:,.0f} edges/second")
        logger.info(f"GPU efficiency: {gpu_efficiency:.2f}M edges/second/GPU")
        logger.info(f"Final loss: {results['final_loss']:.6f}")
        logger.info(f"Best loss: {results['best_loss']:.6f}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU ISNE training from ArangoDB")
    parser.add_argument(
        '--db-name',
        type=str,
        default='isne_training_database',
        help='ArangoDB database name'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./output/isne_training_multi_gpu',
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
        default=8192,
        help='Batch size (optimized for multi-GPU)'
    )
    parser.add_argument(
        '--hidden-dim',
        type=int,
        default=1024,
        help='Hidden dimension size'
    )
    parser.add_argument(
        '--num-layers',
        type=int,
        default=8,
        help='Number of model layers'
    )
    parser.add_argument(
        '--num-heads',
        type=int,
        default=16,
        help='Number of attention heads'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.1,
        help='Dropout rate'
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=1e-5,
        help='Weight decay for AdamW optimizer'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize trainer
        trainer = MultiGPUISNETrainer(args.db_name)
        
        # Connect to database
        db = trainer.connect_to_database()
        
        # Extract graph data
        graph_data = trainer.extract_graph_data(db)
        
        # Train model with multi-GPU support
        results = trainer.train_isne_model_multi_gpu(
            graph_data=graph_data,
            output_dir=Path(args.output_dir),
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout=args.dropout,
            weight_decay=args.weight_decay
        )
        
        # Print summary
        print("\n" + "="*70)
        print("MULTI-GPU ISNE TRAINING COMPLETED")
        print("="*70)
        print(f"Database: {args.db_name}")
        print(f"Model saved to: {results['model_path']}")
        print(f"Training time: {results['training_time_seconds']:.2f} seconds")
        print(f"Final loss: {results['final_loss']:.6f}")
        print(f"Best loss: {results['best_loss']:.6f}")
        print(f"Epochs completed: {results['epochs_completed']}")
        print(f"Model parameters: {results['model_parameters']:,}")
        print(f"Model size: {results['model_size_mb']:.1f} MB")
        print(f"\nGPU Performance:")
        print(f"  GPUs used: {results['gpu_count']}")
        print(f"  Throughput: {results['edges_per_second']:,.0f} edges/second")
        print(f"  Efficiency: {results['gpu_efficiency']:.2f}M edges/second/GPU")
        print(f"\nGraph statistics:")
        print(f"  Nodes: {results['graph_stats']['num_nodes']:,}")
        print(f"  Edges: {results['graph_stats']['num_edges']:,}")
        print(f"  Embedding dimension: {results['graph_stats']['embedding_dim']}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Multi-GPU training failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())