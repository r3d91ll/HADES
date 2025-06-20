#!/usr/bin/env python3
"""
Memory-efficient ISNE training for large graphs.

This version uses gradient accumulation and efficient batching
to train on large graphs without running out of memory.
"""

import argparse
import logging
import sys
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple, Optional

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.database.arango_client import ArangoClient
from src.isne.models.isne_model import ISNEModel
from src.isne.training.trainer import ISNETrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MemoryEfficientISNEModel(nn.Module):
    """Memory-efficient ISNE model without full attention."""
    
    def __init__(self, num_nodes: int, embedding_dim: int, hidden_dim: int = 256, 
                 num_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        
        # Node embeddings (learnable)
        self.node_embeddings = nn.Embedding(num_nodes, embedding_dim)
        nn.init.xavier_uniform_(self.node_embeddings.weight)
        
        # Efficient GNN layers without full attention
        self.layers = nn.ModuleList()
        
        # First layer
        self.layers.append(nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),  # Use original embedding dim
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        ))
        
        # Middle layers
        for _ in range(num_layers - 2):
            self.layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
        
        # Output layer
        self.layers.append(nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim)
        ))
        
        # Edge attention (lightweight)
        self.edge_attention = nn.Linear(hidden_dim * 2, 1)
        
    def forward(self, node_ids: torch.Tensor, edge_index: torch.Tensor, 
                node_features: torch.Tensor = None):
        """
        Forward pass for specific nodes and edges.
        
        Args:
            node_ids: Node indices to compute embeddings for
            edge_index: Edge indices (2, E) connecting nodes
            node_features: Optional pre-computed features
        """
        # Get node embeddings
        node_embeds = self.node_embeddings(node_ids)
        
        # Initialize features
        if node_features is not None:
            # Use the original features, add learnable embeddings as residual
            h = node_features + node_embeds
        else:
            h = node_embeds
        
        # Message passing through layers
        for i, layer in enumerate(self.layers[:-1]):
            # Apply layer
            h = layer(h)
            
            # Simple message passing on edges (memory efficient)
            if edge_index.numel() > 0 and i == len(self.layers) // 2:
                # Aggregate neighbor information
                row, col = edge_index
                
                # Ensure indices are within bounds
                mask = (row < h.size(0)) & (col < h.size(0))
                row = row[mask]
                col = col[mask]
                
                if row.numel() > 0:
                    # Compute edge attention weights
                    edge_features = torch.cat([h[row], h[col]], dim=1)
                    attention_weights = torch.sigmoid(self.edge_attention(edge_features))
                    
                    # Aggregate messages
                    messages = h[col] * attention_weights
                    h_agg = torch.zeros_like(h)
                    h_agg.index_add_(0, row, messages)
                    
                    # Residual connection
                    h = h + 0.5 * h_agg
        
        # Final layer
        output = self.layers[-1](h)
        
        return output
    
    def get_all_embeddings(self, batch_size: int = 1024):
        """Get embeddings for all nodes in batches."""
        all_embeddings = []
        
        for i in range(0, self.num_nodes, batch_size):
            end_idx = min(i + batch_size, self.num_nodes)
            node_ids = torch.arange(i, end_idx, device=next(self.parameters()).device)
            
            with torch.no_grad():
                embeds = self.node_embeddings(node_ids)
                all_embeddings.append(embeds)
        
        return torch.cat(all_embeddings, dim=0)


class MemoryEfficientISNETrainer:
    """Memory-efficient trainer for ISNE models."""
    
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
        
        # Get embeddings count first
        count_query = "RETURN LENGTH(embeddings)"
        cursor = db.aql.execute(count_query)
        total_embeddings = list(cursor)[0]
        logger.info(f"Total embeddings in database: {total_embeddings}")
        
        # Get embeddings in batches to avoid memory issues
        batch_size = 10000
        all_embeddings_data = []
        
        for offset in range(0, total_embeddings, batch_size):
            embeddings_query = f"""
            FOR e IN embeddings
                LIMIT {offset}, {batch_size}
                LET chunk = DOCUMENT(e.chunk_id)
                FILTER chunk != null
                RETURN {{
                    chunk_id: e.chunk_id,
                    embedding: e.embedding,
                    source_file: chunk.source_file_id,
                    chunk_index: chunk.chunk_index
                }}
            """
            
            cursor = db.aql.execute(embeddings_query)
            batch_data = list(cursor)
            all_embeddings_data.extend(batch_data)
            logger.info(f"Loaded {len(all_embeddings_data)} embeddings...")
        
        logger.info(f"Found {len(all_embeddings_data)} embeddings with valid chunks")
        
        # Create nodes
        nodes = []
        chunk_to_node_id = {}
        
        for i, embedding_data in enumerate(all_embeddings_data):
            nodes.append({
                'id': i,
                'chunk_id': embedding_data['chunk_id'],
                'embedding': embedding_data['embedding'],
                'source_file': embedding_data['source_file'],
                'chunk_index': embedding_data['chunk_index']
            })
            chunk_to_node_id[embedding_data['chunk_id']] = i
        
        # Get edge counts
        edge_counts = {}
        for edge_type in ['sequential_edges', 'similarity_edges']:
            count_query = f"RETURN LENGTH({edge_type})"
            cursor = db.aql.execute(count_query)
            edge_counts[edge_type] = list(cursor)[0]
        
        logger.info(f"Edge counts: {edge_counts}")
        
        # Get edges in batches
        edges = []
        
        # Sequential edges (most important for ISNE)
        batch_size = 50000
        for offset in range(0, edge_counts.get('sequential_edges', 0), batch_size):
            edge_query = f"""
            FOR e IN sequential_edges
                LIMIT {offset}, {batch_size}
                RETURN {{
                    source: e._from,
                    target: e._to,
                    weight: e.weight,
                    type: "sequential"
                }}
            """
            
            cursor = db.aql.execute(edge_query)
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
        
        logger.info(f"Loaded {len(edges)} sequential edges")
        
        # Sample similarity edges (to avoid memory explosion)
        max_similarity_edges = 50000
        similarity_query = f"""
        FOR e IN similarity_edges
            SORT RAND()
            LIMIT {max_similarity_edges}
            RETURN {{
                source: e._from,
                target: e._to,
                weight: e.weight,
                type: "similarity"
            }}
        """
        
        cursor = db.aql.execute(similarity_query)
        similarity_edges_added = 0
        
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
                similarity_edges_added += 1
        
        logger.info(f"Added {similarity_edges_added} sampled similarity edges")
        logger.info(f"Total edges: {len(edges)}")
        
        return {
            'nodes': nodes,
            'edges': edges,
            'metadata': {
                'db_name': self.db_name,
                'extraction_timestamp': datetime.now(timezone.utc).isoformat(),
                'num_embeddings': len(nodes),
                'sampled_similarity_edges': similarity_edges_added
            }
        }
    
    def train_memory_efficient(
        self,
        model: MemoryEfficientISNEModel,
        nodes: List[Dict],
        edges: List[Dict],
        epochs: int = 50,
        batch_size: int = 512,
        learning_rate: float = 0.001,
        device: torch.device = torch.device('cuda'),
        accumulation_steps: int = 4
    ) -> Dict[str, Any]:
        """Train model with memory-efficient batching."""
        
        # Prepare node features
        node_features = torch.tensor(
            [node['embedding'] for node in nodes],
            dtype=torch.float32,
            device=device
        )
        
        # Create edge index
        edge_list = [(e['source'], e['target']) for e in edges]
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().to(device)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=learning_rate * 0.01
        )
        
        model.train()
        loss_history = []
        
        logger.info(f"Starting memory-efficient training")
        logger.info(f"Nodes: {len(nodes)}, Edges: {edge_index.shape[1]}")
        logger.info(f"Batch size: {batch_size}, Accumulation steps: {accumulation_steps}")
        
        for epoch in range(epochs):
            epoch_start = time.time()
            total_loss = 0
            num_batches = 0
            
            # Create node batches
            node_indices = torch.randperm(len(nodes), device=device)
            
            for i in range(0, len(nodes), batch_size):
                # Get batch nodes
                batch_end = min(i + batch_size, len(nodes))
                batch_nodes = node_indices[i:batch_end]
                
                # Find edges within this batch
                mask = torch.zeros(edge_index.shape[1], dtype=torch.bool, device=device)
                for j, node_id in enumerate(batch_nodes):
                    mask |= (edge_index[0] == node_id) | (edge_index[1] == node_id)
                
                batch_edges = edge_index[:, mask]
                
                # Remap edge indices to batch-local indices
                unique_nodes = torch.unique(torch.cat([batch_nodes, batch_edges.flatten()]))
                node_map = {int(n): i for i, n in enumerate(unique_nodes)}
                
                # Get features for unique nodes
                batch_features = node_features[unique_nodes]
                
                # Remap edges
                if batch_edges.numel() > 0:
                    remapped_edges = torch.tensor(
                        [[node_map[int(n)] for n in batch_edges[0]],
                         [node_map[int(n)] for n in batch_edges[1]]],
                        device=device
                    )
                else:
                    remapped_edges = torch.empty((2, 0), dtype=torch.long, device=device)
                
                # Forward pass
                embeddings = model(unique_nodes, remapped_edges, batch_features)
                
                # Compute loss using negative sampling
                if remapped_edges.numel() > 0:
                    # Sample positive edges
                    num_pos = min(256, remapped_edges.shape[1])
                    pos_idx = torch.randperm(remapped_edges.shape[1])[:num_pos]
                    pos_edges = remapped_edges[:, pos_idx]
                    
                    pos_src = embeddings[pos_edges[0]]
                    pos_dst = embeddings[pos_edges[1]]
                    pos_scores = (pos_src * pos_dst).sum(dim=1)
                    
                    # Negative sampling
                    neg_dst_idx = torch.randint(0, embeddings.size(0), (num_pos,), device=device)
                    neg_dst = embeddings[neg_dst_idx]
                    neg_scores = (pos_src * neg_dst).sum(dim=1)
                    
                    # Margin loss
                    loss = F.relu(1 - pos_scores + neg_scores).mean()
                    
                    # Regularization
                    reg_loss = 0.01 * embeddings.norm(p=2, dim=1).mean()
                    loss = loss + reg_loss
                    
                    # Gradient accumulation
                    loss = loss / accumulation_steps
                    loss.backward()
                    
                    if (num_batches + 1) % accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        optimizer.zero_grad()
                    
                    total_loss += loss.item() * accumulation_steps
                    num_batches += 1
            
            # Final gradient step
            if num_batches % accumulation_steps != 0:
                optimizer.step()
                optimizer.zero_grad()
            
            scheduler.step()
            
            # Log progress
            avg_loss = total_loss / max(num_batches, 1)
            loss_history.append(avg_loss)
            
            epoch_time = time.time() - epoch_start
            
            if (epoch + 1) % 10 == 0:
                # Memory stats
                if device.type == 'cuda':
                    mem_allocated = torch.cuda.memory_allocated(device) / 1e9
                    mem_reserved = torch.cuda.memory_reserved(device) / 1e9
                    logger.info(
                        f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f} - "
                        f"Time: {epoch_time:.2f}s - "
                        f"Memory: {mem_allocated:.1f}/{mem_reserved:.1f} GB"
                    )
                else:
                    logger.info(
                        f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f} - "
                        f"Time: {epoch_time:.2f}s"
                    )
        
        return {
            'loss_history': loss_history,
            'final_loss': loss_history[-1] if loss_history else 0,
            'best_loss': min(loss_history) if loss_history else 0
        }
    
    def run_training(
        self,
        graph_data: Dict[str, Any],
        output_dir: Path,
        epochs: int = 50,
        batch_size: int = 512,
        learning_rate: float = 0.001,
        hidden_dim: int = 256,
        num_layers: int = 4,
        device: str = 'auto'
    ) -> Dict[str, Any]:
        """Run the complete training pipeline."""
        
        nodes = graph_data['nodes']
        edges = graph_data['edges']
        
        # Device setup
        if device == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device)
        
        logger.info(f"Using device: {device}")
        
        # Create model
        num_nodes = len(nodes)
        embedding_dim = len(nodes[0]['embedding'])
        
        model = MemoryEfficientISNEModel(
            num_nodes=num_nodes,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        ).to(device)
        
        model_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters: {model_params:,}")
        
        # Train
        start_time = time.time()
        
        training_results = self.train_memory_efficient(
            model=model,
            nodes=nodes,
            edges=edges,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device
        )
        
        training_time = time.time() - start_time
        
        # Save model
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / f"isne_model_efficient_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.pth"
        
        save_data = {
            'model_state_dict': model.state_dict(),
            'model_config': {
                'num_nodes': num_nodes,
                'embedding_dim': embedding_dim,
                'hidden_dim': hidden_dim,
                'num_layers': num_layers,
                'memory_efficient': True
            },
            'training_config': {
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'training_time_seconds': training_time
            },
            'graph_metadata': graph_data['metadata'],
            'training_results': training_results
        }
        
        torch.save(save_data, model_path)
        
        return {
            'model_path': str(model_path),
            'training_time': training_time,
            'final_loss': training_results['final_loss'],
            'best_loss': training_results['best_loss'],
            'model_parameters': model_params,
            'num_nodes': num_nodes,
            'num_edges': len(edges)
        }


def main():
    parser = argparse.ArgumentParser(description="Memory-efficient ISNE training")
    parser.add_argument('--db-name', type=str, default='isne_training_database')
    parser.add_argument('--output-dir', type=str, default='./output/isne_training_efficient')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--num-layers', type=int, default=4)
    parser.add_argument('--device', type=str, default='auto')
    
    args = parser.parse_args()
    
    try:
        trainer = MemoryEfficientISNETrainer(args.db_name)
        db = trainer.connect_to_database()
        graph_data = trainer.extract_graph_data(db)
        
        results = trainer.run_training(
            graph_data=graph_data,
            output_dir=Path(args.output_dir),
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            device=args.device
        )
        
        print("\n" + "="*60)
        print("MEMORY-EFFICIENT ISNE TRAINING COMPLETED")
        print("="*60)
        print(f"Model saved to: {results['model_path']}")
        print(f"Training time: {results['training_time']:.2f} seconds")
        print(f"Final loss: {results['final_loss']:.6f}")
        print(f"Best loss: {results['best_loss']:.6f}")
        print(f"Model parameters: {results['model_parameters']:,}")
        print(f"Graph: {results['num_nodes']:,} nodes, {results['num_edges']:,} edges")
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())