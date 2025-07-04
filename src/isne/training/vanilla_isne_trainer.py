"""
Vanilla ISNE trainer that uses .hades metadata-driven graph structure.

This implements the pure ISNE algorithm from the research paper:
- h(v) = (1/|N_v|) * Σ_{n∈N_v} θ_n
- Trained with skip-gram objective on random walks
- Uses edges defined in .hades/relationships.json
"""

import logging
from typing import Dict, Any, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from arango import ArangoClient
import random
import numpy as np

from src.isne.models.isne_model import ISNEModel

logger = logging.getLogger(__name__)


class VanillaISNETrainer:
    """Train ISNE on graph constructed from .hades metadata."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize trainer with configuration.
        
        Args:
            config: Training configuration including:
                - embedding_dim: Dimension of embeddings
                - learning_rate: Learning rate
                - num_epochs: Number of training epochs
                - walks_per_node: Random walks per node
                - walk_length: Length of each walk
                - context_window: Skip-gram context window
                - negative_samples: Number of negative samples
                - arango_config: ArangoDB connection settings
        """
        self.config = config
        self.embedding_dim = config.get('embedding_dim', 128)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.num_epochs = config.get('num_epochs', 100)
        self.walks_per_node = config.get('walks_per_node', 10)
        self.walk_length = config.get('walk_length', 80)
        self.context_window = config.get('context_window', 5)
        self.negative_samples = config.get('negative_samples', 5)
        
        # Connect to ArangoDB
        self.db = self._connect_arango(config.get('arango_config', {}))
        
        # Build adjacency lists from .hades relationships
        self.adjacency_lists: Optional[List[List[int]]] = None
        self.node_to_idx: Dict[str, int] = {}
        self.idx_to_node: Dict[int, str] = {}
        
        # ISNE model
        self.model: Optional[ISNEModel] = None
        
    def _connect_arango(self, arango_config: Dict[str, Any]) -> Any:
        """Connect to ArangoDB."""
        client = ArangoClient(hosts=arango_config.get('url', 'http://localhost:8529'))
        return client.db(
            arango_config.get('database', 'hades'),
            username=arango_config.get('username', 'root'),
            password=arango_config.get('password', '')
        )
    
    def build_adjacency_from_hades(self) -> Tuple[List[List[int]], Dict[str, int]]:
        """Build adjacency lists from .hades-driven graph in ArangoDB.
        
        Returns:
            Tuple of (adjacency_lists, node_mapping)
        """
        logger.info("Building adjacency lists from .hades relationships...")
        
        # Get all nodes with .hades
        nodes_query = """
        FOR node IN nodes
        FILTER node.has_hades == true OR node.type == 'file'
        RETURN node
        """
        nodes = list(self.db.aql.execute(nodes_query))
        
        # Create node mappings
        for idx, node in enumerate(nodes):
            node_key = node['_key']
            self.node_to_idx[node_key] = idx
            self.idx_to_node[idx] = node_key
        
        num_nodes = len(nodes)
        adjacency_lists: List[List[int]] = [[] for _ in range(num_nodes)]
        
        # Get edges from all collections (prioritizing .hades relationships)
        edge_collections = ['filesystem_edges', 'semantic_edges', 'functional_edges']
        
        for collection in edge_collections:
            edges_query = f"""
            FOR edge IN {collection}
            FILTER edge.source == 'hades_metadata' OR edge.strength >= 0.5
            RETURN edge
            """
            
            for edge in self.db.aql.execute(edges_query):
                # Extract node keys from _from and _to
                from_key = edge['_from'].split('/')[1]
                to_key = edge['_to'].split('/')[1]
                
                if from_key in self.node_to_idx and to_key in self.node_to_idx:
                    from_idx = self.node_to_idx[from_key]
                    to_idx = self.node_to_idx[to_key]
                    
                    # Add bidirectional edges for undirected graph
                    adjacency_lists[from_idx].append(to_idx)
                    adjacency_lists[to_idx].append(from_idx)
        
        logger.info(f"Built graph with {num_nodes} nodes and "
                   f"{sum(len(adj) for adj in adjacency_lists)} edges")
        
        self.adjacency_lists = adjacency_lists
        return adjacency_lists, self.node_to_idx
    
    def generate_random_walks(self) -> List[List[int]]:
        """Generate random walks from the graph."""
        if self.adjacency_lists is None:
            raise RuntimeError("Adjacency lists not built. Call build_graph first.")
        
        walks = []
        num_nodes = len(self.adjacency_lists)
        
        for node_idx in range(num_nodes):
            for _ in range(self.walks_per_node):
                walk = self._random_walk(node_idx)
                if len(walk) > 1:  # Only keep meaningful walks
                    walks.append(walk)
        
        return walks
    
    def _random_walk(self, start_node: int) -> List[int]:
        """Perform a random walk starting from a node."""
        if self.adjacency_lists is None:
            raise RuntimeError("Adjacency lists not built. Call build_graph first.")
            
        walk = [start_node]
        current = start_node
        
        for _ in range(self.walk_length - 1):
            neighbors = self.adjacency_lists[current]
            if not neighbors:
                break
            
            # Choose next node randomly
            current = random.choice(neighbors)
            walk.append(current)
        
        return walk
    
    def train(self):
        """Train ISNE model using skip-gram on random walks."""
        # Build graph from .hades relationships
        self.build_adjacency_from_hades()
        
        # Initialize model
        if self.adjacency_lists is None:
            raise RuntimeError("Adjacency lists not built.")
        
        num_nodes = len(self.adjacency_lists)
        self.model = ISNEModel(
            num_nodes=num_nodes,
            embedding_dim=self.embedding_dim,
            learning_rate=self.learning_rate,
            negative_samples=self.negative_samples,
            context_window=self.context_window
        )
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        logger.info(f"Starting ISNE training for {self.num_epochs} epochs...")
        
        for epoch in range(self.num_epochs):
            # Generate new random walks each epoch
            walks = self.generate_random_walks()
            random.shuffle(walks)
            
            total_loss = 0.0
            num_batches = 0
            
            for walk in walks:
                # Skip-gram training on the walk
                for i, center in enumerate(walk):
                    # Get context nodes
                    start = max(0, i - self.context_window)
                    end = min(len(walk), i + self.context_window + 1)
                    context = walk[start:i] + walk[i+1:end]
                    
                    if not context:
                        continue
                    
                    # Convert to tensors
                    center_tensor = torch.tensor([center])
                    context_tensor = torch.tensor(context)
                    
                    # Sample negative nodes
                    negative_nodes = []
                    for _ in range(self.negative_samples):
                        neg = random.randint(0, num_nodes - 1)
                        while neg in context or neg == center:
                            neg = random.randint(0, num_nodes - 1)
                        negative_nodes.append(neg)
                    negative_tensor = torch.tensor(negative_nodes)
                    
                    # Compute loss
                    if self.model is None:
                        raise RuntimeError("Model not initialized")
                    
                    loss = self.model.compute_loss(
                        center_nodes=center_tensor,
                        context_nodes=context_tensor.unsqueeze(0),
                        neighbor_lists=[self.adjacency_lists[center]],
                        negative_nodes=negative_tensor.unsqueeze(0)
                    )
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}/{self.num_epochs}, Loss: {avg_loss:.4f}")
        
        logger.info("ISNE training completed!")
        
        # Save embeddings to ArangoDB
        self.save_embeddings_to_arango()
    
    def save_embeddings_to_arango(self):
        """Save trained ISNE embeddings back to ArangoDB nodes."""
        logger.info("Saving ISNE embeddings to ArangoDB...")
        
        # Get all embeddings
        if self.model is None:
            raise RuntimeError("Model not initialized")
        if self.adjacency_lists is None:
            raise RuntimeError("Adjacency lists not built")
            
        all_embeddings = self.model.get_all_embeddings(self.adjacency_lists)
        
        # Update each node with its ISNE embedding
        for idx, embedding in enumerate(all_embeddings):
            node_key = self.idx_to_node[idx]
            
            # Update node with ISNE embedding
            update_query = """
            FOR node IN nodes
            FILTER node._key == @key
            UPDATE node WITH {
                embeddings: MERGE(node.embeddings || {}, {
                    isne: @embedding
                })
            } IN nodes
            """
            
            self.db.aql.execute(
                update_query,
                bind_vars={
                    'key': node_key,
                    'embedding': embedding.tolist()
                }
            )
        
        logger.info(f"Saved {len(all_embeddings)} ISNE embeddings to ArangoDB")
    
    def get_node_embedding(self, node_key: str) -> Optional[np.ndarray]:
        """Get ISNE embedding for a specific node."""
        if node_key not in self.node_to_idx:
            return None
        
        if self.adjacency_lists is None or self.model is None:
            return None
            
        node_idx = self.node_to_idx[node_key]
        neighbors = self.adjacency_lists[node_idx]
        
        embedding = self.model.get_node_embeddings(
            torch.tensor([node_idx]),
            [neighbors]
        )
        
        return embedding[0].detach().numpy()


def train_isne_from_hades(config_path: str) -> None:
    """Main entry point for training ISNE from .hades metadata."""
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    trainer = VanillaISNETrainer(config)
    trainer.train()
    
    logger.info("ISNE training pipeline completed successfully!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = "src/config/isne/isne_training_config.yaml"
    
    train_isne_from_hades(config_path)