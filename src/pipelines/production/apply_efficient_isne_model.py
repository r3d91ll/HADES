#!/usr/bin/env python3
"""
Apply memory-efficient ISNE model to enhance embeddings and create new edges.

This script loads the MemoryEfficientISNEModel and applies it to create
enhanced embeddings and discover new relationships.
"""

import argparse
import logging
import sys
import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.database.arango_client import ArangoClient

def setup_logging(log_file: Optional[str] = None):
    """Setup comprehensive logging."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=logging.DEBUG,
        format=log_format,
        handlers=handlers
    )

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
        """Forward pass for specific nodes and edges."""
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
    
    def get_all_embeddings_batched(self, batch_size: int = 1000, device: torch.device = None):
        """Get embeddings for all nodes in batches."""
        if device is None:
            device = next(self.parameters()).device
            
        all_embeddings = []
        
        self.eval()
        with torch.no_grad():
            for i in range(0, self.num_nodes, batch_size):
                end_idx = min(i + batch_size, self.num_nodes)
                node_ids = torch.arange(i, end_idx, device=device)
                
                # Get node embeddings and pass through model
                node_embeds = self.node_embeddings(node_ids)
                
                # Apply layers (without edges for simplicity)
                h = node_embeds
                for layer in self.layers:
                    h = layer(h)
                
                all_embeddings.append(h.cpu())
        
        return torch.cat(all_embeddings, dim=0)


class EfficientISNEApplicationPipeline:
    """Apply memory-efficient ISNE model to create production database."""
    
    def __init__(self, source_db: str, target_db: Optional[str] = None):
        self.source_db = source_db
        self.target_db = target_db or source_db
        self.update_in_place = (target_db is None)
        self.arango_client = ArangoClient()
        
    def load_efficient_model(self, model_path: str) -> Tuple[MemoryEfficientISNEModel, Dict[str, Any]]:
        """Load the memory-efficient ISNE model."""
        logger.info(f"Loading memory-efficient ISNE model from: {model_path}")
        
        # Load the saved model data
        save_data = torch.load(model_path, map_location='cpu')
        
        # Extract configuration
        model_config = save_data['model_config']
        num_nodes = model_config['num_nodes']
        embedding_dim = model_config['embedding_dim']
        hidden_dim = model_config.get('hidden_dim', 256)
        num_layers = model_config.get('num_layers', 4)
        
        # Initialize model with same architecture
        model = MemoryEfficientISNEModel(
            num_nodes=num_nodes,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )
        
        # Load state dict
        model.load_state_dict(save_data['model_state_dict'])
        model.eval()
        
        logger.info(f"Model loaded: {num_nodes} nodes, {embedding_dim}D embeddings")
        logger.info(f"Architecture: {hidden_dim}D hidden, {num_layers} layers")
        
        return model, save_data
    
    def create_production_database(self, db_name: str):
        """Create production database with proper schema."""
        logger.info(f"Creating production database: {db_name}")
        
        # Create database using admin connection
        try:
            admin_client = ArangoClient()
            admin_client.create_database(db_name)
            logger.info(f"Created new database: {db_name}")
        except Exception as e:
            logger.warning(f"Database creation failed (may already exist): {e}")
        
        # Connect to database
        success = self.arango_client.connect_to_database(db_name)
        if not success:
            raise Exception(f"Failed to connect to database: {db_name}")
        
        db = self.arango_client._database
        
        # Define collections for production
        collections = {
            # Node collections
            'code_files': {'type': 'document'},
            'documentation_files': {'type': 'document'},
            'config_files': {'type': 'document'},
            'chunks': {'type': 'document'},
            'embeddings': {'type': 'document'},
            'isne_embeddings': {'type': 'document'},  # NEW: Enhanced embeddings
            
            # Edge collections
            'intra_modal_edges': {'type': 'edge'},
            'cross_modal_edges': {'type': 'edge'},
            'directory_edges': {'type': 'edge'},
            'similarity_edges': {'type': 'edge'},
            'sequential_edges': {'type': 'edge'},
            'isne_enhanced_edges': {'type': 'edge'},  # NEW: ISNE-discovered edges
        }
        
        # Create collections
        for name, config in collections.items():
            if not db.has_collection(name):
                if config['type'] == 'edge':
                    db.create_collection(name, edge=True)
                else:
                    db.create_collection(name)
                logger.info(f"Created collection: {name}")
        
        # Create indexes
        chunks_coll = db.collection('chunks')
        if not chunks_coll.get_index('source_file_id'):
            chunks_coll.add_hash_index(fields=['source_file_id'])
        
        embeddings_coll = db.collection('embeddings')
        if not embeddings_coll.get_index('chunk_id'):
            embeddings_coll.add_hash_index(fields=['chunk_id'])
        
        isne_embeddings_coll = db.collection('isne_embeddings')
        if not isne_embeddings_coll.get_index('chunk_id'):
            isne_embeddings_coll.add_hash_index(fields=['chunk_id'])
            isne_embeddings_coll.add_hash_index(fields=['node_id'])
        
        return db
    
    def ensure_collections_exist(self, db):
        """Ensure required collections exist in the database."""
        logger.info("Ensuring required collections exist...")
        
        # Collections needed for ISNE enhancement
        required_collections = {
            'isne_embeddings': {'type': 'document'},
            'isne_enhanced_edges': {'type': 'edge'}
        }
        
        for name, config in required_collections.items():
            if not db.has_collection(name):
                if config['type'] == 'edge':
                    db.create_collection(name, edge=True)
                else:
                    db.create_collection(name)
                logger.info(f"Created collection: {name}")
            else:
                logger.info(f"Collection already exists: {name}")
        
        # Create indexes for new collections
        if db.has_collection('isne_embeddings'):
            isne_coll = db.collection('isne_embeddings')
            try:
                isne_coll.add_hash_index(fields=['chunk_id'])
                isne_coll.add_hash_index(fields=['node_id'])
            except:
                pass  # Indexes may already exist
    
    def copy_base_data(self, source_db, target_db):
        """Copy base data from source to target database."""
        logger.info("Copying base data to production database...")
        
        # Collections to copy (matches actual bootstrap schema)
        collections_to_copy = [
            'code_files', 'documentation_files', 'config_files',
            'chunks', 'embeddings',
            'sequential_edges', 'directory_edges', 'cross_modal_edges', 'similarity_edges',
            'intra_modal_edges'  # Also in your database
        ]
        
        for collection_name in collections_to_copy:
            logger.debug(f"Processing collection: {collection_name}")
            
            if not source_db.has_collection(collection_name):
                logger.warning(f"Collection {collection_name} not found in source database")
                continue
            
            # Get source collection info
            source_coll = source_db.collection(collection_name)
            coll_info = source_coll.properties()
            is_edge = coll_info.get('type') == 3  # Edge collection type
            
            # Create collection in target if it doesn't exist
            if not target_db.has_collection(collection_name):
                logger.info(f"Creating collection {collection_name} (edge: {is_edge})")
                try:
                    target_db.create_collection(collection_name, edge=is_edge)
                    logger.info(f"Successfully created collection {collection_name}")
                except Exception as e:
                    logger.error(f"Failed to create collection {collection_name}: {e}")
                    continue
            else:
                logger.debug(f"Collection {collection_name} already exists in target")
            
            # Get document count
            doc_count = source_coll.count()
            logger.info(f"Collection {collection_name} has {doc_count} documents")
            
            if doc_count > 0:
                # Get all documents from source
                query = f"FOR doc IN {collection_name} RETURN doc"
                cursor = source_db.aql.execute(query)
                docs = list(cursor)
                
                logger.debug(f"Retrieved {len(docs)} documents from {collection_name}")
                
                if docs:
                    # Verify target collection exists before inserting
                    if not target_db.has_collection(collection_name):
                        logger.error(f"Target collection {collection_name} still doesn't exist after creation attempt")
                        continue
                        
                    # Insert into target in batches
                    target_coll = target_db.collection(collection_name)
                    batch_size = 1000
                    total_inserted = 0
                    
                    for i in range(0, len(docs), batch_size):
                        batch = docs[i:i+batch_size]
                        try:
                            target_coll.insert_many(batch, overwrite=False)
                            total_inserted += len(batch)
                            logger.debug(f"Inserted batch {i//batch_size + 1}, total: {total_inserted}")
                        except Exception as e:
                            logger.error(f"Error inserting batch for {collection_name}: {e}")
                            logger.debug(f"First document in failed batch: {batch[0] if batch else 'empty batch'}")
                            
                    logger.info(f"Copied {total_inserted} documents to {collection_name}")
            else:
                logger.info(f"Collection {collection_name} is empty, skipping")
    
    def apply_isne_model(self, model: MemoryEfficientISNEModel, db) -> Dict[str, np.ndarray]:
        """Apply ISNE model to enhance embeddings."""
        logger.info("Applying memory-efficient ISNE model to embeddings...")
        
        # Get mapping from chunk_id to node_id
        embeddings_query = """
        FOR e IN embeddings
            LET chunk = DOCUMENT(e.chunk_id)
            FILTER chunk != null
            SORT e._key
            RETURN {
                chunk_id: e.chunk_id,
                node_id: TO_NUMBER(e._key),
                embedding: e.embedding
            }
        """
        
        cursor = db.aql.execute(embeddings_query)
        embeddings_data = list(cursor)
        
        logger.info(f"Found {len(embeddings_data)} embeddings to enhance")
        
        # Apply model to get enhanced embeddings
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        enhanced_embeddings = model.get_all_embeddings_batched(
            batch_size=1000,
            device=device
        )
        
        # Create mapping
        enhanced_map = {}
        for i, data in enumerate(embeddings_data):
            if i < enhanced_embeddings.shape[0]:
                enhanced_map[data['chunk_id']] = enhanced_embeddings[i].numpy()
        
        logger.info(f"Enhanced {len(enhanced_map)} embeddings")
        return enhanced_map
    
    def store_enhanced_embeddings(self, enhanced_map: Dict[str, np.ndarray], db):
        """Store ISNE-enhanced embeddings in database."""
        logger.info("Storing enhanced embeddings...")
        
        isne_coll = db.collection('isne_embeddings')
        
        docs = []
        for chunk_id, embedding in enhanced_map.items():
            doc = {
                'chunk_id': chunk_id,
                'node_id': chunk_id.split('/')[-1],  # Extract node key
                'embedding': embedding.tolist(),
                'dimensions': len(embedding),
                'model_version': 'memory_efficient_isne_v1',
                'created_at': datetime.now().isoformat()
            }
            docs.append(doc)
        
        # Insert in batches
        batch_size = 1000
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i+batch_size]
            isne_coll.insert_many(batch)
            
        logger.info(f"Stored {len(docs)} enhanced embeddings")
    
    def create_enhanced_edges(
        self,
        enhanced_map: Dict[str, np.ndarray],
        db,
        similarity_threshold: float = 0.85,
        max_edges_per_node: int = 10
    ):
        """Create new edges based on enhanced embedding similarity."""
        logger.info("Creating enhanced edges from ISNE embeddings...")
        
        # Convert to arrays for efficient computation
        chunk_ids = list(enhanced_map.keys())
        embeddings = np.array([enhanced_map[cid] for cid in chunk_ids])
        
        logger.info(f"Computing similarity for {len(embeddings)} embeddings...")
        
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-8)
        
        # Get source file for each chunk
        chunk_to_file = {}
        query = """
        FOR c IN chunks
            RETURN {chunk_id: CONCAT('chunks/', c._key), source_file: c.source_file_id}
        """
        cursor = db.aql.execute(query)
        for result in cursor:
            chunk_to_file[result['chunk_id']] = result['source_file']
        
        # Find high-similarity pairs efficiently
        batch_size = 1000
        new_edges = []
        
        for i in range(0, len(normalized), batch_size):
            batch_end = min(i + batch_size, len(normalized))
            batch_embeddings = normalized[i:batch_end]
            
            # Compute similarities for this batch against all embeddings
            similarities = np.dot(batch_embeddings, normalized.T)
            
            # Find top-k similar nodes for each node in batch
            for batch_idx in range(batch_embeddings.shape[0]):
                node_idx = i + batch_idx
                node_similarities = similarities[batch_idx]
                
                # Exclude self-similarity
                node_similarities[node_idx] = -1
                
                # Find top-k most similar nodes
                if len(node_similarities) > max_edges_per_node:
                    top_k_indices = np.argpartition(node_similarities, -max_edges_per_node)[-max_edges_per_node:]
                    top_k_indices = top_k_indices[node_similarities[top_k_indices] > similarity_threshold]
                else:
                    top_k_indices = np.where(node_similarities > similarity_threshold)[0]
                
                source_chunk = chunk_ids[node_idx]
                source_file = chunk_to_file.get(source_chunk)
                
                for target_idx in top_k_indices:
                    target_chunk = chunk_ids[target_idx]
                    target_file = chunk_to_file.get(target_chunk)
                    
                    # Only create edges between different files
                    if source_file and target_file and source_file != target_file:
                        similarity = float(node_similarities[target_idx])
                        
                        new_edges.append({
                            '_from': source_chunk,
                            '_to': target_chunk,
                            'weight': similarity,
                            'confidence': similarity,
                            'edge_type': 'isne_enhanced',
                            'source_file': source_file,
                            'target_file': target_file,
                            'discovery_method': 'memory_efficient_isne',
                            'created_at': datetime.now().isoformat()
                        })
            
            logger.info(f"Processed batch {i//batch_size + 1}/{(len(normalized) + batch_size - 1)//batch_size}")
        
        # Insert new edges
        if new_edges:
            edge_coll = db.collection('isne_enhanced_edges')
            
            # Remove duplicates (keep highest weight)
            edge_dict = {}
            for edge in new_edges:
                key = (edge['_from'], edge['_to'])
                if key not in edge_dict or edge['weight'] > edge_dict[key]['weight']:
                    edge_dict[key] = edge
            
            unique_edges = list(edge_dict.values())
            
            # Insert in batches
            batch_size = 1000
            total_inserted = 0
            for i in range(0, len(unique_edges), batch_size):
                batch = unique_edges[i:i+batch_size]
                edge_coll.insert_many(batch)
                total_inserted += len(batch)
            
            logger.info(f"Created {total_inserted} new ISNE-enhanced edges")
            
            # Analyze edge distribution
            file_connections = defaultdict(set)
            for edge in unique_edges[:100]:  # Sample for analysis
                source_file = edge['source_file'].split('/')[-1]
                target_file = edge['target_file'].split('/')[-1]
                file_connections[source_file].add(target_file)
            
            logger.info("Sample new cross-file connections discovered:")
            for source, targets in sorted(file_connections.items())[:10]:
                target_list = list(targets)[:3]
                logger.info(f"  {source} → {', '.join(target_list)}")
        else:
            logger.info("No new edges created (similarity threshold too high?)")
    
    def run_pipeline(
        self,
        model_path: str,
        similarity_threshold: float = 0.85,
        max_edges_per_node: int = 10,
        create_new_db: bool = False
    ) -> Dict[str, Any]:
        """Run the complete ISNE application pipeline."""
        
        start_time = time.time()
        
        # Load model
        model, model_data = self.load_efficient_model(model_path)
        
        # Connect to source database
        source_client = ArangoClient()
        success = source_client.connect_to_database(self.source_db)
        if not success:
            raise Exception(f"Failed to connect to source database: {self.source_db}")
        source_db = source_client._database
        
        # Create/connect to target database
        if self.target_db != self.source_db:
            # Connect to target database (should exist)
            target_client = ArangoClient()
            success = target_client.connect_to_database(self.target_db)
            if not success:
                raise Exception(f"Failed to connect to target database: {self.target_db}")
            target_db = target_client._database
            
            # Copy base data if it's a new database
            if create_new_db:
                self.copy_base_data(source_db, target_db)
        else:
            target_db = source_db
        
        # Ensure required collections exist
        self.ensure_collections_exist(target_db)
        
        # Apply ISNE model
        enhanced_embeddings = self.apply_isne_model(model, target_db)
        
        # Store enhanced embeddings
        self.store_enhanced_embeddings(enhanced_embeddings, target_db)
        
        # Create new edges
        self.create_enhanced_edges(
            enhanced_embeddings,
            target_db,
            similarity_threshold,
            max_edges_per_node
        )
        
        # Calculate statistics
        total_time = time.time() - start_time
        
        results = {
            'success': True,
            'source_db': self.source_db,
            'target_db': self.target_db if create_new_db else self.source_db,
            'model_path': model_path,
            'processing_time_seconds': total_time,
            'enhanced_embeddings_count': len(enhanced_embeddings),
            'similarity_threshold': similarity_threshold,
            'max_edges_per_node': max_edges_per_node
        }
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description="Apply memory-efficient ISNE model to create enhanced embeddings and edges"
    )
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to trained memory-efficient ISNE model file'
    )
    parser.add_argument(
        '--source-db',
        type=str,
        default='isne_training_database',
        help='Source database with training data'
    )
    parser.add_argument(
        '--target-db',
        type=str,
        help='Target database for production (optional, updates source if not specified)'
    )
    parser.add_argument(
        '--similarity-threshold',
        type=float,
        default=0.85,
        help='Similarity threshold for creating new edges (0-1)'
    )
    parser.add_argument(
        '--max-edges-per-node',
        type=int,
        default=10,
        help='Maximum number of new edges per node'
    )
    parser.add_argument(
        '--create-new-db',
        action='store_true',
        help='Create new database for production'
    )
    parser.add_argument(
        '--log-file',
        type=str,
        help='Log file for debugging (optional)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_file)
    
    try:
        # Initialize pipeline
        pipeline = EfficientISNEApplicationPipeline(
            source_db=args.source_db,
            target_db=args.target_db
        )
        
        # Run pipeline
        results = pipeline.run_pipeline(
            model_path=args.model_path,
            similarity_threshold=args.similarity_threshold,
            max_edges_per_node=args.max_edges_per_node,
            create_new_db=args.create_new_db
        )
        
        # Print results
        print("\n" + "="*60)
        print("MEMORY-EFFICIENT ISNE MODEL APPLICATION COMPLETED")
        print("="*60)
        print(f"Source database: {results['source_db']}")
        print(f"Target database: {results['target_db']}")
        print(f"Processing time: {results['processing_time_seconds']:.2f} seconds")
        print(f"Enhanced embeddings: {results['enhanced_embeddings_count']:,}")
        print(f"\nISNE-discovered edges added to 'isne_enhanced_edges' collection")
        print("\nYou can now visualize the enhanced graph in ArangoDB!")
        
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())