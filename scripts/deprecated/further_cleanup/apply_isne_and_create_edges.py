#!/usr/bin/env python3
"""
Apply trained ISNE model to enhance embeddings and create new edges.

This script:
1. Loads a trained ISNE model
2. Applies it to embeddings in ArangoDB
3. Creates new edges based on enhanced embedding similarity
4. Can build a production database with enhanced graph structure
"""

import argparse
import logging
import sys
import time
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.arango_client import ArangoClient
from src.isne.models.isne_model import ISNEModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ISNEApplicationPipeline:
    """Apply ISNE model to create production database with enhanced edges."""
    
    def __init__(self, source_db: str, target_db: Optional[str] = None):
        """
        Initialize pipeline.
        
        Args:
            source_db: Source database name (with bootstrapped data)
            target_db: Target database name (production). If None, updates source.
        """
        self.source_db = source_db
        self.target_db = target_db or source_db
        self.update_in_place = (target_db is None)
        self.arango_client = ArangoClient()
        
    def load_trained_model(self, model_path: str) -> Tuple[ISNEModel, Dict[str, Any]]:
        """Load trained ISNE model and its configuration."""
        logger.info(f"Loading ISNE model from: {model_path}")
        
        # Load the saved model data
        save_data = torch.load(model_path, map_location='cpu')
        
        # Extract configuration
        model_config = save_data['model_config']
        num_nodes = model_config['num_nodes']
        embedding_dim = model_config['embedding_dim']
        
        # Initialize model
        model = ISNEModel(num_nodes, embedding_dim)
        model.load_state_dict(save_data['model_state_dict'])
        model.eval()
        
        logger.info(f"Model loaded: {num_nodes} nodes, {embedding_dim}D embeddings")
        
        return model, save_data
    
    def create_production_database(self, db_name: str):
        """Create production database with proper schema."""
        logger.info(f"Creating production database: {db_name}")
        
        # Create database using admin connection
        admin_client = ArangoClient()
        admin_client.create_database(db_name)
        
        # Connect to new database
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
        chunks_coll.add_hash_index(fields=['source_file_id'])
        chunks_coll.add_hash_index(fields=['chunk_index'])
        
        embeddings_coll = db.collection('embeddings')
        embeddings_coll.add_hash_index(fields=['chunk_id'])
        
        isne_embeddings_coll = db.collection('isne_embeddings')
        isne_embeddings_coll.add_hash_index(fields=['chunk_id'])
        isne_embeddings_coll.add_hash_index(fields=['node_id'])
        
        return db
    
    def copy_base_data(self, source_db, target_db):
        """Copy base data from source to target database."""
        logger.info("Copying base data to production database...")
        
        # Collections to copy
        collections_to_copy = [
            'code_files', 'documentation_files', 'config_files',
            'chunks', 'embeddings',
            'sequential_edges', 'directory_edges', 'cross_modal_edges', 'similarity_edges'
        ]
        
        for collection_name in collections_to_copy:
            # Get all documents from source
            query = f"FOR doc IN {collection_name} RETURN doc"
            cursor = source_db.aql.execute(query)
            docs = list(cursor)
            
            if docs:
                # Insert into target
                target_coll = target_db.collection(collection_name)
                target_coll.insert_many(docs, overwrite=False)
                logger.info(f"Copied {len(docs)} documents to {collection_name}")
    
    def apply_isne_model(self, model: ISNEModel, db) -> Dict[str, np.ndarray]:
        """Apply ISNE model to enhance embeddings."""
        logger.info("Applying ISNE model to embeddings...")
        
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
        
        # Prepare data for model
        node_features = []
        chunk_to_node_id = {}
        node_id_to_chunk = {}
        
        for i, data in enumerate(embeddings_data):
            node_features.append(data['embedding'])
            chunk_to_node_id[data['chunk_id']] = i
            node_id_to_chunk[i] = data['chunk_id']
        
        node_features = torch.tensor(node_features, dtype=torch.float32)
        
        # Get edges for model
        edge_list = []
        
        # Sequential edges
        seq_query = """
        FOR e IN sequential_edges
            RETURN {source: e._from, target: e._to}
        """
        cursor = db.aql.execute(seq_query)
        for edge in cursor:
            source_id = chunk_to_node_id.get(edge['source'])
            target_id = chunk_to_node_id.get(edge['target'])
            if source_id is not None and target_id is not None:
                edge_list.append([source_id, target_id])
        
        # Similarity edges
        sim_query = """
        FOR e IN similarity_edges
            RETURN {source: e._from, target: e._to}
        """
        cursor = db.aql.execute(sim_query)
        for edge in cursor:
            source_id = chunk_to_node_id.get(edge['source'])
            target_id = chunk_to_node_id.get(edge['target'])
            if source_id is not None and target_id is not None:
                edge_list.append([source_id, target_id])
        
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        
        # Apply model
        device = next(model.parameters()).device
        node_features = node_features.to(device)
        edge_index = edge_index.to(device)
        
        with torch.no_grad():
            enhanced_embeddings = model(node_features, edge_index)
            enhanced_embeddings = enhanced_embeddings.cpu().numpy()
        
        # Create mapping
        enhanced_map = {}
        for i, chunk_id in node_id_to_chunk.items():
            enhanced_map[chunk_id] = enhanced_embeddings[i]
        
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
                'model_version': 'isne_v1',
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
        logger.info("Computing similarity matrix...")
        
        # Process in batches to manage memory
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
                top_k_indices = np.argpartition(node_similarities, -max_edges_per_node)[-max_edges_per_node:]
                top_k_indices = top_k_indices[node_similarities[top_k_indices] > similarity_threshold]
                
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
                            'discovery_method': 'isne_similarity',
                            'created_at': datetime.now().isoformat()
                        })
        
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
            for edge in unique_edges:
                source_file = edge['source_file'].split('/')[-1]
                target_file = edge['target_file'].split('/')[-1]
                file_connections[source_file].add(target_file)
            
            logger.info("New cross-file connections discovered:")
            for source, targets in sorted(file_connections.items())[:10]:
                logger.info(f"  {source} → {', '.join(list(targets)[:5])}")
        else:
            logger.info("No new edges created (similarity threshold too high?)")
    
    def create_post_training_graph(self, db):
        """Create post-training graph for visualization."""
        logger.info("Creating post-training graph...")
        
        from scripts.create_arango_graph import ArangoGraphCreator
        
        creator = ArangoGraphCreator(db.name)
        
        # Create graph with ISNE edges
        graph_name = "post_training_graph"
        
        # Define edge definitions including ISNE edges
        edge_definitions = [
            {
                'edge_collection': 'directory_edges',
                'from_vertex_collections': ['code_files', 'documentation_files', 'config_files'],
                'to_vertex_collections': ['code_files', 'documentation_files', 'config_files']
            },
            {
                'edge_collection': 'cross_modal_edges',
                'from_vertex_collections': ['documentation_files'],
                'to_vertex_collections': ['code_files']
            },
            {
                'edge_collection': 'sequential_edges',
                'from_vertex_collections': ['chunks'],
                'to_vertex_collections': ['chunks']
            },
            {
                'edge_collection': 'similarity_edges',
                'from_vertex_collections': ['chunks'],
                'to_vertex_collections': ['chunks']
            },
            {
                'edge_collection': 'isne_enhanced_edges',
                'from_vertex_collections': ['chunks'],
                'to_vertex_collections': ['chunks']
            }
        ]
        
        # Check if graph exists
        if db.has_graph(graph_name):
            db.delete_graph(graph_name, drop_collections=False)
        
        # Create the graph
        graph = db.create_graph(
            name=graph_name,
            edge_definitions=edge_definitions,
            orphan_collections=['embeddings', 'isne_embeddings']
        )
        
        logger.info(f"Created post-training graph: {graph_name}")
        
        # Get statistics
        stats = creator.get_graph_statistics(graph_name)
        return stats
    
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
        model, model_data = self.load_trained_model(model_path)
        
        # Connect to source database
        source_client = ArangoClient()
        success = source_client.connect_to_database(self.source_db)
        if not success:
            raise Exception(f"Failed to connect to source database: {self.source_db}")
        source_db = source_client._database
        
        # Create/connect to target database
        if create_new_db and self.target_db != self.source_db:
            target_db = self.create_production_database(self.target_db)
            self.copy_base_data(source_db, target_db)
        else:
            target_db = source_db
        
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
        
        # Create visualization graph
        graph_stats = self.create_post_training_graph(target_db)
        
        # Calculate statistics
        total_time = time.time() - start_time
        
        results = {
            'success': True,
            'source_db': self.source_db,
            'target_db': self.target_db if create_new_db else self.source_db,
            'model_path': model_path,
            'processing_time_seconds': total_time,
            'enhanced_embeddings_count': len(enhanced_embeddings),
            'graph_stats': graph_stats,
            'similarity_threshold': similarity_threshold,
            'max_edges_per_node': max_edges_per_node
        }
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description="Apply ISNE model to create enhanced embeddings and edges"
    )
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to trained ISNE model file'
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
    
    args = parser.parse_args()
    
    try:
        # Initialize pipeline
        pipeline = ISNEApplicationPipeline(
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
        print("ISNE MODEL APPLICATION COMPLETED")
        print("="*60)
        print(f"Source database: {results['source_db']}")
        print(f"Target database: {results['target_db']}")
        print(f"Processing time: {results['processing_time_seconds']:.2f} seconds")
        print(f"Enhanced embeddings: {results['enhanced_embeddings_count']:,}")
        print(f"\nGraph statistics:")
        for key, value in results['graph_stats'].items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v:,}")
            else:
                print(f"  {key}: {value:,}")
        
        print(f"\nISNE-discovered edges added to 'isne_enhanced_edges' collection")
        print(f"Post-training graph created: 'post_training_graph'")
        print("\nYou can now visualize the enhanced graph in ArangoDB!")
        
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())