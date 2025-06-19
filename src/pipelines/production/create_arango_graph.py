#!/usr/bin/env python3
"""
Create ArangoDB Graph definitions for visualization.

This script creates graph structures in ArangoDB from our existing collections
to enable visual exploration and real-time monitoring of graph evolution.
"""

import logging
import argparse
import sys
from pathlib import Path
from typing import Dict, List

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.database.arango_client import ArangoClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ArangoGraphCreator:
    """Create ArangoDB graph definitions for visualization."""
    
    def __init__(self, db_name: str):
        self.db_name = db_name
        self.arango_client = ArangoClient()
        
    def create_bootstrap_graph(self, graph_name: str = "bootstrap_graph"):
        """Create graph definition for bootstrap data (Stage 1)."""
        logger.info(f"Creating bootstrap graph: {graph_name}")
        
        # Connect to database
        success = self.arango_client.connect_to_database(self.db_name)
        if not success:
            raise Exception(f"Failed to connect to database: {self.db_name}")
            
        db = self.arango_client._database
        
        # Define edge definitions for bootstrap graph
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
            }
        ]
        
        # Check if graph already exists
        if db.has_graph(graph_name):
            logger.info(f"Graph {graph_name} already exists, dropping and recreating...")
            db.delete_graph(graph_name, drop_collections=False)
            
        # Create the graph
        graph = db.create_graph(
            name=graph_name,
            edge_definitions=edge_definitions,
            orphan_collections=['embeddings', 'isne_models', 'processing_logs']  # Collections not in edge definitions
        )
        
        logger.info(f"✅ Bootstrap graph '{graph_name}' created successfully!")
        return graph
        
    def create_isne_enhanced_graph(self, graph_name: str = "isne_enhanced_graph"):
        """Create graph definition for ISNE-enhanced data (Stage 2)."""
        logger.info(f"Creating ISNE enhanced graph: {graph_name}")
        
        # Connect to database
        success = self.arango_client.connect_to_database(self.db_name)
        if not success:
            raise Exception(f"Failed to connect to database: {self.db_name}")
            
        db = self.arango_client._database
        
        # Define edge definitions (includes ISNE-discovered edges)
        edge_definitions = [
            # Original bootstrap edges
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
            # ISNE-discovered edges (to be created by Stage 2)
            {
                'edge_collection': 'isne_enhanced_edges',
                'from_vertex_collections': ['chunks', 'code_files', 'documentation_files', 'config_files'],
                'to_vertex_collections': ['chunks', 'code_files', 'documentation_files', 'config_files']
            }
        ]
        
        # Check if graph already exists
        if db.has_graph(graph_name):
            logger.info(f"Graph {graph_name} already exists, dropping and recreating...")
            db.delete_graph(graph_name, drop_collections=False)
            
        # Create the enhanced graph
        graph = db.create_graph(
            name=graph_name,
            edge_definitions=edge_definitions,
            orphan_collections=['embeddings', 'isne_models', 'processing_logs']
        )
        
        logger.info(f"✅ ISNE enhanced graph '{graph_name}' created successfully!")
        return graph
        
    def get_graph_statistics(self, graph_name: str) -> Dict:
        """Get statistics about the current graph."""
        logger.info(f"Getting statistics for graph: {graph_name}")
        
        # Connect to database
        success = self.arango_client.connect_to_database(self.db_name)
        if not success:
            raise Exception(f"Failed to connect to database: {self.db_name}")
            
        db = self.arango_client._database
        
        if not db.has_graph(graph_name):
            logger.error(f"Graph {graph_name} does not exist")
            return {}
            
        graph = db.graph(graph_name)
        
        # Collect statistics
        stats = {
            'graph_name': graph_name,
            'vertex_collections': {},
            'edge_collections': {},
            'total_vertices': 0,
            'total_edges': 0
        }
        
        # Get graph properties
        graph_info = graph.properties()
        
        # Count vertices in each collection
        vertex_collections = [ed['from_vertex_collections'] + ed['to_vertex_collections'] 
                             for ed in graph_info.get('edge_definitions', [])]
        vertex_collections = list(set([item for sublist in vertex_collections for item in sublist]))
        vertex_collections.extend(graph_info.get('orphan_collections', []))
        
        for collection_name in set(vertex_collections):
            try:
                collection = db.collection(collection_name)
                count = collection.count()
                stats['vertex_collections'][collection_name] = count
                stats['total_vertices'] += count
            except:
                continue
            
        # Count edges in each collection
        edge_collections = [ed['edge_collection'] for ed in graph_info.get('edge_definitions', [])]
        
        for collection_name in edge_collections:
            try:
                collection = db.collection(collection_name)
                count = collection.count()
                stats['edge_collections'][collection_name] = count
                stats['total_edges'] += count
            except:
                continue
            
        return stats
        
    def list_available_graphs(self) -> List[str]:
        """List all graphs in the database."""
        # Connect to database
        success = self.arango_client.connect_to_database(self.db_name)
        if not success:
            raise Exception(f"Failed to connect to database: {self.db_name}")
            
        db = self.arango_client._database
        graphs = [graph['name'] for graph in db.graphs()]
        return graphs


def main():
    parser = argparse.ArgumentParser(description="Create ArangoDB graphs for visualization")
    parser.add_argument(
        '--db-name',
        type=str,
        default='isne_training_database',
        help='Database name'
    )
    parser.add_argument(
        '--action',
        type=str,
        choices=['create-bootstrap', 'create-enhanced', 'stats', 'list'],
        default='create-bootstrap',
        help='Action to perform'
    )
    parser.add_argument(
        '--graph-name',
        type=str,
        help='Specific graph name (optional)'
    )
    
    args = parser.parse_args()
    
    creator = ArangoGraphCreator(args.db_name)
    
    try:
        if args.action == 'create-bootstrap':
            graph_name = args.graph_name or 'bootstrap_graph'
            graph = creator.create_bootstrap_graph(graph_name)
            stats = creator.get_graph_statistics(graph_name)
            
            print("\n" + "="*60)
            print("BOOTSTRAP GRAPH CREATED")
            print("="*60)
            print(f"Graph name: {graph_name}")
            print(f"Database: {args.db_name}")
            print(f"Total vertices: {stats['total_vertices']:,}")
            print(f"Total edges: {stats['total_edges']:,}")
            print("\nVertex collections:")
            for coll, count in stats['vertex_collections'].items():
                print(f"  {coll}: {count:,}")
            print("\nEdge collections:")
            for coll, count in stats['edge_collections'].items():
                print(f"  {coll}: {count:,}")
                
        elif args.action == 'create-enhanced':
            graph_name = args.graph_name or 'isne_enhanced_graph'
            graph = creator.create_isne_enhanced_graph(graph_name)
            stats = creator.get_graph_statistics(graph_name)
            
            print("\n" + "="*60)
            print("ISNE ENHANCED GRAPH CREATED")
            print("="*60)
            print(f"Graph name: {graph_name}")
            print(f"Database: {args.db_name}")
            print(f"Total vertices: {stats['total_vertices']:,}")
            print(f"Total edges: {stats['total_edges']:,}")
            print("\nVertex collections:")
            for coll, count in stats['vertex_collections'].items():
                print(f"  {coll}: {count:,}")
            print("\nEdge collections:")
            for coll, count in stats['edge_collections'].items():
                print(f"  {coll}: {count:,}")
                
        elif args.action == 'stats':
            graphs = creator.list_available_graphs()
            if not graphs:
                print("No graphs found in database")
                return
                
            for graph_name in graphs:
                stats = creator.get_graph_statistics(graph_name)
                print(f"\n📊 Graph: {graph_name}")
                print(f"   Vertices: {stats['total_vertices']:,}")
                print(f"   Edges: {stats['total_edges']:,}")
                
        elif args.action == 'list':
            graphs = creator.list_available_graphs()
            print(f"\n📋 Available graphs in '{args.db_name}':")
            if graphs:
                for graph in graphs:
                    print(f"  - {graph}")
            else:
                print("  No graphs found")
                
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()