"""
Graph construction stage for ISNE bootstrap using .hades metadata.
"""

import logging
import os
import json
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
from arango import ArangoClient

logger = logging.getLogger(__name__)


class StageResult:
    """Result from a pipeline stage execution."""
    def __init__(self, success: bool, data: Any = None, error: Optional[str] = None):
        self.success = success
        self.data = data
        self.error = error


class GraphConstructionStage:
    """Stage for constructing knowledge graph.
    
    Constructs the graph using:
    - Chunked embeddings from late chunking
    - .hades metadata for relationships
    - Filesystem structure for hierarchy
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize graph construction stage."""
        self.config = config
        self.similarity_threshold = config.get("similarity_threshold", 0.7)
        self.max_neighbors = config.get("max_neighbors", 10)
        self.arango_url = config.get("arango_url", "http://localhost:8529")
        self.database = config.get("database", "hades")
        self.username = config.get("username", "root")
        self.password = config.get("password", "")
    
    def execute(self, chunked_data: Any, config: Optional[Dict[str, Any]] = None) -> StageResult:
        """Execute graph construction.
        
        Args:
            chunked_data: Chunked embeddings from chunking stage
            config: Optional override configuration
            
        Returns:
            StageResult with graph data
        """
        try:
            # Use provided config or fall back to instance config
            cfg = config or self.config
            
            # TODO: Implement actual graph construction
            # This would:
            # 1. Create nodes from chunks
            # 2. Add embeddings to nodes
            # 3. Create edges from .hades relationships
            # 4. Add filesystem-based edges
            
            graph_data = {
                "nodes": [],
                "edges": [],
                "graph_stats": {
                    "total_nodes": 0,
                    "total_edges": 0,
                    "edge_types": {},
                    "avg_degree": 0.0
                }
            }
            return StageResult(success=True, data=graph_data)
        except Exception as e:
            logger.error(f"Graph construction failed: {e}")
            return StageResult(success=False, error=str(e))
    
    def run(self, embeddings: Dict[str, Any]) -> Dict[str, Any]:
        """Legacy run method for backward compatibility."""
        logger.info("Constructing knowledge graph from embeddings")
        
        # Convert embeddings format for execute method
        result = self.execute(embeddings)
        
        if result.success:
            stats = result.data.get("graph_stats", {})
            return {
                "total_nodes": stats.get("total_nodes", 0),
                "total_edges": stats.get("total_edges", 0),
                "edge_types": stats.get("edge_types", {}),
                "avg_degree": stats.get("avg_degree", 0.0),
                "errors": []
            }
        else:
            return {
                "total_nodes": 0,
                "total_edges": 0,
                "edge_types": {},
                "avg_degree": 0.0,
                "errors": [result.error]
            }


def create_stage(config: Dict[str, Any]) -> GraphConstructionStage:
    """Factory function to create stage."""
    return GraphConstructionStage(config)


async def construct_graph(config: Dict[str, Any], root_path: str) -> Dict[str, Any]:
    """Construct graph from .hades directories for ISNE training.
    
    This creates a multimodal graph in ArangoDB:
    1. Scans for .hades/ directories
    2. Creates nodes with Jina v4 embeddings
    3. Uses .hades/relationships.json for explicit edges
    4. Adds filesystem edges (parent-child, sibling)
    5. Prepares graph for ISNE training
    
    Args:
        config: Configuration dictionary
        root_path: Root directory to scan
        
    Returns:
        Graph construction results
    """
    logger.info("Constructing multimodal graph from .hades metadata...")
    
    # Connect to ArangoDB
    client = ArangoClient(hosts=config.get('arango_url', 'http://localhost:8529'))
    db = client.db(config.get('database', 'hades'), 
                   username=config.get('username', 'root'),
                   password=config.get('password', ''))
    
    # Collections
    nodes_collection = db.collection('nodes')
    filesystem_edges = db.collection('filesystem_edges')
    semantic_edges = db.collection('semantic_edges')
    functional_edges = db.collection('functional_edges')
    
    nodes_created = 0
    edges_created = 0
    hades_dirs = []
    
    # Phase 1: Scan for .hades directories and create nodes
    for root, dirs, files in os.walk(root_path):
        if '.hades' in dirs:
            hades_dirs.append(root)
            
            # Load .hades metadata
            metadata_path = os.path.join(root, '.hades', 'metadata.json')
            relationships_path = os.path.join(root, '.hades', 'relationships.json')
            
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Create directory node
                dir_node = {
                    '_key': path_to_key(root),
                    'filesystem_path': root,
                    'type': 'directory',
                    'has_hades': True,
                    'metadata': metadata,
                    'embeddings': {}  # Will be populated by Jina v4
                }
                
                # Generate Jina v4 embedding for directory
                # (Using metadata content as proxy for directory)
                dir_content = json.dumps(metadata, indent=2)
                dir_node['embeddings']['jina_v4'] = await generate_jina_embedding(dir_content)
                
                nodes_collection.insert(dir_node, overwrite=True)
                nodes_created += 1
                
                # Create nodes for files in this directory
                for file in files:
                    if file.startswith('.'):
                        continue
                        
                    file_path = os.path.join(root, file)
                    file_node: Dict[str, Any] = {
                        '_key': path_to_key(file_path),
                        'filesystem_path': file_path,
                        'type': 'file',
                        'parent_dir': root,
                        'embeddings': {}
                    }
                    
                    # Generate Jina v4 embedding for file
                    if file.endswith('.py'):
                        with open(file_path, 'r') as f:
                            content = f.read()
                        file_node['embeddings']['jina_v4'] = await generate_jina_embedding(content)
                    
                    nodes_collection.insert(file_node, overwrite=True)
                    nodes_created += 1
        else:
            # Don't descend into non-.hades directories
            dirs[:] = []
    
    # Phase 2: Create edges from .hades/relationships.json
    for hades_dir in hades_dirs:
        relationships_path = os.path.join(hades_dir, '.hades', 'relationships.json')
        
        if os.path.exists(relationships_path):
            with open(relationships_path, 'r') as f:
                relationships = json.load(f)
            
            source_key = path_to_key(hades_dir)
            
            # Process explicit relationships for ISNE
            for rel in relationships.get('explicit_relationships', []):
                target_path = rel['target']
                rel_type = rel['type']
                strength = rel.get('strength', 1.0)
                
                # Only create edge if target has .hades
                if os.path.exists(os.path.join(target_path, '.hades')):
                    target_key = path_to_key(target_path)
                    
                    # Create edge in appropriate collection
                    edge_data = {
                        '_from': f'nodes/{source_key}',
                        '_to': f'nodes/{target_key}',
                        'type': rel_type,
                        'strength': strength,
                        'source': 'hades_metadata'
                    }
                    
                    if rel_type in ['depends_on', 'imports', 'calls']:
                        functional_edges.insert(edge_data, overwrite=True)
                    elif rel_type in ['similar_to', 'related_to']:
                        semantic_edges.insert(edge_data, overwrite=True)
                    else:
                        filesystem_edges.insert(edge_data, overwrite=True)
                    
                    edges_created += 1
    
    # Phase 3: Create filesystem edges (parent-child, siblings)
    for hades_dir in hades_dirs:
        dir_key = path_to_key(hades_dir)
        
        # Parent-child edges
        parent_dir = os.path.dirname(hades_dir)
        if parent_dir and os.path.exists(os.path.join(parent_dir, '.hades')):
            edge_data = {
                '_from': f'nodes/{path_to_key(parent_dir)}',
                '_to': f'nodes/{dir_key}',
                'type': 'parent_child',
                'strength': 1.0,
                'source': 'filesystem'
            }
            filesystem_edges.insert(edge_data, overwrite=True)
            edges_created += 1
        
        # Sibling edges (directories at same level)
        if parent_dir:
            for sibling in os.listdir(parent_dir):
                sibling_path = os.path.join(parent_dir, sibling)
                if (os.path.isdir(sibling_path) and 
                    sibling_path != hades_dir and
                    os.path.exists(os.path.join(sibling_path, '.hades'))):
                    
                    edge_data = {
                        '_from': f'nodes/{dir_key}',
                        '_to': f'nodes/{path_to_key(sibling_path)}',
                        'type': 'sibling',
                        'strength': 0.7,
                        'source': 'filesystem'
                    }
                    filesystem_edges.insert(edge_data, overwrite=True)
                    edges_created += 1
    
    logger.info(f"Graph construction completed: {nodes_created} nodes, {edges_created} edges")
    
    return {
        "status": "completed",
        "nodes_created": nodes_created,
        "edges_created": edges_created,
        "hades_directories": len(hades_dirs),
        "collections": {
            "nodes": nodes_created,
            "filesystem_edges": filesystem_edges.count(),
            "semantic_edges": semantic_edges.count(),
            "functional_edges": functional_edges.count()
        },
        "errors": []
    }


def path_to_key(path: str) -> str:
    """Convert filesystem path to ArangoDB key."""
    # Remove leading slash and replace path separators
    return path.strip('/').replace('/', '_').replace('.', '_')


async def generate_jina_embedding(content: str) -> List[float]:
    """Generate Jina v4 embedding for content."""
    # TODO: Integrate with actual Jina v4 processor
    # For now, return placeholder
    import hashlib
    hash_val = int(hashlib.md5(content.encode()).hexdigest()[:8], 16)
    return [float(hash_val % 100) / 100.0] * 768  # Jina v4 dimension


__all__ = [
    "GraphConstructionStage",
    "StageResult",
    "create_stage",
    "construct_graph",
    "path_to_key",
    "generate_jina_embedding"
]
