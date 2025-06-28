"""
Main bootstrap pipeline for creating supra-weight graph in ArangoDB.
"""

from typing import Dict, List, Any, Optional, Callable, Union
from pathlib import Path
import logging
import time
from datetime import datetime
import json
from tqdm import tqdm

from ..core import SupraWeightCalculator, RelationshipDetector, DensityController
from ..core.density_controller import EdgeCandidate
from ..processing import DocumentProcessor, ChunkProcessor, EmbeddingProcessor
from ..storage import ArangoSupraStorage, BatchWriter

logger = logging.getLogger(__name__)


class SupraWeightBootstrapPipeline:
    """
    Main pipeline for bootstrapping a supra-weight graph for ISNE training.
    
    This pipeline:
    1. Processes documents/chunks into nodes
    2. Detects multi-dimensional relationships
    3. Calculates supra-weights
    4. Controls graph density
    5. Stores the graph in ArangoDB
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the pipeline.
        
        Args:
            config: Configuration dictionary with sections:
                - database: ArangoDB connection settings
                - bootstrap: Bootstrap parameters
                - supra_weight: Supra-weight calculation settings
                - density: Density control settings
        """
        self.config = config
        
        # Initialize components
        self._initialize_components()
        
        # Statistics
        self.stats: Dict[str, Any] = {
            'start_time': None,
            'end_time': None,
            'nodes_processed': 0,
            'edges_created': 0,
            'relationships_detected': {},
            'errors': []
        }
        
        logger.info("Initialized SupraWeightBootstrapPipeline")
        
    def _initialize_components(self) -> None:
        """Initialize pipeline components from config."""
        # Storage
        db_config = self.config.get('database', {})
        self.storage = ArangoSupraStorage(
            connection_url=db_config.get('url', 'http://localhost:8529'),
            username=db_config.get('username', 'root'),
            password=db_config.get('password', ''),
            database_name=db_config.get('database', 'isne_bootstrap')
        )
        
        # Bootstrap settings
        bootstrap_config = self.config.get('bootstrap', {})
        self.batch_size = bootstrap_config.get('batch_size', 1000)
        self.max_edges_per_node = bootstrap_config.get('max_edges_per_node', 50)
        self.min_edge_weight = bootstrap_config.get('min_edge_weight', 0.3)
        
        # Supra-weight calculator
        sw_config = self.config.get('supra_weight', {})
        self.supra_calculator = SupraWeightCalculator(
            method=sw_config.get('aggregation_method', 'adaptive'),
            importance_weights=sw_config.get('importance_weights')
        )
        
        # Relationship detector
        self.relationship_detector = RelationshipDetector(
            semantic_threshold=sw_config.get('semantic_threshold', 0.5)
        )
        
        # Density controller
        density_config = self.config.get('density', {})
        self.density_controller = DensityController(
            max_edges_per_node=self.max_edges_per_node,
            min_edge_weight=self.min_edge_weight,
            target_density=density_config.get('target_density'),
            local_density_factor=density_config.get('local_density_factor', 2.0)
        )
        
        # Processors
        self.doc_processor = DocumentProcessor()
        self.chunk_processor = ChunkProcessor()
        self.embedding_processor = EmbeddingProcessor(
            similarity_threshold=sw_config.get('semantic_threshold', 0.5),
            batch_size=bootstrap_config.get('similarity_batch_size', 1000)
        )
        
    def run(self,
            input_source: Union[str, Path, List[Dict[str, Any]]],
            progress_callback: Optional[Callable[[Dict], None]] = None) -> Dict[str, Any]:
        """
        Run the bootstrap pipeline.
        
        Args:
            input_source: Either:
                - Path to directory with documents
                - Path to JSON file with nodes/edges
                - List of node dictionaries
            progress_callback: Optional callback for progress updates
            
        Returns:
            Pipeline statistics and results
        """
        self.stats['start_time'] = datetime.utcnow()
        logger.info(f"Starting bootstrap pipeline at {self.stats['start_time']}")
        
        try:
            # Initialize storage collections
            self.storage.initialize_collections()
            
            # Process input to get nodes
            nodes = self._process_input(input_source)
            logger.info(f"Processed {len(nodes)} nodes")
            
            # Validate embeddings if present
            if any('embedding' in node for node in nodes):
                valid, errors = self.embedding_processor.validate_embeddings(nodes)
                if not valid:
                    logger.warning(f"Embedding validation errors: {errors}")
                    
            # Build graph with batch processing
            self._build_graph_batch(nodes, progress_callback)
            
            # Create named graph
            self.storage.create_graph()
            
            # Get final statistics
            graph_stats = self.storage.get_graph_statistics()
            
            self.stats['end_time'] = datetime.utcnow()
            self.stats['duration_seconds'] = (
                self.stats['end_time'] - self.stats['start_time']
            ).total_seconds()
            
            # Combine statistics
            return {
                **self.stats,
                **graph_stats,
                'density_control': self.density_controller.get_statistics()
            }
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            self.stats['errors'].append(str(e))
            raise
            
    def _process_input(self, input_source: Union[str, Path, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Process various input sources into nodes."""
        if isinstance(input_source, list):
            # Direct node list
            return self.doc_processor.process_nodes(input_source)
            
        input_path = Path(input_source)
        
        if input_path.is_file() and input_path.suffix == '.json':
            # JSON file with nodes
            with open(input_path) as f:
                data = json.load(f)
                if isinstance(data, list):
                    return self.doc_processor.process_nodes(data)
                elif isinstance(data, dict) and 'nodes' in data:
                    return self.doc_processor.process_nodes(data['nodes'])
                else:
                    raise ValueError("JSON file must contain list of nodes or dict with 'nodes' key")
                    
        elif input_path.is_dir():
            # Directory of documents
            nodes = []
            extensions = self.config.get('bootstrap', {}).get('file_extensions', ['.py', '.js', '.md'])
            
            for node in self.doc_processor.process_directory(input_path, extensions=extensions):
                nodes.append(node)
                
            return nodes
            
        else:
            raise ValueError(f"Invalid input source: {input_source}")
            
    def _build_graph_batch(self, 
                          nodes: List[Dict[str, Any]], 
                          progress_callback: Optional[Callable[[Dict], None]] = None) -> None:
        """Build graph using batch processing."""
        logger.info(f"Building graph from {len(nodes)} nodes")
        
        # Initialize batch writer
        with BatchWriter(self.storage, batch_size=self.batch_size) as writer:
            # Add nodes
            writer.add_nodes(nodes)
            self.stats['nodes_processed'] = len(nodes)
            
            # Create embedding index for efficiency
            embedding_index = self.embedding_processor.create_embedding_index(nodes)
            
            # Process relationships in batches
            total_pairs = len(nodes) * (len(nodes) - 1) // 2
            processed_pairs = 0
            
            # Use progress bar if available
            pbar = tqdm(total=total_pairs, desc="Processing relationships")
            
            # Process node pairs
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    # Get embeddings if available
                    embeddings = None
                    if nodes[i]['node_id'] in embedding_index and nodes[j]['node_id'] in embedding_index:
                        embeddings = (
                            embedding_index[nodes[i]['node_id']],
                            embedding_index[nodes[j]['node_id']]
                        )
                        
                    # Detect relationships
                    relationships = self.relationship_detector.detect_all_relationships(
                        nodes[i], nodes[j], embeddings
                    )
                    
                    if relationships:
                        # Calculate supra-weight
                        weight, weight_vector = self.supra_calculator.calculate(relationships)
                        
                        # Create edge candidate
                        edge_candidate = EdgeCandidate(
                            from_node=nodes[i]['node_id'],
                            to_node=nodes[j]['node_id'],
                            weight=weight,
                            relationships=relationships,
                            weight_vector=weight_vector
                        )
                        
                        # Add to density controller
                        self.density_controller.add_edge_candidate(edge_candidate)
                        
                        # Track relationship types
                        for rel in relationships:
                            rel_type = rel.type.value
                            if isinstance(self.stats['relationships_detected'], dict):
                                self.stats['relationships_detected'][rel_type] = \
                                    self.stats['relationships_detected'].get(rel_type, 0) + 1
                                
                    processed_pairs += 1
                    pbar.update(1)
                    
                    # Progress callback
                    if progress_callback and processed_pairs % 1000 == 0:
                        progress_callback({
                            'processed_pairs': processed_pairs,
                            'total_pairs': total_pairs,
                            'progress': processed_pairs / total_pairs
                        })
                        
                # Process batch of edges with density control
                if self.density_controller.edge_candidates:
                    accepted_edges = self.density_controller.process_edge_batch(len(nodes))
                    writer.add_edges(accepted_edges)
                    if isinstance(self.stats['edges_created'], int):
                        self.stats['edges_created'] += len(accepted_edges)
                    
            pbar.close()
            
            # Store final metadata
            writer.add_metadata('pipeline_config', self.config)
            writer.add_metadata('pipeline_stats', self.stats)
            writer.add_metadata('bootstrap_timestamp', datetime.utcnow().isoformat())
            
        logger.info(f"Graph building complete: {self.stats['edges_created']} edges created")
        
    def validate_graph(self) -> Dict[str, Any]:
        """Validate the constructed graph."""
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'errors': []
        }
        
        stats = self.storage.get_graph_statistics()
        
        # Check node count
        if stats['node_count'] == 0:
            if isinstance(validation_results['errors'], list):
                validation_results['errors'].append("No nodes in graph")
            validation_results['is_valid'] = False
            
        # Check edge count
        if stats['edge_count'] == 0:
            if isinstance(validation_results['warnings'], list):
                validation_results['warnings'].append("No edges in graph")
            
        # Check density
        density = stats.get('density', 0)
        if density > 0.5:
            if isinstance(validation_results['warnings'], list):
                validation_results['warnings'].append(f"High graph density: {density:.3f}")
            
        # Check degree distribution
        degree_stats = stats.get('degree_statistics', {})
        max_degree = degree_stats.get('max_out_degree', 0)
        
        if max_degree > self.max_edges_per_node * 1.5:
            if isinstance(validation_results['warnings'], list):
                validation_results['warnings'].append(
                    f"Some nodes exceed expected max degree: {max_degree} > {self.max_edges_per_node * 1.5}"
                )
            
        return validation_results