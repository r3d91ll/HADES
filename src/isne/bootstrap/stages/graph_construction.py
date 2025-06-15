"""
Graph Construction Stage for ISNE Bootstrap Pipeline

Handles construction of knowledge graphs from embeddings and document
relationships for subsequent ISNE training.
"""

import logging
import traceback
import time
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

# Note: GraphConstructionInput not available in contracts, using direct implementation
from .base import BaseBootstrapStage

logger = logging.getLogger(__name__)


@dataclass
class GraphConstructionResult:
    """Result of graph construction stage."""
    success: bool
    graph_data: Dict[str, Any]  # Graph structure and metadata
    stats: Dict[str, Any]
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None


class GraphConstructionStage(BaseBootstrapStage):
    """Graph construction stage for bootstrap pipeline."""
    
    def __init__(self):
        """Initialize graph construction stage."""
        super().__init__("graph_construction")
        
    def execute(self, embeddings: List[Any], config: Any, output_dir: Optional[Path] = None) -> GraphConstructionResult:
        """
        Execute graph construction stage.
        
        Args:
            embeddings: List of embedding objects from embedding stage
            config: GraphConstructionConfig object
            
        Returns:
            GraphConstructionResult with graph data and stats
        """
        logger.info(f"Starting graph construction stage with {len(embeddings)} embeddings")
        
        try:
            # Extract embedding vectors and metadata
            embedding_vectors = []
            node_metadata = []
            
            for i, emb in enumerate(embeddings):
                if hasattr(emb, 'embedding') and len(emb.embedding) > 0:
                    embedding_vectors.append(emb.embedding)
                    node_metadata.append({
                        'node_id': i,
                        'chunk_id': getattr(emb, 'chunk_id', f'chunk_{i}'),
                        'document_id': getattr(emb, 'document_id', f'doc_{i}'),
                        'content': getattr(emb, 'content', ''),
                        'metadata': getattr(emb, 'metadata', {})
                    })
                else:
                    logger.warning(f"Skipping embedding {i} with empty/invalid embedding vector")
            
            if not embedding_vectors:
                error_msg = "No valid embeddings found for graph construction"
                logger.error(error_msg)
                return GraphConstructionResult(
                    success=False,
                    graph_data={},
                    stats={},
                    error_message=error_msg
                )
            
            logger.info(f"Processing {len(embedding_vectors)} valid embeddings")
            
            # Convert to numpy array for similarity computation
            embedding_matrix = np.array(embedding_vectors)
            logger.info(f"Embedding matrix shape: {embedding_matrix.shape}")
            
            # Use GPU-accelerated similarity computation
            construction_start = time.time()
            logger.info("Computing similarities using GPU acceleration...")
            
            # Check if CUDA is available and use it
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"Using device: {device}")
            
            # Convert to torch tensors and move to GPU
            embedding_tensor = torch.from_numpy(embedding_matrix).float().to(device)
            logger.info(f"Moved embeddings to {device}, shape: {embedding_tensor.shape}")
            
            # Normalize embeddings for cosine similarity
            normalized_embeddings = torch.nn.functional.normalize(embedding_tensor, p=2, dim=1)
            logger.info("Normalized embeddings on GPU")
            
            # Construct edges based on similarity threshold using GPU computation
            logger.info(f"Constructing edges with similarity threshold {config.similarity_threshold}")
            edges = []
            num_nodes = len(embedding_vectors)
            batch_size = min(2000, num_nodes)  # Larger batches for GPU
            
            for batch_start in range(0, num_nodes, batch_size):
                batch_end = min(batch_start + batch_size, num_nodes)
                if batch_start % 10000 == 0:  # Log progress every 10k nodes
                    logger.info(f"Processing similarity batch {batch_start}-{batch_end} ({batch_start/num_nodes*100:.1f}%)")
                
                # Compute similarities for this batch against all nodes using GPU
                batch_embeddings = normalized_embeddings[batch_start:batch_end]
                
                # Use torch.mm for GPU matrix multiplication
                with torch.no_grad():  # Save memory during inference
                    batch_similarities = torch.mm(batch_embeddings, normalized_embeddings.T)
                    
                    # Convert back to CPU for processing (smaller chunks)
                    batch_similarities_cpu = batch_similarities.cpu().numpy()
                
                # Clear GPU memory
                del batch_similarities
                torch.cuda.empty_cache() if device.type == 'cuda' else None
                
                for i in range(batch_end - batch_start):
                    global_i = batch_start + i
                    similarities = batch_similarities_cpu[i]
                    
                    # Set self-similarity to -1 to exclude it
                    similarities[global_i] = -1
                    
                    # Get indices of nodes above threshold
                    above_threshold = np.where(similarities >= config.similarity_threshold)[0]
                    
                    if len(above_threshold) > 0:
                        # Sort by similarity and take top-k
                        sorted_indices = above_threshold[np.argsort(similarities[above_threshold])[::-1]]
                        top_k_indices = sorted_indices[:config.max_edges_per_node]
                        
                        # Create edges
                        for j in top_k_indices:
                            edge = {
                                'source': global_i,
                                'target': int(j),
                                'weight': float(similarities[j]),
                                'edge_type': 'semantic_similarity'
                            }
                            edges.append(edge)
            
            # Add metadata-based edges if requested
            if config.include_metadata_edges:
                logger.info("Adding metadata-based edges...")
                metadata_edges = self._create_metadata_edges(node_metadata, config)
                edges.extend(metadata_edges)
            
            construction_duration = time.time() - construction_start
            
            # Create graph data structure
            graph_data = {
                'nodes': [
                    {
                        'id': i,
                        'embedding': embedding_vectors[i].tolist() if hasattr(embedding_vectors[i], 'tolist') else list(embedding_vectors[i]),
                        **node_metadata[i]
                    }
                    for i in range(len(embedding_vectors))
                ],
                'edges': edges,
                'metadata': {
                    'construction_method': 'similarity_threshold',
                    'similarity_threshold': config.similarity_threshold,
                    'max_edges_per_node': config.max_edges_per_node,
                    'include_metadata_edges': config.include_metadata_edges,
                    'embedding_dimension': embedding_matrix.shape[1],
                    'construction_timestamp': time.time()
                }
            }
            
            # Calculate statistics
            total_edges = len(edges)
            semantic_edges = len([e for e in edges if e['edge_type'] == 'semantic_similarity'])
            metadata_edges_count = total_edges - semantic_edges
            
            stats = {
                'num_nodes': num_nodes,
                'num_edges': total_edges,
                'semantic_edges': semantic_edges,
                'metadata_edges': metadata_edges_count,
                'avg_edges_per_node': total_edges / num_nodes if num_nodes > 0 else 0,
                'construction_time_seconds': construction_duration,
                'similarity_threshold': config.similarity_threshold,
                'max_edges_per_node': config.max_edges_per_node,
                'embedding_dimension': embedding_matrix.shape[1],
                'graph_density': (2 * total_edges) / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0,
                'construction_config': {
                    'similarity_threshold': config.similarity_threshold,
                    'max_edges_per_node': config.max_edges_per_node,
                    'include_metadata_edges': config.include_metadata_edges
                }
            }
            
            # Quality checks
            if stats['avg_edges_per_node'] < 1.0:
                warning_msg = f"Low graph connectivity: {stats['avg_edges_per_node']:.2f} edges per node"
                logger.warning(warning_msg)
                stats['quality_warnings'] = [warning_msg]
            
            if stats['graph_density'] < 0.01:
                warning_msg = f"Very sparse graph: density {stats['graph_density']:.4f}"
                logger.warning(warning_msg)
                stats.setdefault('quality_warnings', []).append(warning_msg)
            
            logger.info(f"Graph construction completed:")
            logger.info(f"  Nodes: {stats['num_nodes']}")
            logger.info(f"  Edges: {stats['num_edges']} ({stats['semantic_edges']} semantic, {stats['metadata_edges']} metadata)")
            logger.info(f"  Avg edges per node: {stats['avg_edges_per_node']:.2f}")
            logger.info(f"  Graph density: {stats['graph_density']:.4f}")
            logger.info(f"  Construction time: {construction_duration:.2f}s")
            
            # Save graph data for evaluation
            if config.save_graph_data and output_dir is not None:
                graph_data_path = output_dir / "graph_data.json"
                logger.info(f"Saving graph data to: {graph_data_path}")
                
                try:
                    import json
                    with open(graph_data_path, 'w') as f:
                        json.dump(graph_data, f, indent=2)
                    logger.info(f"Graph data saved successfully")
                    stats['graph_data_path'] = str(graph_data_path)
                except Exception as e:
                    logger.error(f"Failed to save graph data: {e}")
                    stats['graph_data_save_error'] = str(e)
            elif config.save_graph_data and output_dir is None:
                logger.warning("Graph data saving requested but no output directory provided")
            
            return GraphConstructionResult(
                success=True,
                graph_data=graph_data,
                stats=stats
            )
            
        except Exception as e:
            error_msg = f"Graph construction stage failed: {e}"
            error_traceback = traceback.format_exc()
            logger.error(error_msg)
            logger.debug(error_traceback)
            
            return GraphConstructionResult(
                success=False,
                graph_data={},
                stats={},
                error_message=error_msg,
                error_traceback=error_traceback
            )
    
    def _create_metadata_edges(self, node_metadata: List[Dict[str, Any]], config: Any) -> List[Dict[str, Any]]:
        """
        Create edges based on metadata relationships.
        
        Args:
            node_metadata: List of node metadata dictionaries
            config: GraphConstructionConfig object
            
        Returns:
            List of metadata-based edges
        """
        metadata_edges = []
        
        # Group nodes by document
        doc_groups = {}
        for i, node in enumerate(node_metadata):
            doc_id = node.get('document_id', 'unknown')
            if doc_id not in doc_groups:
                doc_groups[doc_id] = []
            doc_groups[doc_id].append(i)
        
        # Create intra-document edges (chunks from same document)
        for doc_id, node_indices in doc_groups.items():
            if len(node_indices) > 1:
                # Connect consecutive chunks with higher weight
                for i in range(len(node_indices) - 1):
                    edge = {
                        'source': node_indices[i],
                        'target': node_indices[i + 1],
                        'weight': 0.9,  # High weight for consecutive chunks
                        'edge_type': 'document_sequence'
                    }
                    metadata_edges.append(edge)
                
                # Connect all chunks within document with lower weight
                for i in range(len(node_indices)):
                    for j in range(i + 2, len(node_indices)):  # Skip consecutive (already connected)
                        edge = {
                            'source': node_indices[i],
                            'target': node_indices[j],
                            'weight': 0.6,  # Medium weight for same document
                            'edge_type': 'document_membership'
                        }
                        metadata_edges.append(edge)
        
        logger.info(f"Created {len(metadata_edges)} metadata-based edges")
        return metadata_edges
    
    def validate_inputs(self, embeddings: List[Any], config: Any) -> List[str]:
        """
        Validate inputs for graph construction stage.
        
        Args:
            embeddings: List of embedding objects
            config: GraphConstructionConfig object
            
        Returns:
            List of validation error messages
        """
        errors = []
        
        if not embeddings:
            errors.append("No embeddings provided for graph construction")
            return errors
        
        # Check embedding structure
        valid_embeddings = 0
        for i, emb in enumerate(embeddings):
            if not hasattr(emb, 'embedding'):
                errors.append(f"Embedding {i} missing 'embedding' attribute")
            elif not hasattr(emb.embedding, '__len__'):
                errors.append(f"Embedding {i} embedding is not a vector")
            elif len(emb.embedding) == 0:
                logger.warning(f"Embedding {i} has empty embedding vector")
            else:
                valid_embeddings += 1
        
        if valid_embeddings == 0:
            errors.append("No valid embedding vectors found")
        
        # Validate configuration
        if config.similarity_threshold < 0 or config.similarity_threshold > 1:
            errors.append("similarity_threshold must be between 0 and 1")
        
        if config.max_edges_per_node <= 0:
            errors.append("max_edges_per_node must be positive")
        
        if config.max_edges_per_node > len(embeddings):
            errors.append(f"max_edges_per_node ({config.max_edges_per_node}) cannot exceed number of embeddings ({len(embeddings)})")
        
        return errors
    
    def get_expected_outputs(self) -> List[str]:
        """Get list of expected output keys."""
        return ["graph_data", "stats"]
    
    def estimate_duration(self, input_size: int) -> float:
        """
        Estimate stage duration based on input size.
        
        Args:
            input_size: Number of embeddings
            
        Returns:
            Estimated duration in seconds
        """
        # Graph construction is O(n^2) for similarity computation
        # Conservative estimate: 0.01 seconds per embedding pair
        base_time = 30  # Base overhead
        computation_time = (input_size ** 2) * 0.00001  # Very conservative
        return max(base_time, computation_time)
    
    def get_resource_requirements(self, input_size: int) -> Dict[str, Any]:
        """
        Get estimated resource requirements.
        
        Args:
            input_size: Number of embeddings
            
        Returns:
            Dictionary with resource estimates
        """
        # Memory requirements for similarity matrix (n x n)
        embedding_dim = 384  # Typical dimension
        matrix_memory = (input_size ** 2) * 8 / 1024 / 1024  # Float64 in MB
        embedding_memory = input_size * embedding_dim * 8 / 1024 / 1024  # Embeddings in MB
        
        return {
            "memory_mb": max(1000, matrix_memory + embedding_memory + 500),  # Extra overhead
            "cpu_cores": 4,  # Benefits from multiple cores for numpy operations
            "disk_mb": input_size * 0.5,  # Graph data storage
            "network_required": False
        }
    
    def pre_execute_checks(self, embeddings: List[Any], config: Any) -> List[str]:
        """
        Perform pre-execution checks specific to graph construction stage.
        
        Args:
            embeddings: List of embedding objects
            config: GraphConstructionConfig object
            
        Returns:
            List of check failure messages
        """
        checks = []
        
        # Check available memory for similarity computation
        try:
            import psutil
            available_memory_mb = psutil.virtual_memory().available / 1024 / 1024
            required_memory = self.get_resource_requirements(len(embeddings))["memory_mb"]
            
            if required_memory > available_memory_mb * 0.9:  # Use max 90% of available memory
                checks.append(f"Insufficient memory for graph construction: need {required_memory:.0f}MB, "
                             f"available {available_memory_mb:.0f}MB")
        except ImportError:
            pass  # Skip memory check if psutil not available
        
        # Check embedding dimension consistency
        dimensions = []
        for emb in embeddings[:100]:  # Check first 100 for efficiency
            if hasattr(emb, 'embedding') and hasattr(emb.embedding, '__len__'):
                dimensions.append(len(emb.embedding))
        
        if dimensions and len(set(dimensions)) > 1:
            checks.append(f"Inconsistent embedding dimensions found: {set(dimensions)}")
        
        # Check for very large graphs that might be problematic
        if len(embeddings) > 100000:
            checks.append(f"Very large graph ({len(embeddings)} nodes) may require significant memory and time")
        
        return checks