"""
Spectral Graph Sparsification Module

Implements research-backed spectral sparsification for ISNE-optimized graph construction.
Based on Spielman-Srivastava "Graph sparsification by effective resistances" (2011).
"""

import logging
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Set, Optional
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
import random

logger = logging.getLogger(__name__)


class SpectralSparsifier:
    """
    Spectral graph sparsification optimized for ISNE training.
    
    Preserves spectral properties while dramatically reducing edge count
    using effective resistance-based sampling.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.target_density = self.config.get('target_density', 0.05)
        self.spectral_preservation = self.config.get('spectral_preservation', 0.95)
        self.max_eigenvalues = self.config.get('max_eigenvalues', 50)
        self.min_edges_per_node = self.config.get('min_edges_per_node', 2)
        
        logger.info(f"Initialized SpectralSparsifier with density target: {self.target_density}")
    
    def sparsify_graph(self, graph: nx.Graph, preserve_edge_types: bool = True) -> nx.Graph:
        """
        Main sparsification method that preserves spectral properties.
        
        Args:
            graph: Dense input graph to sparsify
            preserve_edge_types: Whether to maintain edge type distributions
            
        Returns:
            Sparsified graph with preserved spectral properties
        """
        initial_edges = graph.number_of_edges()
        n_nodes = graph.number_of_nodes()
        target_edges = int(initial_edges * self.target_density)
        
        logger.info(f"Sparsifying graph: {initial_edges} → {target_edges} edges")
        
        if initial_edges <= target_edges:
            logger.info("Graph already sparse enough, returning original")
            return graph.copy()
        
        # Step 1: Calculate effective resistances
        effective_resistances = self._compute_effective_resistance(graph)
        
        # Step 2: Sample edges preserving spectral properties
        if preserve_edge_types:
            sparse_graph = self._sample_edges_by_type(graph, effective_resistances, target_edges)
        else:
            sparse_graph = self._sample_edges_uniform(graph, effective_resistances, target_edges)
        
        # Step 3: Ensure connectivity constraints
        sparse_graph = self._ensure_connectivity(sparse_graph, graph)
        
        final_edges = sparse_graph.number_of_edges()
        logger.info(f"Sparsification complete: {final_edges} edges (reduction: {1-final_edges/initial_edges:.2%})")
        
        return sparse_graph
    
    def _compute_effective_resistance(self, graph: nx.Graph) -> Dict[Tuple[int, int], float]:
        """
        Compute effective resistance for all edges using spectral methods.
        
        Uses approximation for large graphs to maintain computational feasibility.
        """
        logger.info("Computing effective resistances...")
        
        # Convert to adjacency matrix
        adj_matrix = nx.adjacency_matrix(graph, weight='weight')
        n = adj_matrix.shape[0]
        
        # Compute Laplacian matrix
        degree_matrix = np.diag(np.array(adj_matrix.sum(axis=1)).flatten())
        laplacian = degree_matrix - adj_matrix.toarray()
        
        # For large graphs, use approximation
        if n > 1000:
            return self._approximate_effective_resistance(graph, laplacian)
        else:
            return self._exact_effective_resistance(graph, laplacian)
    
    def _exact_effective_resistance(self, graph: nx.Graph, laplacian: np.ndarray) -> Dict[Tuple[int, int], float]:
        """Exact effective resistance calculation for small graphs."""
        try:
            # Compute pseudoinverse of Laplacian
            laplacian_pinv = np.linalg.pinv(laplacian)
            
            resistances = {}
            node_list = list(graph.nodes())
            
            for u, v in graph.edges():
                # Get node indices
                i = node_list.index(u)
                j = node_list.index(v)
                
                # Effective resistance formula: (e_i - e_j)^T L^+ (e_i - e_j)
                e_diff = np.zeros(len(node_list))
                e_diff[i] = 1
                e_diff[j] = -1
                
                resistance = e_diff.T @ laplacian_pinv @ e_diff
                resistances[(u, v)] = max(resistance, 1e-10)  # Avoid zero resistance
                
            return resistances
            
        except Exception as e:
            logger.warning(f"Exact calculation failed: {e}, falling back to approximation")
            return self._approximate_effective_resistance(graph, laplacian)
    
    def _approximate_effective_resistance(self, graph: nx.Graph, laplacian: np.ndarray) -> Dict[Tuple[int, int], float]:
        """Approximate effective resistance using eigendecomposition."""
        try:
            # Use sparse eigensolver for top-k eigenvalues
            k = min(self.max_eigenvalues, laplacian.shape[0] - 2)
            laplacian_sparse = csr_matrix(laplacian)
            
            # Get smallest non-zero eigenvalues and eigenvectors
            eigenvals, eigenvecs = eigsh(laplacian_sparse, k=k, which='SM', sigma=1e-8)
            
            # Build pseudoinverse approximation
            laplacian_pinv_approx = np.zeros_like(laplacian)
            for i in range(k):
                if eigenvals[i] > 1e-10:  # Skip near-zero eigenvalues
                    v = eigenvecs[:, i]
                    laplacian_pinv_approx += np.outer(v, v) / eigenvals[i]
            
            # Calculate resistances
            resistances = {}
            node_list = list(graph.nodes())
            
            for u, v in graph.edges():
                i = node_list.index(u)
                j = node_list.index(v)
                
                e_diff = np.zeros(len(node_list))
                e_diff[i] = 1
                e_diff[j] = -1
                
                resistance = e_diff.T @ laplacian_pinv_approx @ e_diff
                resistances[(u, v)] = max(resistance, 1e-10)
            
            return resistances
            
        except Exception as e:
            logger.error(f"Approximation failed: {e}, using fallback method")
            return self._fallback_resistance(graph)
    
    def _fallback_resistance(self, graph: nx.Graph) -> Dict[Tuple[int, int], float]:
        """Fallback resistance calculation using graph properties."""
        resistances = {}
        
        for u, v, data in graph.edges(data=True):
            # Use inverse of edge weight as resistance proxy
            weight = data.get('weight', 1.0)
            resistance = 1.0 / max(weight, 1e-10)
            
            # Add path-based component (longer paths = higher resistance)
            try:
                shortest_path = nx.shortest_path_length(graph, u, v)
                resistance *= shortest_path
            except:
                resistance *= 2.0  # Default path multiplier
            
            resistances[(u, v)] = resistance
        
        return resistances
    
    def _sample_edges_by_type(self, graph: nx.Graph, resistances: Dict, target_edges: int) -> nx.Graph:
        """
        Sample edges while preserving edge type distributions.
        
        Maintains your multi-dimensional relationship structure.
        """
        # Group edges by type
        edge_types = {}
        for u, v, data in graph.edges(data=True):
            edge_type = data.get('relation_type', 'unknown')
            if edge_type not in edge_types:
                edge_types[edge_type] = []
            edge_types[edge_type].append((u, v, data))
        
        # Calculate target edges per type (proportional to current distribution)
        type_quotas = {}
        total_edges = graph.number_of_edges()
        
        for edge_type, edges in edge_types.items():
            proportion = len(edges) / total_edges
            type_quotas[edge_type] = int(target_edges * proportion)
        
        # Sample edges within each type
        sampled_edges = []
        
        for edge_type, edges in edge_types.items():
            quota = type_quotas.get(edge_type, 0)
            if quota == 0 or len(edges) == 0:
                continue
            
            # Calculate sampling probabilities based on effective resistance
            probabilities = []
            for u, v, data in edges:
                edge_weight = data.get('weight', 1.0)
                resistance = resistances.get((u, v), resistances.get((v, u), 1.0))
                
                # Higher resistance = more important = higher probability
                importance = edge_weight * resistance
                probabilities.append(importance)
            
            # Normalize probabilities
            if sum(probabilities) > 0:
                probabilities = np.array(probabilities) / sum(probabilities)
                
                # Sample edges
                n_sample = min(quota, len(edges))
                if n_sample > 0:
                    selected_indices = np.random.choice(
                        len(edges), 
                        size=n_sample, 
                        replace=False, 
                        p=probabilities
                    )
                    
                    for idx in selected_indices:
                        sampled_edges.append(edges[idx])
        
        # Build sparse graph
        sparse_graph = nx.Graph()
        sparse_graph.add_nodes_from(graph.nodes(data=True))
        
        for u, v, data in sampled_edges:
            sparse_graph.add_edge(u, v, **data)
        
        return sparse_graph
    
    def _sample_edges_uniform(self, graph: nx.Graph, resistances: Dict, target_edges: int) -> nx.Graph:
        """Sample edges uniformly based on effective resistance."""
        all_edges = list(graph.edges(data=True))
        
        # Calculate probabilities
        probabilities = []
        for u, v, data in all_edges:
            edge_weight = data.get('weight', 1.0)
            resistance = resistances.get((u, v), resistances.get((v, u), 1.0))
            importance = edge_weight * resistance
            probabilities.append(importance)
        
        # Normalize and sample
        if sum(probabilities) > 0:
            probabilities = np.array(probabilities) / sum(probabilities)
            
            n_sample = min(target_edges, len(all_edges))
            selected_indices = np.random.choice(
                len(all_edges), 
                size=n_sample, 
                replace=False, 
                p=probabilities
            )
            
            # Build sparse graph
            sparse_graph = nx.Graph()
            sparse_graph.add_nodes_from(graph.nodes(data=True))
            
            for idx in selected_indices:
                u, v, data = all_edges[idx]
                sparse_graph.add_edge(u, v, **data)
            
            return sparse_graph
        
        return graph.copy()
    
    def _ensure_connectivity(self, sparse_graph: nx.Graph, original_graph: nx.Graph) -> nx.Graph:
        """
        Ensure minimum connectivity requirements for ISNE training.
        """
        # Check minimum degree constraint
        low_degree_nodes = [
            node for node in sparse_graph.nodes() 
            if sparse_graph.degree(node) < self.min_edges_per_node
        ]
        
        if low_degree_nodes:
            logger.info(f"Adding edges for {len(low_degree_nodes)} low-degree nodes")
            
            for node in low_degree_nodes:
                current_degree = sparse_graph.degree(node)
                needed_edges = self.min_edges_per_node - current_degree
                
                # Find available connections from original graph
                original_neighbors = set(original_graph.neighbors(node))
                current_neighbors = set(sparse_graph.neighbors(node))
                available_neighbors = original_neighbors - current_neighbors
                
                # Add edges to meet minimum degree
                if available_neighbors:
                    selected_neighbors = random.sample(
                        list(available_neighbors), 
                        min(needed_edges, len(available_neighbors))
                    )
                    
                    for neighbor in selected_neighbors:
                        # Copy edge data from original graph
                        edge_data = original_graph[node][neighbor]
                        sparse_graph.add_edge(node, neighbor, **edge_data)
        
        return sparse_graph
    
    def validate_spectral_properties(self, original_graph: nx.Graph, sparse_graph: nx.Graph) -> Dict[str, float]:
        """
        Validate that spectral properties are preserved after sparsification.
        """
        try:
            # Calculate spectral gaps
            original_eigenvals = nx.laplacian_spectrum(original_graph)
            sparse_eigenvals = nx.laplacian_spectrum(sparse_graph)
            
            # Spectral gap (difference between two smallest eigenvalues)
            original_gap = original_eigenvals[1] - original_eigenvals[0] if len(original_eigenvals) > 1 else 0
            sparse_gap = sparse_eigenvals[1] - sparse_eigenvals[0] if len(sparse_eigenvals) > 1 else 0
            
            # Gap preservation ratio
            gap_preservation = min(sparse_gap / original_gap, original_gap / sparse_gap) if original_gap > 0 else 1.0
            
            # Connectivity preservation
            original_components = nx.number_connected_components(original_graph)
            sparse_components = nx.number_connected_components(sparse_graph)
            
            return {
                'spectral_gap_preservation': gap_preservation,
                'connected_components_original': original_components,
                'connected_components_sparse': sparse_components,
                'connectivity_preserved': sparse_components <= original_components,
                'edge_reduction_ratio': 1 - sparse_graph.number_of_edges() / original_graph.number_of_edges(),
                'density_original': nx.density(original_graph),
                'density_sparse': nx.density(sparse_graph)
            }
            
        except Exception as e:
            logger.error(f"Spectral validation failed: {e}")
            return {'validation_failed': True, 'error': str(e)}