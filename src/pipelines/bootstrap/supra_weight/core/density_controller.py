"""
Density controller for managing graph density during construction.
"""

from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
import numpy as np
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EdgeCandidate:
    """Candidate edge with its weight and metadata."""
    from_node: str
    to_node: str
    weight: float
    relationships: List
    weight_vector: np.ndarray
    

class DensityController:
    """
    Controls graph density to prevent quadratic explosion while preserving important relationships.
    
    Uses multiple strategies:
    1. Degree limiting - max edges per node
    2. Weight thresholding - minimum edge weight
    3. Top-k selection - keep only best edges
    4. Local density control - prevent local clusters from becoming too dense
    """
    
    def __init__(self, 
                 max_edges_per_node: int = 50,
                 min_edge_weight: float = 0.3,
                 target_density: Optional[float] = None,
                 local_density_factor: float = 2.0):
        """
        Initialize the density controller.
        
        Args:
            max_edges_per_node: Maximum edges allowed per node
            min_edge_weight: Minimum weight threshold for edges
            target_density: Target graph density (edges / possible edges)
            local_density_factor: Factor for local density limits
        """
        self.max_edges_per_node = max_edges_per_node
        self.min_edge_weight = min_edge_weight
        self.target_density = target_density
        self.local_density_factor = local_density_factor
        
        # Track node degrees
        self.node_degrees: Dict[str, int] = defaultdict(int)
        
        # Track edge candidates for batch processing
        self.edge_candidates: List[EdgeCandidate] = []
        
        # Statistics
        self.stats = {
            'edges_proposed': 0,
            'edges_accepted': 0,
            'edges_rejected_weight': 0,
            'edges_rejected_degree': 0,
            'edges_rejected_density': 0
        }
        
        logger.info(f"Initialized DensityController: max_edges={max_edges_per_node}, "
                   f"min_weight={min_edge_weight}, target_density={target_density}")
        
    def should_add_edge(self, from_node: str, to_node: str, weight: float, 
                       check_only: bool = False) -> bool:
        """
        Check if an edge should be added based on density constraints.
        
        Args:
            from_node: Source node ID
            to_node: Target node ID
            weight: Edge weight
            check_only: If True, don't update internal state
            
        Returns:
            True if edge should be added
        """
        self.stats['edges_proposed'] += 1
        
        # Check weight threshold
        if weight < self.min_edge_weight:
            self.stats['edges_rejected_weight'] += 1
            return False
            
        # Check degree limits
        if (self.node_degrees[from_node] >= self.max_edges_per_node or
            self.node_degrees[to_node] >= self.max_edges_per_node):
            self.stats['edges_rejected_degree'] += 1
            return False
            
        # If not just checking, update state
        if not check_only:
            self.node_degrees[from_node] += 1
            self.node_degrees[to_node] += 1
            self.stats['edges_accepted'] += 1
            
        return True
        
    def add_edge_candidate(self, candidate: EdgeCandidate) -> None:
        """Add an edge candidate for batch processing."""
        self.edge_candidates.append(candidate)
        
    def process_edge_batch(self, num_nodes: int) -> List[EdgeCandidate]:
        """
        Process a batch of edge candidates with global optimization.
        
        This method considers all candidates together and selects the best subset
        that satisfies density constraints.
        
        Args:
            num_nodes: Total number of nodes in the graph
            
        Returns:
            List of accepted edge candidates
        """
        if not self.edge_candidates:
            return []
            
        logger.info(f"Processing batch of {len(self.edge_candidates)} edge candidates")
        
        # Sort candidates by weight (descending)
        sorted_candidates = sorted(self.edge_candidates, 
                                 key=lambda x: x.weight, 
                                 reverse=True)
        
        # Reset for batch processing
        batch_degrees: Dict[str, int] = defaultdict(int)
        accepted_edges: List[EdgeCandidate] = []
        
        # Calculate target number of edges if density specified
        max_edges = float('inf')
        if self.target_density and num_nodes > 1:
            max_possible_edges = num_nodes * (num_nodes - 1) / 2
            max_edges = int(self.target_density * max_possible_edges)
            logger.info(f"Target density {self.target_density} allows {max_edges} edges")
        
        # Process candidates in order of weight
        for candidate in sorted_candidates:
            # Check if we've reached density limit
            if len(accepted_edges) >= max_edges:
                self.stats['edges_rejected_density'] += 1
                continue
                
            # Check weight threshold
            if candidate.weight < self.min_edge_weight:
                self.stats['edges_rejected_weight'] += 1
                continue
                
            # Check degree constraints
            if (batch_degrees[candidate.from_node] >= self.max_edges_per_node or
                batch_degrees[candidate.to_node] >= self.max_edges_per_node):
                self.stats['edges_rejected_degree'] += 1
                continue
                
            # Accept edge
            accepted_edges.append(candidate)
            batch_degrees[candidate.from_node] += 1
            batch_degrees[candidate.to_node] += 1
            self.stats['edges_accepted'] += 1
            
        # Update global degree counts
        self.node_degrees.update(batch_degrees)
        
        # Clear candidates
        self.edge_candidates = []
        
        logger.info(f"Accepted {len(accepted_edges)} edges from batch")
        return accepted_edges
        
    def apply_local_density_control(self, edges: List[EdgeCandidate], 
                                  neighborhood_size: int = 2) -> List[EdgeCandidate]:
        """
        Apply local density control to prevent dense clusters.
        
        Args:
            edges: List of edge candidates
            neighborhood_size: Size of local neighborhood to consider
            
        Returns:
            Filtered list of edges
        """
        # Build adjacency lists
        adjacency = defaultdict(set)
        for edge in edges:
            adjacency[edge.from_node].add(edge.to_node)
            adjacency[edge.to_node].add(edge.from_node)
            
        # Calculate local densities
        local_densities = {}
        for node in adjacency:
            # Get k-hop neighborhood
            neighborhood = self._get_k_hop_neighbors(node, adjacency, neighborhood_size)
            
            # Calculate density in neighborhood
            edge_count = sum(len(adjacency[n]) for n in neighborhood) / 2
            possible_edges = len(neighborhood) * (len(neighborhood) - 1) / 2
            
            if possible_edges > 0:
                local_densities[node] = edge_count / possible_edges
            else:
                local_densities[node] = 0
                
        # Filter edges in over-dense regions
        filtered_edges = []
        max_local_density = 1.0 / self.local_density_factor
        
        for edge in edges:
            local_density_from = local_densities.get(edge.from_node, 0)
            local_density_to = local_densities.get(edge.to_node, 0)
            
            # Keep edge if local density is acceptable
            if max(local_density_from, local_density_to) <= max_local_density:
                filtered_edges.append(edge)
            else:
                # Keep only very strong edges in dense regions
                if edge.weight > 0.8:
                    filtered_edges.append(edge)
                    
        return filtered_edges
        
    def _get_k_hop_neighbors(self, node: str, adjacency: Dict[str, Set[str]], 
                           k: int) -> Set[str]:
        """Get k-hop neighbors of a node."""
        neighbors = {node}
        for _ in range(k):
            new_neighbors = set()
            for n in neighbors:
                new_neighbors.update(adjacency.get(n, set()))
            neighbors.update(new_neighbors)
        return neighbors
        
    def get_statistics(self) -> Dict:
        """Get density control statistics."""
        total_proposed = self.stats['edges_proposed']
        if total_proposed > 0:
            accept_rate = self.stats['edges_accepted'] / total_proposed
            weight_reject_rate = self.stats['edges_rejected_weight'] / total_proposed
            degree_reject_rate = self.stats['edges_rejected_degree'] / total_proposed
            density_reject_rate = self.stats['edges_rejected_density'] / total_proposed
        else:
            accept_rate = weight_reject_rate = degree_reject_rate = density_reject_rate = 0
            
        return {
            **self.stats,
            'acceptance_rate': accept_rate,
            'weight_rejection_rate': weight_reject_rate,
            'degree_rejection_rate': degree_reject_rate,
            'density_rejection_rate': density_reject_rate,
            'avg_node_degree': np.mean(list(self.node_degrees.values())) if self.node_degrees else 0,
            'max_node_degree': max(self.node_degrees.values()) if self.node_degrees else 0,
            'num_nodes_with_edges': len(self.node_degrees)
        }
        
    def reset_statistics(self) -> None:
        """Reset statistics counters."""
        self.stats = {
            'edges_proposed': 0,
            'edges_accepted': 0,
            'edges_rejected_weight': 0,
            'edges_rejected_degree': 0,
            'edges_rejected_density': 0
        }