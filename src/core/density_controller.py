"""
Density Controller for HADES.

This module provides density-based control mechanisms for managing
the information flow and graph density in the HADES system.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DensityMetrics:
    """Metrics for density measurement."""
    node_count: int
    edge_count: int
    avg_degree: float
    density: float
    clustering_coefficient: float
    connected_components: int


@dataclass
class EdgeCandidate:
    """Candidate edge for addition to the graph."""
    source_id: str
    target_id: str
    weight: float
    edge_type: str
    metadata: Dict[str, Any]
    # Additional fields for arango_supra_storage
    from_node: Optional[str] = None
    to_node: Optional[str] = None
    weight_vector: Optional[List[float]] = None
    relationships: Optional[List[str]] = None
    
    def __post_init__(self) -> None:
        """Initialize from_node and to_node from source_id and target_id if not set."""
        if self.from_node is None:
            self.from_node = self.source_id
        if self.to_node is None:
            self.to_node = self.target_id


class DensityController:
    """Controls graph density for optimal performance."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize density controller."""
        self.config = config or {}
        self.max_density = self.config.get("max_density", 0.1)
        self.min_density = self.config.get("min_density", 0.001)
        self.target_avg_degree = self.config.get("target_avg_degree", 10)
        self.pruning_threshold = self.config.get("pruning_threshold", 0.3)
    
    def compute_metrics(self, graph_data: Dict[str, Any]) -> DensityMetrics:
        """Compute density metrics for a graph."""
        node_count = graph_data.get("node_count", 0)
        edge_count = graph_data.get("edge_count", 0)
        
        if node_count == 0:
            return DensityMetrics(0, 0, 0.0, 0.0, 0.0, 0)
        
        # Average degree
        avg_degree = (2 * edge_count) / node_count if node_count > 0 else 0
        
        # Graph density
        max_edges = node_count * (node_count - 1) / 2
        density = edge_count / max_edges if max_edges > 0 else 0
        
        # Placeholder for other metrics
        clustering_coefficient = graph_data.get("clustering_coefficient", 0.0)
        connected_components = graph_data.get("connected_components", 1)
        
        return DensityMetrics(
            node_count=node_count,
            edge_count=edge_count,
            avg_degree=avg_degree,
            density=density,
            clustering_coefficient=clustering_coefficient,
            connected_components=connected_components
        )
    
    def should_prune(self, metrics: DensityMetrics) -> bool:
        """Determine if graph should be pruned based on metrics."""
        return bool(
            metrics.density > self.max_density or
            metrics.avg_degree > self.target_avg_degree * 2
        )
    
    def prune_edges(
        self,
        edges: List[Tuple[str, str, float]],
        target_count: Optional[int] = None
    ) -> List[Tuple[str, str, float]]:
        """Prune edges based on weights."""
        if not edges:
            return []
        
        # Sort by weight (ascending, lower weights pruned first)
        sorted_edges = sorted(edges, key=lambda x: x[2], reverse=True)
        
        if target_count is None:
            # Prune based on threshold
            return [e for e in sorted_edges if e[2] >= self.pruning_threshold]
        else:
            # Keep top target_count edges
            return sorted_edges[:target_count]
    
    def recommend_actions(self, metrics: DensityMetrics) -> Dict[str, Any]:
        """Recommend actions based on density metrics."""
        recommendations: Dict[str, Any] = {
            "prune_edges": False,
            "add_edges": False,
            "recompute_weights": False,
            "actions": []
        }
        actions: List[str] = recommendations["actions"]
        
        if metrics.density > self.max_density:
            recommendations["prune_edges"] = True
            actions.append(
                f"Prune edges to reduce density from {metrics.density:.3f} to {self.max_density}"
            )
        
        if metrics.density < self.min_density and metrics.node_count > 10:
            recommendations["add_edges"] = True
            actions.append(
                f"Add edges to increase density from {metrics.density:.3f} to {self.min_density}"
            )
        
        if metrics.avg_degree > self.target_avg_degree * 2:
            recommendations["recompute_weights"] = True
            actions.append(
                f"Recompute edge weights, avg degree {metrics.avg_degree:.1f} exceeds target"
            )
        
        return recommendations


def create_density_controller(config: Optional[Dict[str, Any]] = None) -> DensityController:
    """Factory function to create density controller."""
    return DensityController(config)


__all__ = [
    "DensityMetrics",
    "DensityController",
    "create_density_controller"
]