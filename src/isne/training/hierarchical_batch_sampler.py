"""
Hierarchical batch sampler for directory-aware ISNE training.

This module implements batch sampling strategies that preserve directory structure
and co-location relationships during training.
"""

import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

import numpy as np
import torch
from torch_geometric.data import Data

from src.types.isne.training import BatchSample, DirectoryMetadata


@dataclass
class DirectoryNode:
    """Represents a directory in the hierarchical structure."""
    path: Path
    depth: int
    node_ids: Set[int] = field(default_factory=set)
    subdirectories: List['DirectoryNode'] = field(default_factory=list)
    parent: Optional['DirectoryNode'] = None
    
    @property
    def is_leaf(self) -> bool:
        """Check if this is a leaf directory."""
        return len(self.subdirectories) == 0
    
    @property
    def total_nodes(self) -> int:
        """Total number of nodes in this directory and subdirectories."""
        count = len(self.node_ids)
        for subdir in self.subdirectories:
            count += subdir.total_nodes
        return count


class HierarchicalBatchSampler:
    """
    Samples batches that preserve directory structure and relationships.
    
    This sampler ensures that files from the same directory are more likely
    to appear in the same batch, preserving co-location signals during training.
    """
    
    def __init__(
        self,
        graph_data: Data,
        directory_metadata: Dict[int, DirectoryMetadata],
        batch_size: int = 256,
        same_dir_weight: float = 2.0,
        negative_sampling_ratio: int = 5,
        debug_mode: bool = False,
        debug_output_dir: Optional[Path] = None
    ):
        """
        Initialize the hierarchical batch sampler.
        
        Args:
            graph_data: PyTorch Geometric graph data
            directory_metadata: Mapping from node ID to directory metadata
            batch_size: Target batch size
            same_dir_weight: Weight for sampling from same directory
            negative_sampling_ratio: Ratio of negative to positive samples
            debug_mode: Enable debug output
            debug_output_dir: Directory for debug outputs
        """
        self.graph_data = graph_data
        self.directory_metadata = directory_metadata
        self.batch_size = batch_size
        self.same_dir_weight = same_dir_weight
        self.negative_sampling_ratio = negative_sampling_ratio
        self.debug_mode = debug_mode
        self.debug_output_dir = debug_output_dir
        
        # Build directory tree
        self.root_directories = self._build_directory_tree()
        
        # Precompute directory groups for efficiency
        self.directory_groups = self._group_nodes_by_directory()
        
        if self.debug_mode and self.debug_output_dir:
            self._save_debug_info("initialization")
    
    def _build_directory_tree(self) -> List[DirectoryNode]:
        """Build hierarchical directory structure from metadata."""
        directory_nodes: Dict[Path, DirectoryNode] = {}
        
        # Create directory nodes
        for node_id, metadata in self.directory_metadata.items():
            dir_path = Path(metadata.directory_path)
            
            # Create nodes for all parent directories
            current_path = Path("/")
            depth = 0
            
            for part in dir_path.parts[1:]:  # Skip root
                current_path = current_path / part
                depth += 1
                
                if current_path not in directory_nodes:
                    directory_nodes[current_path] = DirectoryNode(
                        path=current_path,
                        depth=depth
                    )
                
                # Add node to leaf directory
                if current_path == dir_path:
                    directory_nodes[current_path].node_ids.add(node_id)
        
        # Build parent-child relationships
        for path, node in directory_nodes.items():
            if path.parent != path:  # Not root
                parent_path = path.parent
                if parent_path in directory_nodes:
                    parent_node = directory_nodes[parent_path]
                    parent_node.subdirectories.append(node)
                    node.parent = parent_node
        
        # Find root directories
        roots = [
            node for node in directory_nodes.values()
            if node.parent is None
        ]
        
        return roots
    
    def _group_nodes_by_directory(self) -> Dict[Path, List[int]]:
        """Group node IDs by their directory."""
        groups = defaultdict(list)
        
        for node_id, metadata in self.directory_metadata.items():
            groups[Path(metadata.directory_path)].append(node_id)
        
        return dict(groups)
    
    def sample(self, num_batches: int) -> List[BatchSample]:
        """
        Sample batches using hierarchical strategy.
        
        Args:
            num_batches: Number of batches to generate
            
        Returns:
            List of batch samples with directory awareness
        """
        batches = []
        
        for batch_idx in range(num_batches):
            batch_nodes = self._sample_batch()
            
            # Create batch sample
            batch = self._create_batch_sample(batch_nodes, batch_idx)
            batches.append(batch)
            
            if self.debug_mode:
                self._save_debug_batch(batch, batch_idx)
        
        return batches
    
    def _sample_batch(self) -> List[int]:
        """Sample a single batch with directory awareness."""
        batch_nodes = []
        remaining_size = self.batch_size
        
        # Sample anchor directories based on node count
        anchor_dirs = self._sample_anchor_directories()
        
        for anchor_dir in anchor_dirs:
            if remaining_size <= 0:
                break
            
            # Sample nodes from this directory
            dir_nodes = self.directory_groups.get(anchor_dir, [])
            if not dir_nodes:
                continue
            
            # Determine how many nodes to sample from this directory
            num_to_sample = min(
                remaining_size,
                max(1, int(len(dir_nodes) * self.same_dir_weight))
            )
            
            # Sample nodes
            sampled = random.sample(
                dir_nodes,
                min(num_to_sample, len(dir_nodes))
            )
            
            batch_nodes.extend(sampled)
            remaining_size -= len(sampled)
            
            # Add some nodes from sibling directories
            if remaining_size > 0:
                sibling_nodes = self._sample_sibling_nodes(
                    anchor_dir,
                    num_samples=remaining_size // 4
                )
                batch_nodes.extend(sibling_nodes)
                remaining_size -= len(sibling_nodes)
        
        # Fill remaining with random nodes (negative samples)
        if remaining_size > 0:
            all_nodes = list(range(self.graph_data.num_nodes))
            remaining_nodes = [n for n in all_nodes if n not in batch_nodes]
            
            if remaining_nodes:
                negative_samples = random.sample(
                    remaining_nodes,
                    min(remaining_size, len(remaining_nodes))
                )
                batch_nodes.extend(negative_samples)
        
        return batch_nodes[:self.batch_size]
    
    def _sample_anchor_directories(self) -> List[Path]:
        """Sample anchor directories weighted by node count."""
        # Calculate weights based on directory size
        dir_weights = {
            path: len(nodes) 
            for path, nodes in self.directory_groups.items()
        }
        
        if not dir_weights:
            return []
        
        # Sample directories
        directories = list(dir_weights.keys())
        weights = list(dir_weights.values())
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            normalized_weights = [float(w) / total_weight for w in weights]
        else:
            normalized_weights = [1.0 / len(weights)] * len(weights)
        
        # Sample up to 5 anchor directories per batch
        num_anchors = min(5, len(directories))
        
        try:
            sampled_indices = np.random.choice(
                len(directories),
                size=num_anchors,
                replace=False,
                p=normalized_weights
            )
            sampled = [directories[i] for i in sampled_indices]
        except ValueError:
            # Fallback to uniform sampling if weights are problematic
            sampled = random.sample(directories, num_anchors)
        
        return sampled
    
    def _sample_sibling_nodes(
        self, 
        directory: Path, 
        num_samples: int
    ) -> List[int]:
        """Sample nodes from sibling directories."""
        sibling_nodes = []
        
        # Find parent directory
        parent_dir = directory.parent
        if parent_dir == directory:  # Root directory
            return []
        
        # Find sibling directories
        sibling_dirs = [
            d for d in self.directory_groups.keys()
            if d.parent == parent_dir and d != directory
        ]
        
        if not sibling_dirs:
            return []
        
        # Sample from siblings
        for sibling_dir in random.sample(
            sibling_dirs, 
            min(len(sibling_dirs), 3)
        ):
            nodes = self.directory_groups.get(sibling_dir, [])
            if nodes:
                sampled = random.sample(
                    nodes,
                    min(num_samples // 3, len(nodes))
                )
                sibling_nodes.extend(sampled)
        
        return sibling_nodes[:num_samples]
    
    def _create_batch_sample(
        self, 
        node_ids: List[int], 
        batch_idx: int
    ) -> BatchSample:
        """Create a BatchSample object with directory features."""
        # Extract node features
        node_features = self.graph_data.x[node_ids]
        
        # Extract edges within this batch
        edge_mask = torch.tensor([
            (src.item() in node_ids and dst.item() in node_ids)
            for src, dst in self.graph_data.edge_index.t()
        ])
        batch_edges = self.graph_data.edge_index[:, edge_mask]
        
        # Create node ID mapping for batch
        node_id_map = {old_id: new_id for new_id, old_id in enumerate(node_ids)}
        
        # Remap edge indices
        remapped_edges = torch.zeros_like(batch_edges)
        for i, (src, dst) in enumerate(batch_edges.t()):
            remapped_edges[0, i] = node_id_map[src.item()]
            remapped_edges[1, i] = node_id_map[dst.item()]
        
        # Extract directory features
        directory_features = self._extract_directory_features(node_ids)
        
        return BatchSample(
            node_ids=torch.tensor(node_ids),
            features=node_features,
            edge_index=remapped_edges,
            directory_features=directory_features,
            batch_idx=batch_idx
        )
    
    def _extract_directory_features(
        self, 
        node_ids: List[int]
    ) -> Dict[str, torch.Tensor]:
        """Extract directory-specific features for nodes."""
        depths = []
        sibling_counts = []
        is_leaf = []
        
        for node_id in node_ids:
            metadata = self.directory_metadata.get(node_id)
            if metadata:
                depths.append(metadata.depth)
                sibling_counts.append(metadata.sibling_count)
                is_leaf.append(1.0 if metadata.is_leaf else 0.0)
            else:
                # Default values for nodes without metadata
                depths.append(0)
                sibling_counts.append(0)
                is_leaf.append(0.0)
        
        return {
            'depth': torch.tensor(depths, dtype=torch.float32),
            'sibling_count': torch.tensor(sibling_counts, dtype=torch.float32),
            'is_leaf': torch.tensor(is_leaf, dtype=torch.float32)
        }
    
    def _save_debug_info(self, stage: str) -> None:
        """Save debug information to file."""
        if not self.debug_output_dir:
            return
        
        import json
        from datetime import datetime
        
        debug_dir = Path(self.debug_output_dir) / "hierarchical_batch_sampler"
        debug_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().isoformat()
        
        debug_info = {
            "component": "hierarchical_batch_sampler",
            "timestamp": timestamp,
            "stage": stage,
            "data": {
                "num_nodes": self.graph_data.num_nodes,
                "num_edges": self.graph_data.edge_index.shape[1],
                "num_directories": len(self.directory_groups),
                "batch_size": self.batch_size,
                "same_dir_weight": self.same_dir_weight,
                "directory_distribution": {
                    str(path): len(nodes)
                    for path, nodes in list(self.directory_groups.items())[:10]
                }
            },
            "metadata": {
                "memory_usage_mb": self._estimate_memory_usage(),
                "warnings": []
            }
        }
        
        output_path = debug_dir / f"{timestamp}_{stage}.json"
        with open(output_path, 'w') as f:
            json.dump(debug_info, f, indent=2)
    
    def _save_debug_batch(self, batch: BatchSample, batch_idx: int) -> None:
        """Save batch information for debugging."""
        if not self.debug_output_dir:
            return
        
        import json
        from datetime import datetime
        
        debug_dir = Path(self.debug_output_dir) / "hierarchical_batch_sampler" / "batches"
        debug_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().isoformat()
        
        # Analyze batch composition
        directories_in_batch: defaultdict[str, int] = defaultdict(int)
        for node_id in batch.node_ids.tolist():
            metadata = self.directory_metadata.get(node_id)
            if metadata:
                directories_in_batch[metadata.directory_path] += 1
        
        debug_info = {
            "component": "hierarchical_batch_sampler",
            "timestamp": timestamp,
            "stage": "batch_sampling",
            "batch_idx": batch_idx,
            "data": {
                "batch_size": len(batch.node_ids),
                "num_edges": batch.edge_index.shape[1],
                "directory_distribution": dict(directories_in_batch),
                "depth_distribution": batch.directory_features['depth'].tolist()[:10]
            },
            "metadata": {
                "processing_time_ms": 0,  # Would need timing logic
                "item_count": len(batch.node_ids)
            }
        }
        
        output_path = debug_dir / f"{timestamp}_batch_{batch_idx}.json"
        with open(output_path, 'w') as f:
            json.dump(debug_info, f, indent=2)
    
    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB."""
        # Rough estimation
        node_memory = self.graph_data.x.element_size() * self.graph_data.x.nelement()
        edge_memory = self.graph_data.edge_index.element_size() * self.graph_data.edge_index.nelement()
        
        # Directory metadata
        metadata_memory = len(self.directory_metadata) * 1024  # Rough estimate
        
        total_bytes = node_memory + edge_memory + metadata_memory
        return float(total_bytes / (1024 * 1024))