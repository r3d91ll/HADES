"""
Batch writer for efficient database operations.
"""

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
import logging
import time
from collections import defaultdict

from ..core.density_controller import EdgeCandidate
from .arango_supra_storage import ArangoSupraStorage

logger = logging.getLogger(__name__)


@dataclass
class WriteBatch:
    """Container for batch write operations."""
    nodes: List[Dict[str, Any]]
    edges: List[EdgeCandidate]
    metadata: Dict[str, Any]
    
    def __init__(self) -> None:
        self.nodes = []
        self.edges = []
        self.metadata = {}
        
    def size(self) -> int:
        """Get total size of batch."""
        return len(self.nodes) + len(self.edges)
        
    def clear(self) -> None:
        """Clear the batch."""
        self.nodes.clear()
        self.edges.clear()
        self.metadata.clear()


class BatchWriter:
    """
    Manages efficient batch writing to storage.
    
    Accumulates writes and flushes them in batches to improve performance.
    """
    
    def __init__(self, 
                 storage: ArangoSupraStorage,
                 batch_size: int = 1000,
                 flush_interval: float = 30.0,
                 progress_callback: Optional[Callable[[Dict], None]] = None):
        """
        Initialize batch writer.
        
        Args:
            storage: Storage adapter instance
            batch_size: Maximum batch size before automatic flush
            flush_interval: Maximum time between flushes (seconds)
            progress_callback: Optional callback for progress updates
        """
        self.storage = storage
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.progress_callback = progress_callback
        
        # Current batch
        self.current_batch = WriteBatch()
        
        # Tracking
        self.last_flush_time = time.time()
        self.total_nodes_written = 0
        self.total_edges_written = 0
        self.flush_count = 0
        
        # Node ID mapping
        self.id_mapping: Dict[str, str] = {}
        
        # Statistics
        self.stats: Dict[str, Any] = defaultdict(int)
        
        logger.info(f"Initialized BatchWriter with batch_size={batch_size}")
        
    def add_node(self, node: Dict[str, Any]) -> None:
        """Add a node to the batch."""
        self.current_batch.nodes.append(node)
        self._check_flush()
        
    def add_nodes(self, nodes: List[Dict[str, Any]]) -> None:
        """Add multiple nodes to the batch."""
        self.current_batch.nodes.extend(nodes)
        self._check_flush()
        
    def add_edge(self, edge: EdgeCandidate) -> None:
        """Add an edge to the batch."""
        self.current_batch.edges.append(edge)
        self._check_flush()
        
    def add_edges(self, edges: List[EdgeCandidate]) -> None:
        """Add multiple edges to the batch."""
        self.current_batch.edges.extend(edges)
        self._check_flush()
        
    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the batch."""
        self.current_batch.metadata[key] = value
        
    def _check_flush(self) -> None:
        """Check if batch should be flushed."""
        should_flush = False
        
        # Check batch size
        if self.current_batch.size() >= self.batch_size:
            should_flush = True
            self.stats['flush_reason_size'] += 1
            
        # Check time interval
        if time.time() - self.last_flush_time >= self.flush_interval:
            should_flush = True
            self.stats['flush_reason_time'] += 1
            
        if should_flush:
            self.flush()
            
    def flush(self) -> None:
        """Flush current batch to storage."""
        if self.current_batch.size() == 0 and not self.current_batch.metadata:
            return
            
        start_time = time.time()
        
        # Write nodes first (to get ID mapping)
        if self.current_batch.nodes:
            node_mapping = self.storage.store_nodes(self.current_batch.nodes)
            self.id_mapping.update(node_mapping)
            self.total_nodes_written += len(self.current_batch.nodes)
            logger.info(f"Flushed {len(self.current_batch.nodes)} nodes")
            
        # Write edges
        if self.current_batch.edges:
            self.storage.store_edges(self.current_batch.edges, self.id_mapping)
            self.total_edges_written += len(self.current_batch.edges)
            logger.info(f"Flushed {len(self.current_batch.edges)} edges")
            
        # Write metadata
        if self.current_batch.metadata:
            self.storage.store_metadata(self.current_batch.metadata)
            
        # Update statistics
        flush_time = time.time() - start_time
        self.stats['total_flush_time'] += flush_time
        self.stats['flush_count'] += 1
        
        # Report progress
        if self.progress_callback:
            self.progress_callback({
                'nodes_written': self.total_nodes_written,
                'edges_written': self.total_edges_written,
                'flush_count': self.flush_count,
                'last_flush_time': flush_time
            })
            
        # Clear batch
        self.current_batch.clear()
        self.last_flush_time = time.time()
        self.flush_count += 1
        
    def finalize(self) -> Dict[str, Any]:
        """
        Finalize writing and return statistics.
        
        This flushes any remaining data and returns final statistics.
        """
        # Final flush
        if self.current_batch.size() > 0 or self.current_batch.metadata:
            self.stats['flush_reason_final'] = 1
            self.flush()
            
        # Calculate statistics
        avg_flush_time = (
            self.stats['total_flush_time'] / self.stats['flush_count']
            if self.stats['flush_count'] > 0 else 0
        )
        
        return {
            'total_nodes_written': self.total_nodes_written,
            'total_edges_written': self.total_edges_written,
            'flush_count': self.flush_count,
            'average_flush_time': avg_flush_time,
            'total_flush_time': self.stats['total_flush_time'],
            'flush_reasons': {
                'size': self.stats.get('flush_reason_size', 0),
                'time': self.stats.get('flush_reason_time', 0),
                'final': self.stats.get('flush_reason_final', 0)
            }
        }
        
    def __enter__(self) -> 'BatchWriter':
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - ensures final flush."""
        self.finalize()