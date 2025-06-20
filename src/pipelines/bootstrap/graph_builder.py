"""
Graph Builder for Bootstrap Pipeline

Builds Sequential-ISNE graph structure in ArangoDB using discovered files and relationships.
"""

import logging
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timezone
from collections import defaultdict

from .config import BootstrapConfig, BootstrapMetrics
from src.storage.incremental.sequential_isne_types import (
    EdgeType, FileType, ProcessingStatus, Embedding
)
from src.components.storage.arangodb.storage_v2 import ArangoStorageV2

logger = logging.getLogger(__name__)


class GraphBuilder:
    """
    Builds Sequential-ISNE graph structure in ArangoDB.
    
    Handles:
    - Storage of modality-specific files
    - Creation of cross-modal and intra-modal edges
    - Graph validation and statistics
    - Batch processing for performance
    """
    
    def __init__(self, config: BootstrapConfig, storage: ArangoStorageV2):
        self.config = config
        self.storage = storage
        self.logger = logging.getLogger(__name__)
        
        # Track created nodes and edges
        self.created_files: Dict[str, str] = {}  # file_path -> storage_id
        self.created_edges: List[str] = []
        self.metrics = BootstrapMetrics()
        
        self.logger.info("Initialized GraphBuilder")
    
    def build_graph(
        self, 
        processed_files: List[Tuple[str, Any, List[Any], List[Embedding]]], 
        relationships: List[Tuple[str, str, str, float, Dict[str, Any]]]
    ) -> BootstrapMetrics:
        """
        Build complete graph from processed files and relationships with embeddings.
        
        Args:
            processed_files: List of (file_path, file_object, chunks, embeddings) tuples
            relationships: List of (from_path, to_path, edge_type, weight, metadata) tuples
            
        Returns:
            BootstrapMetrics with build statistics
        """
        self.logger.info(f"Building graph from {len(processed_files)} files and {len(relationships)} relationships")
        
        start_time = datetime.now(timezone.utc)
        
        try:
            # Phase 1: Store all files in modality collections
            self._store_files(processed_files)
            
            # Phase 2: Create relationships/edges
            self._create_edges(relationships)
            
            # Phase 3: Validate graph and collect metrics
            self._collect_final_metrics()
            
            # Calculate total processing time
            end_time = datetime.now(timezone.utc)
            self.metrics.total_processing_time = (end_time - start_time).total_seconds()
            
            self.logger.info(f"Graph build completed in {self.metrics.total_processing_time:.2f}s")
            self.logger.info(f"Created {self.metrics.total_nodes_created} nodes and {self.metrics.total_edges_created} edges")
            
        except Exception as e:
            self.logger.error(f"Graph build failed: {e}")
            raise
        
        return self.metrics
    
    def _store_files(self, processed_files: List[Tuple[str, Any, List[Any], List[Embedding]]]) -> None:
        """Store all files in their respective modality collections with embeddings."""
        self.logger.info("Storing files in modality collections...")
        
        files_by_type = defaultdict(list)
        all_embeddings = []
        
        # Group files by type for batch processing and collect all embeddings
        for file_path, file_obj, chunks, embeddings in processed_files:
            file_type = file_obj.__class__.__name__.replace('File', '').lower()
            files_by_type[file_type].append((file_path, file_obj, chunks))
            all_embeddings.extend(embeddings)  # Collect all embeddings for batch storage
        
        # Store files by type
        for file_type, files in files_by_type.items():
            self.logger.info(f"Storing {len(files)} {file_type} files...")
            
            for file_path, file_obj, chunks in files:
                try:
                    # Store the file
                    storage_id = self.storage.store_modality_file(file_obj)
                    self.created_files[file_path] = storage_id
                    
                    # Update metrics
                    self.metrics.total_files_processed += 1
                    self.metrics.total_nodes_created += 1
                    
                    if file_type not in self.metrics.files_by_type:
                        self.metrics.files_by_type[file_type] = 0
                    self.metrics.files_by_type[file_type] += 1
                    
                    # Store semantic chunks
                    if chunks:
                        chunk_ids = self._store_chunks(chunks, storage_id, file_path)
                        self.logger.debug(f"Stored {len(chunk_ids)} chunks for {file_path}")
                        self.metrics.total_chunks_created += len(chunk_ids)
                    
                except Exception as e:
                    self.logger.error(f"Failed to store file {file_path}: {e}")
                    self.metrics.processing_errors += 1
        
        self.logger.info(f"Stored {len(self.created_files)} files successfully")
        
        # Store all embeddings in batch
        if all_embeddings:
            self._store_embeddings(all_embeddings)
    
    def _store_embeddings(self, embeddings: List[Embedding]) -> None:
        """Store all embeddings in the embeddings collection."""
        self.logger.info(f"Storing {len(embeddings)} embeddings...")
        
        stored_count = 0
        for embedding in embeddings:
            try:
                # Store embedding using storage component
                embedding_id = self.storage.store_embedding(embedding)
                if embedding_id:
                    stored_count += 1
                    self.metrics.total_embeddings_created += 1
                    
            except Exception as e:
                self.logger.error(f"Failed to store embedding for {embedding.source_id}: {e}")
                self.metrics.processing_errors += 1
        
        self.logger.info(f"Stored {stored_count}/{len(embeddings)} embeddings successfully")
    
    def _create_edges(self, relationships: List[Tuple[str, str, str, float, Dict[str, Any]]]) -> None:
        """Create edges from discovered relationships."""
        self.logger.info(f"Creating {len(relationships)} edges...")
        
        edge_type_counts = defaultdict(int)
        cross_modal_count = 0
        
        for from_path, to_path, edge_type, weight, metadata in relationships:
            try:
                # Get storage IDs for the files
                from_id = self.created_files.get(from_path)
                to_id = self.created_files.get(to_path)
                
                if not from_id or not to_id:
                    self.logger.debug(f"Skipping edge {from_path} -> {to_path}: file not found in storage")
                    continue
                
                # Determine if this is a cross-modal edge
                is_cross_modal = self._is_cross_modal_edge(edge_type)
                
                if is_cross_modal:
                    # Create cross-modal edge
                    edge_id = self.storage.create_cross_modal_edge(
                        from_file_id=from_id,
                        to_file_id=to_id,
                        edge_type=EdgeType(edge_type),
                        weight=weight,
                        metadata=metadata
                    )
                    cross_modal_count += 1
                else:
                    # Create intra-modal edge
                    # For now, we'll use the cross-modal method but mark it differently
                    edge_id = self.storage.create_cross_modal_edge(
                        from_file_id=from_id,
                        to_file_id=to_id,
                        edge_type=EdgeType(edge_type),
                        weight=weight,
                        metadata={**metadata, 'cross_modal': False}
                    )
                
                self.created_edges.append(edge_id)
                edge_type_counts[edge_type] += 1
                self.metrics.total_edges_created += 1
                
            except Exception as e:
                self.logger.error(f"Failed to create edge {from_path} -> {to_path}: {e}")
                self.metrics.processing_errors += 1
        
        # Update metrics
        self.metrics.edges_by_type = dict(edge_type_counts)
        self.metrics.cross_modal_edges = cross_modal_count
        
        self.logger.info(f"Created {self.metrics.total_edges_created} edges ({cross_modal_count} cross-modal)")
    
    def _is_cross_modal_edge(self, edge_type: str) -> bool:
        """Check if an edge type represents a cross-modal relationship."""
        cross_modal_types = {
            EdgeType.CODE_TO_DOC.value,
            EdgeType.DOC_TO_CODE.value,
            EdgeType.CODE_TO_CONFIG.value,
            EdgeType.CONFIG_TO_CODE.value,
            EdgeType.DOC_TO_CONFIG.value,
            EdgeType.ISNE_SIMILARITY.value,
            EdgeType.SEMANTIC_SIMILARITY.value
        }
        return edge_type in cross_modal_types
    
    def _store_chunks(self, chunks: List[Any], file_storage_id: str, file_path: str) -> List[str]:
        """Store semantic chunks for a file."""
        chunk_ids = []
        
        for chunk in chunks:
            try:
                # Update chunk with file reference
                chunk.source_file_id = file_storage_id
                
                # Store chunk in database
                chunk_id = self.storage.store_chunk(chunk)
                chunk_ids.append(chunk_id)
                
            except Exception as e:
                self.logger.error(f"Failed to store chunk for {file_path}: {e}")
                self.metrics.processing_errors += 1
        
        return chunk_ids
    
    def _collect_final_metrics(self) -> None:
        """Collect final graph metrics and statistics."""
        self.logger.info("Collecting final graph metrics...")
        
        try:
            # Get storage statistics
            storage_stats = self.storage.get_statistics()
            modality_stats = self.storage.get_modality_statistics()
            
            # Update metrics with storage data
            if 'total_files' in modality_stats:
                self.metrics.total_nodes_created = modality_stats['total_files']
            
            if 'cross_modal_edges' in modality_stats:
                self.metrics.cross_modal_edges = modality_stats['cross_modal_edges']
            
            if 'modality_distribution' in modality_stats:
                self.metrics.modality_distribution = modality_stats['modality_distribution']
            
            # Calculate graph density (simplified)
            if self.metrics.total_nodes_created > 1:
                max_edges = self.metrics.total_nodes_created * (self.metrics.total_nodes_created - 1)
                self.metrics.graph_density = self.metrics.total_edges_created / max_edges
            
            # For now, assume single connected component (can be improved with actual analysis)
            self.metrics.connected_components = 1
            self.metrics.largest_component_size = self.metrics.total_nodes_created
            
            # Calculate average processing time
            if self.metrics.total_files_processed > 0:
                self.metrics.avg_file_processing_time = (
                    self.metrics.total_processing_time / self.metrics.total_files_processed
                )
            
        except Exception as e:
            self.logger.warning(f"Failed to collect some metrics: {e}")
    
    def validate_graph(self) -> bool:
        """Validate the created graph structure."""
        self.logger.info("Validating graph structure...")
        
        try:
            # Check if storage is healthy
            if not self.storage.health_check():
                self.logger.error("Storage health check failed")
                return False
            
            # Check if we have nodes and edges
            if self.metrics.total_nodes_created == 0:
                self.logger.error("No nodes were created")
                return False
            
            # Check for cross-modal edges if enabled
            if self.config.enable_cross_modal_discovery and self.metrics.cross_modal_edges == 0:
                self.logger.warning("No cross-modal edges were created")
            
            # Validate modality distribution
            expected_types = {'code', 'documentation', 'config'}
            found_types = set(self.metrics.modality_distribution.keys())
            
            if not found_types.intersection(expected_types):
                self.logger.warning("No files of expected types were processed")
            
            self.logger.info("Graph validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Graph validation failed: {e}")
            return False
    
    def get_graph_summary(self) -> Dict[str, Any]:
        """Get a summary of the created graph."""
        return {
            'nodes': {
                'total': self.metrics.total_nodes_created,
                'by_type': self.metrics.files_by_type,
                'modality_distribution': self.metrics.modality_distribution
            },
            'edges': {
                'total': self.metrics.total_edges_created,
                'by_type': self.metrics.edges_by_type,
                'cross_modal': self.metrics.cross_modal_edges
            },
            'performance': {
                'processing_time': self.metrics.total_processing_time,
                'avg_file_time': self.metrics.avg_file_processing_time,
                'errors': self.metrics.processing_errors
            },
            'quality': {
                'graph_density': self.metrics.graph_density,
                'connected_components': self.metrics.connected_components,
                'largest_component': self.metrics.largest_component_size
            }
        }