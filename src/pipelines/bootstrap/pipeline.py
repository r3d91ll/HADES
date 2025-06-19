"""
Bootstrap Pipeline for Sequential-ISNE

Main pipeline implementation that orchestrates the bootstrap process for Sequential-ISNE
graph construction from directory structure and file analysis.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from .config import BootstrapConfig, BootstrapResult, BootstrapMetrics
from .directory_analyzer import DirectoryAnalyzer
from .file_processor import FileProcessor
from .graph_builder import GraphBuilder

from src.components.storage.factory import create_storage_component
from src.components.storage.arangodb.storage_v2 import ArangoStorageV2

logger = logging.getLogger(__name__)


class BootstrapPipeline:
    """
    Sequential-ISNE Bootstrap Pipeline.
    
    Creates initial graph structure by:
    1. Analyzing directory structure and file relationships
    2. Processing files into modality-specific objects
    3. Building graph with cross-modal and intra-modal edges
    4. Storing in ArangoDB for Sequential-ISNE training
    """
    
    def __init__(self, config: BootstrapConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.directory_analyzer = DirectoryAnalyzer(config)
        self.file_processor = FileProcessor(config)
        
        # Storage will be initialized during execution
        self.storage: Optional[ArangoStorageV2] = None
        self.graph_builder: Optional[GraphBuilder] = None
        
        self.logger.info(f"Initialized Bootstrap Pipeline for {config.input_directory}")
    
    def execute(self) -> BootstrapResult:
        """
        Execute the complete bootstrap pipeline.
        
        Returns:
            BootstrapResult with execution details and metrics
        """
        start_time = datetime.now()
        errors = []
        warnings = []
        
        self.logger.info("Starting Bootstrap Pipeline execution")
        
        try:
            # Phase 1: Initialize storage
            self.logger.info("Phase 1: Initializing storage...")
            self._initialize_storage()
            
            # Phase 2: Analyze directory structure
            self.logger.info("Phase 2: Analyzing directory structure...")
            discovered_files, relationships = self.directory_analyzer.analyze_directory(
                self.config.input_directory
            )
            
            if not discovered_files:
                raise ValueError(f"No files discovered in {self.config.input_directory}")
            
            # Phase 3: Process files
            self.logger.info("Phase 3: Processing files...")
            processed_files = list(self.file_processor.process_files(discovered_files))
            
            if not processed_files:
                raise ValueError("No files could be processed successfully")
            
            # Phase 4: Build graph
            self.logger.info("Phase 4: Building graph structure...")
            self.graph_builder = GraphBuilder(self.config, self.storage)
            metrics = self.graph_builder.build_graph(processed_files, relationships)
            
            # Phase 5: Validate graph
            self.logger.info("Phase 5: Validating graph...")
            if not self.graph_builder.validate_graph():
                warnings.append("Graph validation failed - results may be incomplete")
            
            # Calculate final metrics
            end_time = datetime.now()
            total_time = (end_time - start_time).total_seconds()
            
            # Update metrics with discovered files
            metrics.total_files_discovered = len(discovered_files)
            
            # Create result
            result = BootstrapResult(
                success=True,
                config=self.config,
                metrics=metrics,
                database_name=self.config.output_database,
                node_count=metrics.total_nodes_created,
                edge_count=metrics.total_edges_created,
                errors=errors,
                warnings=warnings
            )
            
            self.logger.info(f"Bootstrap Pipeline completed successfully in {total_time:.2f}s")
            self.logger.info(f"Created graph with {result.node_count} nodes and {result.edge_count} edges")
            
            return result
            
        except Exception as e:
            error_msg = f"Bootstrap Pipeline failed: {e}"
            self.logger.error(error_msg)
            errors.append(error_msg)
            
            # Create failure result
            result = BootstrapResult(
                success=False,
                config=self.config,
                metrics=BootstrapMetrics(),  # Empty metrics
                database_name=self.config.output_database,
                node_count=0,
                edge_count=0,
                errors=errors,
                warnings=warnings
            )
            
            return result
        
        finally:
            # Cleanup
            if self.storage:
                self.storage.disconnect()
    
    def _initialize_storage(self) -> None:
        """Initialize storage component."""
        try:
            # Create storage configuration
            storage_config = {
                **self.config.storage_config,
                "database": {
                    "host": "127.0.0.1",
                    "port": 8529,
                    "username": "root",
                    "password": "",
                    "database": self.config.output_database
                },
                "sequential_isne": {
                    "db_name": self.config.output_database,
                    "batch_size": self.config.batch_size,
                    "similarity_threshold": self.config.semantic_similarity_threshold,
                    "embedding_dim": 768
                }
            }
            
            # Create and connect storage
            self.storage = create_storage_component(
                self.config.storage_component, 
                storage_config
            )
            
            if not self.storage.connect():
                raise RuntimeError("Failed to connect to storage")
            
            self.logger.info(f"Storage connected to database: {self.config.output_database}")
            
        except Exception as e:
            raise RuntimeError(f"Storage initialization failed: {e}")
    
    def get_progress_info(self) -> Dict[str, Any]:
        """Get current progress information."""
        info = {
            "pipeline_status": "running",
            "current_phase": "unknown",
            "directory_analyzer": {},
            "file_processor": {},
            "graph_builder": {}
        }
        
        # Get file processor stats
        if self.file_processor:
            info["file_processor"] = self.file_processor.get_processing_stats()
        
        # Get graph builder stats
        if self.graph_builder:
            info["graph_builder"] = self.graph_builder.get_graph_summary()
        
        return info
    
    @classmethod
    def create_from_directory(
        cls, 
        input_directory: Path, 
        output_database: str = "sequential_isne_bootstrap",
        **kwargs
    ) -> 'BootstrapPipeline':
        """
        Create pipeline from directory with sensible defaults.
        
        Args:
            input_directory: Directory to bootstrap from
            output_database: Target database name
            **kwargs: Additional configuration options
            
        Returns:
            Configured BootstrapPipeline
        """
        config_dict = {
            "input_directory": input_directory,
            "output_database": output_database,
            **kwargs
        }
        
        config = BootstrapConfig(**config_dict)
        return cls(config)
    
    def dry_run(self) -> Dict[str, Any]:
        """
        Perform a dry run to analyze what would be processed.
        
        Returns:
            Analysis results without actually building the graph
        """
        self.logger.info("Performing dry run analysis...")
        
        try:
            # Analyze directory structure
            discovered_files, relationships = self.directory_analyzer.analyze_directory(
                self.config.input_directory
            )
            
            # Analyze file processing (without actually processing)
            file_type_counts = {}
            total_size = 0
            
            for file_path, file_info in discovered_files.items():
                file_type = file_info.get('file_type', 'unknown')
                if file_type not in file_type_counts:
                    file_type_counts[file_type] = 0
                file_type_counts[file_type] += 1
                total_size += file_info.get('size', 0)
            
            # Analyze relationships
            edge_type_counts = {}
            cross_modal_count = 0
            
            for _, _, edge_type, _, metadata in relationships:
                if edge_type not in edge_type_counts:
                    edge_type_counts[edge_type] = 0
                edge_type_counts[edge_type] += 1
                
                # Check if cross-modal
                if any(modal in edge_type for modal in ['_to_', 'cross', 'semantic']):
                    cross_modal_count += 1
            
            return {
                "analysis": {
                    "total_files": len(discovered_files),
                    "file_types": file_type_counts,
                    "total_size_bytes": total_size,
                    "total_relationships": len(relationships),
                    "edge_types": edge_type_counts,
                    "cross_modal_edges": cross_modal_count
                },
                "config": self.config.dict(),
                "estimated_processing_time": self._estimate_processing_time(len(discovered_files)),
                "recommendations": self._generate_recommendations(discovered_files, relationships)
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _estimate_processing_time(self, file_count: int) -> float:
        """Estimate processing time based on file count."""
        # Rough estimation: 0.1 seconds per file + overhead
        base_time = file_count * 0.1
        overhead = 30  # Database initialization, etc.
        
        if self.config.enable_parallel_processing:
            base_time = base_time / self.config.max_workers
        
        return base_time + overhead
    
    def _generate_recommendations(
        self, 
        discovered_files: Dict[str, Dict[str, Any]], 
        relationships: List[Any]
    ) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        # File count recommendations
        file_count = len(discovered_files)
        if file_count < 10:
            recommendations.append("Very few files discovered - consider including more file types")
        elif file_count > 10000:
            recommendations.append("Large number of files - consider using parallel processing")
        
        # Relationship recommendations
        relationship_count = len(relationships)
        if relationship_count < file_count * 0.1:
            recommendations.append("Few relationships discovered - consider enabling more edge discovery methods")
        
        # File type recommendations
        file_types = set()
        for file_info in discovered_files.values():
            file_types.add(file_info.get('file_type', 'unknown'))
        
        if len(file_types) == 1:
            recommendations.append("Only one file type found - cross-modal analysis will be limited")
        
        # Size recommendations
        total_size = sum(info.get('size', 0) for info in discovered_files.values())
        if total_size > 100 * 1024 * 1024:  # 100MB
            recommendations.append("Large total file size - consider adjusting max_file_size or batch_size")
        
        return recommendations