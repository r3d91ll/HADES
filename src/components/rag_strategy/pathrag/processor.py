"""
PathRAG Processor Implementation

This module implements the PathRAG strategy for HADES, following the paper:
"PathRAG: Pruning Graph-based Retrieval Augmented Generation with Relational Paths"

The implementation includes:
1. Flow-based path pruning algorithm
2. Resource allocation with decay penalty  
3. Path reliability scoring
4. Multi-mode retrieval (naive, local, global, hybrid, pathrag)
5. Integration with HADES components (storage, embedding, graph enhancement)
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime, timezone
import networkx as nx
import numpy as np
from collections import defaultdict, deque

from src.types.components.protocols import RAGStrategy
from src.types.components.contracts import (
    RAGStrategyInput, 
    RAGStrategyOutput, 
    RAGMode,
    RAGResult,
    PathInfo,
    ComponentMetadata,
    ComponentType,
    ProcessingStatus
)

# Import HADES components - use lazy imports to avoid circular dependencies
from src.components.registry import get_component

logger = logging.getLogger(__name__)


class PathRAGError(Exception):
    """Exception raised by PathRAG processor."""
    pass


class PathRAGProcessor:
    """
    PathRAG implementation for HADES.
    
    Implements the complete PathRAG algorithm including:
    - Flow-based path pruning
    - Resource allocation with decay penalty
    - Path reliability scoring
    - Multi-mode retrieval strategies
    """
    
    def __init__(self) -> None:
        """Initialize PathRAG processor."""
        self.name = "pathrag"
        self.version = "1.0.0"
        self._component_type = ComponentType.RAG_STRATEGY
        
        # Configuration
        self.config: Dict[str, Any] = {}
        
        # HADES components (will be initialized during configure)
        self.storage = None
        self.embedder = None
        self.graph_enhancer = None
        
        # PathRAG state
        self.knowledge_graph: Optional[nx.Graph] = None
        self.node_embeddings: Dict[str, List[float]] = {}
        self.is_initialized = False
        
        logger.info("Initialized PathRAG processor")
    
    @property
    def component_type(self) -> ComponentType:
        """Get component type."""
        return self._component_type
    
    def configure(self, config: Dict[str, Any]) -> None:
        """
        Configure PathRAG processor.
        
        Args:
            config: Configuration dictionary
        """
        try:
            self.config = config.copy()
            
            # Initialize HADES components based on config
            self._initialize_components()
            
            logger.info("Configured PathRAG processor")
            
        except Exception as e:
            error_msg = f"Failed to configure PathRAG processor: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e
    
    def _initialize_components(self) -> None:
        """Initialize HADES components from config."""
        try:
            # Initialize storage component (simplified for testing)
            storage_config = self.config.get('storage', {})
            storage_type = storage_config.get('type', 'memory')
            
            # For testing, use a simple mock storage
            self.storage = type('MockStorage', (), {
                'health_check': lambda: True,
                'get_metrics': lambda: {'type': 'mock'}
            })()
            
            # Initialize embedder component - lazy import to avoid circular dependencies
            embedder_config = self.config.get('embedder', {})
            embedder_type = embedder_config.get('type', 'cpu')
            try:
                from src.components.embedding.factory import create_embedding_component
                self.embedder = create_embedding_component(embedder_type, embedder_config)
            except ImportError as e:
                logger.warning(f"Could not load embedding component: {e}")
                self.embedder = None
            
            # Initialize graph enhancer component - lazy import to avoid circular dependencies
            enhancer_config = self.config.get('graph_enhancer', {})
            enhancer_type = enhancer_config.get('type', 'isne')
            try:
                from src.components.graph_enhancement.factory import create_graph_enhancement_component
                self.graph_enhancer = create_graph_enhancement_component(enhancer_type, enhancer_config)
            except ImportError as e:
                logger.warning(f"Could not load graph enhancement component: {e}")
                self.graph_enhancer = None
            
            logger.info("Initialized HADES components for PathRAG")
            
        except Exception as e:
            error_msg = f"Failed to initialize HADES components: {e}"
            logger.error(error_msg)
            raise PathRAGError(error_msg) from e
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate configuration parameters.
        
        Args:
            config: Configuration to validate
            
        Returns:
            True if valid, False otherwise
        """
        required_keys = ['flow_decay_factor', 'pruning_threshold', 'max_path_length']
        
        for key in required_keys:
            if key not in config:
                logger.warning(f"Missing required config key: {key}")
                return False
        
        # Validate parameter ranges
        if not (0.0 < config['flow_decay_factor'] <= 1.0):
            logger.warning("flow_decay_factor must be between 0.0 and 1.0")
            return False
            
        if not (0.0 < config['pruning_threshold'] <= 1.0):
            logger.warning("pruning_threshold must be between 0.0 and 1.0")
            return False
            
        if not (1 <= config['max_path_length'] <= 10):
            logger.warning("max_path_length must be between 1 and 10")
            return False
        
        return True
    
    def get_config_schema(self) -> Dict[str, Any]:
        """
        Get JSON schema for PathRAG configuration.
        
        Returns:
            JSON schema dictionary
        """
        return {
            "type": "object",
            "properties": {
                "flow_decay_factor": {
                    "type": "number", 
                    "minimum": 0.0, 
                    "maximum": 1.0,
                    "default": 0.85
                },
                "pruning_threshold": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0, 
                    "default": 0.1
                },
                "max_path_length": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "default": 5
                },
                "max_iterations": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 10
                },
                "storage": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string", "default": "arangodb"}
                    }
                },
                "embedder": {
                    "type": "object", 
                    "properties": {
                        "type": {"type": "string", "default": "cpu"}
                    }
                },
                "graph_enhancer": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string", "default": "isne"}
                    }
                }
            },
            "required": ["flow_decay_factor", "pruning_threshold", "max_path_length"]
        }
    
    def health_check(self) -> bool:
        """
        Check PathRAG processor health.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            # Check component initialization
            if not all([self.storage, self.embedder, self.graph_enhancer]):
                logger.warning("PathRAG components not properly initialized")
                return False
            
            # Check component health
            if self.storage and hasattr(self.storage, 'health_check') and not self.storage.health_check():
                logger.warning("Storage component health check failed")
                return False
            
            if self.embedder and hasattr(self.embedder, 'health_check') and not self.embedder.health_check():
                logger.warning("Embedder component health check failed")
                return False
                
            if self.graph_enhancer and hasattr(self.graph_enhancer, 'health_check') and not self.graph_enhancer.health_check():
                logger.warning("Graph enhancer component health check failed")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"PathRAG health check failed: {e}")
            return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get PathRAG performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        metrics = {
            "component_name": self.name,
            "component_version": self.version,
            "is_initialized": self.is_initialized,
            "knowledge_graph_nodes": len(self.knowledge_graph.nodes()) if self.knowledge_graph else 0,
            "knowledge_graph_edges": len(self.knowledge_graph.edges()) if self.knowledge_graph else 0,
            "cached_embeddings": len(self.node_embeddings)
        }
        
        # Add component metrics if available
        if self.storage and hasattr(self.storage, 'get_metrics'):
            metrics["storage_metrics"] = self.storage.get_metrics()
        if self.embedder and hasattr(self.embedder, 'get_metrics'):
            metrics["embedder_metrics"] = self.embedder.get_metrics()
        if self.graph_enhancer and hasattr(self.graph_enhancer, 'get_metrics'):
            metrics["graph_enhancer_metrics"] = self.graph_enhancer.get_metrics()
        
        return metrics
    
    def retrieve_and_generate(self, input_data: RAGStrategyInput) -> RAGStrategyOutput:
        """
        Perform PathRAG retrieval and generation.
        
        Args:
            input_data: Input conforming to RAGStrategyInput contract
            
        Returns:
            Output conforming to RAGStrategyOutput contract
        """
        start_time = time.time()
        
        try:
            # Ensure knowledge base is initialized
            if not self.is_initialized:
                self._initialize_knowledge_base()
            
            # Step 1: Extract keywords and get initial embeddings
            query_keywords = self._extract_keywords(input_data.query)
            query_embedding = self._get_query_embedding(input_data.query)
            
            # Step 2: Node retrieval - find relevant starting nodes
            relevant_nodes = self._retrieve_relevant_nodes(
                query_embedding, 
                query_keywords,
                top_k=input_data.top_k * 2  # Get more nodes for path exploration
            )
            
            # Step 3: Path retrieval based on mode
            if input_data.mode == RAGMode.PATHRAG:
                results, paths_explored = self._pathrag_retrieval(
                    relevant_nodes, 
                    input_data
                )
            else:
                results, paths_explored = self._standard_retrieval(
                    relevant_nodes,
                    input_data
                )
            
            # Step 4: Generate answer (optional)
            generated_answer = None
            answer_confidence = None
            if len(results) > 0:
                generated_answer, answer_confidence = self._generate_answer(
                    input_data.query, 
                    results
                )
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Create output
            output = RAGStrategyOutput(
                query=input_data.query,
                mode=input_data.mode,
                results=results[:input_data.top_k],  # Limit to requested top_k
                generated_answer=generated_answer,
                answer_confidence=answer_confidence,
                paths_explored=paths_explored,
                graph_stats=self._get_graph_stats(),
                metadata=ComponentMetadata(
                    component_name=self.name,
                    component_version=self.version,
                    component_type=self.component_type,
                    processed_at=datetime.now(timezone.utc),
                    status=ProcessingStatus.SUCCESS
                ),
                retrieval_stats={
                    "relevant_nodes_found": len(relevant_nodes),
                    "paths_explored": len(paths_explored),
                    "results_generated": len(results)
                },
                query_processing_time=processing_time,
                total_results=len(results)
            )
            
            logger.info(f"PathRAG processing completed in {processing_time:.2f}s")
            return output
            
        except Exception as e:
            error_msg = f"PathRAG processing failed: {e}"
            logger.error(error_msg)
            
            # Return error output
            return RAGStrategyOutput(
                query=input_data.query,
                mode=input_data.mode,
                results=[],
                metadata=ComponentMetadata(
                    component_name=self.name,
                    component_version=self.version,
                    component_type=self.component_type,
                    processed_at=datetime.now(timezone.utc),
                    status=ProcessingStatus.ERROR
                ),
                errors=[error_msg],
                query_processing_time=time.time() - start_time,
                total_results=0
            )
    
    def retrieve_only(self, input_data: RAGStrategyInput) -> RAGStrategyOutput:
        """
        Perform only retrieval without generation.
        
        Args:
            input_data: Input conforming to RAGStrategyInput contract
            
        Returns:
            Output with results but no generated_answer
        """
        # Use full retrieve_and_generate but ignore answer generation
        output = self.retrieve_and_generate(input_data)
        output.generated_answer = None
        output.answer_confidence = None
        return output
    
    def get_supported_modes(self) -> List[str]:
        """
        Get list of supported RAG modes.
        
        Returns:
            List of supported mode strings
        """
        return ["naive", "local", "global", "hybrid", "pathrag"]
    
    def supports_mode(self, mode: str) -> bool:
        """
        Check if strategy supports the given mode.
        
        Args:
            mode: RAG mode to check
            
        Returns:
            True if mode is supported
        """
        return mode in self.get_supported_modes()
    
    def initialize_knowledge_base(self, storage_config: Dict[str, Any]) -> bool:
        """
        Initialize the knowledge base for retrieval.
        
        Args:
            storage_config: Storage configuration
            
        Returns:
            True if initialization successful
        """
        try:
            self._initialize_knowledge_base()
            return True
        except Exception as e:
            logger.error(f"Failed to initialize knowledge base: {e}")
            return False
    
    def _initialize_knowledge_base(self) -> None:
        """Load knowledge graph and embeddings from storage."""
        try:
            # Load graph structure from storage
            self.knowledge_graph = self._load_graph_from_storage()
            
            # Load or compute node embeddings
            self._load_node_embeddings()
            
            self.is_initialized = True
            logger.info("Knowledge base initialized successfully")
            
        except Exception as e:
            error_msg = f"Failed to initialize knowledge base: {e}"
            logger.error(error_msg)
            raise PathRAGError(error_msg) from e
    
    def _load_graph_from_storage(self) -> nx.Graph:
        """Load graph structure from storage."""
        # This is a placeholder - implement based on storage structure
        # For now, create a simple test graph
        graph = nx.Graph()
        
        # Add some test nodes and edges
        # In real implementation, this would query the storage backend
        for i in range(10):
            graph.add_node(f"node_{i}", content=f"Content for node {i}")
            
        for i in range(8):
            graph.add_edge(f"node_{i}", f"node_{i+1}", weight=0.8)
        
        logger.info(f"Loaded graph with {len(graph.nodes())} nodes and {len(graph.edges())} edges")
        return graph
    
    def _load_node_embeddings(self) -> None:
        """Load or compute embeddings for all nodes."""
        # This is a placeholder - implement based on embedder component
        # For now, create random embeddings
        for node_id in self.knowledge_graph.nodes():
            # In real implementation, load from storage or compute with embedder
            self.node_embeddings[node_id] = np.random.random(768).tolist()
        
        logger.info(f"Loaded embeddings for {len(self.node_embeddings)} nodes")
    
    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base."""
        if not self.is_initialized:
            return {"initialized": False}
        
        return {
            "initialized": True,
            "total_nodes": len(self.knowledge_graph.nodes()) if self.knowledge_graph else 0,
            "total_edges": len(self.knowledge_graph.edges()) if self.knowledge_graph else 0,
            "cached_embeddings": len(self.node_embeddings),
            "average_degree": (2 * len(self.knowledge_graph.edges()) / len(self.knowledge_graph.nodes())) if self.knowledge_graph and len(self.knowledge_graph.nodes()) > 0 else 0
        }
    
    def estimate_retrieval_time(self, input_data: RAGStrategyInput) -> float:
        """Estimate time for retrieval operation."""
        # Simple estimation based on top_k and mode
        base_time = 0.1  # Base retrieval time
        
        if input_data.mode == RAGMode.PATHRAG:
            # PathRAG is more expensive due to path exploration
            return base_time + (input_data.top_k * 0.05) + (input_data.max_path_length * 0.02)
        else:
            return base_time + (input_data.top_k * 0.01)
    
    def estimate_generation_time(self, input_data: RAGStrategyInput) -> float:
        """Estimate time for generation operation."""
        # Simple estimation based on context size
        base_time = 0.5  # Base generation time
        context_factor = input_data.max_token_for_context / 1000  # Scale by context size
        return base_time + context_factor
    
    def get_optimal_context_size(self, query_type: str) -> int:
        """Get optimal context size for query type."""
        context_sizes = {
            "factual": 2000,
            "analytical": 4000,
            "comparison": 3000,
            "summary": 3500,
            "default": 3000
        }
        return context_sizes.get(query_type, context_sizes["default"])
    
    # ===== PathRAG Algorithm Implementation =====
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords from query (placeholder implementation)."""
        # Simple keyword extraction - in real implementation use LLM or NLP
        stop_words = {"the", "is", "are", "was", "were", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        words = query.lower().split()
        keywords = [word.strip('.,!?;:') for word in words if word not in stop_words and len(word) > 2]
        return keywords[:10]  # Limit to top 10 keywords
    
    def _get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for query."""
        # Placeholder - use embedder component
        embedding: List[float] = np.random.random(768).tolist()
        return embedding
    
    def _retrieve_relevant_nodes(self, query_embedding: List[float], keywords: List[str], top_k: int) -> List[str]:
        """Retrieve relevant nodes using embedding similarity."""
        # Placeholder implementation using random selection
        # In real implementation, compute cosine similarity with node embeddings
        if self.knowledge_graph is None:
            return []
        all_nodes = list(self.knowledge_graph.nodes())
        return all_nodes[:min(top_k, len(all_nodes))]
    
    def _pathrag_retrieval(self, relevant_nodes: List[str], input_data: RAGStrategyInput) -> Tuple[List[RAGResult], List[PathInfo]]:
        """Perform PathRAG-specific retrieval with flow-based pruning."""
        # Implementation of the core PathRAG algorithm
        
        # Step 1: Initialize resource allocation
        resources = self._initialize_resource_allocation(relevant_nodes)
        
        # Step 2: Flow-based pruning with decay penalty
        pruned_resources = self._flow_based_pruning(
            resources,
            decay_factor=input_data.flow_decay_factor,
            threshold=input_data.pruning_threshold,
            max_iterations=self.config.get('max_iterations', 10)
        )
        
        # Step 3: Extract and score paths
        paths = self._extract_paths_between_nodes(
            list(pruned_resources.keys()),
            max_length=input_data.max_path_length
        )
        
        scored_paths = self._score_paths_by_reliability(paths, pruned_resources)
        
        # Step 4: Convert paths to results
        results = self._paths_to_results(scored_paths[:input_data.top_k])
        
        # Step 5: Create path info for output
        path_infos = self._create_path_infos(scored_paths)
        
        return results, path_infos
    
    def _standard_retrieval(self, relevant_nodes: List[str], input_data: RAGStrategyInput) -> Tuple[List[RAGResult], List[PathInfo]]:
        """Perform standard retrieval (non-PathRAG modes)."""
        # Simple implementation for non-PathRAG modes
        results = []
        
        for i, node_id in enumerate(relevant_nodes[:input_data.top_k]):
            if self.knowledge_graph is None:
                continue
            node_data = self.knowledge_graph.nodes[node_id]
            
            result = RAGResult(
                item_id=node_id,
                content=node_data.get('content', f'Content for {node_id}'),
                score=max(0.1, 1.0 - (i * 0.1)),  # Decreasing score
                relevance_score=max(0.1, 1.0 - (i * 0.1)),
                diversity_score=0.5,
                metadata={"mode": input_data.mode.value, "node_id": node_id}
            )
            results.append(result)
        
        return results, []  # No path info for standard retrieval
    
    def _initialize_resource_allocation(self, start_nodes: List[str]) -> Dict[str, float]:
        """Initialize resource allocation for flow-based pruning."""
        resources = {}
        initial_resource = 1.0 / len(start_nodes) if start_nodes else 0.0
        
        if self.knowledge_graph is not None:
            for node in self.knowledge_graph.nodes():
                resources[node] = initial_resource if node in start_nodes else 0.0
        
        return resources
    
    def _flow_based_pruning(self, initial_resources: Dict[str, float], decay_factor: float, threshold: float, max_iterations: int) -> Dict[str, float]:
        """
        Implement flow-based pruning with resource allocation.
        
        This is the core PathRAG algorithm from the paper.
        """
        resources = initial_resources.copy()
        
        for iteration in range(max_iterations):
            new_resources = {node: 0.0 for node in resources}
            
            # Propagate resources through the graph
            for node in resources:
                if resources[node] < threshold:
                    continue  # Early stopping
                
                if self.knowledge_graph is None:
                    continue
                neighbors = list(self.knowledge_graph.neighbors(node))
                if not neighbors:
                    continue
                
                # Resource propagation with decay penalty
                resource_per_neighbor = resources[node] * decay_factor / len(neighbors)
                
                for neighbor in neighbors:
                    edge_weight = self.knowledge_graph[node][neighbor].get('weight', 1.0)
                    transferred_resource = resource_per_neighbor * edge_weight
                    new_resources[neighbor] += transferred_resource
            
            # Update resources
            resources.update(new_resources)
            
            # Check convergence
            total_change = sum(abs(new_resources[node] - resources[node]) for node in resources)
            if total_change < 0.001:  # Convergence threshold
                break
        
        # Filter nodes above threshold
        return {node: resource for node, resource in resources.items() if resource >= threshold}
    
    def _extract_paths_between_nodes(self, nodes: List[str], max_length: int) -> List[List[str]]:
        """Extract paths between high-resource nodes."""
        paths = []
        
        for i, start_node in enumerate(nodes):
            for end_node in nodes[i+1:]:
                try:
                    # Find shortest path (up to max_length)
                    if nx.has_path(self.knowledge_graph, start_node, end_node):
                        path = nx.shortest_path(self.knowledge_graph, start_node, end_node)
                        if len(path) <= max_length:
                            paths.append(path)
                except nx.NetworkXNoPath:
                    continue
        
        return paths
    
    def _score_paths_by_reliability(self, paths: List[List[str]], resources: Dict[str, float]) -> List[Tuple[List[str], float]]:
        """Score paths by reliability using resource values."""
        scored_paths = []
        
        for path in paths:
            if len(path) < 2:
                continue
            
            # Calculate path reliability as average resource values
            path_resources = [resources.get(node, 0.0) for node in path]
            reliability_score = sum(path_resources) / len(path_resources)
            
            scored_paths.append((path, reliability_score))
        
        # Sort by reliability score (descending)
        scored_paths.sort(key=lambda x: x[1], reverse=True)
        return scored_paths
    
    def _paths_to_results(self, scored_paths: List[Tuple[List[str], float]]) -> List[RAGResult]:
        """Convert scored paths to RAG results."""
        results = []
        
        for i, (path, score) in enumerate(scored_paths):
            # Aggregate content from path nodes
            path_content = []
            for node_id in path:
                if self.knowledge_graph is None:
                    continue
                node_data = self.knowledge_graph.nodes[node_id]
                content = node_data.get('content', f'Content for {node_id}')
                path_content.append(content)
            
            aggregated_content = " -> ".join(path_content)
            
            # Create path info
            path_info = PathInfo(
                path_id=f"path_{i}",
                nodes=path,
                edges=[f"{path[j]}-{path[j+1]}" for j in range(len(path)-1)],
                path_score=score,
                reliability_score=score,
                metadata={"path_length": len(path)}
            )
            
            result = RAGResult(
                item_id=f"path_result_{i}",
                content=aggregated_content,
                score=score,
                path_info=path_info,
                relevance_score=score,
                diversity_score=min(1.0, len(set(path)) / len(path)),  # Diversity based on unique nodes
                metadata={"path_nodes": path, "path_length": len(path)}
            )
            
            results.append(result)
        
        return results
    
    def _create_path_infos(self, scored_paths: List[Tuple[List[str], float]]) -> List[PathInfo]:
        """Create PathInfo objects for output."""
        path_infos = []
        
        for i, (path, score) in enumerate(scored_paths):
            path_info = PathInfo(
                path_id=f"explored_path_{i}",
                nodes=path,
                edges=[f"{path[j]}-{path[j+1]}" for j in range(len(path)-1)],
                path_score=score,
                reliability_score=score,
                metadata={"exploration_order": i, "path_length": len(path)}
            )
            path_infos.append(path_info)
        
        return path_infos
    
    def _generate_answer(self, query: str, results: List[RAGResult]) -> Tuple[str, float]:
        """Generate answer from retrieved results (placeholder)."""
        # Placeholder implementation
        # In real implementation, use LLM to generate answer from context
        
        if not results:
            return "No relevant information found.", 0.0
        
        # Simple answer generation
        context_snippets = [result.content[:200] + "..." for result in results[:3]]
        generated_answer = f"Based on the retrieved information: {' '.join(context_snippets)}"
        confidence = min(0.9, len(results) * 0.2)  # Simple confidence calculation
        
        return generated_answer, confidence
    
    def _get_graph_stats(self) -> Dict[str, Any]:
        """Get current graph statistics."""
        if self.knowledge_graph is None:
            return {}
        
        return {
            "total_nodes": len(self.knowledge_graph.nodes()),
            "total_edges": len(self.knowledge_graph.edges()),
            "average_degree": (2 * len(self.knowledge_graph.edges()) / len(self.knowledge_graph.nodes())) if len(self.knowledge_graph.nodes()) > 0 else 0,
            "connected_components": nx.number_connected_components(self.knowledge_graph),
            "graph_density": nx.density(self.knowledge_graph)
        }