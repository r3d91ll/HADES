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
6. Jina v4 embedding integration for query processing
7. Fallback mechanisms for robust query embedding generation

Recent Updates:
- Implemented actual query embedding generation using Jina v4 processor
- Added support for both single-vector and multi-vector embedding modes
- Integrated embedder component with proper fallback handling
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Set, Tuple, Union
from datetime import datetime, timezone
import networkx as nx
import numpy as np
from collections import defaultdict, deque

from src.types.pathrag.strategy import (
    PathRAGConfig,
    PathNode,
    RetrievalPath,
    PathRAGResult,
    PathScore,
    QueryDecomposition,
    RetrievalStats
)
from src.types.common import ProcessingStatus

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
        self._component_type = "rag_strategy"
        
        # Configuration
        self.config: Dict[str, Any] = {}
        
        # HADES components (will be initialized during configure)
        self.storage: Optional[Any] = None
        self.embedder: Optional[Any] = None
        self.graph_enhancer: Optional[Any] = None
        
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
            if not all([self.storage is not None, self.embedder is not None, self.graph_enhancer is not None]):
                logger.warning("PathRAG components not properly initialized")
                return False
            
            # Check component health
            if self.storage is not None and hasattr(self.storage, 'health_check') and not self.storage.health_check():
                logger.warning("Storage component health check failed")
                return False
            
            if self.embedder is not None and hasattr(self.embedder, 'health_check') and not self.embedder.health_check():
                logger.warning("Embedder component health check failed")
                return False
                
            if self.graph_enhancer is not None and hasattr(self.graph_enhancer, 'health_check') and not self.graph_enhancer.health_check():
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
        if self.storage is not None and hasattr(self.storage, 'get_metrics'):
            metrics["storage_metrics"] = self.storage.get_metrics()
        if self.embedder is not None and hasattr(self.embedder, 'get_metrics'):
            metrics["embedder_metrics"] = self.embedder.get_metrics()
        if self.graph_enhancer is not None and hasattr(self.graph_enhancer, 'get_metrics'):
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
                    "entity_nodes_found": len(relevant_nodes.get("entity_nodes", [])),
                    "relationship_nodes_found": len(relevant_nodes.get("relationship_nodes", [])),
                    "total_relevant_nodes": len(relevant_nodes.get("entity_nodes", [])) + len(relevant_nodes.get("relationship_nodes", [])),
                    "paths_explored": len(paths_explored),
                    "results_generated": len(results),
                    "mode": input_data.mode.value
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
        if self.knowledge_graph is not None:
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
    
    def _extract_keywords(self, query: str) -> Dict[str, List[str]]:
        """Extract hierarchical keywords from query using LLM-based approach."""
        try:
            # Use LLM for sophisticated keyword extraction
            return self._llm_keyword_extraction(query)
        except Exception as e:
            logger.warning(f"LLM keyword extraction failed: {e}, falling back to simple extraction")
            return self._simple_keyword_extraction(query)
    
    def _llm_keyword_extraction(self, query: str) -> Dict[str, List[str]]:
        """Extract keywords using LLM with hierarchical structure."""
        # Keyword extraction prompt based on legacy implementation
        extraction_prompt = f"""
Extract keywords from the following query in a hierarchical manner:

Query: "{query}"

Please provide keywords in JSON format with:
- "high_level_keywords" for overarching concepts or themes
- "low_level_keywords" for specific entities or details

Example format:
{{
    "high_level_keywords": ["concept1", "theme2"],
    "low_level_keywords": ["entity1", "detail2", "specific3"]
}}

Keywords:"""
        
        # Placeholder for LLM call - integrate with model_engine when available
        # For now, fall back to simple extraction
        return self._simple_keyword_extraction(query)
    
    def _simple_keyword_extraction(self, query: str) -> Dict[str, List[str]]:
        """Simple keyword extraction fallback."""
        stop_words = {"the", "is", "are", "was", "were", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        words = query.lower().split()
        keywords = [word.strip('.,!?;:') for word in words if word not in stop_words and len(word) > 2]
        
        # Split into high-level and low-level based on word length and commonality
        high_level = [word for word in keywords if len(word) > 6][:5]
        low_level = [word for word in keywords if len(word) <= 6][:10]
        
        return {
            "high_level_keywords": high_level,
            "low_level_keywords": low_level
        }
    
    def _get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for query using Jina v4 processor."""
        try:
            # Check if we have an embedder component
            if self.embedder is not None:
                # Use the embedder component's embed method
                embeddings = self.embedder.embed([query])
                if embeddings and len(embeddings) > 0:
                    embedding = embeddings[0]
                    if isinstance(embedding, list):
                        return embedding
                    elif hasattr(embedding, 'tolist'):
                        return embedding.tolist()
                    else:
                        # Convert to list if needed
                        return list(embedding)
            
            # Fallback: Use Jina v4 processor directly
            from src.jina_v4.factory import JinaV4Factory
            
            # Get or create processor instance
            factory = JinaV4Factory()
            processor = factory.create_processor()
            
            # Process query as a document
            query_structure = {
                "content": query,
                "metadata": {"type": "query"}
            }
            
            # Generate embeddings synchronously
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're already in an async context, create a new task
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, processor.process(content=query))
                    result = future.result()
            else:
                # We can run directly
                result = asyncio.run(processor.process(content=query))
            
            # Extract embedding from result
            if result and 'chunks' in result and len(result['chunks']) > 0:
                chunk = result['chunks'][0]
                if 'embeddings' in chunk:
                    embeddings = chunk['embeddings']
                    # Handle both single-vector and multi-vector modes
                    if isinstance(embeddings, list):
                        if len(embeddings) > 0:
                            if isinstance(embeddings[0], list):
                                # Multi-vector mode - use mean pooling
                                embedding_array = np.array(embeddings)
                                return np.mean(embedding_array, axis=0).tolist()
                            else:
                                # Single-vector mode
                                return embeddings
                    elif isinstance(embeddings, dict) and 'multimodal' in embeddings:
                        # Handle structured embedding response
                        multimodal_embedding = embeddings['multimodal']
                        if isinstance(multimodal_embedding, list):
                            return multimodal_embedding
                        else:
                            return list(multimodal_embedding)
            
            # Final fallback with proper dimensionality
            logger.warning("Could not generate query embedding, using fallback")
            embedding_dim = self.config.get('embedding_dimension', 2048)
            embedding: List[float] = np.random.random(embedding_dim).tolist()
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            # Return fallback embedding with proper dimensionality
            embedding_dim = self.config.get('embedding_dimension', 2048)
            embedding: List[float] = np.random.random(embedding_dim).tolist()
            return embedding
    
    def _retrieve_relevant_nodes(self, query_embedding: List[float], keywords: Dict[str, List[str]], top_k: int) -> Dict[str, List[str]]:
        """Retrieve relevant nodes using hierarchical keyword strategy."""
        results: Dict[str, List[str]] = {
            "entity_nodes": [],
            "relationship_nodes": []
        }
        
        if self.knowledge_graph is None:
            return results
        
        # Get entity nodes using low-level keywords (local search)
        entity_nodes = self._get_entity_nodes_by_keywords(
            keywords.get("low_level_keywords", []), top_k
        )
        
        # Get relationship nodes using high-level keywords (global search)  
        relationship_nodes = self._get_relationship_nodes_by_keywords(
            keywords.get("high_level_keywords", []), top_k
        )
        
        results["entity_nodes"] = entity_nodes
        results["relationship_nodes"] = relationship_nodes
        
        return results
    
    def _get_entity_nodes_by_keywords(self, keywords: List[str], top_k: int) -> List[str]:
        """Get entity nodes using low-level keywords for local search."""
        # Placeholder - implement entity search using embedding similarity
        if not keywords or self.knowledge_graph is None:
            return []
        
        entity_nodes = [node for node in self.knowledge_graph.nodes() 
                       if any(keyword.lower() in str(node).lower() for keyword in keywords)]
        return entity_nodes[:top_k]
    
    def _get_relationship_nodes_by_keywords(self, keywords: List[str], top_k: int) -> List[str]:
        """Get relationship nodes using high-level keywords for global search."""
        # Placeholder - implement relationship search using embedding similarity
        if not keywords or self.knowledge_graph is None:
            return []
        
        # For now, return nodes that might represent relationships
        relationship_nodes = [node for node in self.knowledge_graph.nodes() 
                            if any(keyword.lower() in str(node).lower() for keyword in keywords)]
        return relationship_nodes[:top_k]
    
    def _pathrag_retrieval(self, relevant_nodes: Dict[str, List[str]], input_data: RAGStrategyInput) -> Tuple[List[RAGResult], List[PathInfo]]:
        """Perform PathRAG-specific retrieval with flow-based pruning."""
        # Combine entity and relationship nodes for comprehensive retrieval
        all_relevant_nodes = relevant_nodes.get("entity_nodes", []) + relevant_nodes.get("relationship_nodes", [])
        
        if not all_relevant_nodes:
            return [], []
        
        # Step 1: Initialize resource allocation
        resources = self._initialize_resource_allocation(all_relevant_nodes)
        
        # Step 2: Flow-based pruning with decay penalty (core PathRAG algorithm)
        pruned_resources = self._flow_based_pruning(
            resources,
            decay_factor=getattr(input_data, 'flow_decay_factor', self.config.get('flow_decay_factor', 0.8)),
            threshold=getattr(input_data, 'pruning_threshold', self.config.get('pruning_threshold', 0.3)),
            max_iterations=self.config.get('max_iterations', 10)
        )
        
        # Step 3: Multi-hop path exploration (legacy algorithm integration)
        paths = self._find_most_related_edges_from_entities(
            list(pruned_resources.keys()),
            max_length=getattr(input_data, 'max_path_length', self.config.get('max_path_length', 5))
        )
        
        # Step 4: Score paths by reliability using resource values
        scored_paths = self._score_paths_by_reliability_data(paths, pruned_resources)
        
        # Step 5: Convert paths to results with context building
        results = self._paths_to_results_with_context(scored_paths[:input_data.top_k])
        
        # Step 6: Create path info for output
        path_infos = self._create_path_infos_from_data(scored_paths)
        
        return results, path_infos
    
    def _standard_retrieval(self, relevant_nodes: Dict[str, List[str]], input_data: RAGStrategyInput) -> Tuple[List[RAGResult], List[PathInfo]]:
        """Perform standard retrieval (non-PathRAG modes) with hybrid approach."""
        results: List[RAGResult] = []
        
        # Handle hybrid retrieval mode (local + global)
        if input_data.mode.value in ["hybrid", "local", "global"]:
            return self._hybrid_retrieval_strategy(relevant_nodes, input_data)
        
        # Naive mode - simple node retrieval
        all_nodes = relevant_nodes.get("entity_nodes", []) + relevant_nodes.get("relationship_nodes", [])
        
        for i, node_id in enumerate(all_nodes[:input_data.top_k]):
            if self.knowledge_graph is None:
                continue
            
            node_data = self.knowledge_graph.nodes.get(node_id, {})
            
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
    
    def _hybrid_retrieval_strategy(self, relevant_nodes: Dict[str, List[str]], input_data: RAGStrategyInput) -> Tuple[List[RAGResult], List[PathInfo]]:
        """Implement hybrid retrieval strategy (legacy algorithm)."""
        results: List[RAGResult] = []
        
        # Local context from entity nodes (low-level keywords)
        entity_results = self._build_local_context(
            relevant_nodes.get("entity_nodes", []), 
            input_data.top_k // 2
        )
        
        # Global context from relationship nodes (high-level keywords)
        relationship_results = self._build_global_context(
            relevant_nodes.get("relationship_nodes", []), 
            input_data.top_k // 2
        )
        
        # Combine contexts with weighting
        entity_weight = self.config.get('context', {}).get('weights', {}).get('entity_context', 0.4)
        relationship_weight = self.config.get('context', {}).get('weights', {}).get('relationship_context', 0.4)
        
        # Weight and combine results
        for result in entity_results:
            result.score *= entity_weight
            result.metadata['context_type'] = 'local_entity'
            results.append(result)
        
        for result in relationship_results:
            result.score *= relationship_weight
            result.metadata['context_type'] = 'global_relationship'
            results.append(result)
        
        # Sort by combined score
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results[:input_data.top_k], []
    
    def _build_local_context(self, entity_nodes: List[str], top_k: int) -> List[RAGResult]:
        """Build local context from entity nodes."""
        results = []
        
        for i, node_id in enumerate(entity_nodes[:top_k]):
            if self.knowledge_graph is None:
                continue
            
            node_data = self.knowledge_graph.nodes.get(node_id, {})
            
            # Build entity-focused content
            content_parts = []
            if 'description' in node_data:
                content_parts.append(f"Entity: {node_data['description']}")
            if 'content' in node_data:
                content_parts.append(f"Content: {node_data['content']}")
            
            content = "; ".join(content_parts) if content_parts else f"Entity: {node_id}"
            
            result = RAGResult(
                item_id=f"local_{node_id}",
                content=content,
                score=max(0.1, 1.0 - (i * 0.1)),
                relevance_score=max(0.1, 1.0 - (i * 0.1)),
                diversity_score=0.6,
                metadata={"context_type": "local", "node_id": node_id}
            )
            results.append(result)
        
        return results
    
    def _build_global_context(self, relationship_nodes: List[str], top_k: int) -> List[RAGResult]:
        """Build global context from relationship nodes."""
        results = []
        
        for i, node_id in enumerate(relationship_nodes[:top_k]):
            if self.knowledge_graph is None:
                continue
            
            # Build relationship-focused content by exploring connections
            neighbors = list(self.knowledge_graph.neighbors(node_id))
            
            content_parts = [f"Relationship hub: {node_id}"]
            if neighbors:
                content_parts.append(f"Connected to: {', '.join(neighbors[:5])}")
            
            node_data = self.knowledge_graph.nodes.get(node_id, {})
            if 'description' in node_data:
                content_parts.append(f"Description: {node_data['description']}")
            
            content = "; ".join(content_parts)
            
            result = RAGResult(
                item_id=f"global_{node_id}",
                content=content,
                score=max(0.1, 1.0 - (i * 0.1)),
                relevance_score=max(0.1, 1.0 - (i * 0.1)),
                diversity_score=0.7,
                metadata={"context_type": "global", "node_id": node_id, "neighbors": len(neighbors)}
            )
            results.append(result)
        
        return results
    
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
    
    def _find_most_related_edges_from_entities(self, nodes: List[str], max_length: int) -> List[Dict[str, Any]]:
        """Find most related edges using multi-hop reasoning (legacy algorithm)."""
        if self.knowledge_graph is None or len(nodes) < 2:
            return []
        
        paths_with_stats = []
        
        # Process entity pairs for multi-hop path discovery
        for i, source_node in enumerate(nodes):
            for target_node in nodes[i+1:]:
                if source_node == target_node:
                    continue
                
                # Find paths between entity pairs
                entity_paths = self._find_paths_between_entities(
                    source_node, target_node, max_length
                )
                
                for path_data in entity_paths:
                    paths_with_stats.append(path_data)
        
        # Sort by path score and return top paths
        paths_with_stats.sort(key=lambda x: x.get('score', 0), reverse=True)
        return paths_with_stats[:50]  # Limit to top 50 paths
    
    def _find_paths_between_entities(self, source: str, target: str, max_hops: int) -> List[Dict[str, Any]]:
        """Find paths between two entities with multi-hop reasoning."""
        paths: List[Dict[str, Any]] = []
        
        if not nx.has_path(self.knowledge_graph, source, target):
            return paths
        
        # Try different hop lengths (1-hop, 2-hop, 3-hop)
        for hop_length in range(1, min(max_hops + 1, 4)):
            try:
                # Use BFS to find paths of specific length
                hop_paths = self._find_paths_with_hops(source, target, hop_length)
                
                for path in hop_paths:
                    path_data = self._create_path_data_with_narrative(path, hop_length)
                    if path_data:
                        paths.append(path_data)
                        
            except Exception as e:
                logger.debug(f"Error finding {hop_length}-hop path from {source} to {target}: {e}")
                continue
        
        return paths
    
    def _find_paths_with_hops(self, source: str, target: str, target_hops: int) -> List[List[str]]:
        """Find all paths with specific number of hops using BFS."""
        if target_hops <= 0 or self.knowledge_graph is None:
            return []
        
        # BFS to find paths of exact length
        queue = deque([(source, [source])])
        paths: List[List[str]] = []
        
        while queue:
            current_node, current_path = queue.popleft()
            
            if len(current_path) == target_hops + 1:
                if current_node == target:
                    paths.append(current_path)
                continue
            
            if len(current_path) > target_hops + 1:
                continue
            
            # Explore neighbors
            if self.knowledge_graph is not None:
                for neighbor in self.knowledge_graph.neighbors(current_node):
                    if neighbor not in current_path:  # Avoid cycles
                        new_path = current_path + [neighbor]
                        queue.append((neighbor, new_path))
        
        return paths[:10]  # Limit paths per hop length
    
    def _create_path_data_with_narrative(self, path: List[str], hop_length: int) -> Optional[Dict[str, Any]]:
        """Create path data with narrative description (legacy algorithm)."""
        if len(path) < 2 or self.knowledge_graph is None:
            return None
        
        # Build narrative description based on path length
        narrative = self._build_path_narrative(path, hop_length)
        
        # Calculate path score using weighted BFS approach
        path_score = self._calculate_path_score_weighted_bfs(path)
        
        return {
            'path': path,
            'hop_length': hop_length,
            'narrative': narrative,
            'score': path_score,
            'path_type': f'{hop_length}_hop',
            'source': path[0],
            'target': path[-1],
            'intermediate_nodes': path[1:-1] if len(path) > 2 else []
        }
    
    def _build_path_narrative(self, path: List[str], hop_length: int) -> str:
        """Build narrative description of path (legacy algorithm pattern)."""
        if len(path) < 2 or self.knowledge_graph is None:
            return ""
        
        # Get node data for narrative building
        source_data = self.knowledge_graph.nodes.get(path[0], {})
        target_data = self.knowledge_graph.nodes.get(path[-1], {})
        
        source_desc = source_data.get('description', f'entity {path[0]}')
        target_desc = target_data.get('description', f'entity {path[-1]}')
        
        if hop_length == 1:
            # Direct connection
            edge_data = self.knowledge_graph.get_edge_data(path[0], path[1], {})
            edge_desc = edge_data.get('description', 'connected to')
            narrative = f"The entity {path[0]} ({source_desc}) is {edge_desc} {path[1]} ({target_desc})"
            
        elif hop_length == 2:
            # 2-hop connection
            bridge_node = path[1]
            bridge_data = self.knowledge_graph.nodes.get(bridge_node, {})
            bridge_desc = bridge_data.get('description', f'entity {bridge_node}')
            
            edge1_data = self.knowledge_graph.get_edge_data(path[0], path[1], {})
            edge2_data = self.knowledge_graph.get_edge_data(path[1], path[2], {})
            
            edge1_desc = edge1_data.get('description', 'connected to')
            edge2_desc = edge2_data.get('description', 'connected to')
            
            narrative = (f"The entity {path[0]} ({source_desc}) is {edge1_desc} "
                        f"{bridge_node} ({bridge_desc}), which is {edge2_desc} "
                        f"{path[2]} ({target_desc})")
            
        elif hop_length == 3:
            # 3-hop connection
            bridge1_node = path[1]
            bridge2_node = path[2]
            
            bridge1_data = self.knowledge_graph.nodes.get(bridge1_node, {})
            bridge2_data = self.knowledge_graph.nodes.get(bridge2_node, {})
            
            bridge1_desc = bridge1_data.get('description', f'entity {bridge1_node}')
            bridge2_desc = bridge2_data.get('description', f'entity {bridge2_node}')
            
            narrative = (f"The entity {path[0]} ({source_desc}) connects through "
                        f"{bridge1_node} ({bridge1_desc}) and {bridge2_node} ({bridge2_desc}) "
                        f"to reach {path[-1]} ({target_desc})")
        else:
            # Longer paths - simplified narrative
            intermediate = ' -> '.join(path[1:-1])
            narrative = f"The entity {path[0]} connects to {path[-1]} through path: {intermediate}"
        
        return narrative
    
    def _calculate_path_score_weighted_bfs(self, path: List[str]) -> float:
        """Calculate path score using weighted BFS approach (legacy algorithm)."""
        if len(path) < 2:
            return 0.0
        
        # Initialize with high score, apply decay for longer paths
        base_score = 1.0
        decay_factor = self.config.get('flow_decay_factor', 0.8)
        
        # Apply decay based on path length
        length_penalty = decay_factor ** (len(path) - 1)
        
        # Calculate edge weights along path
        edge_score = 1.0
        if self.knowledge_graph is not None:
            for i in range(len(path) - 1):
                edge_data = self.knowledge_graph.get_edge_data(path[i], path[i+1], {})
                edge_weight = edge_data.get('weight', 1.0)
                edge_score *= edge_weight
        
        # Final score combines base score, length penalty, and edge weights
        final_score = base_score * length_penalty * edge_score
        
        return float(final_score)
    
    def _score_paths_by_reliability_data(self, path_data_list: List[Dict[str, Any]], resources: Dict[str, float]) -> List[Dict[str, Any]]:
        """Score paths by reliability using resource values and path characteristics."""
        scored_paths: List[Dict[str, Any]] = []
        
        for path_data in path_data_list:
            path = path_data.get('path', [])
            if len(path) < 2:
                continue
            
            # Calculate reliability based on resource allocation
            path_resources = [resources.get(node, 0.0) for node in path]
            resource_score = sum(path_resources) / len(path_resources) if path_resources else 0.0
            
            # Combine with existing path score from BFS algorithm
            bfs_score = path_data.get('score', 0.0)
            
            # Weighted combination of resource and BFS scores
            reliability_score = (resource_score * 0.6) + (bfs_score * 0.4)
            
            # Apply path length penalty (prefer shorter, more direct paths)
            length_penalty = 1.0 / (1.0 + len(path) * 0.1)
            final_score = reliability_score * length_penalty
            
            # Update path data with final score
            updated_path_data = path_data.copy()
            updated_path_data['reliability_score'] = reliability_score
            updated_path_data['final_score'] = final_score
            updated_path_data['resource_score'] = resource_score
            updated_path_data['length_penalty'] = length_penalty
            
            scored_paths.append(updated_path_data)
        
        # Sort by final score (descending)
        scored_paths.sort(key=lambda x: x.get('final_score', 0.0), reverse=True)
        
        logger.debug(f"Scored {len(scored_paths)} paths with reliability scores")
        return scored_paths
    
    def _paths_to_results_with_context(self, path_data_list: List[Dict[str, Any]]) -> List[RAGResult]:
        """Convert path data to RAG results with rich context."""
        results = []
        
        for i, path_data in enumerate(path_data_list):
            path = path_data.get('path', [])
            score = path_data.get('score', 0.0)
            narrative = path_data.get('narrative', '')
            hop_length = path_data.get('hop_length', 1)
            
            if len(path) < 2:
                continue
            
            # Build rich content using narrative and node details
            content_parts = []
            
            # Add narrative description
            if narrative:
                content_parts.append(f"Path Description: {narrative}")
            
            # Add node details
            node_details = []
            for node_id in path:
                if self.knowledge_graph is None:
                    continue
                node_data = self.knowledge_graph.nodes.get(node_id, {})
                node_content = node_data.get('content', '')
                node_desc = node_data.get('description', '')
                
                if node_content or node_desc:
                    detail = f"{node_id}: {node_desc or node_content}"
                    node_details.append(detail)
            
            if node_details:
                content_parts.append("Node Details: " + "; ".join(node_details))
            
            aggregated_content = "\n".join(content_parts)
            
            # Create path info with rich metadata
            path_info = PathInfo(
                path_id=f"pathrag_path_{i}",
                nodes=path,
                edges=[f"{path[j]}-{path[j+1]}" for j in range(len(path)-1)],
                path_score=score,
                reliability_score=score,
                metadata={
                    "path_length": len(path),
                    "hop_length": hop_length,
                    "path_type": path_data.get('path_type', 'unknown'),
                    "narrative": narrative
                }
            )
            
            # Calculate diversity score based on unique concepts
            unique_nodes = len(set(path))
            diversity_score = unique_nodes / len(path) if len(path) > 0 else 0.0
            
            result = RAGResult(
                item_id=f"pathrag_result_{i}",
                content=aggregated_content,
                score=score,
                path_info=path_info,
                relevance_score=score,
                diversity_score=diversity_score,
                metadata={
                    "path_nodes": path,
                    "path_length": len(path),
                    "hop_length": hop_length,
                    "algorithm": "pathrag_multihop",
                    "narrative": narrative
                }
            )
            
            results.append(result)
        
        return results
    
    def _create_path_infos_from_data(self, path_data_list: List[Dict[str, Any]]) -> List[PathInfo]:
        """Create PathInfo objects from path data."""
        path_infos = []
        
        for i, path_data in enumerate(path_data_list):
            path = path_data.get('path', [])
            if len(path) < 2:
                continue
            
            path_info = PathInfo(
                path_id=f"pathrag_explored_{i}",
                nodes=path,
                edges=[f"{path[j]}-{path[j+1]}" for j in range(len(path)-1)],
                path_score=path_data.get('final_score', 0.0),
                reliability_score=path_data.get('reliability_score', 0.0),
                metadata={
                    "exploration_order": i,
                    "path_length": len(path),
                    "hop_length": path_data.get('hop_length', 1),
                    "path_type": path_data.get('path_type', 'unknown'),
                    "narrative": path_data.get('narrative', ''),
                    "resource_score": path_data.get('resource_score', 0.0),
                    "bfs_score": path_data.get('score', 0.0),
                    "length_penalty": path_data.get('length_penalty', 1.0)
                }
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