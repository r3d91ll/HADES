"""
PathRAG engine implementation for the multi-engine RAG platform.

This module integrates the PathRAG processor with the engine framework
to enable A/B testing and comparison with other RAG strategies.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

from .base import (
    BaseRAGEngine, EngineType, RetrievalMode,
    EngineRetrievalRequest, EngineRetrievalResult
)
from src.pathrag.pathrag_rag_strategy import PathRAGProcessor
from src.types.pathrag.strategy import (
    PathRAGConfig, PathRAGResult
)
from src.types.pathrag.rag_types import (
    RAGMode,
    RAGResult,
    RAGStrategyInput,
    RAGStrategyOutput,
    PathInfo
)
# TODO: Import component registry - need to implement new component system

logger = logging.getLogger(__name__)


class PathRAGEngine(BaseRAGEngine):
    """
    PathRAG engine implementation for multi-engine platform.
    
    This engine wraps the PathRAG processor to provide:
    - Flow-based path pruning
    - Multi-hop reasoning
    - Hierarchical keyword extraction
    - Resource allocation algorithms
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize PathRAG engine."""
        super().__init__(config or {})
        
        # Default PathRAG configuration
        self._default_config = {
            "flow_decay_factor": 0.8,
            "pruning_threshold": 0.3,
            "max_path_length": 5,
            "max_iterations": 10,
            "convergence_threshold": 0.001,
            
            # Multi-hop settings
            "multihop": {
                "enable_1hop": True,
                "enable_2hop": True,
                "enable_3hop": True,
                "max_paths_per_pair": 5
            },
            
            # Context settings
            "context": {
                "weights": {
                    "entity_context": 0.4,
                    "relationship_context": 0.4,
                    "source_text_context": 0.2
                },
                "combination_method": "weighted_merge"
            },
            
            # Component settings
            "storage": {"type": "arangodb"},
            "embedder": {"type": "modernbert"},
            "graph_enhancer": {"type": "isne"},
            
            # Performance settings
            "enable_caching": True,
            "batch_size": 32,
            "timeout_seconds": 30
        }
        
        # Merge with provided config
        if config:
            self._merge_config(self._default_config, config)
        else:
            self._config = self._default_config.copy()
        
        # Initialize PathRAG processor
        self._processor: Optional[PathRAGProcessor] = None
        self._initialized = False
        
        logger.info("PathRAG engine initialized")
    
    @property
    def engine_type(self) -> EngineType:
        """Get engine type."""
        return EngineType.PATHRAG
    
    @property
    def supported_modes(self) -> List[RetrievalMode]:
        """Get supported retrieval modes for PathRAG."""
        return [
            RetrievalMode.VECTOR,
            RetrievalMode.GRAPH,
            RetrievalMode.HYBRID,
            RetrievalMode.PATHRAG,
            RetrievalMode.SEMANTIC
        ]
    
    async def _ensure_initialized(self):
        """Ensure PathRAG processor is initialized."""
        if not self._initialized:
            try:
                # Create PathRAG processor
                self._processor = PathRAGProcessor()
                
                # Configure processor
                self._processor.configure(self._config)
                
                self._initialized = True
                logger.info("PathRAG processor initialized successfully")
                
            except Exception as e:
                logger.error(f"Failed to initialize PathRAG processor: {e}")
                raise RuntimeError(f"PathRAG engine initialization failed: {e}")
    
    async def _execute_retrieval(self, request: EngineRetrievalRequest) -> List[EngineRetrievalResult]:
        """Execute PathRAG retrieval."""
        await self._ensure_initialized()
        
        if not self._processor:
            raise RuntimeError("PathRAG processor not initialized")
        
        try:
            # Map retrieval mode to RAG mode
            rag_mode = self._map_retrieval_mode(request.mode)
            
            # Create PathRAG input
            pathrag_input = RAGStrategyInput(
                query=request.query,
                mode=rag_mode,
                top_k=request.top_k,
                filters=request.filters or {},
                parameters={
                    "max_token_for_context": request.max_token_for_context,
                    "search_options": request.engine_params or {}
                }
            )
            
            # Execute retrieval
            pathrag_output = self._processor.retrieve_and_generate(pathrag_input)
            
            # Convert results to engine format
            results = []
            for result in pathrag_output.results:
                engine_result = EngineRetrievalResult(
                    content=result.content,
                    score=result.score,
                    source=result.metadata.get("source", "unknown"),
                    metadata={
                        "chunk_metadata": result.metadata.get("chunk_metadata", {}),
                        "document_metadata": result.metadata.get("document_metadata", {}),
                        "path_info": result.paths[0].model_dump() if result.paths else None,
                        "retrieval_mode": request.mode.value,
                        "pathrag_metadata": result.metadata
                    },
                    engine_metadata={
                        "pathrag_mode": rag_mode.value,
                        "processing_time": pathrag_output.metadata.get("processing_time", 0),
                        "component_version": pathrag_output.metadata.get("component_version", "unknown"),
                        "retrieval_stats": pathrag_output.metadata.get("retrieval_stats", {}),
                        "context_info": pathrag_output.metadata.get("context_info", {})
                    }
                )
                results.append(engine_result)
            
            logger.info(f"PathRAG retrieval completed: {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"PathRAG retrieval failed: {e}")
            # Return error result
            return [EngineRetrievalResult(
                content=f"PathRAG retrieval failed: {str(e)}",
                score=0.0,
                source="error",
                metadata={"error": str(e), "mode": request.mode.value},
                engine_metadata={"error_type": type(e).__name__}
            )]
    
    def _map_retrieval_mode(self, mode: RetrievalMode) -> RAGMode:
        """Map engine retrieval mode to PathRAG mode."""
        mapping = {
            RetrievalMode.VECTOR: RAGMode.NAIVE,
            RetrievalMode.GRAPH: RAGMode.GLOBAL,
            RetrievalMode.HYBRID: RAGMode.HYBRID,
            RetrievalMode.PATHRAG: RAGMode.PATHRAG,
            RetrievalMode.SEMANTIC: RAGMode.LOCAL
        }
        return mapping.get(mode, RAGMode.PATHRAG)
    
    def _merge_config(self, base: Dict[str, Any], updates: Dict[str, Any]) -> None:
        """Recursively merge configuration dictionaries."""
        for key, value in updates.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value
    
    async def configure(self, config: Dict[str, Any]) -> bool:
        """Update PathRAG engine configuration."""
        try:
            # Merge new config
            self._merge_config(self._config, config)
            
            # Reconfigure processor if initialized
            if self._processor:
                self._processor.configure(self._config)
            
            self._last_config_update = datetime.now(timezone.utc)
            logger.info("PathRAG engine configuration updated")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update PathRAG configuration: {e}")
            return False
    
    async def get_config_schema(self) -> Dict[str, Any]:
        """Get PathRAG configuration schema."""
        return {
            "type": "object",
            "properties": {
                "flow_decay_factor": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.8,
                    "description": "Resource flow decay factor"
                },
                "pruning_threshold": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.3,
                    "description": "Threshold for pruning low-resource nodes"
                },
                "max_path_length": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "default": 5,
                    "description": "Maximum path length for exploration"
                },
                "max_iterations": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 10,
                    "description": "Maximum resource allocation iterations"
                },
                "multihop": {
                    "type": "object",
                    "properties": {
                        "enable_1hop": {"type": "boolean", "default": True},
                        "enable_2hop": {"type": "boolean", "default": True},
                        "enable_3hop": {"type": "boolean", "default": True},
                        "max_paths_per_pair": {"type": "integer", "minimum": 1, "default": 5}
                    }
                },
                "context": {
                    "type": "object",
                    "properties": {
                        "weights": {
                            "type": "object",
                            "properties": {
                                "entity_context": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                "relationship_context": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                "source_text_context": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                            }
                        }
                    }
                },
                "storage": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string", "enum": ["arangodb", "networkx", "memory"]}
                    }
                },
                "embedder": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string", "enum": ["modernbert", "cpu", "vllm"]}
                    }
                }
            },
            "required": ["flow_decay_factor", "pruning_threshold", "max_path_length"]
        }
    
    async def health_check(self) -> bool:
        """Check if PathRAG engine is healthy."""
        try:
            await self._ensure_initialized()
            
            if not self._processor:
                return False
            
            # Check processor health
            return self._processor.health_check()
            
        except Exception as e:
            logger.error(f"PathRAG health check failed: {e}")
            return False
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get PathRAG-specific metrics."""
        base_metrics = await super().get_metrics()
        
        pathrag_metrics = {
            "initialized": self._initialized,
            "processor_available": self._processor is not None
        }
        
        # Add processor metrics if available
        if self._processor:
            try:
                processor_metrics = self._processor.get_metrics()
                pathrag_metrics.update({
                    "knowledge_graph_nodes": processor_metrics.get("knowledge_graph_nodes", 0),
                    "knowledge_graph_edges": processor_metrics.get("knowledge_graph_edges", 0),
                    "last_resource_allocation_time": processor_metrics.get("last_resource_allocation_time", None),
                    "total_paths_explored": processor_metrics.get("total_paths_explored", 0),
                    "avg_path_length": processor_metrics.get("avg_path_length", 0.0)
                })
            except Exception as e:
                pathrag_metrics["metrics_error"] = str(e)  # type: ignore[assignment]
        
        return {**base_metrics, **pathrag_metrics}
    
    async def analyze_paths(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Analyze path construction for a query (PathRAG-specific feature).
        
        This method provides insight into how PathRAG constructs and scores paths.
        """
        await self._ensure_initialized()
        
        if not self._processor:
            raise RuntimeError("PathRAG processor not initialized")
        
        try:
            # Create analysis request
            analysis_input = RAGStrategyInput(
                query=query,
                mode=RAGMode.PATHRAG,
                top_k=top_k,
                max_token_for_context=4000,
                search_options={"analyze_paths": True}
            )
            
            # Execute with path analysis
            result = self._processor.retrieve_and_generate(analysis_input)
            
            # Extract path analysis data
            path_analysis = {
                "query": query,
                "total_paths_explored": len(result.results),
                "path_details": [],
                "resource_allocation": result.metadata.get("retrieval_stats", {}).get("resource_allocation", {}),
                "keyword_extraction": result.metadata.get("retrieval_stats", {}).get("keywords", {}),
                "execution_stats": {
                    "total_time_ms": result.execution_time_ms,
                    "resource_allocation_time": result.metadata.get("retrieval_stats", {}).get("resource_allocation_time", 0),
                    "path_construction_time": result.metadata.get("retrieval_stats", {}).get("path_construction_time", 0)
                }
            }
            
            # Add detailed path information
            for result_item in result.results:
                if result_item.paths:
                    for path_info in result_item.paths:
                        path_details = {
                            "path_nodes": path_info.nodes,
                            "path_score": result_item.score,
                            "hop_count": len(path_info.nodes) - 1,
                            "narrative": path_info.metadata.get("narrative", ""),
                            "resource_flow": path_info.metadata.get("resource_flow", {})
                        }
                    path_analysis["path_details"].append(path_details)
            
            return path_analysis
            
        except Exception as e:
            logger.error(f"Path analysis failed: {e}")
            return {
                "query": query,
                "error": str(e),
                "total_paths_explored": 0,
                "path_details": []
            }
    
    async def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics (PathRAG-specific feature)."""
        await self._ensure_initialized()
        
        if not self._processor:
            return {"error": "PathRAG processor not initialized"}
        
        try:
            return self._processor.get_knowledge_base_stats()
        except Exception as e:
            logger.error(f"Failed to get knowledge base stats: {e}")
            return {"error": str(e)}