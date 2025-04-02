"""
Retrieval pipeline interfaces for the HADES modular pipeline architecture.

This module defines the interfaces for the retrieval pipeline components,
including query processors, vector search, path finding, and reranking.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from hades.pipeline.interfaces import Component, Configurable, Pipeline, Pluggable, Serializable
from hades.ingest.interface import EmbeddedChunk


class Query:
    """Class representing a query in the retrieval pipeline."""
    
    def __init__(self, query_id: str, text: str, metadata: Optional[Dict[str, Any]] = None):
        """Initialize a query.
        
        Args:
            query_id: Unique identifier for the query
            text: Query text
            metadata: Optional metadata for the query
        """
        self.query_id = query_id
        self.text = text
        self.metadata = metadata or {}
    
    def __repr__(self) -> str:
        return f"Query(id={self.query_id}, text='{self.text[:30]}...')"


class EmbeddedQuery:
    """Class representing an embedded query in the retrieval pipeline."""
    
    def __init__(self, query_id: str, text: str, embedding: np.ndarray, 
                 metadata: Optional[Dict[str, Any]] = None):
        """Initialize an embedded query.
        
        Args:
            query_id: Unique identifier for the query
            text: Query text
            embedding: Vector embedding of the query
            metadata: Optional metadata for the query
        """
        self.query_id = query_id
        self.text = text
        self.embedding = embedding
        self.metadata = metadata or {}
    
    def __repr__(self) -> str:
        return f"EmbeddedQuery(id={self.query_id}, text='{self.text[:30]}...')"


class RetrievalResult:
    """Class representing a retrieval result in the retrieval pipeline."""
    
    def __init__(self, chunk: EmbeddedChunk, score: float, 
                 metadata: Optional[Dict[str, Any]] = None):
        """Initialize a retrieval result.
        
        Args:
            chunk: Retrieved chunk
            score: Relevance score
            metadata: Optional metadata for the result
        """
        self.chunk = chunk
        self.score = score
        self.metadata = metadata or {}
    
    def __repr__(self) -> str:
        return f"RetrievalResult(chunk_id={self.chunk.chunk_id}, score={self.score:.4f})"


class PathNode:
    """Class representing a node in a retrieval path."""
    
    def __init__(self, chunk: EmbeddedChunk, score: float, 
                 metadata: Optional[Dict[str, Any]] = None):
        """Initialize a path node.
        
        Args:
            chunk: Chunk at this node
            score: Relevance score
            metadata: Optional metadata for the node
        """
        self.chunk = chunk
        self.score = score
        self.metadata = metadata or {}
    
    def __repr__(self) -> str:
        return f"PathNode(chunk_id={self.chunk.chunk_id}, score={self.score:.4f})"


class RetrievalPath:
    """Class representing a path of retrieved chunks."""
    
    def __init__(self, path_id: str, nodes: List[PathNode], 
                 total_score: float, metadata: Optional[Dict[str, Any]] = None):
        """Initialize a retrieval path.
        
        Args:
            path_id: Unique identifier for the path
            nodes: List of nodes in the path
            total_score: Total relevance score for the path
            metadata: Optional metadata for the path
        """
        self.path_id = path_id
        self.nodes = nodes
        self.total_score = total_score
        self.metadata = metadata or {}
    
    def __repr__(self) -> str:
        return f"RetrievalPath(id={self.path_id}, nodes={len(self.nodes)}, score={self.total_score:.4f})"


class QueryProcessor(Component, Configurable, Pluggable, ABC):
    """Interface for query processors in the retrieval pipeline."""
    
    @property
    def component_type(self) -> str:
        """Return the type of this component."""
        return "query_processor"
    
    @classmethod
    def get_plugin_type(cls) -> str:
        """Return the plugin type for this component."""
        return "query_processor"
    
    @abstractmethod
    def process_query(self, query_text: str, 
                     metadata: Optional[Dict[str, Any]] = None) -> Query:
        """Process a query text and convert it to a Query object.
        
        Args:
            query_text: Query text to process
            metadata: Optional metadata for the query
            
        Returns:
            Processed query
        """
        pass
    
    @abstractmethod
    def embed_query(self, query: Query) -> EmbeddedQuery:
        """Generate embedding for a query.
        
        Args:
            query: Query to embed
            
        Returns:
            Embedded query
        """
        pass


class VectorSearcher(Component, Configurable, Pluggable, ABC):
    """Interface for vector searchers in the retrieval pipeline."""
    
    @property
    def component_type(self) -> str:
        """Return the type of this component."""
        return "vector_searcher"
    
    @classmethod
    def get_plugin_type(cls) -> str:
        """Return the plugin type for this component."""
        return "vector_searcher"
    
    @abstractmethod
    def search(self, query: EmbeddedQuery, top_k: int = 10, 
              filter_criteria: Optional[Dict[str, Any]] = None) -> List[RetrievalResult]:
        """Search for chunks similar to the query.
        
        Args:
            query: Embedded query to search for
            top_k: Number of results to return
            filter_criteria: Optional criteria to filter results
            
        Returns:
            List of retrieval results
        """
        pass


class PathFinder(Component, Configurable, Pluggable, ABC):
    """Interface for path finders in the retrieval pipeline."""
    
    @property
    def component_type(self) -> str:
        """Return the type of this component."""
        return "path_finder"
    
    @classmethod
    def get_plugin_type(cls) -> str:
        """Return the plugin type for this component."""
        return "path_finder"
    
    @abstractmethod
    def find_paths(self, query: EmbeddedQuery, initial_results: List[RetrievalResult], 
                  max_paths: int = 3, max_path_length: int = 5) -> List[RetrievalPath]:
        """Find paths of related chunks starting from initial results.
        
        Args:
            query: Embedded query
            initial_results: Initial retrieval results
            max_paths: Maximum number of paths to return
            max_path_length: Maximum length of each path
            
        Returns:
            List of retrieval paths
        """
        pass


class Reranker(Component, Configurable, Pluggable, ABC):
    """Interface for rerankers in the retrieval pipeline."""
    
    @property
    def component_type(self) -> str:
        """Return the type of this component."""
        return "reranker"
    
    @classmethod
    def get_plugin_type(cls) -> str:
        """Return the plugin type for this component."""
        return "reranker"
    
    @abstractmethod
    def rerank_results(self, query: Query, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Rerank retrieval results.
        
        Args:
            query: Original query
            results: Retrieval results to rerank
            
        Returns:
            Reranked retrieval results
        """
        pass
    
    @abstractmethod
    def rerank_paths(self, query: Query, paths: List[RetrievalPath]) -> List[RetrievalPath]:
        """Rerank retrieval paths.
        
        Args:
            query: Original query
            paths: Retrieval paths to rerank
            
        Returns:
            Reranked retrieval paths
        """
        pass


class RetrievalPipeline(Pipeline, ABC):
    """Interface for the retrieval pipeline."""
    
    @property
    def component_type(self) -> str:
        """Return the type of this component."""
        return "retrieval_pipeline"
    
    @abstractmethod
    def retrieve(self, query_text: str, top_k: int = 10, 
                use_paths: bool = True, 
                filter_criteria: Optional[Dict[str, Any]] = None) -> Union[List[RetrievalResult], List[RetrievalPath]]:
        """Retrieve relevant chunks or paths for a query.
        
        Args:
            query_text: Query text
            top_k: Number of results to return
            use_paths: Whether to use path finding
            filter_criteria: Optional criteria to filter results
            
        Returns:
            List of retrieval results or paths
        """
        pass
    
    @abstractmethod
    def retrieve_and_format(self, query_text: str, top_k: int = 10, 
                          use_paths: bool = True, 
                          filter_criteria: Optional[Dict[str, Any]] = None) -> str:
        """Retrieve and format results as a string.
        
        Args:
            query_text: Query text
            top_k: Number of results to return
            use_paths: Whether to use path finding
            filter_criteria: Optional criteria to filter results
            
        Returns:
            Formatted retrieval results as a string
        """
        pass