"""Query type definitions for storage systems.

This module defines types for database queries, including filters, sorting,
pagination, and query results for different query operations.
"""

from typing import Dict, List, Any, Optional, Union, TypedDict, Literal, Tuple, Generic, TypeVar
from enum import Enum
from datetime import datetime

from src.types.common import NodeData, EdgeData, NodeID, EdgeID, EmbeddingVector


# Define a type variable for query results
T = TypeVar('T')


class SortDirection(str, Enum):
    """Enum for sort directions."""
    
    ASC = "asc"
    """Ascending sort order."""
    
    DESC = "desc"
    """Descending sort order."""


class FilterOperator(str, Enum):
    """Enum for filter operators."""
    
    EQUALS = "eq"
    """Equality operator."""
    
    NOT_EQUALS = "ne"
    """Inequality operator."""
    
    GREATER_THAN = "gt"
    """Greater than operator."""
    
    GREATER_THAN_OR_EQUAL = "gte"
    """Greater than or equal operator."""
    
    LESS_THAN = "lt"
    """Less than operator."""
    
    LESS_THAN_OR_EQUAL = "lte"
    """Less than or equal operator."""
    
    IN = "in"
    """In list operator."""
    
    NOT_IN = "nin"
    """Not in list operator."""
    
    CONTAINS = "contains"
    """Contains substring operator."""
    
    STARTS_WITH = "starts_with"
    """Starts with substring operator."""
    
    ENDS_WITH = "ends_with"
    """Ends with substring operator."""
    
    REGEX = "regex"
    """Regular expression operator."""
    
    EXISTS = "exists"
    """Field exists operator."""


class FilterCondition(TypedDict, total=False):
    """Filter condition for queries."""
    
    field: str
    """Field to filter on."""
    
    operator: FilterOperator
    """Filter operator."""
    
    value: Any
    """Value to compare against."""


class CompoundFilter(TypedDict, total=False):
    """Compound filter with logical operators."""
    
    and_filters: List[Union['CompoundFilter', FilterCondition]]
    """List of filters to AND together."""
    
    or_filters: List[Union['CompoundFilter', FilterCondition]]
    """List of filters to OR together."""
    
    not_filter: Union['CompoundFilter', FilterCondition]
    """Filter to negate."""


# Filter type for queries - can be simple or compound
QueryFilter = Union[FilterCondition, CompoundFilter]


class SortOption(TypedDict):
    """Sort option for queries."""
    
    field: str
    """Field to sort on."""
    
    direction: SortDirection
    """Sort direction."""


class PaginationOptions(TypedDict, total=False):
    """Pagination options for queries."""
    
    limit: int
    """Maximum number of results to return."""
    
    offset: int
    """Number of results to skip."""
    
    page: int
    """Page number (alternative to offset)."""
    
    page_size: int
    """Page size (alternative to limit)."""
    
    cursor: str
    """Cursor for cursor-based pagination."""


class QueryOptions(TypedDict, total=False):
    """Options for database queries."""
    
    filters: Optional[QueryFilter]
    """Filter conditions."""
    
    sort: Optional[List[SortOption]]
    """Sort options."""
    
    pagination: Optional[PaginationOptions]
    """Pagination options."""
    
    include_fields: Optional[List[str]]
    """Fields to include in results."""
    
    exclude_fields: Optional[List[str]]
    """Fields to exclude from results."""
    
    timeout: Optional[float]
    """Query timeout in seconds."""
    
    cache: bool
    """Whether to use cached results if available."""


class VectorQueryOptions(TypedDict, total=False):
    """Options for vector queries."""
    
    embedding: EmbeddingVector
    """Query embedding vector."""
    
    top_k: int
    """Number of nearest neighbors to return."""
    
    min_score: float
    """Minimum similarity score (0-1)."""
    
    include_metadata: bool
    """Whether to include node metadata in results."""
    
    include_embeddings: bool
    """Whether to include embeddings in results."""
    
    filter: Optional[QueryFilter]
    """Additional filters to apply."""


class HybridQueryOptions(TypedDict, total=False):
    """Options for hybrid (text + vector) queries."""
    
    text_query: str
    """Text search query."""
    
    embedding: Optional[EmbeddingVector]
    """Query embedding vector."""
    
    top_k: int
    """Number of results to return."""
    
    text_weight: float
    """Weight for text search relevance (0-1)."""
    
    vector_weight: float
    """Weight for vector similarity (0-1)."""
    
    filter: Optional[QueryFilter]
    """Additional filters to apply."""


class TraversalQueryOptions(TypedDict, total=False):
    """Options for graph traversal queries."""
    
    start_node: NodeID
    """ID of the starting node."""
    
    edge_types: Optional[List[str]]
    """Types of edges to traverse."""
    
    direction: Literal["outbound", "inbound", "any"]
    """Direction of traversal."""
    
    min_depth: int
    """Minimum traversal depth."""
    
    max_depth: int
    """Maximum traversal depth."""
    
    node_filter: Optional[QueryFilter]
    """Filter for nodes during traversal."""
    
    edge_filter: Optional[QueryFilter]
    """Filter for edges during traversal."""
    
    max_results: int
    """Maximum number of results to return."""


class ShortestPathOptions(TypedDict, total=False):
    """Options for shortest path queries."""
    
    start_node: NodeID
    """ID of the starting node."""
    
    end_node: NodeID
    """ID of the ending node."""
    
    edge_types: Optional[List[str]]
    """Types of edges to traverse."""
    
    direction: Literal["outbound", "inbound", "any"]
    """Direction of traversal."""
    
    max_depth: int
    """Maximum path length."""
    
    weight_attribute: Optional[str]
    """Edge attribute to use as weight for weighted paths."""


class QueryResult(Generic[T]):
    """Generic query result with pagination info."""
    
    def __init__(
        self,
        results: List[T],
        total_count: int,
        page: int = 1,
        page_size: int = 0,
        next_cursor: Optional[str] = None,
        execution_time: float = 0.0
    ):
        """Initialize a QueryResult instance."""
        self.results = results
        self.total_count = total_count
        self.page = page
        self.page_size = page_size
        self.next_cursor = next_cursor
        self.execution_time = execution_time
        self.has_more = total_count > (page * page_size) if page_size > 0 else False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "results": self.results,
            "total_count": self.total_count,
            "page": self.page,
            "page_size": self.page_size,
            "next_cursor": self.next_cursor,
            "execution_time": self.execution_time,
            "has_more": self.has_more
        }


class VectorSearchResult(TypedDict):
    """Result of a vector similarity search."""
    
    node_id: NodeID
    """ID of the node."""
    
    node_data: NodeData
    """Node data."""
    
    score: float
    """Similarity score (0-1)."""
    
    embedding: Optional[EmbeddingVector]
    """Node embedding (if requested)."""


class HybridSearchResult(TypedDict):
    """Result of a hybrid search."""
    
    node_id: NodeID
    """ID of the node."""
    
    node_data: NodeData
    """Node data."""
    
    text_score: float
    """Text relevance score (0-1)."""
    
    vector_score: float
    """Vector similarity score (0-1)."""
    
    combined_score: float
    """Combined relevance score (0-1)."""


class TraversalResult(TypedDict):
    """Result of a graph traversal."""
    
    nodes: List[NodeData]
    """Nodes in the traversal."""
    
    edges: List[EdgeData]
    """Edges in the traversal."""
    
    paths: List[List[Union[NodeID, EdgeID]]]
    """Paths found in the traversal."""
    
    stats: Dict[str, Any]
    """Traversal statistics."""


class PathResult(TypedDict):
    """Result of a path query."""
    
    path: List[Union[NodeData, EdgeData]]
    """Nodes and edges in the path, alternating."""
    
    length: int
    """Length of the path (number of edges)."""
    
    total_weight: float
    """Total weight of the path (for weighted paths)."""
