"""
XnX Traversal Functions for HADES-PathRAG

This module provides the implementation of XnX-enhanced graph traversal functions,
including weight filtering, directional flow, and temporal filtering as described
in the XnX documentation.
"""

import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

class XnXTraversalError(Exception):
    """Base exception for XnX traversal errors."""
    pass


class InvalidNodeError(XnXTraversalError):
    """Raised when a node does not exist in the graph."""
    pass


class WeightThresholdError(XnXTraversalError):
    """Raised when no paths meet the weight threshold."""
    pass


class TemporalConstraintError(XnXTraversalError):
    """Raised when no paths are valid at the requested time."""
    pass


class DirectionalityError(XnXTraversalError):
    """Raised when the direction parameter is invalid."""
    pass


def traverse_with_xnx_constraints(
    db_connection,
    start_node: str,
    min_weight: float = 0.8,
    max_distance: int = 3,
    direction: str = "any",  # 'inbound', 'outbound', or 'any'
    graph_name: str = "pathrag_graph",
    nodes_collection: str = "pathrag_nodes",
    edges_collection: str = "pathrag_edges"
) -> List[Dict[str, Any]]:
    """
    Traverse the graph with XnX constraints on weight and direction.

    Args:
        db_connection: ArangoDB connection object
        start_node: ID of the starting node
        min_weight: Minimum edge weight (0.0-1.0)
        max_distance: Maximum path distance
        direction: Path direction ('inbound', 'outbound', or 'any')
        graph_name: Name of the graph to traverse
        nodes_collection: Name of the nodes collection
        edges_collection: Name of the edges collection

    Returns:
        List of paths matching the constraints

    Raises:
        InvalidNodeError: If the start_node does not exist
        WeightThresholdError: If no paths meet the weight threshold
        DirectionalityError: If the direction parameter is invalid
    """
    # Validate parameters
    if not 0.0 <= min_weight <= 1.0:
        raise ValueError("min_weight must be between 0.0 and 1.0")
    
    if max_distance < 1:
        raise ValueError("max_distance must be at least 1")
    
    if direction not in ["inbound", "outbound", "any"]:
        raise DirectionalityError(f"Invalid direction: {direction}. Must be 'inbound', 'outbound', or 'any'")

    # Check if the start node exists
    if not db_connection.collection_exists(nodes_collection):
        raise InvalidNodeError(f"Nodes collection {nodes_collection} does not exist")
    
    # Set up direction filter for the query
    direction_filter = ""
    if direction == "outbound":
        direction_filter = "FILTER e.direction == -1"
    elif direction == "inbound":
        direction_filter = "FILTER e.direction == 1"

    # Construct and execute the AQL query
    query = f"""
    FOR v, e, p IN 1..{max_distance}
        OUTBOUND @start_node
        GRAPH @graph_name
        FILTER e.weight >= @min_weight
        {direction_filter}
        RETURN {{
            node: v._key,
            path_weight: SUM(p.edges[*].weight) / LENGTH(p.edges),  // Average weight
            path_length: LENGTH(p.edges),
            edges: p.edges,
            nodes: p.vertices
        }}
    """

    bind_vars = {
        "start_node": start_node,
        "graph_name": graph_name,
        "min_weight": min_weight
    }

    try:
        results = db_connection.query(query, bind_vars=bind_vars)
        
        if not results:
            logger.warning(f"No paths found from {start_node} with weight >= {min_weight}")
            raise WeightThresholdError(f"No paths found from {start_node} with weight >= {min_weight}")
        
        return results
    except Exception as e:
        if "not found" in str(e).lower():
            raise InvalidNodeError(f"Start node {start_node} not found") from e
        raise


def traverse_with_temporal_xnx(
    db_connection,
    start_node: str,
    min_weight: float = 0.8,
    max_distance: int = 3,
    direction: str = "any",
    valid_at: Optional[Union[str, datetime]] = None,
    graph_name: str = "pathrag_graph",
    nodes_collection: str = "pathrag_nodes",
    edges_collection: str = "pathrag_edges"
) -> List[Dict[str, Any]]:
    """
    Traverse the graph with XnX constraints including temporal validity.

    Args:
        db_connection: ArangoDB connection object
        start_node: ID of the starting node
        min_weight: Minimum edge weight (0.0-1.0)
        max_distance: Maximum path distance
        direction: Path direction ('inbound', 'outbound', or 'any')
        valid_at: Time point for which edges should be valid (ISO format string or datetime)
        graph_name: Name of the graph to traverse
        nodes_collection: Name of the nodes collection
        edges_collection: Name of the edges collection

    Returns:
        List of paths matching the constraints

    Raises:
        InvalidNodeError: If the start_node does not exist
        WeightThresholdError: If no paths meet the weight threshold
        TemporalConstraintError: If no paths are valid at the requested time
        DirectionalityError: If the direction parameter is invalid
    """
    # Validate parameters
    if not 0.0 <= min_weight <= 1.0:
        raise ValueError("min_weight must be between 0.0 and 1.0")
    
    if max_distance < 1:
        raise ValueError("max_distance must be at least 1")
    
    if direction not in ["inbound", "outbound", "any"]:
        raise DirectionalityError(f"Invalid direction: {direction}. Must be 'inbound', 'outbound', or 'any'")

    # If valid_at is a datetime, convert to ISO string
    if isinstance(valid_at, datetime):
        valid_at = valid_at.isoformat()
    
    # Use current time if not specified
    if valid_at is None:
        valid_at = datetime.now().isoformat()

    # Check if the start node exists
    if not db_connection.collection_exists(nodes_collection):
        raise InvalidNodeError(f"Nodes collection {nodes_collection} does not exist")
    
    # Set up direction filter for the query
    direction_filter = ""
    if direction == "outbound":
        direction_filter = "FILTER e.direction == -1"
    elif direction == "inbound":
        direction_filter = "FILTER e.direction == 1"

    # Construct temporal filter
    temporal_filter = """
    FILTER (@valid_at >= e.valid_from OR e.valid_from == NULL) 
    AND (@valid_at <= e.valid_to OR e.valid_to == NULL)
    """

    # Construct and execute the AQL query
    query = f"""
    FOR v, e, p IN 1..{max_distance}
        OUTBOUND @start_node
        GRAPH @graph_name
        FILTER e.weight >= @min_weight
        {direction_filter}
        {temporal_filter}
        RETURN {{
            node: v._key,
            path_weight: SUM(p.edges[*].weight) / LENGTH(p.edges),  // Average weight
            path_length: LENGTH(p.edges),
            edges: p.edges,
            nodes: p.vertices,
            temporal_valid: true
        }}
    """

    bind_vars = {
        "start_node": start_node,
        "graph_name": graph_name,
        "min_weight": min_weight,
        "valid_at": valid_at
    }

    try:
        results = db_connection.query(query, bind_vars=bind_vars)
        
        if not results:
            logger.warning(f"No temporally valid paths found from {start_node} at time {valid_at}")
            raise TemporalConstraintError(f"No temporally valid paths found from {start_node} at time {valid_at}")
        
        return results
    except TemporalConstraintError:
        raise
    except Exception as e:
        if "not found" in str(e).lower():
            raise InvalidNodeError(f"Start node {start_node} not found") from e
        raise


def format_xnx_output(edge: Dict[str, Any]) -> str:
    """
    Format an edge dictionary into an XnX notation string.

    Args:
        edge: Edge dictionary from the graph query

    Returns:
        Formatted XnX string in the format: "weight node_id direction"
    """
    # Extract target node from the _to field (format: "collection/node_id")
    node_parts = edge.get('_to', '').split('/')
    target_node = node_parts[-1] if len(node_parts) > 1 else edge.get('_to', '')
    
    # Get weight and direction
    weight = edge.get('weight', 0.0)
    direction = edge.get('direction', 0)
    
    # Temporal window formatting (optional)
    temporal_window = ""
    valid_from = edge.get('valid_from')
    valid_to = edge.get('valid_to')
    
    if valid_from and valid_to:
        # Format timestamps for readability
        if isinstance(valid_from, str) and isinstance(valid_to, str):
            # Extract date portion if full ISO timestamps
            if 'T' in valid_from:
                valid_from = valid_from.split('T')[0]
            if 'T' in valid_to:
                valid_to = valid_to.split('T')[0]
            
            temporal_window = f" [{valid_from} → {valid_to}]"
    
    # Format the XnX string
    return f"{weight:.2f} {target_node} {direction}{temporal_window}"


def calculate_path_score(path: Dict[str, Any], use_log_scale: bool = False) -> float:
    """
    Calculate the total score for a path based on edge weights.
    
    Args:
        path: Path dictionary from the traversal query
        use_log_scale: Whether to use log-scale for numerical stability
        
    Returns:
        Path score (product of weights, or sum of log weights if log_scale=True)
    """
    import math
    
    edges = path.get('edges', [])
    
    if not edges:
        return 0.0
    
    if use_log_scale:
        # Log-scale for numerical stability: sum of log(weights)
        try:
            return sum(math.log(edge.get('weight', 0.001)) for edge in edges)
        except ValueError:
            # Handle case where weight is 0 or negative
            return float('-inf')
    else:
        # Raw scale: product of weights
        product = 1.0
        for edge in edges:
            product *= edge.get('weight', 0.0)
        return product
