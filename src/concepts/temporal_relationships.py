"""
Temporal Relationships for Knowledge Graphs

This module implements time-bound relationships that can expire or
be valid only within certain time windows, enabling dynamic knowledge graphs.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class TemporalBoundType(Enum):
    """Types of temporal bounds for relationships."""
    PERMANENT = "permanent"          # No expiration
    EXPIRING = "expiring"           # Has expiration date
    WINDOWED = "windowed"           # Valid within time window
    PERIODIC = "periodic"           # Recurring validity
    DECAYING = "decaying"           # Strength decreases over time


@dataclass
class TemporalBound:
    """Represents temporal constraints on a relationship."""
    bound_type: TemporalBoundType
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    valid_from: Optional[datetime] = None
    valid_to: Optional[datetime] = None
    decay_rate: float = 0.0  # For decaying relationships
    period: Optional[timedelta] = None  # For periodic relationships
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_valid(self, at_time: Optional[datetime] = None) -> bool:
        """Check if the temporal bound is valid at a given time."""
        current_time = at_time or datetime.now(timezone.utc)
        
        if self.bound_type == TemporalBoundType.PERMANENT:
            return True
        
        elif self.bound_type == TemporalBoundType.EXPIRING:
            return current_time < self.expires_at if self.expires_at else True
        
        elif self.bound_type == TemporalBoundType.WINDOWED:
            if self.valid_from and current_time < self.valid_from:
                return False
            if self.valid_to and current_time > self.valid_to:
                return False
            return True
        
        elif self.bound_type == TemporalBoundType.PERIODIC:
            if not self.period or not self.valid_from:
                return False
            # Calculate if we're in a valid period
            elapsed = current_time - self.valid_from
            period_number = elapsed // self.period
            period_start = self.valid_from + (period_number * self.period)
            period_end = period_start + self.period
            return period_start <= current_time < period_end
        
        else:  # TemporalBoundType.DECAYING
            # Still valid but with reduced strength
            return True
    
    def get_strength(self, 
                    base_strength: float = 1.0, 
                    at_time: Optional[datetime] = None) -> float:
        """Get the effective strength of a relationship at a given time."""
        if not self.is_valid(at_time):
            return 0.0
        
        current_time = at_time or datetime.now(timezone.utc)
        
        if self.bound_type == TemporalBoundType.DECAYING:
            # Exponential decay
            age = (current_time - self.created_at).total_seconds()
            decay_factor = 2.0 ** (-self.decay_rate * age / 86400)  # Daily decay
            return float(base_strength * decay_factor)
        else:
            return base_strength


class TemporalRelationship:
    """
    Manages temporal relationships in the knowledge graph.
    
    Handles creation, validation, and decay of time-bound connections.
    """
    
    def __init__(self) -> None:
        """Initialize temporal relationship manager."""
        self.temporal_bounds: Dict[str, TemporalBound] = {}
        self.relationship_history: List[Dict[str, Any]] = []
        
        logger.info("Initialized Temporal Relationship manager")
    
    def create_temporal_edge(self,
                           from_node: str,
                           to_node: str,
                           edge_type: str,
                           weight: float,
                           temporal_bound: TemporalBound,
                           metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create an edge with temporal constraints.
        
        Args:
            from_node: Source node ID
            to_node: Target node ID
            edge_type: Type of relationship
            weight: Base weight of the edge
            temporal_bound: Temporal constraints
            metadata: Additional metadata
            
        Returns:
            Edge dictionary with temporal information
        """
        edge_id = f"{from_node}-{to_node}-{datetime.now().timestamp()}"
        
        edge = {
            '_id': edge_id,
            '_from': from_node,
            '_to': to_node,
            'type': edge_type,
            'base_weight': weight,
            'temporal_bound': temporal_bound,
            'metadata': metadata or {},
            'created_at': datetime.now(timezone.utc).isoformat()
        }
        
        # Store temporal bound
        self.temporal_bounds[edge_id] = temporal_bound
        
        # Add to history
        self.relationship_history.append({
            'edge_id': edge_id,
            'action': 'created',
            'timestamp': datetime.now(timezone.utc),
            'bound_type': temporal_bound.bound_type.value
        })
        
        logger.info(f"Created temporal edge {edge_id} with {temporal_bound.bound_type.value} bound")
        
        return edge
    
    def validate_edges(self, 
                      edges: List[Dict[str, Any]], 
                      at_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Filter edges based on temporal validity.
        
        Args:
            edges: List of edges to validate
            at_time: Time to check validity (default: now)
            
        Returns:
            List of valid edges with updated weights
        """
        valid_edges = []
        current_time = at_time or datetime.now(timezone.utc)
        
        for edge in edges:
            edge_id = edge.get('_id')
            if not edge_id:
                # Skip edges without ID
                continue
            
            # Get temporal bound
            temporal_bound = self.temporal_bounds.get(edge_id)
            if not temporal_bound:
                # No temporal bound means permanent
                valid_edges.append(edge)
                continue
            
            # Check validity
            if temporal_bound.is_valid(current_time):
                # Update weight based on temporal effects
                base_weight = edge.get('base_weight', edge.get('weight', 1.0))
                effective_weight = temporal_bound.get_strength(base_weight, current_time)
                
                # Create updated edge
                updated_edge = edge.copy()
                updated_edge['weight'] = effective_weight
                updated_edge['temporal_status'] = {
                    'is_valid': True,
                    'bound_type': temporal_bound.bound_type.value,
                    'effective_weight': effective_weight,
                    'checked_at': current_time.isoformat()
                }
                
                valid_edges.append(updated_edge)
            else:
                # Log expired edge
                logger.debug(f"Edge {edge_id} expired at {current_time}")
                self.relationship_history.append({
                    'edge_id': edge_id,
                    'action': 'expired',
                    'timestamp': current_time,
                    'bound_type': temporal_bound.bound_type.value
                })
        
        return valid_edges
    
    def create_query_temporal_bound(self, ttl_seconds: int = 3600) -> TemporalBound:
        """
        Create a temporal bound for query nodes.
        
        Query nodes should expire after a certain time to prevent
        accumulation in the graph.
        """
        return TemporalBound(
            bound_type=TemporalBoundType.EXPIRING,
            expires_at=datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds),
            metadata={'purpose': 'query_node_ttl'}
        )
    
    def create_knowledge_decay_bound(self, 
                                   half_life_days: float = 365) -> TemporalBound:
        """
        Create a decay bound for knowledge that becomes less relevant over time.
        
        Args:
            half_life_days: Days until strength is halved
            
        Returns:
            Temporal bound with decay
        """
        # Calculate decay rate: strength = e^(-decay_rate * time)
        # At half_life: 0.5 = e^(-decay_rate * half_life)
        # decay_rate = ln(2) / half_life
        decay_rate = 0.693 / half_life_days
        
        return TemporalBound(
            bound_type=TemporalBoundType.DECAYING,
            decay_rate=decay_rate,
            metadata={
                'half_life_days': half_life_days,
                'purpose': 'knowledge_decay'
            }
        )
    
    def create_version_window_bound(self,
                                  version: str,
                                  valid_from: datetime,
                                  valid_to: Optional[datetime] = None) -> TemporalBound:
        """
        Create a window bound for version-specific relationships.
        
        Useful for documentation or API versions.
        """
        return TemporalBound(
            bound_type=TemporalBoundType.WINDOWED,
            valid_from=valid_from,
            valid_to=valid_to,
            metadata={
                'version': version,
                'purpose': 'version_specific'
            }
        )
    
    def create_periodic_bound(self,
                            period: timedelta,
                            start_time: Optional[datetime] = None) -> TemporalBound:
        """
        Create a periodic bound for recurring relationships.
        
        Useful for scheduled content or periodic reviews.
        """
        return TemporalBound(
            bound_type=TemporalBoundType.PERIODIC,
            valid_from=start_time or datetime.now(timezone.utc),
            period=period,
            metadata={'purpose': 'periodic_relationship'}
        )
    
    def analyze_temporal_patterns(self) -> Dict[str, Any]:
        """
        Analyze temporal patterns in relationship history.
        
        Returns insights about relationship dynamics.
        """
        analysis: Dict[str, Any] = {
            'total_relationships': len(self.temporal_bounds),
            'by_type': {},
            'expiration_timeline': [],
            'decay_analysis': {},
            'patterns': []
        }
        
        # Count by type
        type_counts: Dict[str, int] = {}
        for bound in self.temporal_bounds.values():
            bound_type = bound.bound_type.value
            type_counts[bound_type] = type_counts.get(bound_type, 0) + 1
        analysis['by_type'] = type_counts
        
        # Analyze expirations
        current_time = datetime.now(timezone.utc)
        future_expirations: List[Dict[str, Any]] = []
        
        for edge_id, bound in self.temporal_bounds.items():
            if bound.bound_type == TemporalBoundType.EXPIRING and bound.expires_at:
                if bound.expires_at > current_time:
                    days_until = (bound.expires_at - current_time).days
                    future_expirations.append({
                        'edge_id': edge_id,
                        'expires_in_days': days_until,
                        'expires_at': bound.expires_at.isoformat()
                    })
        
        # Sort by expiration
        future_expirations.sort(key=lambda x: int(x['expires_in_days']))
        analysis['expiration_timeline'] = future_expirations[:10]  # Next 10
        
        # Analyze decay
        decaying_edges = [
            (eid, b) for eid, b in self.temporal_bounds.items() 
            if b.bound_type == TemporalBoundType.DECAYING
        ]
        
        if decaying_edges:
            half_lives = [b.metadata.get('half_life_days', 365) for _, b in decaying_edges]
            analysis['decay_analysis'] = {
                'count': len(decaying_edges),
                'avg_half_life_days': sum(half_lives) / len(half_lives),
                'min_half_life_days': min(half_lives),
                'max_half_life_days': max(half_lives)
            }
        
        # Identify patterns
        if len(self.relationship_history) > 100:
            # Simple pattern: creation/expiration rate
            creations = sum(1 for h in self.relationship_history[-100:] if h['action'] == 'created')
            expirations = sum(1 for h in self.relationship_history[-100:] if h['action'] == 'expired')
            
            analysis['patterns'].append({
                'pattern': 'creation_expiration_rate',
                'last_100_events': {
                    'creations': creations,
                    'expirations': expirations,
                    'ratio': creations / max(expirations, 1)
                }
            })
        
        return analysis
    
    def cleanup_expired(self, cutoff_time: Optional[datetime] = None) -> int:
        """
        Remove expired temporal bounds from memory.
        
        Returns number of bounds removed.
        """
        current_time = cutoff_time or datetime.now(timezone.utc)
        expired_ids = []
        
        for edge_id, bound in self.temporal_bounds.items():
            if not bound.is_valid(current_time):
                expired_ids.append(edge_id)
        
        # Remove expired bounds
        for edge_id in expired_ids:
            del self.temporal_bounds[edge_id]
        
        if expired_ids:
            logger.info(f"Cleaned up {len(expired_ids)} expired temporal bounds")
        
        return len(expired_ids)