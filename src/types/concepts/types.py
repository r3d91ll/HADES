"""Concept type definitions."""

from typing import Dict, List, Any, Optional, Tuple
from pydantic import BaseModel, Field
from datetime import datetime

from ..common import BaseSchema, NodeID, EdgeID


class SupraWeight(BaseSchema):
    """Multi-dimensional edge weight."""
    semantic_similarity: float = Field(..., description="Semantic similarity score", ge=0.0, le=1.0)
    structural_proximity: float = Field(..., description="Structural proximity score", ge=0.0, le=1.0)
    temporal_relevance: float = Field(..., description="Temporal relevance score", ge=0.0, le=1.0)
    causal_strength: float = Field(..., description="Causal relationship strength", ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional weight metadata")
    
    @property
    def composite_weight(self) -> float:
        """Calculate composite weight from all dimensions."""
        return (
            self.semantic_similarity * 0.4 +
            self.structural_proximity * 0.3 +
            self.temporal_relevance * 0.2 +
            self.causal_strength * 0.1
        )


class QueryNode(BaseSchema):
    """Query represented as a graph node."""
    query_id: str = Field(..., description="Unique query identifier")
    query_text: str = Field(..., description="Query text")
    embedding: Optional[List[float]] = Field(None, description="Query embedding")
    decomposed_concepts: List[str] = Field(default_factory=list, description="Decomposed query concepts")
    temporal_context: Optional[datetime] = Field(None, description="Temporal context of query")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Query metadata")


class LCCClassification(BaseSchema):
    """Library of Congress Classification result."""
    main_class: str = Field(..., description="Main LCC class")
    subclass: Optional[str] = Field(None, description="LCC subclass")
    confidence: float = Field(..., description="Classification confidence", ge=0.0, le=1.0)
    related_classes: List[Tuple[str, float]] = Field(default_factory=list, description="Related classes with scores")


class TemporalRelationship(BaseSchema):
    """Temporal relationship between entities."""
    source_id: NodeID = Field(..., description="Source node ID")
    target_id: NodeID = Field(..., description="Target node ID")
    relationship_type: str = Field(..., description="Type of temporal relationship")
    temporal_distance: Optional[float] = Field(None, description="Temporal distance/lag")
    strength: float = Field(1.0, description="Relationship strength", ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Relationship metadata")


class TheoryPracticeBridge(BaseSchema):
    """Bridge between theoretical concepts and practical implementations."""
    theory_node: NodeID = Field(..., description="Theoretical concept node")
    practice_node: NodeID = Field(..., description="Practical implementation node")
    bridge_type: str = Field(..., description="Type of bridge (implements, exemplifies, etc.)")
    alignment_score: float = Field(..., description="Theory-practice alignment", ge=0.0, le=1.0)
    evidence: List[str] = Field(default_factory=list, description="Supporting evidence")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Bridge metadata")