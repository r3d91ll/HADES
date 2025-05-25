"""
ISNE relationship schemas for HADES-PathRAG.

This module provides Pydantic models for ISNE document relationships,
used for graph construction and embedding enhancement.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import Field, field_validator

from ..common.base import BaseSchema
from ..common.types import MetadataDict
from .models import ISNERelationType


class ISNEDocumentRelationSchema(BaseSchema):
    """Relationship between document entities in the ISNE system."""
    
    source_id: str = Field(..., description="ID of the source entity")
    target_id: str = Field(..., description="ID of the target entity")
    relation_type: ISNERelationType = Field(..., description="Type of relationship")
    weight: float = Field(default=1.0, description="Weight of the relationship")
    metadata: MetadataDict = Field(default_factory=dict, description="Additional relationship metadata")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique ID for this relation")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    bidirectional: bool = Field(default=False, description="Whether the relationship is bidirectional")
    
    @field_validator("relation_type")
    @classmethod
    def validate_relation_type(cls, v: Any) -> ISNERelationType:
        """Validate relation type."""
        if isinstance(v, str) and not isinstance(v, ISNERelationType):
            try:
                return ISNERelationType(v)
            except ValueError:
                # Default to custom relationship type if not found
                return ISNERelationType.CUSTOM
        return v
    
    @field_validator("weight")
    @classmethod
    def validate_weight(cls, v: float) -> float:
        """Validate weight is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError(f"Weight must be between 0 and 1, got {v}")
        return v
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.
        
        Returns:
            Dictionary representation of the relation
        """
        return self.model_dump_safe()


class ISNEGraphSchema(BaseSchema):
    """Graph representation for ISNE processing."""
    
    nodes: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Graph nodes by ID")
    edges: List[ISNEDocumentRelationSchema] = Field(default_factory=list, description="Graph edges")
    metadata: MetadataDict = Field(default_factory=dict, description="Graph metadata")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique graph ID")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    name: Optional[str] = Field(default=None, description="Graph name")
    description: Optional[str] = Field(default=None, description="Graph description")
    
    def add_node(self, node_id: str, attributes: Dict[str, Any]) -> None:
        """Add a node to the graph.
        
        Args:
            node_id: Unique identifier for the node
            attributes: Node attributes
        """
        self.nodes[node_id] = attributes
    
    def add_edge(self, relation: ISNEDocumentRelationSchema) -> None:
        """Add an edge to the graph.
        
        Args:
            relation: Relationship to add
        """
        self.edges.append(relation)
        
        # Add bidirectional edge if specified
        if relation.bidirectional:
            reverse_relation = ISNEDocumentRelationSchema(
                source_id=relation.target_id,
                target_id=relation.source_id,
                relation_type=relation.relation_type,
                weight=relation.weight,
                metadata=relation.metadata.copy()
            )
            self.edges.append(reverse_relation)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.
        
        Returns:
            Dictionary representation of the graph
        """
        return self.model_dump_safe()
