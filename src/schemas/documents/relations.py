"""
Document relationship schemas for HADES-PathRAG.

This module defines the schemas for representing and validating relationships
between documents in the system.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import Field, field_validator

from ..common.base import BaseSchema
from ..common.enums import RelationType
from ..common.types import MetadataDict


class DocumentRelationSchema(BaseSchema):
    """Schema for document relations."""
    
    source_id: str = Field(..., description="ID of the source document")
    target_id: str = Field(..., description="ID of the target document")
    relation_type: RelationType = Field(..., description="Type of relationship")
    weight: float = Field(default=1.0, description="Weight or strength of the relationship")
    bidirectional: bool = Field(default=False, description="Whether the relationship is bidirectional")
    metadata: MetadataDict = Field(default_factory=dict, description="Additional relation metadata")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique ID for this relation")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "source_id": "doc123",
                    "target_id": "doc456",
                    "relation_type": "contains",
                    "weight": 0.8,
                    "bidirectional": False
                }
            ]
        }
    }
    
    @field_validator("relation_type")
    @classmethod
    def validate_relation_type(cls, v: Any) -> RelationType:
        """Validate relation type."""
        if isinstance(v, str):
            try:
                return RelationType(v)
            except ValueError:
                # Custom relation type if not in enum
                if v not in [r.value for r in RelationType]:
                    return RelationType.CUSTOM
        return v
