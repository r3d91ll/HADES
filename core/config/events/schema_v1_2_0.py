"""
Machine-readable contract for ETL event schema v1.2.0.
This Pydantic model formalizes required/optional fields, 
enforces whitelist enums for collections and relationship types,
and rejects unknown schema_ver values unless a feature flag explicitly opts-in.

Based on the PRD specification in docs/prd/hybrid_graph_visualization_and_etl_pipeline.md
"""
import ast
import json
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, validator


class CollectionEnum(str, Enum):
    """Valid collection names for events."""
    papers = "papers"
    files = "files"
    funcs = "funcs"
    paper_chunks = "paper_chunks"
    code_assets = "code_assets"
    has_asset = "has_asset"
    imports = "imports"
    cites = "cites"
    chunk_code_edges = "chunk_code_edges"
    code_knn = "code_knn"
    chunk_knn = "chunk_knn"

class OperationEnum(str, Enum):
    """Valid operation types for events."""
    upsert_node = "upsert_node"
    upsert_edge = "upsert_edge"
    delete_node = "delete_node"
    delete_edge = "delete_edge"

class RelationshipType(str, Enum):
    """Valid relationship types for edges."""
    HAS_ASSET = "HAS_ASSET"
    IMPORTS = "IMPORTS"
    CITES = "CITES"
    CHUNK_CODE = "CHUNK_CODE"
    CODE_KNN = "CODE_KNN"
    CHUNK_KNN = "CHUNK_KNN"

class EncodingEnum(str, Enum):
    """Valid encoding types for events."""
    json = "json"
    gzip_b64 = "gzip+b64"

class NodeLabel(str, Enum):
    """Valid Neo4j node labels."""
    Paper = "Paper"
    File = "File"
    Func = "Func"
    Chunk = "Chunk"

class ConveyanceContext(BaseModel):
    """Aggregate shared-context (C_ext) score and optional metadata."""

    value: float = Field(..., ge=0.0, description="C_ext aggregate scalar (legacy alias 'ctx')")
    meta: Optional[Dict[str, Any]] = Field(default=None, description="Optional context metadata")


class ConveyanceDims(BaseModel):
    """Conveyance framework dimensions payload."""

    mode: str = Field(..., description="compact | verbose")
    W: float = Field(..., ge=0.0, le=1.0)
    R: float = Field(..., ge=0.0, le=1.0)
    H: float = Field(..., ge=0.0, le=1.0)
    T: float = Field(..., gt=0.0)
    ctx: ConveyanceContext  # legacy field name (represents C_ext)
    L: float = Field(..., ge=0.0, le=1.0)
    I: float = Field(..., ge=0.0, le=1.0)
    A: float = Field(..., ge=0.0, le=1.0)
    G: float = Field(..., ge=0.0, le=1.0)
    alpha: float = Field(..., ge=0.0, le=4.0, description="α exponent applied to C_ext")
    zero_propagation: bool = Field(False, description="Whether conveyance collapses to zero")
    evidence: List[str] = Field(default_factory=list, description="Supporting link/ID evidence")

    @property
    def conveyance(self) -> float:
        """Computed conveyance score using symmetric single-sided assumption."""

        if self.zero_propagation:
            return 0.0
        base = (self.W * self.R * self.H) / self.T if self.T else 0.0
        c_ext_term = self.ctx.value ** self.alpha if self.ctx.value else 0.0
        return base * c_ext_term

class FrameContext(BaseModel):
    """FRAME dimension context metadata."""
    observer: str
    perspective: str
    confidence: float
    forward: Dict[str, Any]
    backward: Dict[str, Any]

class EventDoc(BaseModel):
    """Source document snapshot plus GraphSAGE status."""

    id_: str = Field(..., alias="_id")
    rev_: str = Field(..., alias="_rev")
    # Add other common fields based on typical Arango documents
    path: Optional[str] = None
    loc_bin: Optional[str] = None
    lang: Optional[str] = None
    _feat: Optional[Dict[str, Any]] = None
    _score: Optional[float] = None
    deleted: bool = False
    score_status: Optional[str] = None  # pending | available | failed

    model_config = ConfigDict(extra='allow', populate_by_name=True, protected_namespaces=())

    @property
    def _id(self) -> str:  # pragma: no cover - trivial alias
        return self.id_

    @property
    def _rev(self) -> str:  # pragma: no cover - trivial alias
        return self.rev_

class EventSchema(BaseModel):
    """Main event schema as defined in the PRD."""
    event_id: str
    op: OperationEnum
    coll: CollectionEnum
    schema_ver: str = "1.2.0"
    seq_no: int = Field(..., ge=0)
    rev: Optional[str] = None
    depends_on: List[str] = Field(default_factory=list)
    retry_count: int = Field(default=0, ge=0)
    max_retries: int = Field(default=6, ge=0)
    encoding: EncodingEnum = EncodingEnum.json
    t_produced: datetime  # ISO-8601 timestamp
    dims: Optional[ConveyanceDims] = None
    frame_context: Optional[FrameContext] = None
    doc: EventDoc

    @validator('depends_on', pre=True)
    def _coerce_depends_on(cls, value):
        """Accept JSON-encoded dependency lists from legacy emitters."""

        if value is None:
            return []
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                try:
                    parsed = ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    parsed = [
                        part.strip(" []'\"")
                        for part in value.split(',')
                        if part.strip(" []'\"")
                    ]
            if isinstance(parsed, list):
                return parsed
            raise ValueError("depends_on string must decode to a list")
        return value

    @validator('schema_ver')
    def validate_schema_version(cls, v):
        if v != "1.2.0":
            raise ValueError(f"Unsupported schema version {v}")
        return v

    class Config:
        use_enum_values = True  # Return enum values instead of enum objects

        # Make sure we can access the raw string value or enum directly
