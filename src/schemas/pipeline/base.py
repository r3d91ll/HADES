"""
Base pipeline schemas for HADES-PathRAG.

This module defines the foundational pipeline schemas used across
different pipeline implementations in the system.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import Field, field_validator

from ..common.base import BaseSchema
from ..common.types import MetadataDict


class PipelineStage(str, Enum):
    """Stages in a document processing pipeline."""
    LOAD = "load"
    PREPROCESS = "preprocess"
    CHUNK = "chunk"
    EMBED = "embed"
    INDEX = "index"
    STORE = "store"
    COMPLETE = "complete"
    FAILED = "failed"


class PipelineStatus(str, Enum):
    """Status of a pipeline execution."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


class PipelineConfigSchema(BaseSchema):
    """Base configuration for pipeline execution."""
    
    name: str = Field(..., description="Name of the pipeline")
    description: Optional[str] = Field(default=None, description="Description of the pipeline")
    version: str = Field(default="1.0.0", description="Pipeline version")
    enabled_stages: List[PipelineStage] = Field(default_factory=lambda: list(PipelineStage), 
                                               description="Enabled pipeline stages")
    parallel: bool = Field(default=False, description="Whether to run stages in parallel")
    max_workers: int = Field(default=4, description="Maximum number of parallel workers")
    timeout_seconds: Optional[int] = Field(default=3600, description="Pipeline timeout in seconds")
    retry_count: int = Field(default=3, description="Number of retries for failed operations")
    metadata: MetadataDict = Field(default_factory=dict, description="Additional pipeline metadata")
    
    @field_validator("max_workers")
    @classmethod
    def validate_max_workers(cls, v: int) -> int:
        """Validate max_workers is positive."""
        if v <= 0:
            raise ValueError(f"max_workers must be positive, got {v}")
        return v
    
    @field_validator("retry_count")
    @classmethod
    def validate_retry_count(cls, v: int) -> int:
        """Validate retry_count is non-negative."""
        if v < 0:
            raise ValueError(f"retry_count must be non-negative, got {v}")
        return v


class PipelineStatsSchema(BaseSchema):
    """Statistics for pipeline execution."""
    
    start_time: Optional[datetime] = Field(default=None, description="Pipeline start time")
    end_time: Optional[datetime] = Field(default=None, description="Pipeline end time")
    duration_seconds: Optional[float] = Field(default=None, description="Pipeline duration in seconds")
    documents_processed: int = Field(default=0, description="Number of documents processed")
    documents_failed: int = Field(default=0, description="Number of documents that failed processing")
    documents_skipped: int = Field(default=0, description="Number of documents skipped")
    current_stage: Optional[PipelineStage] = Field(default=None, description="Current pipeline stage")
    stage_stats: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Statistics by stage")
    errors: List[Dict[str, Any]] = Field(default_factory=list, description="List of errors encountered")
    
    def record_start(self) -> None:
        """Record pipeline start time."""
        self.start_time = datetime.now()
    
    def record_end(self) -> None:
        """Record pipeline end time and calculate duration."""
        self.end_time = datetime.now()
        if self.start_time:
            self.duration_seconds = (self.end_time - self.start_time).total_seconds()
    
    def update_stage(self, stage: Union[PipelineStage, str]) -> None:
        """Update current pipeline stage.
        
        Args:
            stage: New pipeline stage (can be enum or string)
        """
        self.current_stage = stage
        # Get the stage value, handling both enum and string inputs
        stage_value = stage.value if hasattr(stage, 'value') else stage
        if stage_value not in self.stage_stats:
            self.stage_stats[stage_value] = {
                "start_time": datetime.now().isoformat(),
                "documents_processed": 0,
                "errors": 0
            }
    
    def record_document_processed(self, stage: Optional[Union[PipelineStage, str]] = None) -> None:
        """Record a document as processed.
        
        Args:
            stage: Optional stage to record for (defaults to current_stage)
        """
        self.documents_processed += 1
        
        # Handle different types of stage inputs (enum or string)
        if stage is not None:
            stage_key = stage.value if hasattr(stage, 'value') else stage
        elif self.current_stage is not None:
            stage_key = self.current_stage.value if hasattr(self.current_stage, 'value') else self.current_stage
        else:
            stage_key = None
            
        if stage_key:
            # Initialize the stage if it doesn't exist
            if stage_key not in self.stage_stats:
                self.stage_stats[stage_key] = {
                    "start_time": datetime.now().isoformat(),
                    "documents_processed": 0,
                    "errors": 0
                }
            # Increment the document count
            self.stage_stats[stage_key]["documents_processed"] = self.stage_stats[stage_key].get("documents_processed", 0) + 1
    
    def record_error(self, error: str, document_id: Optional[str] = None, stage: Optional[Union[PipelineStage, str]] = None) -> None:
        """Record an error during pipeline execution.
        
        Args:
            error: Error message
            document_id: Optional ID of document that caused the error
            stage: Optional stage where the error occurred (defaults to current_stage)
        """
        error_entry = {
            "error": error,
            "timestamp": datetime.now().isoformat()
        }
        if document_id:
            error_entry["document_id"] = document_id
        
        # Handle different types of stage inputs (enum or string)
        if stage is not None:
            stage_key = stage.value if hasattr(stage, 'value') else stage
        elif self.current_stage is not None:
            stage_key = self.current_stage.value if hasattr(self.current_stage, 'value') else self.current_stage
        else:
            stage_key = None
            
        if stage_key:
            error_entry["stage"] = stage_key
            # Initialize the stage if it doesn't exist
            if stage_key not in self.stage_stats:
                self.stage_stats[stage_key] = {
                    "start_time": datetime.now().isoformat(),
                    "documents_processed": 0,
                    "errors": 0
                }
            # Increment the error count
            self.stage_stats[stage_key]["errors"] = self.stage_stats[stage_key].get("errors", 0) + 1
                
        self.errors.append(error_entry)
        self.documents_failed += 1
