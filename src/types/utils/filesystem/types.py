"""Filesystem utility type definitions."""

from typing import List, Dict, Any, Optional, Set
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel, Field

from ...common import BaseSchema


class FileMetadata(BaseSchema):
    """Metadata for a file."""
    path: Path = Field(..., description="File path")
    size: int = Field(..., description="File size in bytes")
    created_at: datetime = Field(..., description="File creation time")
    modified_at: datetime = Field(..., description="File modification time")
    extension: str = Field(..., description="File extension")
    mime_type: Optional[str] = Field(None, description="MIME type")
    is_binary: bool = Field(False, description="Whether file is binary")
    encoding: Optional[str] = Field(None, description="Text encoding if applicable")


class DirectoryMetadata(BaseSchema):
    """Metadata for a directory."""
    path: Path = Field(..., description="Directory path")
    file_count: int = Field(0, description="Number of files")
    total_size: int = Field(0, description="Total size in bytes")
    subdirectory_count: int = Field(0, description="Number of subdirectories")
    file_types: Dict[str, int] = Field(default_factory=dict, description="File type counts")
    created_at: datetime = Field(..., description="Directory creation time")
    modified_at: datetime = Field(..., description="Directory modification time")


class IgnorePattern(BaseSchema):
    """Pattern for ignoring files/directories."""
    pattern: str = Field(..., description="Glob or regex pattern")
    pattern_type: str = Field("glob", description="Pattern type: glob or regex")
    source: str = Field(..., description="Source of pattern (e.g., .gitignore)")
    is_negation: bool = Field(False, description="Whether pattern negates previous patterns")


class FileSystemScanResult(BaseSchema):
    """Result of scanning a filesystem location."""
    root_path: Path = Field(..., description="Root path that was scanned")
    files: List[FileMetadata] = Field(default_factory=list, description="Files found")
    directories: List[DirectoryMetadata] = Field(default_factory=list, description="Directories found")
    ignored_paths: Set[Path] = Field(default_factory=set, description="Paths that were ignored")
    scan_time: datetime = Field(default_factory=datetime.utcnow, description="When scan was performed")
    total_files: int = Field(0, description="Total files scanned")
    total_size: int = Field(0, description="Total size in bytes")
    errors: List[Dict[str, Any]] = Field(default_factory=list, description="Errors encountered")