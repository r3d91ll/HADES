"""
Type definitions for document validation and enhanced document structures.

This module contains types for documents with validation capabilities
and validation metadata.
"""

from typing import Any, Dict, List, Protocol, runtime_checkable

from .results import ValidationSummary


@runtime_checkable
class ValidatedDocumentList(Protocol):
    """Protocol for document lists with attached validation summaries."""
    
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> Dict[str, Any]: ...
    def __iter__(self) -> Any: ...
    
    @property
    def validation_summary(self) -> ValidationSummary: ...


class DocumentListWithValidation(list):
    """
    Enhanced document list that can carry validation summary metadata.
    
    This class extends list to allow attaching validation summaries
    to document collections without affecting serialization.
    """
    
    def __init__(self, documents: List[Dict[str, Any]]):
        super().__init__(documents)
        self._validation_summary: ValidationSummary | None = None
    
    @property
    def validation_summary(self) -> ValidationSummary | None:
        """Get the attached validation summary."""
        return self._validation_summary
    
    @validation_summary.setter
    def validation_summary(self, summary: ValidationSummary) -> None:
        """Set the validation summary."""
        self._validation_summary = summary