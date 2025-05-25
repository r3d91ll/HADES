"""
Base Loader for ISNE.

This module provides the base class and configuration for all ISNE data loaders.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any, Tuple, Set
from pathlib import Path

from ..types.models import (
    DocumentType,
    RelationType, 
    IngestDocument,
    DocumentRelation,
    LoaderResult
)


@dataclass
class LoaderConfig:
    """
    Configuration for ISNE data loaders.
    
    This class contains common configuration options for all loaders,
    which can be extended by specific loader implementations.
    """
    data_dir: Optional[Union[str, Path]] = None
    input_file: Optional[Union[str, Path]] = None
    embedding_dim: int = 768
    include_metadata: bool = True
    batch_size: int = 32
    max_documents: Optional[int] = None
    document_types: List[str] = field(default_factory=list)
    
    def __post_init__(self) -> None:
        """Convert string paths to Path objects."""
        if self.data_dir and isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)
        
        if self.input_file and isinstance(self.input_file, str):
            self.input_file = Path(self.input_file)


class BaseLoader:
    """
    Base class for all ISNE data loaders.
    
    This abstract class defines the interface that all loaders must implement,
    as well as providing common utility functions.
    """
    
    def __init__(self, config: Optional[LoaderConfig] = None):
        """
        Initialize the loader with configuration.
        
        Args:
            config: Configuration for the loader. If None, default config is used.
        """
        self.config = config or LoaderConfig()
    
    def load(self) -> LoaderResult:
        """
        Load documents and relations from the data source.
        
        Returns:
            LoaderResult containing documents and relations.
        
        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement the 'load' method")
    
    def load_document(self, document_id: str) -> Optional[IngestDocument]:
        """
        Load a single document by ID.
        
        Args:
            document_id: ID of the document to load.
            
        Returns:
            Loaded document, or None if not found.
        
        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement the 'load_document' method")
    
    def save_results(self, result: LoaderResult, output_path: Union[str, Path]) -> None:
        """
        Save loader results to a file.
        
        Args:
            result: LoaderResult to save.
            output_path: Path to save the results to.
        """
        import json
        from pathlib import Path
        
        # Convert path to Path object if needed
        if isinstance(output_path, str):
            output_path = Path(output_path)
        
        # Create parent directories if they don't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for serialization
        data = {
            "documents": [
                {
                    "id": doc.id,
                    "content": doc.content,
                    "source": doc.source,
                    "document_type": doc.document_type,
                    "metadata": doc.metadata,
                    "embedding": doc.embedding.tolist() if doc.embedding is not None else None,
                    "enhanced_embedding": doc.enhanced_embedding.tolist() if doc.enhanced_embedding is not None else None,
                    "chunks": doc.chunks
                } for doc in result.documents
            ],
            "relations": [
                {
                    "source_id": rel.source_id,
                    "target_id": rel.target_id,
                    "relation_type": rel.relation_type,
                    "weight": rel.weight,
                    "metadata": rel.metadata
                } for rel in result.relations
            ],
            "metadata": result.metadata
        }
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    @staticmethod
    def create_document_relation(
        source_id: str,
        target_id: str,
        relation_type: Union[RelationType, str],
        weight: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> DocumentRelation:
        """
        Create a document relation with appropriate type handling.
        
        Args:
            source_id: ID of the source document.
            target_id: ID of the target document.
            relation_type: Type of relation.
            weight: Weight of the relation (0-1).
            metadata: Additional metadata for the relation.
            
        Returns:
            DocumentRelation object.
        """
        # Convert string relation type to enum if needed
        if isinstance(relation_type, str):
            try:
                relation_type = RelationType[relation_type.upper()]
            except KeyError:
                relation_type = RelationType.GENERIC
        
        # Create relation with default weight if not specified
        return DocumentRelation(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            weight=weight if weight is not None else 1.0,
            metadata=metadata or {}
        )
