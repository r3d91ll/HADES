"""
Conflict Resolver for Incremental Storage

Handles detection and resolution of document conflicts during incremental ingestion,
ensuring data consistency and maintaining an audit trail of all resolutions.
"""

import logging
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json
import difflib

from arango.database import StandardDatabase

from .types import (
    IncrementalConfig,
    DocumentInfo,
    DocumentState, 
    ConflictStrategy,
    ConflictResolution
)

logger = logging.getLogger(__name__)


class ConflictResolver:
    """
    Resolves conflicts during incremental document ingestion.
    
    Handles various conflict types:
    1. Content changes in existing documents
    2. Duplicate documents with different content
    3. Version mismatches
    4. Metadata conflicts
    """
    
    def __init__(self, db: StandardDatabase, config: IncrementalConfig):
        """
        Initialize conflict resolver.
        
        Args:
            db: ArangoDB database connection
            config: Incremental storage configuration
        """
        self.db = db
        self.config = config
        self.similarity_threshold = config.similarity_threshold
        
    async def resolve_conflict(
        self,
        document_info: DocumentInfo,
        strategy: ConflictStrategy
    ) -> ConflictResolution:
        """
        Resolve a document conflict using the specified strategy.
        
        Args:
            document_info: Information about the conflicting document
            strategy: Resolution strategy to use
            
        Returns:
            Conflict resolution details
        """
        logger.info(f"Resolving conflict for {document_info.file_path} using strategy: {strategy}")
        
        # Get existing document
        existing_doc = await self._get_existing_document(document_info.file_path)
        
        if not existing_doc:
            logger.warning(f"No existing document found for {document_info.file_path}")
            # Treat as new document
            resolution = ConflictResolution(
                document_id=document_info.file_path,
                file_path=document_info.file_path,
                strategy_used=ConflictStrategy.UPDATE,
                original_hash="",
                new_hash=document_info.content_hash,
                resolution_time=datetime.now(),
                changes_summary="New document (no existing version found)"
            )
            return resolution
        
        # Detect conflict type
        conflict_type = self._detect_conflict_type(document_info, existing_doc)
        
        # Apply resolution strategy
        if strategy == ConflictStrategy.SKIP:
            resolution = await self._skip_document(document_info, existing_doc, conflict_type)
        elif strategy == ConflictStrategy.UPDATE:
            resolution = await self._update_document(document_info, existing_doc, conflict_type)
        elif strategy == ConflictStrategy.MERGE:
            resolution = await self._merge_document(document_info, existing_doc, conflict_type)
        elif strategy == ConflictStrategy.KEEP_BOTH:
            resolution = await self._keep_both_versions(document_info, existing_doc, conflict_type)
        else:
            raise ValueError(f"Unknown conflict strategy: {strategy}")
        
        # Log conflict resolution
        await self._log_conflict_resolution(resolution, conflict_type)
        
        return resolution
    
    async def _get_existing_document(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Get existing document from database.
        
        Args:
            file_path: Path to the document
            
        Returns:
            Existing document or None if not found
        """
        collection = self.db.collection('documents')
        
        query = """
        FOR doc IN documents
        FILTER doc.file_path == @file_path
        LIMIT 1
        RETURN doc
        """
        
        results = list(self.db.query(query, bind_vars={'file_path': file_path}))
        
        return results[0] if results else None
    
    def _detect_conflict_type(
        self, 
        new_doc: DocumentInfo, 
        existing_doc: Dict[str, Any]
    ) -> str:
        """
        Detect the type of conflict between documents.
        
        Args:
            new_doc: New document information
            existing_doc: Existing document from database
            
        Returns:
            Conflict type string
        """
        if new_doc.content_hash != existing_doc['content_hash']:
            return "content_change"
        elif new_doc.size != existing_doc['size']:
            return "size_mismatch"
        elif new_doc.modified_time != existing_doc['modified_time']:
            return "timestamp_mismatch"
        else:
            return "metadata_conflict"
    
    async def _skip_document(
        self,
        new_doc: DocumentInfo,
        existing_doc: Dict[str, Any],
        conflict_type: str
    ) -> ConflictResolution:
        """
        Skip the conflicting document (keep existing version).
        
        Args:
            new_doc: New document information
            existing_doc: Existing document
            conflict_type: Type of conflict detected
            
        Returns:
            Conflict resolution details
        """
        logger.info(f"Skipping document: {new_doc.file_path}")
        
        return ConflictResolution(
            document_id=existing_doc['_key'],
            file_path=new_doc.file_path,
            strategy_used=ConflictStrategy.SKIP,
            original_hash=existing_doc['content_hash'],
            new_hash=new_doc.content_hash,
            resolution_time=datetime.now(),
            changes_summary=f"Document skipped due to {conflict_type}",
            metadata={
                'conflict_type': conflict_type,
                'existing_size': existing_doc['size'],
                'new_size': new_doc.size,
                'auto_resolved': True
            }
        )
    
    async def _update_document(
        self,
        new_doc: DocumentInfo,
        existing_doc: Dict[str, Any],
        conflict_type: str
    ) -> ConflictResolution:
        """
        Update existing document with new version.
        
        Args:
            new_doc: New document information
            existing_doc: Existing document
            conflict_type: Type of conflict detected
            
        Returns:
            Conflict resolution details
        """
        logger.info(f"Updating document: {new_doc.file_path}")
        
        # Analyze changes
        changes_summary = await self._analyze_document_changes(new_doc, existing_doc)
        
        # Update document in database
        collection = self.db.collection('documents')
        
        update_data = {
            'content_hash': new_doc.content_hash,
            'size': new_doc.size,
            'modified_time': new_doc.modified_time.isoformat(),
            'ingestion_time': datetime.now().isoformat(),
            'processing_status': 'pending',
            'previous_hash': existing_doc['content_hash'],
            'update_reason': conflict_type
        }
        
        collection.update(existing_doc['_key'], update_data)
        
        # Mark related chunks for reprocessing
        await self._mark_chunks_for_reprocessing(existing_doc['_key'])
        
        return ConflictResolution(
            document_id=existing_doc['_key'],
            file_path=new_doc.file_path,
            strategy_used=ConflictStrategy.UPDATE,
            original_hash=existing_doc['content_hash'],
            new_hash=new_doc.content_hash,
            resolution_time=datetime.now(),
            changes_summary=changes_summary,
            metadata={
                'conflict_type': conflict_type,
                'size_change': new_doc.size - existing_doc['size'],
                'auto_resolved': True
            }
        )
    
    async def _merge_document(
        self,
        new_doc: DocumentInfo,
        existing_doc: Dict[str, Any],
        conflict_type: str
    ) -> ConflictResolution:
        """
        Merge new document with existing version.
        
        Args:
            new_doc: New document information
            existing_doc: Existing document
            conflict_type: Type of conflict detected
            
        Returns:
            Conflict resolution details
        """
        logger.info(f"Merging document: {new_doc.file_path}")
        
        # For now, merging is complex and depends on document type
        # We'll implement a simple strategy: update with merge metadata
        
        merge_metadata = {
            'merge_strategy': 'replace_with_newer',
            'original_version': existing_doc['content_hash'],
            'merged_version': new_doc.content_hash,
            'merge_timestamp': datetime.now().isoformat(),
            'conflict_type': conflict_type
        }
        
        # Update document (similar to update strategy but with merge metadata)
        collection = self.db.collection('documents')
        
        update_data = {
            'content_hash': new_doc.content_hash,
            'size': new_doc.size,
            'modified_time': new_doc.modified_time.isoformat(),
            'ingestion_time': datetime.now().isoformat(),
            'processing_status': 'pending',
            'merge_metadata': merge_metadata
        }
        
        collection.update(existing_doc['_key'], update_data)
        
        return ConflictResolution(
            document_id=existing_doc['_key'],
            file_path=new_doc.file_path,
            strategy_used=ConflictStrategy.MERGE,
            original_hash=existing_doc['content_hash'],
            new_hash=new_doc.content_hash,
            resolution_time=datetime.now(),
            changes_summary=f"Document merged (strategy: replace_with_newer)",
            metadata=merge_metadata
        )
    
    async def _keep_both_versions(
        self,
        new_doc: DocumentInfo,
        existing_doc: Dict[str, Any],
        conflict_type: str
    ) -> ConflictResolution:
        """
        Keep both versions of the document.
        
        Args:
            new_doc: New document information
            existing_doc: Existing document
            conflict_type: Type of conflict detected
            
        Returns:
            Conflict resolution details
        """
        logger.info(f"Keeping both versions: {new_doc.file_path}")
        
        # Create new document entry with versioned key
        collection = self.db.collection('documents')
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        new_key = f"{existing_doc['_key']}_v{timestamp}"
        
        new_doc_data = {
            '_key': new_key,
            'file_path': f"{new_doc.file_path}_v{timestamp}",
            'original_path': new_doc.file_path,
            'content_hash': new_doc.content_hash,
            'size': new_doc.size,
            'modified_time': new_doc.modified_time.isoformat(),
            'ingestion_time': datetime.now().isoformat(),
            'processing_status': 'pending',
            'version_metadata': {
                'version_type': 'conflict_duplicate',
                'original_document_id': existing_doc['_key'],
                'conflict_type': conflict_type,
                'created_due_to_conflict': True
            }
        }
        
        collection.insert(new_doc_data)
        
        return ConflictResolution(
            document_id=new_key,
            file_path=new_doc.file_path,
            strategy_used=ConflictStrategy.KEEP_BOTH,
            original_hash=existing_doc['content_hash'],
            new_hash=new_doc.content_hash,
            resolution_time=datetime.now(),
            changes_summary=f"Created new version: {new_key}",
            metadata={
                'conflict_type': conflict_type,
                'original_document_id': existing_doc['_key'],
                'new_document_id': new_key,
                'auto_resolved': True
            }
        )
    
    async def _analyze_document_changes(
        self,
        new_doc: DocumentInfo,
        existing_doc: Dict[str, Any]
    ) -> str:
        """
        Analyze changes between document versions.
        
        Args:
            new_doc: New document information
            existing_doc: Existing document
            
        Returns:
            Summary of changes
        """
        changes = []
        
        # Size change
        size_change = new_doc.size - existing_doc['size']
        if size_change != 0:
            changes.append(f"Size: {size_change:+d} bytes")
        
        # Timestamp change
        existing_time = datetime.fromisoformat(existing_doc['modified_time'].replace('Z', '+00:00'))
        time_diff = (new_doc.modified_time - existing_time).total_seconds()
        if abs(time_diff) > 1:  # More than 1 second difference
            changes.append(f"Modified: {time_diff:+.0f}s")
        
        # Hash change (always present in conflicts)
        changes.append(f"Content: {existing_doc['content_hash'][:8]} -> {new_doc.content_hash[:8]}")
        
        if not changes:
            return "No significant changes detected"
        
        return "; ".join(changes)
    
    async def _mark_chunks_for_reprocessing(self, document_id: str) -> None:
        """
        Mark chunks associated with document for reprocessing.
        
        Args:
            document_id: Document identifier
        """
        chunks_collection = self.db.collection('chunks')
        
        # Update all chunks for this document
        query = """
        FOR chunk IN chunks
        FILTER chunk.document_id == @document_id
        UPDATE chunk WITH {
            processing_status: 'reprocess_needed',
            marked_for_reprocessing: @timestamp
        } IN chunks
        """
        
        self.db.query(query, bind_vars={
            'document_id': document_id,
            'timestamp': datetime.now().isoformat()
        })
        
        logger.debug(f"Marked chunks for reprocessing: document {document_id}")
    
    async def _log_conflict_resolution(
        self,
        resolution: ConflictResolution,
        conflict_type: str
    ) -> None:
        """
        Log conflict resolution to database.
        
        Args:
            resolution: Conflict resolution details
            conflict_type: Type of conflict
        """
        conflicts_collection = self.db.collection('conflicts')
        
        conflict_log = {
            'document_id': resolution.document_id,
            'file_path': resolution.file_path,
            'conflict_type': conflict_type,
            'detection_time': resolution.resolution_time.isoformat(),
            'resolution_time': resolution.resolution_time.isoformat(),
            'strategy_used': resolution.strategy_used.value,
            'original_hash': resolution.original_hash,
            'new_hash': resolution.new_hash,
            'resolution_details': {
                'changes_summary': resolution.changes_summary,
                'auto_resolved': resolution.metadata.get('auto_resolved', True),
                'metadata': resolution.metadata
            },
            'auto_resolved': True,
            'reviewer': 'system',
            'metadata': resolution.metadata
        }
        
        conflicts_collection.insert(conflict_log)
        logger.debug(f"Logged conflict resolution: {resolution.document_id}")
    
    async def detect_potential_duplicates(
        self,
        documents: List[DocumentInfo]
    ) -> List[Tuple[DocumentInfo, DocumentInfo, float]]:
        """
        Detect potential duplicate documents based on content similarity.
        
        Args:
            documents: List of documents to analyze
            
        Returns:
            List of (doc1, doc2, similarity_score) tuples for potential duplicates
        """
        logger.info(f"Analyzing {len(documents)} documents for potential duplicates")
        
        duplicates = []
        
        # Simple duplicate detection based on file names and sizes
        for i, doc1 in enumerate(documents):
            for doc2 in documents[i+1:]:
                similarity = self._calculate_document_similarity(doc1, doc2)
                
                if similarity >= self.similarity_threshold:
                    duplicates.append((doc1, doc2, similarity))
        
        logger.info(f"Found {len(duplicates)} potential duplicate pairs")
        return duplicates
    
    def _calculate_document_similarity(
        self,
        doc1: DocumentInfo,
        doc2: DocumentInfo
    ) -> float:
        """
        Calculate similarity between two documents.
        
        Args:
            doc1: First document
            doc2: Second document
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        # Simple similarity calculation based on:
        # 1. File name similarity
        # 2. Size similarity
        # 3. Content hash (if same, similarity = 1.0)
        
        if doc1.content_hash == doc2.content_hash:
            return 1.0
        
        # File name similarity
        name1 = doc1.file_path.split('/')[-1]  # Get filename
        name2 = doc2.file_path.split('/')[-1]
        
        name_similarity = difflib.SequenceMatcher(None, name1, name2).ratio()
        
        # Size similarity
        size_diff = abs(doc1.size - doc2.size)
        max_size = max(doc1.size, doc2.size)
        size_similarity = 1.0 - (size_diff / max_size) if max_size > 0 else 1.0
        
        # Combined similarity (weighted average)
        similarity = (name_similarity * 0.3 + size_similarity * 0.7)
        
        return similarity
    
    async def get_conflict_statistics(self) -> Dict[str, Any]:
        """
        Get conflict resolution statistics.
        
        Returns:
            Dictionary with conflict statistics
        """
        conflicts_collection = self.db.collection('conflicts')
        
        # Get conflict counts by type
        query = """
        FOR conflict IN conflicts
        COLLECT type = conflict.conflict_type WITH COUNT INTO count
        RETURN { type: type, count: count }
        """
        
        type_counts = {}
        for result in self.db.query(query):
            type_counts[result['type']] = result['count']
        
        # Get strategy usage
        query = """
        FOR conflict IN conflicts
        COLLECT strategy = conflict.strategy_used WITH COUNT INTO count
        RETURN { strategy: strategy, count: count }
        """
        
        strategy_counts = {}
        for result in self.db.query(query):
            strategy_counts[result['strategy']] = result['count']
        
        # Get recent conflicts
        query = """
        FOR conflict IN conflicts
        SORT conflict.detection_time DESC
        LIMIT 10
        RETURN {
            file_path: conflict.file_path,
            conflict_type: conflict.conflict_type,
            strategy_used: conflict.strategy_used,
            detection_time: conflict.detection_time
        }
        """
        
        recent_conflicts = list(self.db.query(query))
        
        return {
            'conflict_counts_by_type': type_counts,
            'strategy_usage': strategy_counts,
            'recent_conflicts': recent_conflicts,
            'total_conflicts': sum(type_counts.values())
        }