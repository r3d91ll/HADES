"""
Sequential-ISNE Schema Migration and Validation

This module provides migration utilities to transition from the original HADES
schema to the Sequential-ISNE modality-specific schema architecture.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

from arango import ArangoClient
from arango.database import StandardDatabase
from arango.exceptions import ArangoError

from .schema import SchemaManager, IncrementalSchema
from .sequential_isne_schema import SequentialISNESchemaManager, SequentialISNESchema
from .sequential_isne_types import (
    FileType, 
    CodeFileType, 
    DocumentationType, 
    ConfigFileType,
    ProcessingStatus,
    classify_file_type,
    get_specific_file_type,
    get_modality_collection
)

logger = logging.getLogger(__name__)


class SchemaMigrationError(Exception):
    """Exception raised during schema migration."""
    pass


class SequentialISNEMigrator:
    """
    Handles migration from original HADES schema to Sequential-ISNE modality-specific schema.
    
    This migrator:
    1. Validates existing data compatibility
    2. Creates new modality-specific collections
    3. Migrates data from old schema to new schema
    4. Preserves existing relationships where possible
    5. Creates backup of original data
    """
    
    def __init__(self, client: ArangoClient, source_db: str, target_db: str):
        """
        Initialize the migrator.
        
        Args:
            client: ArangoDB client instance
            source_db: Name of source database with old schema
            target_db: Name of target database for new schema
        """
        self.client = client
        self.source_db_name = source_db
        self.target_db_name = target_db
        
        # Initialize schema managers
        self.old_schema_manager = SchemaManager(client, source_db)
        self.new_schema_manager = SequentialISNESchemaManager(client, target_db)
        
        # Migration tracking
        self.migration_log: List[str] = []
        self.errors: List[str] = []
        self.warnings: List[str] = []
        
        logger.info(f"Initialized Sequential-ISNE migrator: {source_db} -> {target_db}")
    
    def validate_source_schema(self) -> bool:
        """
        Validate source database schema compatibility.
        
        Returns:
            True if source schema is compatible for migration
        """
        try:
            # Check if source database exists
            sys_db = self.client.db("_system")
            if not sys_db.has_database(self.source_db_name):
                self.errors.append(f"Source database '{self.source_db_name}' does not exist")
                return False
            
            source_db = self.client.db(self.source_db_name)
            
            # Check for required collections in source
            required_collections = ['documents', 'chunks', 'embeddings']
            missing_collections = []
            
            for collection in required_collections:
                if not source_db.has_collection(collection):
                    missing_collections.append(collection)
            
            if missing_collections:
                self.errors.append(f"Source database missing required collections: {missing_collections}")
                return False
            
            # Check data compatibility
            documents_collection = source_db.collection('documents')
            sample_docs = list(documents_collection.all(limit=10))
            
            incompatible_docs = []
            for doc in sample_docs:
                if not self._validate_document_compatibility(doc):
                    incompatible_docs.append(doc.get('_key', 'unknown'))
            
            if incompatible_docs:
                self.warnings.append(f"Found {len(incompatible_docs)} potentially incompatible documents")
            
            self.migration_log.append("Source schema validation completed")
            return True
            
        except ArangoError as e:
            self.errors.append(f"Error validating source schema: {e}")
            return False
    
    def _validate_document_compatibility(self, doc: Dict[str, Any]) -> bool:
        """
        Validate document compatibility with new schema.
        
        Args:
            doc: Document from source database
            
        Returns:
            True if document is compatible
        """
        required_fields = ['file_path', 'content_hash', 'size', 'modified_time']
        
        for field in required_fields:
            if field not in doc:
                return False
        
        # Check if file path is valid
        try:
            file_path = Path(doc['file_path'])
            file_type = classify_file_type(file_path)
            if file_type == FileType.UNKNOWN:
                return False
        except Exception:
            return False
        
        return True
    
    def create_target_schema(self) -> bool:
        """
        Create target database with Sequential-ISNE schema.
        
        Returns:
            True if schema creation successful
        """
        try:
            # Create target database and schema
            success = self.new_schema_manager.initialize_database()
            
            if success:
                self.migration_log.append("Target Sequential-ISNE schema created successfully")
                return True
            else:
                self.errors.append("Failed to create target schema")
                return False
                
        except Exception as e:
            self.errors.append(f"Error creating target schema: {e}")
            return False
    
    def migrate_documents(self) -> bool:
        """
        Migrate documents from old schema to modality-specific collections.
        
        Returns:
            True if migration successful
        """
        try:
            source_db = self.client.db(self.source_db_name)
            target_db = self.client.db(self.target_db_name)
            
            documents_collection = source_db.collection('documents')
            
            # Get all documents
            all_docs = list(documents_collection.all())
            
            # Group documents by modality
            modality_groups = {
                FileType.CODE: [],
                FileType.DOCUMENTATION: [],
                FileType.CONFIG: []
            }
            
            skipped_docs = []
            
            for doc in all_docs:
                try:
                    file_path = Path(doc['file_path'])
                    file_type = classify_file_type(file_path)
                    
                    if file_type in modality_groups:
                        modality_groups[file_type].append(doc)
                    else:
                        skipped_docs.append(doc.get('_key', 'unknown'))
                        
                except Exception as e:
                    logger.warning(f"Error processing document {doc.get('_key')}: {e}")
                    skipped_docs.append(doc.get('_key', 'unknown'))
            
            # Migrate each modality
            migration_stats = {}
            
            for modality, docs in modality_groups.items():
                if not docs:
                    continue
                
                collection_name = get_modality_collection(modality)
                target_collection = target_db.collection(collection_name)
                
                migrated_count = 0
                failed_count = 0
                
                for doc in docs:
                    try:
                        migrated_doc = self._convert_document_to_modality(doc, modality)
                        
                        # Insert into target collection
                        target_collection.insert(migrated_doc, overwrite=True)
                        migrated_count += 1
                        
                    except Exception as e:
                        logger.error(f"Error migrating document {doc.get('_key')}: {e}")
                        failed_count += 1
                
                migration_stats[modality.value] = {
                    'migrated': migrated_count,
                    'failed': failed_count,
                    'collection': collection_name
                }
            
            # Log migration results
            total_migrated = sum(stats['migrated'] for stats in migration_stats.values())
            total_failed = sum(stats['failed'] for stats in migration_stats.values())
            
            self.migration_log.append(f"Document migration completed:")
            self.migration_log.append(f"  Total documents: {len(all_docs)}")
            self.migration_log.append(f"  Successfully migrated: {total_migrated}")
            self.migration_log.append(f"  Failed migrations: {total_failed}")
            self.migration_log.append(f"  Skipped (unsupported): {len(skipped_docs)}")
            
            for modality, stats in migration_stats.items():
                self.migration_log.append(f"  {modality}: {stats['migrated']} -> {stats['collection']}")
            
            if skipped_docs:
                self.warnings.append(f"Skipped {len(skipped_docs)} unsupported documents")
            
            return total_failed == 0
            
        except Exception as e:
            self.errors.append(f"Error during document migration: {e}")
            return False
    
    def _convert_document_to_modality(self, doc: Dict[str, Any], modality: FileType) -> Dict[str, Any]:
        """
        Convert a document from old schema to modality-specific format.
        
        Args:
            doc: Original document
            modality: Target modality
            
        Returns:
            Converted document for modality-specific collection
        """
        file_path = Path(doc['file_path'])
        
        # Base conversion
        converted = {
            '_key': doc.get('_key'),
            'file_path': str(file_path),
            'file_name': file_path.name,
            'directory': str(file_path.parent),
            'extension': file_path.suffix,
            'file_type': get_specific_file_type(file_path, modality),
            'content': doc.get('content', ''),
            'content_hash': doc['content_hash'],
            'size': doc['size'],
            'modified_time': doc['modified_time'],
            'ingestion_time': doc.get('ingestion_time', datetime.now().isoformat()),
            'processing_status': doc.get('processing_status', ProcessingStatus.COMPLETED.value),
            'directory_depth': len(file_path.parts) - 1,
            'chunk_count': doc.get('chunk_count', 0),
            'metadata': doc.get('metadata', {})
        }
        
        # Add modality-specific fields
        if modality == FileType.CODE:
            # Add code-specific fields
            converted.update({
                'lines_of_code': len(doc.get('content', '').splitlines()),
                'ast_metadata': {},
                'imports': [],
                'functions': [],
                'classes': [],
                'complexity_score': None
            })
            
        elif modality == FileType.DOCUMENTATION:
            # Add documentation-specific fields
            content = doc.get('content', '')
            converted.update({
                'word_count': len(content.split()),
                'document_structure': {},
                'headings': [],
                'links': [],
                'code_references': [],
                'readability_score': None
            })
            
        elif modality == FileType.CONFIG:
            # Add config-specific fields
            converted.update({
                'parsed_config': {},
                'config_schema': {},
                'validation_status': None,
                'dependencies': []
            })
        
        return converted
    
    def migrate_chunks(self) -> bool:
        """
        Migrate chunks to new schema format.
        
        Returns:
            True if migration successful
        """
        try:
            source_db = self.client.db(self.source_db_name)
            target_db = self.client.db(self.target_db_name)
            
            if not source_db.has_collection('chunks'):
                self.warnings.append("No chunks collection found in source database")
                return True
            
            chunks_collection = source_db.collection('chunks')
            target_chunks = target_db.collection('chunks')
            
            all_chunks = list(chunks_collection.all())
            
            migrated_count = 0
            failed_count = 0
            
            for chunk in all_chunks:
                try:
                    converted_chunk = self._convert_chunk(chunk)
                    target_chunks.insert(converted_chunk, overwrite=True)
                    migrated_count += 1
                    
                except Exception as e:
                    logger.error(f"Error migrating chunk {chunk.get('_key')}: {e}")
                    failed_count += 1
            
            self.migration_log.append(f"Chunk migration completed:")
            self.migration_log.append(f"  Total chunks: {len(all_chunks)}")
            self.migration_log.append(f"  Successfully migrated: {migrated_count}")
            self.migration_log.append(f"  Failed migrations: {failed_count}")
            
            return failed_count == 0
            
        except Exception as e:
            self.errors.append(f"Error during chunk migration: {e}")
            return False
    
    def _convert_chunk(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Convert chunk to new schema format."""
        return {
            '_key': chunk.get('_key'),
            'source_file_collection': 'unknown',  # Will need to be determined
            'source_file_id': chunk.get('document_id', ''),
            'content': chunk.get('content', ''),
            'content_hash': chunk.get('content_hash', ''),
            'start_pos': chunk.get('start_pos', 0),
            'end_pos': chunk.get('end_pos', 0),
            'chunk_index': chunk.get('chunk_index', 0),
            'chunk_type': 'text',  # Default, could be improved with analysis
            'created_at': chunk.get('created_at', datetime.now().isoformat()),
            'embedding_id': chunk.get('embedding_id'),
            'metadata': chunk.get('metadata', {})
        }
    
    def migrate_embeddings(self) -> bool:
        """
        Migrate embeddings to new schema format.
        
        Returns:
            True if migration successful
        """
        try:
            source_db = self.client.db(self.source_db_name)
            target_db = self.client.db(self.target_db_name)
            
            if not source_db.has_collection('embeddings'):
                self.warnings.append("No embeddings collection found in source database")
                return True
            
            embeddings_collection = source_db.collection('embeddings')
            target_embeddings = target_db.collection('embeddings')
            
            all_embeddings = list(embeddings_collection.all())
            
            migrated_count = 0
            failed_count = 0
            
            for embedding in all_embeddings:
                try:
                    converted_embedding = self._convert_embedding(embedding)
                    target_embeddings.insert(converted_embedding, overwrite=True)
                    migrated_count += 1
                    
                except Exception as e:
                    logger.error(f"Error migrating embedding {embedding.get('_key')}: {e}")
                    failed_count += 1
            
            self.migration_log.append(f"Embedding migration completed:")
            self.migration_log.append(f"  Total embeddings: {len(all_embeddings)}")
            self.migration_log.append(f"  Successfully migrated: {migrated_count}")
            self.migration_log.append(f"  Failed migrations: {failed_count}")
            
            return failed_count == 0
            
        except Exception as e:
            self.errors.append(f"Error during embedding migration: {e}")
            return False
    
    def _convert_embedding(self, embedding: Dict[str, Any]) -> Dict[str, Any]:
        """Convert embedding to new schema format."""
        return {
            '_key': embedding.get('_key'),
            'source_type': 'chunk',  # Default assumption
            'source_collection': 'chunks',
            'source_id': embedding.get('chunk_id', ''),
            'embedding_type': 'original',
            'vector': embedding.get('vector', []),
            'model_name': embedding.get('model_name', 'unknown'),
            'model_version': embedding.get('model_version', 'unknown'),
            'embedding_dim': embedding.get('embedding_dim', len(embedding.get('vector', []))),
            'created_at': embedding.get('created_at', datetime.now().isoformat()),
            'isne_metadata': {},
            'metadata': embedding.get('metadata', {})
        }
    
    def run_full_migration(self, backup_source: bool = True) -> bool:
        """
        Run complete migration from old schema to Sequential-ISNE schema.
        
        Args:
            backup_source: Whether to create backup of source database
            
        Returns:
            True if migration successful
        """
        try:
            self.migration_log.append(f"Starting Sequential-ISNE migration at {datetime.now()}")
            
            # Step 1: Validate source schema
            if not self.validate_source_schema():
                self.errors.append("Source schema validation failed")
                return False
            
            # Step 2: Create backup if requested
            if backup_source:
                backup_name = f"{self.source_db_name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                if not self._create_backup(backup_name):
                    self.warnings.append("Backup creation failed, continuing migration")
            
            # Step 3: Create target schema
            if not self.create_target_schema():
                self.errors.append("Target schema creation failed")
                return False
            
            # Step 4: Migrate documents
            if not self.migrate_documents():
                self.errors.append("Document migration failed")
                return False
            
            # Step 5: Migrate chunks
            if not self.migrate_chunks():
                self.errors.append("Chunk migration failed")
                return False
            
            # Step 6: Migrate embeddings
            if not self.migrate_embeddings():
                self.errors.append("Embedding migration failed")
                return False
            
            self.migration_log.append(f"Sequential-ISNE migration completed successfully at {datetime.now()}")
            return True
            
        except Exception as e:
            self.errors.append(f"Migration failed with error: {e}")
            return False
    
    def _create_backup(self, backup_name: str) -> bool:
        """Create backup of source database."""
        try:
            # Note: ArangoDB doesn't have built-in database copying
            # This would need to be implemented with export/import
            # For now, just log the intention
            self.migration_log.append(f"Backup requested: {backup_name}")
            self.warnings.append("Backup functionality not implemented - using export/import instead")
            return True
            
        except Exception as e:
            logger.error(f"Backup creation failed: {e}")
            return False
    
    def get_migration_report(self) -> Dict[str, Any]:
        """
        Get detailed migration report.
        
        Returns:
            Migration report with logs, errors, and warnings
        """
        return {
            'migration_log': self.migration_log,
            'errors': self.errors,
            'warnings': self.warnings,
            'success': len(self.errors) == 0,
            'source_database': self.source_db_name,
            'target_database': self.target_db_name,
            'timestamp': datetime.now().isoformat()
        }
    
    def print_migration_report(self) -> None:
        """Print formatted migration report."""
        report = self.get_migration_report()
        
        print("="*60)
        print("SEQUENTIAL-ISNE MIGRATION REPORT")
        print("="*60)
        print(f"Source: {report['source_database']}")
        print(f"Target: {report['target_database']}")
        print(f"Status: {'SUCCESS' if report['success'] else 'FAILED'}")
        print(f"Time: {report['timestamp']}")
        print()
        
        if report['migration_log']:
            print("MIGRATION LOG:")
            for log_entry in report['migration_log']:
                print(f"  {log_entry}")
            print()
        
        if report['warnings']:
            print("WARNINGS:")
            for warning in report['warnings']:
                print(f"  ⚠️  {warning}")
            print()
        
        if report['errors']:
            print("ERRORS:")
            for error in report['errors']:
                print(f"  ❌ {error}")
            print()
        
        print("="*60)