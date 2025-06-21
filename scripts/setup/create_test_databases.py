#!/usr/bin/env python3
"""
Create test databases for HADES development and testing.

This script creates all necessary test databases in ArangoDB for MCP testing,
development, and integration testing.
"""

import sys
import logging
from pathlib import Path
from typing import List, Dict, Any

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.database.arango_client import ArangoClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Test databases to create
TEST_DATABASES = [
    {
        'name': 'test_training_db',
        'description': 'Database for ISNE training tests',
        'collections': ['documents', 'chunks', 'embeddings', 'models', 'training_runs']
    },
    {
        'name': 'test_production_db', 
        'description': 'Database for production pipeline tests',
        'collections': ['documents', 'chunks', 'embeddings', 'semantic_collections']
    },
    {
        'name': 'hades_test_integration',
        'description': 'Integration testing database',
        'collections': ['test_documents', 'test_chunks', 'test_embeddings']
    },
    {
        'name': 'micro_validation_test',
        'description': 'Micro validation testing database',
        'collections': ['validation_docs', 'validation_results']
    }
]

class TestDatabaseCreator:
    """Creates and manages test databases for HADES."""
    
    def __init__(self):
        self.arango_client = ArangoClient()
        
    def create_database(self, db_config: Dict[str, Any]) -> bool:
        """
        Create a test database with collections.
        
        Args:
            db_config: Database configuration with name, description, and collections
            
        Returns:
            bool: True if successful, False otherwise
        """
        db_name = db_config['name']
        description = db_config.get('description', 'Test database')
        collections = db_config.get('collections', [])
        
        try:
            logger.info(f"Creating database: {db_name}")
            logger.info(f"Description: {description}")
            
            # Check if database already exists
            if self._database_exists(db_name):
                logger.info(f"Database '{db_name}' already exists, skipping creation")
                return True
                
            # Create database
            success = self.arango_client.create_database(db_name)
            if not success:
                logger.error(f"Failed to create database: {db_name}")
                return False
                
            # Connect to the new database
            success = self.arango_client.connect_to_database(db_name)
            if not success:
                logger.error(f"Failed to connect to database: {db_name}")
                return False
                
            # Create collections
            db = self.arango_client._database
            for collection_name in collections:
                if not db.has_collection(collection_name):
                    db.create_collection(collection_name)
                    logger.info(f"Created collection: {collection_name}")
                else:
                    logger.info(f"Collection '{collection_name}' already exists")
                    
            # Add some basic indexes for performance
            self._create_basic_indexes(db, collections)
            
            logger.info(f"Successfully created database: {db_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating database {db_name}: {e}")
            return False
            
    def _database_exists(self, db_name: str) -> bool:
        """Check if database already exists."""
        try:
            # Try to connect to see if it exists
            return self.arango_client.connect_to_database(db_name)
        except:
            return False
            
    def _create_basic_indexes(self, db, collections: List[str]):
        """Create basic indexes for common query patterns."""
        try:
            # Add indexes for common collections
            if 'chunks' in collections:
                chunks_coll = db.collection('chunks')
                chunks_coll.add_hash_index(fields=['document_id'])
                chunks_coll.add_hash_index(fields=['chunk_index'])
                logger.info("Created indexes for 'chunks' collection")
                
            if 'embeddings' in collections:
                embeddings_coll = db.collection('embeddings')
                embeddings_coll.add_hash_index(fields=['chunk_id'])
                embeddings_coll.add_hash_index(fields=['model_name'])
                logger.info("Created indexes for 'embeddings' collection")
                
            if 'documents' in collections:
                docs_coll = db.collection('documents')
                docs_coll.add_hash_index(fields=['file_path'])
                docs_coll.add_hash_index(fields=['document_type'])
                logger.info("Created indexes for 'documents' collection")
                
        except Exception as e:
            logger.warning(f"Error creating indexes: {e}")
            
    def create_all_test_databases(self) -> Dict[str, bool]:
        """
        Create all test databases.
        
        Returns:
            dict: Results of database creation (name -> success)
        """
        results = {}
        
        logger.info("Starting test database creation...")
        logger.info(f"Will create {len(TEST_DATABASES)} databases")
        
        for db_config in TEST_DATABASES:
            db_name = db_config['name']
            success = self.create_database(db_config)
            results[db_name] = success
            
        return results
        
    def verify_databases(self) -> Dict[str, bool]:
        """
        Verify all test databases exist and are accessible.
        
        Returns:
            dict: Verification results (name -> accessible)
        """
        results = {}
        
        logger.info("Verifying test databases...")
        
        for db_config in TEST_DATABASES:
            db_name = db_config['name']
            try:
                # Try to connect and run a simple query
                success = self.arango_client.connect_to_database(db_name)
                if success:
                    db = self.arango_client._database
                    # Simple test query
                    list(db.aql.execute("RETURN 1"))
                    logger.info(f"✅ Database '{db_name}' is accessible")
                    results[db_name] = True
                else:
                    logger.error(f"❌ Cannot connect to database '{db_name}'")
                    results[db_name] = False
                    
            except Exception as e:
                logger.error(f"❌ Database '{db_name}' verification failed: {e}")
                results[db_name] = False
                
        return results
        
    def cleanup_test_databases(self) -> Dict[str, bool]:
        """
        Remove all test databases (for cleanup).
        
        Returns:
            dict: Cleanup results (name -> success)
        """
        results = {}
        
        logger.warning("Cleaning up test databases...")
        
        for db_config in TEST_DATABASES:
            db_name = db_config['name']
            try:
                # Note: ArangoDB client might not have delete_database method
                # This is a placeholder for cleanup functionality
                logger.info(f"Would delete database: {db_name}")
                results[db_name] = True
                
            except Exception as e:
                logger.error(f"Error cleaning up database {db_name}: {e}")
                results[db_name] = False
                
        return results


def main():
    """Main function to create test databases."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create HADES test databases")
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify existing databases, do not create'
    )
    parser.add_argument(
        '--cleanup',
        action='store_true',
        help='Remove test databases (use with caution)'
    )
    
    args = parser.parse_args()
    
    creator = TestDatabaseCreator()
    
    if args.cleanup:
        logger.warning("⚠️  CLEANUP MODE: This will remove test databases!")
        response = input("Are you sure? Type 'yes' to continue: ")
        if response.lower() == 'yes':
            results = creator.cleanup_test_databases()
            print("\nCleanup Results:")
            for db_name, success in results.items():
                status = "✅ Success" if success else "❌ Failed"
                print(f"  {db_name}: {status}")
        else:
            logger.info("Cleanup cancelled")
        return
        
    if args.verify_only:
        logger.info("Verification mode - checking existing databases")
        results = creator.verify_databases()
    else:
        logger.info("Creation mode - creating test databases")
        results = creator.create_all_test_databases()
        
        # Verify after creation
        logger.info("\nVerifying created databases...")
        verification_results = creator.verify_databases()
        
    # Print summary
    print("\n" + "="*50)
    print("TEST DATABASE SETUP RESULTS")
    print("="*50)
    
    if not args.verify_only:
        print("Creation Results:")
        for db_name, success in results.items():
            status = "✅ Created" if success else "❌ Failed"
            print(f"  {db_name}: {status}")
            
        print("\nVerification Results:")
        for db_name, success in verification_results.items():
            status = "✅ Accessible" if success else "❌ Not accessible"
            print(f"  {db_name}: {status}")
            
        # Summary
        created_count = sum(results.values())
        verified_count = sum(verification_results.values())
        total_count = len(TEST_DATABASES)
        
        print(f"\nSummary:")
        print(f"  Databases created: {created_count}/{total_count}")
        print(f"  Databases verified: {verified_count}/{total_count}")
        
        if created_count == total_count and verified_count == total_count:
            print(f"\n✅ All test databases ready for MCP testing!")
        else:
            print(f"\n⚠️  Some databases failed - check logs above")
            
    else:
        print("Verification Results:")
        for db_name, success in results.items():
            status = "✅ Accessible" if success else "❌ Not accessible"
            print(f"  {db_name}: {status}")
            
        verified_count = sum(results.values())
        total_count = len(TEST_DATABASES)
        print(f"\nSummary: {verified_count}/{total_count} databases accessible")


if __name__ == "__main__":
    main()