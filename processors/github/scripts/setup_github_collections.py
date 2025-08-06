#!/usr/bin/env python3
"""
Setup GitHub base container collections in ArangoDB.

Theory Connection:
Base containers represent pure observation without interpretation - 
the raw material reality before theoretical frameworks are applied.
Like ethnographic field notes before analysis.
"""

import os
import sys
import logging
from pathlib import Path
from arango import ArangoClient
from typing import Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GitHubCollectionSetup:
    """Setup GitHub base container collections."""
    
    def __init__(self, db_host: str = "192.168.1.69", db_name: str = "academy_store"):
        self.db_host = db_host
        self.db_name = db_name
        self._init_database()
    
    def _init_database(self):
        """Initialize database connection."""
        password = os.environ.get('ARANGO_PASSWORD')
        if not password:
            raise ValueError("ARANGO_PASSWORD environment variable required")
        
        client = ArangoClient(hosts=f'http://{self.db_host}:8529')
        self.db = client.db(self.db_name, username='root', password=password)
    
    def create_collections(self) -> Dict:
        """Create all GitHub base container collections."""
        collections_created = []
        
        # Define collections with their configurations
        collections = [
            {
                'name': 'base_github_repos',
                'type': 'document',
                'indexes': [
                    {'fields': ['language'], 'type': 'persistent'},
                    {'fields': ['topics[*]'], 'type': 'persistent'},
                    {'fields': ['stargazers_count'], 'type': 'persistent'},
                    {'fields': ['clone_status'], 'type': 'persistent'},
                    {'fields': ['owner', 'name'], 'type': 'persistent', 'unique': True}
                ]
            },
            {
                'name': 'base_github_files', 
                'type': 'document',
                'indexes': [
                    {'fields': ['repo_key'], 'type': 'persistent'},
                    {'fields': ['language'], 'type': 'persistent'},
                    {'fields': ['file_type'], 'type': 'persistent'},
                    {'fields': ['file_extension'], 'type': 'persistent'},
                    {'fields': ['repo_key', 'file_path'], 'type': 'persistent', 'unique': True}
                ]
            },
            {
                'name': 'base_github_embeddings',
                'type': 'document',
                'indexes': [
                    {'fields': ['repo_key'], 'type': 'persistent'},
                    {'fields': ['file_key'], 'type': 'persistent'},
                    {'fields': ['chunk_type'], 'type': 'persistent'},
                    {'fields': ['repo_key', 'file_key', 'chunk_index'], 'type': 'persistent', 'unique': True}
                ]
            }
        ]
        
        for collection_config in collections:
            try:
                # Check if collection exists
                if self.db.has_collection(collection_config['name']):
                    logger.info(f"Collection {collection_config['name']} already exists")
                    # Ensure indexes are created
                    collection = self.db.collection(collection_config['name'])
                else:
                    # Create collection
                    collection = self.db.create_collection(
                        name=collection_config['name'],
                        sync=True  # Ensure writes are synced to disk
                    )
                    logger.info(f"Created collection: {collection_config['name']}")
                    collections_created.append(collection_config['name'])
                
                # Create indexes
                for index_config in collection_config.get('indexes', []):
                    try:
                        collection.add_persistent_index(
                            fields=index_config['fields'],
                            unique=index_config.get('unique', False),
                            sparse=index_config.get('sparse', False)
                        )
                        logger.info(f"Created index on {index_config['fields']} for {collection_config['name']}")
                    except Exception as e:
                        # Index might already exist
                        logger.debug(f"Index creation note: {e}")
                        
            except Exception as e:
                logger.error(f"Failed to create collection {collection_config['name']}: {e}")
                raise
        
        return {
            'status': 'success',
            'collections_created': collections_created,
            'total_collections': len(collections)
        }
    
    def verify_setup(self) -> Dict:
        """Verify all collections and indexes are properly configured."""
        verification = {}
        
        expected_collections = ['base_github_repos', 'base_github_files', 'base_github_embeddings']
        
        for coll_name in expected_collections:
            if self.db.has_collection(coll_name):
                collection = self.db.collection(coll_name)
                indexes = collection.indexes()
                
                verification[coll_name] = {
                    'exists': True,
                    'count': collection.count(),
                    'indexes': len(indexes),
                    'index_details': [
                        {
                            'type': idx.get('type'),
                            'fields': idx.get('fields', []),
                            'unique': idx.get('unique', False)
                        }
                        for idx in indexes
                    ]
                }
            else:
                verification[coll_name] = {'exists': False}
        
        return verification


def main():
    """Setup GitHub collections."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Setup GitHub base container collections')
    parser.add_argument('--db-host', default='192.168.1.69', help='ArangoDB host')
    parser.add_argument('--db-name', default='academy_store', help='Database name')
    parser.add_argument('--verify-only', action='store_true', help='Only verify existing setup')
    
    args = parser.parse_args()
    
    setup = GitHubCollectionSetup(args.db_host, args.db_name)
    
    if args.verify_only:
        print("\n🔍 Verifying GitHub collections setup...")
        verification = setup.verify_setup()
        
        for coll_name, info in verification.items():
            if info['exists']:
                print(f"✅ {coll_name}: {info['count']} documents, {info['indexes']} indexes")
                for idx in info['index_details']:
                    if idx['fields']:  # Skip default _key index
                        print(f"   - Index on {idx['fields']} (unique={idx['unique']})")
            else:
                print(f"❌ {coll_name}: NOT FOUND")
    else:
        print("\n🚀 Setting up GitHub base container collections...")
        result = setup.create_collections()
        
        if result['collections_created']:
            print(f"✅ Created {len(result['collections_created'])} new collections:")
            for coll in result['collections_created']:
                print(f"   - {coll}")
        else:
            print("✅ All collections already exist")
        
        # Verify setup
        print("\n🔍 Verifying setup...")
        verification = setup.verify_setup()
        all_good = all(info['exists'] for info in verification.values())
        
        if all_good:
            print("✅ All GitHub base container collections are ready!")
        else:
            print("❌ Some collections are missing - please check logs")


if __name__ == "__main__":
    main()