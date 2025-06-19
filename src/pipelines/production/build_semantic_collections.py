#!/usr/bin/env python3
"""
Build semantic collections for production use.

This script creates organized collections based on content type, purpose,
and semantic meaning, using ISNE-enhanced embeddings for better organization.
"""

import argparse
import logging
import sys
import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Set, Tuple
from collections import defaultdict
import re

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.database.arango_client import ArangoClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SemanticCollectionBuilder:
    """Build semantic collections from enhanced graph data."""
    
    def __init__(self, db_name: str):
        self.db_name = db_name
        self.arango_client = ArangoClient()
        
    def connect_to_database(self):
        """Connect to database."""
        success = self.arango_client.connect_to_database(self.db_name)
        if not success:
            raise Exception(f"Failed to connect to database: {self.db_name}")
        return self.arango_client._database
    
    def create_semantic_collections(self, db):
        """Create all semantic collections with proper schema."""
        logger.info("Creating semantic collections...")
        
        collections = {
            # Code-specific collections
            'code_functions': {'type': 'document'},      # Individual functions/methods
            'code_classes': {'type': 'document'},        # Class definitions
            'code_modules': {'type': 'document'},        # Python modules/files
            'code_tests': {'type': 'document'},          # Test functions/classes
            
            # Documentation collections
            'doc_concepts': {'type': 'document'},        # Key concepts explained
            'doc_tutorials': {'type': 'document'},       # How-to guides
            'doc_api_references': {'type': 'document'},  # API documentation
            'doc_readmes': {'type': 'document'},         # README files
            
            # Configuration collections
            'config_schemas': {'type': 'document'},      # Configuration schemas
            'config_settings': {'type': 'document'},     # Actual settings
            'config_examples': {'type': 'document'},     # Example configs
            
            # Semantic grouping collections
            'semantic_topics': {'type': 'document'},     # High-level topics
            'semantic_clusters': {'type': 'document'},   # Embedding clusters
            'semantic_keywords': {'type': 'document'},   # Key terms/concepts
            
            # Relationship collections (edges)
            'implements_edges': {'type': 'edge'},        # Code implements concept
            'documents_edges': {'type': 'edge'},         # Documentation describes code
            'configures_edges': {'type': 'edge'},        # Config configures component
            'related_to_edges': {'type': 'edge'},        # General semantic relationship
            'belongs_to_edges': {'type': 'edge'},        # Hierarchical membership
            'semantic_similarity_edges': {'type': 'edge'} # High-level semantic similarity
        }
        
        # Create collections
        for name, config in collections.items():
            if not db.has_collection(name):
                if config['type'] == 'edge':
                    db.create_collection(name, edge=True)
                else:
                    db.create_collection(name)
                logger.info(f"Created collection: {name}")
        
        # Create indexes for efficient queries
        self.create_indexes(db)
    
    def create_indexes(self, db):
        """Create indexes for semantic collections."""
        # Code collections indexes
        if db.has_collection('code_functions'):
            coll = db.collection('code_functions')
            coll.add_hash_index(fields=['name'])
            coll.add_hash_index(fields=['module'])
            coll.add_fulltext_index(fields=['docstring'])
        
        if db.has_collection('code_classes'):
            coll = db.collection('code_classes')
            coll.add_hash_index(fields=['name'])
            coll.add_hash_index(fields=['module'])
            coll.add_fulltext_index(fields=['docstring'])
        
        # Documentation indexes
        if db.has_collection('doc_concepts'):
            coll = db.collection('doc_concepts')
            coll.add_hash_index(fields=['concept'])
            coll.add_fulltext_index(fields=['description'])
        
        # Semantic indexes
        if db.has_collection('semantic_topics'):
            coll = db.collection('semantic_topics')
            coll.add_hash_index(fields=['topic_name'])
            coll.add_hash_index(fields=['category'])
    
    def extract_code_entities(self, db) -> Dict[str, List[Dict]]:
        """Extract code entities from Python files."""
        logger.info("Extracting code entities...")
        
        entities = {
            'functions': [],
            'classes': [],
            'modules': [],
            'tests': []
        }
        
        # Get all Python files
        python_files_query = """
        FOR f IN code_files
            FILTER f.file_name =~ ".py$"
            RETURN {
                id: CONCAT('code_files/', f._key),
                path: f.file_path,
                name: f.file_name,
                content: f.content
            }
        """
        
        cursor = db.aql.execute(python_files_query)
        
        for file_data in cursor:
            file_path = file_data['path']
            file_name = file_data['name']
            content = file_data.get('content', '')
            
            # Extract module info
            module_name = file_name.replace('.py', '')
            module_doc = {
                '_key': file_data['id'].split('/')[-1],
                'name': module_name,
                'file_path': file_path,
                'file_id': file_data['id'],
                'is_test': 'test' in file_path.lower(),
                'is_script': file_path.endswith('scripts/'),
                'imports': self.extract_imports(content),
                'docstring': self.extract_module_docstring(content)
            }
            entities['modules'].append(module_doc)
            
            # Extract functions and classes
            functions = self.extract_functions(content, module_name, file_data['id'])
            classes = self.extract_classes(content, module_name, file_data['id'])
            
            # Separate test functions
            for func in functions:
                if func['name'].startswith('test_') or 'test' in file_path.lower():
                    func['test_type'] = 'unit' if 'unit' in file_path else 'integration'
                    entities['tests'].append(func)
                else:
                    entities['functions'].append(func)
            
            entities['classes'].extend(classes)
        
        logger.info(f"Extracted: {len(entities['functions'])} functions, "
                   f"{len(entities['classes'])} classes, "
                   f"{len(entities['modules'])} modules, "
                   f"{len(entities['tests'])} tests")
        
        return entities
    
    def extract_imports(self, content: str) -> List[str]:
        """Extract import statements from Python code."""
        imports = []
        import_pattern = re.compile(r'^\s*(from\s+[\w.]+\s+)?import\s+(.+)$', re.MULTILINE)
        
        for match in import_pattern.finditer(content):
            import_stmt = match.group(0).strip()
            imports.append(import_stmt)
        
        return imports[:20]  # Limit to first 20 imports
    
    def extract_module_docstring(self, content: str) -> str:
        """Extract module-level docstring."""
        docstring_pattern = re.compile(r'^"""(.*?)"""', re.DOTALL)
        match = docstring_pattern.match(content.strip())
        
        if match:
            return match.group(1).strip()
        return ""
    
    def extract_functions(self, content: str, module: str, file_id: str) -> List[Dict]:
        """Extract function definitions from Python code."""
        functions = []
        
        # Simple regex for function extraction (production would use AST)
        func_pattern = re.compile(
            r'^(async\s+)?def\s+(\w+)\s*\([^)]*\)\s*(?:->\s*[^:]+)?:\s*\n\s*"""(.*?)"""',
            re.MULTILINE | re.DOTALL
        )
        
        for match in func_pattern.finditer(content):
            is_async = bool(match.group(1))
            func_name = match.group(2)
            docstring = match.group(3).strip()
            
            functions.append({
                'name': func_name,
                'module': module,
                'file_id': file_id,
                'is_async': is_async,
                'is_private': func_name.startswith('_'),
                'is_dunder': func_name.startswith('__') and func_name.endswith('__'),
                'docstring': docstring[:500]  # Limit docstring length
            })
        
        return functions
    
    def extract_classes(self, content: str, module: str, file_id: str) -> List[Dict]:
        """Extract class definitions from Python code."""
        classes = []
        
        # Simple regex for class extraction
        class_pattern = re.compile(
            r'^class\s+(\w+)(?:\([^)]*\))?\s*:\s*\n\s*"""(.*?)"""',
            re.MULTILINE | re.DOTALL
        )
        
        for match in class_pattern.finditer(content):
            class_name = match.group(1)
            docstring = match.group(2).strip()
            
            classes.append({
                'name': class_name,
                'module': module,
                'file_id': file_id,
                'is_exception': 'Exception' in class_name or 'Error' in class_name,
                'is_test': class_name.startswith('Test'),
                'docstring': docstring[:500]
            })
        
        return classes
    
    def extract_documentation_entities(self, db) -> Dict[str, List[Dict]]:
        """Extract documentation entities."""
        logger.info("Extracting documentation entities...")
        
        entities = {
            'concepts': [],
            'tutorials': [],
            'api_references': [],
            'readmes': []
        }
        
        # Get all documentation files
        doc_files_query = """
        FOR f IN documentation_files
            RETURN {
                id: CONCAT('documentation_files/', f._key),
                path: f.file_path,
                name: f.file_name,
                content: f.content
            }
        """
        
        cursor = db.aql.execute(doc_files_query)
        
        for doc_data in cursor:
            file_name = doc_data['name'].lower()
            content = doc_data.get('content', '')
            
            # Classify documentation type
            if 'readme' in file_name:
                entities['readmes'].append({
                    'file_id': doc_data['id'],
                    'title': doc_data['name'],
                    'path': doc_data['path'],
                    'summary': content[:500],
                    'sections': self.extract_markdown_sections(content)
                })
            elif 'tutorial' in file_name or 'guide' in file_name or 'how' in file_name:
                entities['tutorials'].append({
                    'file_id': doc_data['id'],
                    'title': self.extract_title(content) or doc_data['name'],
                    'path': doc_data['path'],
                    'topics': self.extract_topics(content),
                    'steps': self.extract_tutorial_steps(content)
                })
            elif 'api' in file_name or 'reference' in file_name:
                entities['api_references'].append({
                    'file_id': doc_data['id'],
                    'title': self.extract_title(content) or doc_data['name'],
                    'path': doc_data['path'],
                    'endpoints': self.extract_api_endpoints(content),
                    'parameters': self.extract_parameters(content)
                })
            else:
                # General concepts
                concepts = self.extract_concepts(content, doc_data['id'])
                entities['concepts'].extend(concepts)
        
        logger.info(f"Extracted: {len(entities['concepts'])} concepts, "
                   f"{len(entities['tutorials'])} tutorials, "
                   f"{len(entities['api_references'])} API refs, "
                   f"{len(entities['readmes'])} READMEs")
        
        return entities
    
    def extract_markdown_sections(self, content: str) -> List[str]:
        """Extract section headers from markdown."""
        sections = []
        section_pattern = re.compile(r'^#{1,3}\s+(.+)$', re.MULTILINE)
        
        for match in section_pattern.finditer(content):
            sections.append(match.group(1).strip())
        
        return sections[:10]  # First 10 sections
    
    def extract_title(self, content: str) -> str:
        """Extract title from markdown document."""
        title_pattern = re.compile(r'^#\s+(.+)$', re.MULTILINE)
        match = title_pattern.search(content)
        
        if match:
            return match.group(1).strip()
        return ""
    
    def extract_topics(self, content: str) -> List[str]:
        """Extract topics from content."""
        # Simple topic extraction based on headers and keywords
        topics = []
        
        # Extract from headers
        header_pattern = re.compile(r'^#{2,3}\s+(.+)$', re.MULTILINE)
        for match in header_pattern.finditer(content):
            topics.append(match.group(1).strip())
        
        return list(set(topics))[:10]
    
    def extract_tutorial_steps(self, content: str) -> List[str]:
        """Extract tutorial steps."""
        steps = []
        
        # Look for numbered lists or step patterns
        step_pattern = re.compile(r'^\d+\.\s+(.+)$', re.MULTILINE)
        for match in step_pattern.finditer(content):
            steps.append(match.group(1).strip())
        
        return steps[:10]
    
    def extract_api_endpoints(self, content: str) -> List[str]:
        """Extract API endpoints from documentation."""
        endpoints = []
        
        # Look for HTTP methods and paths
        endpoint_pattern = re.compile(r'(GET|POST|PUT|DELETE|PATCH)\s+(/[\w/{}:-]+)')
        for match in endpoint_pattern.finditer(content):
            endpoints.append(f"{match.group(1)} {match.group(2)}")
        
        return list(set(endpoints))[:20]
    
    def extract_parameters(self, content: str) -> List[str]:
        """Extract parameter definitions."""
        params = []
        
        # Look for parameter patterns
        param_pattern = re.compile(r'`(\w+)`\s*[:\-]\s*([^.]+\.)')
        for match in param_pattern.finditer(content):
            params.append({
                'name': match.group(1),
                'description': match.group(2).strip()
            })
        
        return params[:20]
    
    def extract_concepts(self, content: str, file_id: str) -> List[Dict]:
        """Extract key concepts from documentation."""
        concepts = []
        
        # Extract bold/emphasized terms as potential concepts
        concept_pattern = re.compile(r'\*\*([^*]+)\*\*|__([^_]+)__')
        
        seen_concepts = set()
        for match in concept_pattern.finditer(content):
            concept = match.group(1) or match.group(2)
            concept = concept.strip()
            
            if len(concept) > 3 and concept not in seen_concepts:
                seen_concepts.add(concept)
                
                # Find context around the concept
                start = max(0, match.start() - 100)
                end = min(len(content), match.end() + 100)
                context = content[start:end].strip()
                
                concepts.append({
                    'concept': concept,
                    'file_id': file_id,
                    'context': context,
                    'category': self.categorize_concept(concept)
                })
        
        return concepts[:20]  # Limit concepts per file
    
    def categorize_concept(self, concept: str) -> str:
        """Categorize a concept based on keywords."""
        concept_lower = concept.lower()
        
        if any(kw in concept_lower for kw in ['class', 'object', 'instance']):
            return 'object_oriented'
        elif any(kw in concept_lower for kw in ['function', 'method', 'call']):
            return 'functional'
        elif any(kw in concept_lower for kw in ['api', 'endpoint', 'request']):
            return 'api'
        elif any(kw in concept_lower for kw in ['config', 'setting', 'parameter']):
            return 'configuration'
        elif any(kw in concept_lower for kw in ['test', 'assert', 'mock']):
            return 'testing'
        else:
            return 'general'
    
    def create_semantic_clusters(self, db) -> List[Dict]:
        """Create semantic clusters from ISNE embeddings."""
        logger.info("Creating semantic clusters...")
        
        # Get ISNE embeddings if available
        has_isne = db.has_collection('isne_embeddings')
        
        if has_isne:
            embeddings_query = """
            FOR e IN isne_embeddings
                RETURN {
                    chunk_id: e.chunk_id,
                    embedding: e.embedding
                }
            """
        else:
            # Fall back to regular embeddings
            embeddings_query = """
            FOR e IN embeddings
                RETURN {
                    chunk_id: e.chunk_id,
                    embedding: e.embedding
                }
            """
        
        cursor = db.aql.execute(embeddings_query)
        embeddings_data = list(cursor)
        
        if not embeddings_data:
            logger.warning("No embeddings found for clustering")
            return []
        
        # Convert to numpy array
        embeddings = np.array([e['embedding'] for e in embeddings_data])
        chunk_ids = [e['chunk_id'] for e in embeddings_data]
        
        # Simple clustering using cosine similarity
        # (In production, use sklearn KMeans or DBSCAN)
        num_clusters = min(20, len(embeddings) // 100)  # Adaptive cluster count
        
        logger.info(f"Creating {num_clusters} semantic clusters from {len(embeddings)} embeddings")
        
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-8)
        
        # Simple k-means style clustering
        clusters = []
        cluster_centers = []
        
        # Initialize random centers
        indices = np.random.choice(len(normalized), num_clusters, replace=False)
        for idx in indices:
            cluster_centers.append(normalized[idx])
        
        # Assign points to clusters
        cluster_assignments = []
        for embedding in normalized:
            similarities = [np.dot(embedding, center) for center in cluster_centers]
            cluster_id = np.argmax(similarities)
            cluster_assignments.append(cluster_id)
        
        # Create cluster documents
        for i in range(num_clusters):
            cluster_chunks = [chunk_ids[j] for j, c in enumerate(cluster_assignments) if c == i]
            
            if cluster_chunks:
                # Get sample content from cluster
                sample_query = f"""
                FOR c IN chunks
                    FILTER c._id IN {cluster_chunks[:5]}
                    RETURN SUBSTRING(c.content, 0, 100)
                """
                cursor = db.aql.execute(sample_query)
                samples = list(cursor)
                
                clusters.append({
                    'cluster_id': i,
                    'size': len(cluster_chunks),
                    'chunk_ids': cluster_chunks[:100],  # Limit stored chunks
                    'center_embedding': cluster_centers[i].tolist(),
                    'sample_content': samples,
                    'created_at': datetime.now().isoformat()
                })
        
        logger.info(f"Created {len(clusters)} semantic clusters")
        return clusters
    
    def create_relationships(self, db, entities: Dict[str, Any]):
        """Create semantic relationships between entities."""
        logger.info("Creating semantic relationships...")
        
        relationship_count = 0
        
        # Connect functions to their modules
        if 'functions' in entities['code']:
            implements_edges = []
            for func in entities['code']['functions']:
                if 'file_id' in func:
                    implements_edges.append({
                        '_from': f"code_functions/{func['name']}",
                        '_to': func['file_id'],
                        'relationship': 'implements',
                        'function_name': func['name']
                    })
            
            if implements_edges:
                db.collection('implements_edges').insert_many(implements_edges[:1000])
                relationship_count += len(implements_edges)
        
        # Connect documentation to code
        if 'api_references' in entities['documentation']:
            documents_edges = []
            for api_ref in entities['documentation']['api_references']:
                # Find related code files based on path similarity
                file_name = Path(api_ref['path']).stem
                
                # Look for matching code files
                code_query = """
                FOR c IN code_files
                    FILTER CONTAINS(LOWER(c.file_name), @pattern)
                    LIMIT 5
                    RETURN c._id
                """
                cursor = db.aql.execute(
                    code_query,
                    bind_vars={'pattern': file_name.lower()}
                )
                
                for code_id in cursor:
                    documents_edges.append({
                        '_from': api_ref['file_id'],
                        '_to': f"code_files/{code_id}",
                        'relationship': 'documents',
                        'doc_type': 'api_reference'
                    })
            
            if documents_edges:
                db.collection('documents_edges').insert_many(documents_edges[:1000])
                relationship_count += len(documents_edges)
        
        # Connect concepts to their source files
        if 'concepts' in entities['documentation']:
            belongs_to_edges = []
            for concept in entities['documentation']['concepts']:
                if 'file_id' in concept:
                    belongs_to_edges.append({
                        '_from': f"doc_concepts/{concept['concept']}",
                        '_to': concept['file_id'],
                        'relationship': 'defined_in',
                        'category': concept.get('category', 'general')
                    })
            
            if belongs_to_edges:
                db.collection('belongs_to_edges').insert_many(belongs_to_edges[:1000])
                relationship_count += len(belongs_to_edges)
        
        logger.info(f"Created {relationship_count} semantic relationships")
    
    def build_semantic_collections(self, db) -> Dict[str, Any]:
        """Build all semantic collections."""
        start_time = time.time()
        stats = {}
        
        # Create collection schema
        self.create_semantic_collections(db)
        
        # Extract entities
        code_entities = self.extract_code_entities(db)
        doc_entities = self.extract_documentation_entities(db)
        
        # Store code entities
        if code_entities['functions']:
            db.collection('code_functions').insert_many(code_entities['functions'])
            stats['code_functions'] = len(code_entities['functions'])
        
        if code_entities['classes']:
            db.collection('code_classes').insert_many(code_entities['classes'])
            stats['code_classes'] = len(code_entities['classes'])
        
        if code_entities['modules']:
            db.collection('code_modules').insert_many(code_entities['modules'])
            stats['code_modules'] = len(code_entities['modules'])
        
        if code_entities['tests']:
            db.collection('code_tests').insert_many(code_entities['tests'])
            stats['code_tests'] = len(code_entities['tests'])
        
        # Store documentation entities
        if doc_entities['concepts']:
            db.collection('doc_concepts').insert_many(doc_entities['concepts'])
            stats['doc_concepts'] = len(doc_entities['concepts'])
        
        if doc_entities['tutorials']:
            db.collection('doc_tutorials').insert_many(doc_entities['tutorials'])
            stats['doc_tutorials'] = len(doc_entities['tutorials'])
        
        if doc_entities['api_references']:
            db.collection('doc_api_references').insert_many(doc_entities['api_references'])
            stats['doc_api_references'] = len(doc_entities['api_references'])
        
        if doc_entities['readmes']:
            db.collection('doc_readmes').insert_many(doc_entities['readmes'])
            stats['doc_readmes'] = len(doc_entities['readmes'])
        
        # Create semantic clusters
        clusters = self.create_semantic_clusters(db)
        if clusters:
            db.collection('semantic_clusters').insert_many(clusters)
            stats['semantic_clusters'] = len(clusters)
        
        # Create relationships
        all_entities = {
            'code': code_entities,
            'documentation': doc_entities
        }
        self.create_relationships(db, all_entities)
        
        stats['processing_time'] = time.time() - start_time
        return stats


def main():
    parser = argparse.ArgumentParser(
        description="Build semantic collections for production use"
    )
    parser.add_argument(
        '--db-name',
        type=str,
        default='isne_production_database',
        help='Database name'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize builder
        builder = SemanticCollectionBuilder(args.db_name)
        
        # Connect to database
        db = builder.connect_to_database()
        
        # Build semantic collections
        stats = builder.build_semantic_collections(db)
        
        # Print results
        print("\n" + "="*60)
        print("SEMANTIC COLLECTIONS BUILT")
        print("="*60)
        print(f"Database: {args.db_name}")
        print(f"Processing time: {stats.get('processing_time', 0):.2f} seconds")
        print("\nCollections created:")
        
        for collection, count in sorted(stats.items()):
            if collection != 'processing_time':
                print(f"  {collection}: {count:,} documents")
        
        print("\nYour production database now has:")
        print("- Code analysis (functions, classes, modules, tests)")
        print("- Documentation structure (concepts, tutorials, API refs)")
        print("- Semantic clusters from embeddings")
        print("- Relationship edges connecting everything")
        
        return 0
        
    except Exception as e:
        logger.error(f"Failed to build semantic collections: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())