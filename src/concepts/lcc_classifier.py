"""
Library of Congress Classification (LCC) for Code and Technical Documents

This module implements LCC classification to add semantic categorization
beyond directory structure, enabling cross-domain knowledge discovery.
"""

import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict

logger = logging.getLogger(__name__)


class LCCClassifier:
    """
    Library of Congress Classification system for technical content.
    
    Maps code and documents to standard knowledge domains, creating
    semantic relationships that transcend project boundaries.
    """
    
    def __init__(self) -> None:
        """Initialize the LCC classifier with technical categories."""
        self.categories = self._init_categories()
        self.keyword_mappings = self._build_keyword_mappings()
        self.directory_mappings = self._init_directory_mappings()
        
        logger.info("Initialized LCC Classifier for technical content")
    
    def _init_categories(self) -> Dict[str, Dict[str, Any]]:
        """Initialize LCC categories relevant to technical content."""
        return {
            # Computer Science (QA76)
            'QA76.1': {
                'name': 'General Computer Science',
                'description': 'Computer science theory, algorithms, computational complexity',
                'keywords': ['algorithm', 'complexity', 'theory', 'computation', 'computer science'],
                'patterns': [r'algorith\w*', r'complex\w*', r'computat\w*']
            },
            'QA76.5': {
                'name': 'Computer Architecture',
                'description': 'Hardware design, system architecture, performance',
                'keywords': ['architecture', 'hardware', 'system design', 'performance', 'cpu', 'gpu'],
                'patterns': [r'architect\w*', r'hardware', r'cpu|gpu', r'performance']
            },
            'QA76.6': {
                'name': 'Programming Languages',
                'description': 'Programming languages, compilers, interpreters',
                'keywords': ['programming', 'language', 'compiler', 'interpreter', 'syntax'],
                'patterns': [r'programm\w*', r'language', r'compil\w*', r'syntax']
            },
            'QA76.7': {
                'name': 'Software Engineering',
                'description': 'Software development methodologies, design patterns, testing',
                'keywords': ['software engineering', 'design pattern', 'testing', 'agile', 'devops'],
                'patterns': [r'software\s+engineer\w*', r'design\s+pattern', r'test\w*', r'agile']
            },
            'QA76.9.D26': {
                'name': 'Machine Learning & AI',
                'description': 'Machine learning, artificial intelligence, neural networks',
                'keywords': ['machine learning', 'ai', 'neural network', 'deep learning', 'model'],
                'patterns': [r'machine\s+learning', r'\bai\b', r'neural', r'deep\s+learning']
            },
            'QA76.9.D3': {
                'name': 'Database Systems',
                'description': 'Database design, data management, query systems',
                'keywords': ['database', 'sql', 'nosql', 'query', 'storage', 'transaction'],
                'patterns': [r'database', r'\bsql\b', r'nosql', r'query', r'transaction']
            },
            'QA76.9.D35': {
                'name': 'Data Processing & Analytics',
                'description': 'Data processing, analytics, visualization, ETL',
                'keywords': ['data processing', 'analytics', 'visualization', 'etl', 'pipeline'],
                'patterns': [r'data\s+process\w*', r'analytic\w*', r'visualiz\w*', r'etl']
            },
            'QA76.9.I52': {
                'name': 'Information Retrieval',
                'description': 'Search systems, indexing, retrieval algorithms',
                'keywords': ['search', 'retrieval', 'indexing', 'ranking', 'query processing'],
                'patterns': [r'search', r'retriev\w*', r'index\w*', r'rank\w*']
            },
            'QA76.9.N38': {
                'name': 'Natural Language Processing',
                'description': 'NLP, text processing, language models',
                'keywords': ['nlp', 'natural language', 'text processing', 'tokenization', 'embedding'],
                'patterns': [r'nlp', r'natural\s+language', r'text\s+process\w*', r'embedd\w*']
            },
            'QA76.9.N4': {
                'name': 'Computer Networks',
                'description': 'Networking, distributed systems, protocols',
                'keywords': ['network', 'distributed', 'protocol', 'tcp', 'http', 'api'],
                'patterns': [r'network\w*', r'distribut\w*', r'protocol', r'tcp|http', r'api']
            },
            
            # Mathematics (QA)
            'QA273': {
                'name': 'Probability Theory',
                'description': 'Probability, statistics, stochastic processes',
                'keywords': ['probability', 'statistics', 'stochastic', 'random', 'distribution'],
                'patterns': [r'probabilit\w*', r'statistic\w*', r'stochastic', r'random']
            },
            'QA276': {
                'name': 'Mathematical Statistics',
                'description': 'Statistical methods, inference, hypothesis testing',
                'keywords': ['statistical', 'inference', 'hypothesis', 'regression', 'correlation'],
                'patterns': [r'statistic\w*', r'inference', r'hypothesis', r'regress\w*']
            },
            
            # Technology (T)
            'T57': {
                'name': 'Operations Research',
                'description': 'Optimization, operations research, decision theory',
                'keywords': ['optimization', 'operations research', 'linear programming', 'decision'],
                'patterns': [r'optimiz\w*', r'operations\s+research', r'linear\s+programm\w*']
            },
            'T58.5': {
                'name': 'Information Technology',
                'description': 'IT systems, infrastructure, cloud computing',
                'keywords': ['information technology', 'infrastructure', 'cloud', 'devops', 'deployment'],
                'patterns': [r'infrastructure', r'cloud', r'devops', r'deploy\w*']
            },
            
            # Information Science (Z)
            'Z665': {
                'name': 'Library & Information Science',
                'description': 'Information organization, knowledge management',
                'keywords': ['information science', 'knowledge management', 'taxonomy', 'ontology'],
                'patterns': [r'information\s+science', r'knowledge\s+manag\w*', r'taxonom\w*', r'ontolog\w*']
            }
        }
    
    def _build_keyword_mappings(self) -> Dict[str, List[Tuple[str, float]]]:
        """Build keyword to category mappings."""
        mappings = defaultdict(list)
        
        for lcc_code, category in self.categories.items():
            for keyword in category['keywords']:
                mappings[keyword.lower()].append((lcc_code, 1.0))
        
        return dict(mappings)
    
    def _init_directory_mappings(self) -> Dict[str, str]:
        """Map common directory names to LCC categories."""
        return {
            # ML/AI directories
            'models': 'QA76.9.D26',
            'ml': 'QA76.9.D26',
            'ai': 'QA76.9.D26',
            'neural': 'QA76.9.D26',
            
            # Algorithm directories
            'algorithms': 'QA76.1',
            'algo': 'QA76.1',
            
            # Data directories
            'data': 'QA76.9.D35',
            'analytics': 'QA76.9.D35',
            'etl': 'QA76.9.D35',
            
            # Database directories
            'database': 'QA76.9.D3',
            'db': 'QA76.9.D3',
            'sql': 'QA76.9.D3',
            
            # Network/API directories
            'api': 'QA76.9.N4',
            'network': 'QA76.9.N4',
            'web': 'QA76.9.N4',
            'rest': 'QA76.9.N4',
            
            # NLP directories
            'nlp': 'QA76.9.N38',
            'text': 'QA76.9.N38',
            'language': 'QA76.9.N38',
            
            # Software engineering
            'src': 'QA76.7',
            'lib': 'QA76.7',
            'tests': 'QA76.7',
            'test': 'QA76.7',
            
            # Infrastructure
            'infra': 'T58.5',
            'deploy': 'T58.5',
            'docker': 'T58.5',
            'k8s': 'T58.5'
        }
    
    def classify(self, 
                content: str,
                file_path: Optional[Path] = None,
                metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Classify content into LCC categories.
        
        Args:
            content: Text content to classify
            file_path: Optional file path for directory-based hints
            metadata: Optional metadata (AST analysis, etc.)
            
        Returns:
            Classification results with categories and confidence scores
        """
        scores: defaultdict[str, float] = defaultdict(float)
        
        # Content-based classification
        content_lower = content.lower()
        
        # Check keywords
        words = set(content_lower.split())
        for word in words:
            if word in self.keyword_mappings:
                for lcc_code, weight in self.keyword_mappings[word]:
                    scores[lcc_code] += weight
        
        # Check patterns
        for lcc_code, category in self.categories.items():
            for pattern in category.get('patterns', []):
                matches = len(re.findall(pattern, content_lower))
                if matches:
                    scores[lcc_code] += matches * 0.5
        
        # Directory-based hints
        if file_path:
            path_parts = str(file_path).lower().split('/')
            for part in path_parts:
                if part in self.directory_mappings:
                    lcc_code = self.directory_mappings[part]
                    scores[lcc_code] += 2.0  # Strong signal from directory
        
        # AST-based classification for code
        if metadata and metadata.get('ast_analysis'):
            self._classify_from_ast(metadata['ast_analysis'], scores)
        
        # Normalize scores and select top categories
        total_score = sum(scores.values())
        if total_score > 0:
            normalized_scores = {
                code: score / total_score 
                for code, score in scores.items()
            }
            
            # Select categories with significant scores
            selected = {
                code: score 
                for code, score in normalized_scores.items() 
                if score > 0.1
            }
            
            # Sort by score
            top_categories = sorted(
                selected.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:3]
            
            return {
                'primary_category': top_categories[0][0] if top_categories else None,
                'categories': dict(top_categories),
                'confidence': top_categories[0][1] if top_categories else 0.0,
                'metadata': {
                    'classification_method': 'content_and_structure',
                    'total_signals': len(scores)
                }
            }
        
        return {
            'primary_category': None,
            'categories': {},
            'confidence': 0.0,
            'metadata': {'classification_method': 'none'}
        }
    
    def _classify_from_ast(self, 
                          ast_analysis: Dict[str, Any], 
                          scores: Dict[str, float]) -> None:
        """Add classification signals from AST analysis."""
        # Check imports for library-specific categories
        imports = ast_analysis.get('imports', {})
        
        for module in imports.values():
            module_lower = module.lower()
            
            # ML libraries
            if any(lib in module_lower for lib in ['torch', 'tensorflow', 'sklearn', 'keras']):
                scores['QA76.9.D26'] += 3.0
            
            # Database libraries
            elif any(lib in module_lower for lib in ['sqlalchemy', 'pymongo', 'redis', 'psycopg']):
                scores['QA76.9.D3'] += 3.0
            
            # Data processing
            elif any(lib in module_lower for lib in ['pandas', 'numpy', 'scipy']):
                scores['QA76.9.D35'] += 2.0
            
            # NLP libraries
            elif any(lib in module_lower for lib in ['nltk', 'spacy', 'transformers']):
                scores['QA76.9.N38'] += 3.0
            
            # Network libraries
            elif any(lib in module_lower for lib in ['requests', 'aiohttp', 'flask', 'fastapi']):
                scores['QA76.9.N4'] += 2.0
    
    def get_category_info(self, lcc_code: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about an LCC category."""
        return self.categories.get(lcc_code)
    
    def find_related_categories(self, lcc_code: str) -> List[str]:
        """Find related LCC categories based on hierarchy."""
        related = []
        
        # Find categories with same prefix (same general area)
        prefix = lcc_code.split('.')[0]
        for code in self.categories:
            if code.startswith(prefix) and code != lcc_code:
                related.append(code)
        
        return related
    
    def create_cross_domain_edges(self, 
                                 classified_nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Create edges between nodes in the same LCC category.
        
        This enables cross-project knowledge discovery.
        """
        edges = []
        
        # Group nodes by primary category
        category_groups = defaultdict(list)
        for node in classified_nodes:
            if node.get('lcc_primary'):
                category_groups[node['lcc_primary']].append(node)
        
        # Create edges within categories
        for category, nodes in category_groups.items():
            if len(nodes) < 2:
                continue
            
            # Create edges between nodes in same category
            for i, node1 in enumerate(nodes):
                for node2 in nodes[i+1:]:
                    # Don't connect if already in same project
                    if self._same_project(node1, node2):
                        continue
                    
                    edge = {
                        '_from': f"nodes/{node1['id']}",
                        '_to': f"nodes/{node2['id']}",
                        'type': 'lcc_cross_domain',
                        'weight': 0.6,  # Moderate weight
                        'metadata': {
                            'lcc_category': category,
                            'category_name': self.categories[category]['name'],
                            'cross_project': True
                        }
                    }
                    edges.append(edge)
        
        return edges
    
    def _same_project(self, node1: Dict[str, Any], node2: Dict[str, Any]) -> bool:
        """Check if two nodes are from the same project."""
        path1 = node1.get('file_path', '')
        path2 = node2.get('file_path', '')
        
        # Simple heuristic: same top-level directory
        parts1 = path1.split('/')[:2]
        parts2 = path2.split('/')[:2]
        
        return bool(parts1 == parts2)