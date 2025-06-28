"""
Library of Congress Classification (LCC) System for Bootstrap Graph Builder


Assigns semantic classifications to documents and code files to enable
enhanced relationship discovery beyond directory structure.
"""

import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


class LCCClassifier:
    """
    Library of Congress Classification system for technical documents and code.
    
    Focuses on computer science, technology, and information science categories
    relevant to software development and research projects.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        self.fallback_to_directory = self.config.get('fallback_to_directory', True)
        
        # Type hints for attributes
        self.lcc_categories: Dict[str, Dict[str, Any]] = {}
        self.keyword_to_lcc: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        self.directory_mappings: Dict[str, str] = {}
        
        # Initialize classification system
        self._init_lcc_categories()
        self._init_keyword_mappings()
        self._init_directory_mappings()
        
        logger.info("Initialized LCC Classifier for technical content")
    
    def _init_lcc_categories(self) -> None:
        """Initialize LCC category definitions"""
        self.lcc_categories = {
            # Computer Science (QA76)
            'QA76.1': {
                'name': 'General Computer Science',
                'description': 'Computer science theory, algorithms, computational complexity',
                'keywords': ['algorithm', 'complexity', 'theory', 'computation', 'computer science'],
                'weight': 1.0
            },
            'QA76.5': {
                'name': 'Computer Architecture', 
                'description': 'Hardware design, system architecture, performance',
                'keywords': ['architecture', 'hardware', 'system design', 'performance', 'cpu', 'gpu'],
                'weight': 0.8
            },
            'QA76.6': {
                'name': 'Programming Languages',
                'description': 'Programming languages, software development, coding',
                'keywords': ['programming', 'software', 'code', 'development', 'language', 'python', 'javascript'],
                'weight': 1.0
            },
            'QA76.7': {
                'name': 'Software Engineering',
                'description': 'Software engineering methodologies, design patterns, testing',
                'keywords': ['software engineering', 'design pattern', 'testing', 'methodology', 'agile', 'devops'],
                'weight': 0.9
            },
            'QA76.9.D26': {
                'name': 'Machine Learning & AI',
                'description': 'Machine learning, artificial intelligence, neural networks',
                'keywords': ['machine learning', 'ai', 'artificial intelligence', 'neural network', 'deep learning', 'tensorflow', 'pytorch'],
                'weight': 1.0
            },
            'QA76.9.D3': {
                'name': 'Database Systems',
                'description': 'Database design, data management, storage systems',
                'keywords': ['database', 'data', 'storage', 'sql', 'nosql', 'query', 'mongodb', 'postgresql'],
                'weight': 0.9
            },
            'QA76.9.D35': {
                'name': 'Data Processing & Analytics',
                'description': 'Data processing, analytics, visualization, big data',
                'keywords': ['data processing', 'analytics', 'visualization', 'big data', 'pandas', 'numpy', 'statistics'],
                'weight': 0.9
            },
            'QA76.9.N4': {
                'name': 'Computer Networks',
                'description': 'Networking, distributed systems, web development',
                'keywords': ['network', 'distributed', 'web', 'api', 'http', 'tcp', 'networking'],
                'weight': 0.8
            },
            
            # Technology (T)
            'T57': {
                'name': 'Operations Research',
                'description': 'Optimization, operations research, systems analysis',
                'keywords': ['optimization', 'operations research', 'linear programming', 'systems analysis'],
                'weight': 0.7
            },
            'T58': {
                'name': 'Management Technology', 
                'description': 'Project management, workflow, automation',
                'keywords': ['project management', 'workflow', 'automation', 'process', 'management'],
                'weight': 0.6
            },
            'T59': {
                'name': 'Technical Writing',
                'description': 'Documentation, technical writing, communication',
                'keywords': ['documentation', 'technical writing', 'manual', 'guide', 'tutorial'],
                'weight': 0.7
            },
            
            # Information Science (Z)
            'Z665': {
                'name': 'Information Science',
                'description': 'Information science, knowledge management',
                'keywords': ['information science', 'knowledge management', 'information retrieval'],
                'weight': 0.8
            },
            'Z699.5': {
                'name': 'Information Retrieval',
                'description': 'Search systems, information retrieval, indexing',
                'keywords': ['search', 'retrieval', 'indexing', 'information retrieval', 'elasticsearch'],
                'weight': 0.8
            }
        }
    
    def _init_keyword_mappings(self) -> None:
        """Create keyword to LCC mappings for efficient lookup"""
        self.keyword_to_lcc = defaultdict(list)
        
        for lcc_code, category in self.lcc_categories.items():
            for keyword in category['keywords']:
                self.keyword_to_lcc[keyword.lower()].append((lcc_code, category['weight']))
    
    def _init_directory_mappings(self) -> None:
        """Map common directory names to LCC categories"""
        self.directory_mappings = {
            # Code directories
            'models': 'QA76.9.D26',  # ML models
            'algorithms': 'QA76.1',  # Algorithm implementations
            'data': 'QA76.9.D35',   # Data processing
            'database': 'QA76.9.D3', # Database code
            'db': 'QA76.9.D3',
            'api': 'QA76.9.N4',     # Web APIs
            'web': 'QA76.9.N4',     # Web development
            'network': 'QA76.9.N4', # Networking
            'ml': 'QA76.9.D26',     # Machine learning
            'ai': 'QA76.9.D26',     # AI
            'neural': 'QA76.9.D26', # Neural networks
            'optimization': 'T57',   # Optimization
            'analysis': 'QA76.9.D35', # Data analysis
            'visualization': 'QA76.9.D35', # Data viz
            'tests': 'QA76.7',      # Software testing
            'testing': 'QA76.7',
            'utils': 'QA76.6',      # Utilities
            'tools': 'QA76.6',      # Tools
            'core': 'QA76.6',       # Core programming
            'src': 'QA76.6',        # Source code
            
            # Documentation directories  
            'docs': 'T59',          # Documentation
            'documentation': 'T59',
            'manual': 'T59',
            'tutorials': 'T59',
            'examples': 'T59',
            'guides': 'T59'
        }
    
    def classify_document(self, content: str, file_path: Path) -> Tuple[str, float]:
        """
        Classify a document using content analysis and directory context.
        
        Args:
            content: Document content
            file_path: Path to the document
            
        Returns:
            Tuple of (lcc_classification, confidence_score)
        """
        # Method 1: Content-based classification
        content_classification, content_confidence = self._classify_by_content(content)
        
        # Method 2: Directory-based classification  
        directory_classification, directory_confidence = self._classify_by_directory(file_path)
        
        # Method 3: File type augmentation
        filetype_classification, filetype_confidence = self._classify_by_filetype(file_path, content)
        
        # Combine classifications
        classifications = [
            (content_classification, content_confidence * 1.0),      # High weight for content
            (directory_classification, directory_confidence * 0.6),  # Medium weight for directory
            (filetype_classification, filetype_confidence * 0.4)     # Lower weight for file type
        ]
        
        # Filter out None classifications and sort by confidence
        valid_classifications = [(lcc, conf) for lcc, conf in classifications if lcc is not None]
        
        if not valid_classifications:
            # Fallback to generic programming classification for code files
            if file_path.suffix.lower() == '.py':
                return 'QA76.6', 0.3  # Low confidence fallback
            else:
                return 'T59', 0.2     # Documentation fallback
        
        # Return highest confidence classification
        best_classification = max(valid_classifications, key=lambda x: x[1])
        return best_classification
    
    def _classify_by_content(self, content: str) -> Tuple[Optional[str], float]:
        """Classify based on content analysis"""
        if not content.strip():
            return None, 0.0
        
        content_lower = content.lower()
        
        # Score each LCC category based on keyword matches
        category_scores: Dict[str, float] = defaultdict(float)
        total_matches = 0
        
        for keyword, lcc_entries in self.keyword_to_lcc.items():
            # Count keyword occurrences (with word boundaries)
            pattern = r'\b' + re.escape(keyword) + r'\b'
            matches = len(re.findall(pattern, content_lower))
            
            if matches > 0:
                total_matches += matches
                for lcc_code, weight in lcc_entries:
                    # Score based on frequency and keyword importance
                    category_scores[lcc_code] += matches * weight
        
        if not category_scores:
            return None, 0.0
        
        # Find best classification
        best_lcc = max(category_scores.items(), key=lambda x: x[1])
        lcc_code, score = best_lcc
        
        # Calculate confidence based on score dominance
        total_score = sum(category_scores.values())
        confidence = min(score / total_score, 1.0)  # Normalize to [0,1]
        
        # Apply minimum confidence threshold
        if confidence < self.confidence_threshold:
            return None, confidence
        
        return lcc_code, confidence
    
    def _classify_by_directory(self, file_path: Path) -> Tuple[Optional[str], float]:
        """Classify based on directory structure"""
        path_parts = [part.lower() for part in file_path.parts]
        
        # Check each directory part for matches
        for part in path_parts:
            if part in self.directory_mappings:
                return self.directory_mappings[part], 0.8
        
        # Check for partial matches
        for part in path_parts:
            for dir_keyword, lcc_code in self.directory_mappings.items():
                if dir_keyword in part or part in dir_keyword:
                    return lcc_code, 0.6
        
        return None, 0.0
    
    def _classify_by_filetype(self, file_path: Path, content: str) -> Tuple[Optional[str], float]:
        """Classify based on file type and extension"""
        extension = file_path.suffix.lower()
        
        # Programming files
        if extension == '.py':
            # Further classify Python files based on content
            if any(keyword in content.lower() for keyword in ['tensorflow', 'pytorch', 'sklearn', 'neural']):
                return 'QA76.9.D26', 0.9  # ML/AI
            elif any(keyword in content.lower() for keyword in ['pandas', 'numpy', 'matplotlib']):
                return 'QA76.9.D35', 0.8  # Data processing
            elif any(keyword in content.lower() for keyword in ['flask', 'django', 'fastapi', 'http']):
                return 'QA76.9.N4', 0.8   # Web development
            else:
                return 'QA76.6', 0.7      # General programming
        
        elif extension in {'.js', '.ts', '.html', '.css'}:
            return 'QA76.9.N4', 0.7      # Web development
        
        elif extension in {'.sql'}:
            return 'QA76.9.D3', 0.9      # Database
        
        elif extension in {'.md', '.txt', '.rst'}:
            return 'T59', 0.6             # Documentation
        
        elif extension in {'.json', '.yaml', '.yml', '.toml'}:
            return 'QA76.6', 0.5          # Configuration/Programming
        
        return None, 0.0
    
    def get_lcc_hierarchy(self, lcc_code: str) -> List[str]:
        """Get the hierarchical path for an LCC code"""
        parts = lcc_code.split('.')
        hierarchy = []
        
        current = ""
        for i, part in enumerate(parts):
            if i == 0:
                current = part
            else:
                current += f".{part}"
            hierarchy.append(current)
        
        return hierarchy
    
    def calculate_semantic_similarity(self, lcc_a: str, lcc_b: str) -> float:
        """Calculate semantic similarity between two LCC classifications"""
        if lcc_a == lcc_b:
            return 1.0
        
        hierarchy_a = self.get_lcc_hierarchy(lcc_a)
        hierarchy_b = self.get_lcc_hierarchy(lcc_b)
        
        # Find common prefix length
        common_length = 0
        for i, (a, b) in enumerate(zip(hierarchy_a, hierarchy_b)):
            if a == b:
                common_length = i + 1
            else:
                break
        
        # Similarity based on shared hierarchy depth
        max_depth = max(len(hierarchy_a), len(hierarchy_b))
        if max_depth == 0:
            return 0.0
        
        similarity = common_length / max_depth
        
        # Boost similarity for same top-level category
        if common_length > 0:
            similarity = min(similarity + 0.3, 1.0)
        
        return similarity
    
    def get_related_classifications(self, lcc_code: str, min_similarity: float = 0.5) -> List[Tuple[str, float]]:
        """Get LCC classifications related to the given one"""
        related = []
        
        for other_lcc in self.lcc_categories.keys():
            if other_lcc != lcc_code:
                similarity = self.calculate_semantic_similarity(lcc_code, other_lcc)
                if similarity >= min_similarity:
                    related.append((other_lcc, similarity))
        
        # Sort by similarity
        related.sort(key=lambda x: x[1], reverse=True)
        return related
    
    def classify_batch(self, documents: List[Tuple[str, Path]]) -> Dict[str, Tuple[str, float]]:
        """Classify multiple documents efficiently"""
        results = {}
        
        for content, file_path in documents:
            file_key = str(file_path)
            classification, confidence = self.classify_document(content, file_path)
            results[file_key] = (classification, confidence)
        
        return results
    
    def get_classification_stats(self, classifications: Dict[str, Tuple[str, float]]) -> Dict[str, Any]:
        """Get statistics about a set of classifications"""
        # Create properly typed intermediate variables
        classification_dist: Dict[str, int] = defaultdict(int)
        confidence_dist = {
            'high': 0,    # > 0.8
            'medium': 0,  # 0.5 - 0.8
            'low': 0      # < 0.5
        }
        
        classified_documents = 0
        confidences: List[float] = []
        
        for file_path, (lcc_code, confidence) in classifications.items():
            if lcc_code is not None:
                classified_documents += 1
                classification_dist[lcc_code] += 1
                confidences.append(confidence)
                
                # Confidence distribution
                if confidence > 0.8:
                    confidence_dist['high'] += 1
                elif confidence >= 0.5:
                    confidence_dist['medium'] += 1
                else:
                    confidence_dist['low'] += 1
        
        average_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        # Build final stats dict
        stats: Dict[str, Any] = {
            'total_documents': len(classifications),
            'classified_documents': classified_documents,
            'average_confidence': average_confidence,
            'classification_distribution': dict(classification_dist),
            'confidence_distribution': confidence_dist
        }
        
        return stats
    
    def save_classifications(self, classifications: Dict[str, Tuple[str, float]], output_path: Path) -> None:
        """Save classifications to file"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to serializable format
        serializable = {}
        for file_path, (lcc_code, confidence) in classifications.items():
            serializable[file_path] = {
                'lcc_classification': lcc_code,
                'confidence': confidence
            }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(classifications)} LCC classifications to {output_path}")
    
    def load_classifications(self, input_path: Path) -> Dict[str, Tuple[str, float]]:
        """Load classifications from file"""
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        classifications = {}
        for file_path, classification_data in data.items():
            lcc_code = classification_data['lcc_classification']
            confidence = classification_data['confidence']
            classifications[file_path] = (lcc_code, confidence)
        
        logger.info(f"Loaded {len(classifications)} LCC classifications from {input_path}")
        return classifications