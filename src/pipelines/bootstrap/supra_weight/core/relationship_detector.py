"""
Relationship detector for identifying multi-dimensional relationships between nodes.
"""

from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path
import re
import numpy as np
from datetime import datetime
import logging

from .supra_weight_calculator import Relationship, RelationType
from .theory_practice_detector import TheoryPracticeDetector

logger = logging.getLogger(__name__)


class RelationshipDetector:
    """
    Detects various types of relationships between nodes (files, chunks, etc.).
    """
    
    def __init__(self, semantic_threshold: float = 0.5, enable_theory_practice: bool = True):
        """
        Initialize the detector.
        
        Args:
            semantic_threshold: Minimum similarity for semantic relationships
            enable_theory_practice: Whether to enable theory-practice bridge detection
        """
        self.semantic_threshold = semantic_threshold
        self.enable_theory_practice = enable_theory_practice
        
        # Initialize theory-practice detector
        if self.enable_theory_practice:
            self.theory_practice_detector = TheoryPracticeDetector(
                semantic_threshold=0.4,  # Lower threshold for novel bridges
                cross_domain_threshold=0.6
            )
        
        # Compile regex patterns for import detection
        self.import_patterns = {
            'python': re.compile(r'(?:from\s+(\S+)\s+)?import\s+(\S+)'),
            'javascript': re.compile(r'(?:import|require)\s*\(?\s*[\'"]([^\'"\)]+)[\'"]'),
            'java': re.compile(r'import\s+(\S+);'),
            'cpp': re.compile(r'#include\s*[<"]([^>"]+)[>"]'),
        }
        
        logger.info("Initialized RelationshipDetector")
        
    def detect_all_relationships(self, node_a: Dict, node_b: Dict, 
                               embeddings: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> List[Relationship]:
        """
        Detect all relationships between two nodes.
        
        Args:
            node_a: First node with metadata
            node_b: Second node with metadata
            embeddings: Optional tuple of (embedding_a, embedding_b)
            
        Returns:
            List of detected relationships
        """
        relationships = []
        
        # Check each relationship type
        if self._check_co_location(node_a, node_b):
            relationships.append(
                Relationship(
                    type=RelationType.CO_LOCATION,
                    strength=1.0,
                    metadata={'directory': node_a.get('directory')}
                )
            )
            
        if self._check_sequential(node_a, node_b):
            relationships.append(
                Relationship(
                    type=RelationType.SEQUENTIAL,
                    strength=1.0,
                    metadata={'distance': abs(node_a.get('chunk_index', -1) - node_b.get('chunk_index', -1))}
                )
            )
            
        import_rel = self._check_import(node_a, node_b)
        if import_rel:
            relationships.append(import_rel)
            
        if embeddings:
            semantic_rel = self._check_semantic_similarity(node_a, node_b, embeddings)
            if semantic_rel:
                relationships.append(semantic_rel)
                
        temporal_rel = self._check_temporal(node_a, node_b)
        if temporal_rel:
            relationships.append(temporal_rel)
            
        reference_rel = self._check_reference(node_a, node_b)
        if reference_rel:
            relationships.append(reference_rel)
            
        structural_rel = self._check_structural(node_a, node_b)
        if structural_rel:
            relationships.append(structural_rel)
            
        # Check theory-practice bridges
        if self.enable_theory_practice:
            bridge_rels = self.theory_practice_detector.detect_bridges(
                node_a, node_b, embeddings
            )
            relationships.extend(bridge_rels)
            
        return relationships
        
    def _check_co_location(self, node_a: Dict, node_b: Dict) -> bool:
        """Check if nodes are in the same directory."""
        dir_a = node_a.get('directory') or node_a.get('metadata', {}).get('directory')
        dir_b = node_b.get('directory') or node_b.get('metadata', {}).get('directory')
        
        if dir_a and dir_b:
            return Path(dir_a) == Path(dir_b)
        return False
        
    def _check_sequential(self, node_a: Dict, node_b: Dict) -> bool:
        """Check if nodes are sequential chunks from the same file."""
        # Check if both are chunks from same source
        source_a = node_a.get('source_file_id') or node_a.get('document_id')
        source_b = node_b.get('source_file_id') or node_b.get('document_id')
        
        if source_a and source_b and source_a == source_b:
            idx_a = node_a.get('chunk_index', -1)
            idx_b = node_b.get('chunk_index', -1)
            
            # Adjacent chunks
            if idx_a >= 0 and idx_b >= 0 and abs(idx_a - idx_b) == 1:
                return True
                
        return False
        
    def _check_import(self, node_a: Dict, node_b: Dict) -> Optional[Relationship]:
        """Check for import relationships between files."""
        # Only check file-level nodes
        if node_a.get('node_type') != 'file' or node_b.get('node_type') != 'file':
            return None
            
        content_a = node_a.get('content', '')
        file_b = node_b.get('file_name', '')
        
        if not content_a or not file_b:
            return None
            
        # Determine language from file extension
        ext_a = Path(node_a.get('file_path', '')).suffix.lower()
        language = self._get_language_from_extension(ext_a)
        
        if language and language in self.import_patterns:
            pattern = self.import_patterns[language]
            
            # Check if file_a imports file_b
            for match in pattern.finditer(content_a):
                imported = match.group(1) if match.lastindex else match.group(0)
                
                # Simple check - could be enhanced
                if file_b.replace('.py', '') in imported or imported in file_b:
                    return Relationship(
                        type=RelationType.IMPORT,
                        strength=0.9,
                        confidence=0.8,
                        metadata={'import_type': 'direct', 'language': language}
                    )
                    
        return None
        
    def _check_semantic_similarity(self, node_a: Dict, node_b: Dict, 
                                 embeddings: Tuple[np.ndarray, np.ndarray]) -> Optional[Relationship]:
        """Check semantic similarity using embeddings."""
        emb_a, emb_b = embeddings
        
        # Cosine similarity
        norm_a = np.linalg.norm(emb_a)
        norm_b = np.linalg.norm(emb_b)
        
        if norm_a == 0 or norm_b == 0:
            return None
            
        similarity = np.dot(emb_a, emb_b) / (norm_a * norm_b)
        
        if similarity > self.semantic_threshold:
            return Relationship(
                type=RelationType.SEMANTIC,
                strength=float(similarity),
                confidence=0.9,  # High confidence in embedding quality
                metadata={'similarity_score': float(similarity)}
            )
            
        return None
        
    def _check_temporal(self, node_a: Dict, node_b: Dict) -> Optional[Relationship]:
        """Check temporal relationships based on timestamps."""
        time_a = self._get_timestamp(node_a)
        time_b = self._get_timestamp(node_b)
        
        if time_a and time_b:
            # Calculate time difference in seconds
            time_diff = abs((time_a - time_b).total_seconds())
            
            # Within 1 hour is considered temporally related
            if time_diff < 3600:
                strength = 1.0 - (time_diff / 3600)  # Linear decay
                return Relationship(
                    type=RelationType.TEMPORAL,
                    strength=strength,
                    confidence=1.0,
                    metadata={'time_difference_seconds': time_diff}
                )
                
        return None
        
    def _check_reference(self, node_a: Dict, node_b: Dict) -> Optional[Relationship]:
        """Check for documentation references between nodes."""
        # Look for references in content
        content_a = node_a.get('content', '')
        content_b = node_b.get('content', '')
        name_a = node_a.get('file_name', '')
        name_b = node_b.get('file_name', '')
        
        # Simple reference detection - could be enhanced
        if name_b and name_b in content_a:
            return Relationship(
                type=RelationType.REFERENCE,
                strength=0.8,
                confidence=0.7,
                metadata={'reference_type': 'filename_mention'}
            )
            
        # Check for class/function references
        if self._contains_reference(content_a, content_b):
            return Relationship(
                type=RelationType.REFERENCE,
                strength=0.7,
                confidence=0.6,
                metadata={'reference_type': 'code_reference'}
            )
            
        return None
        
    def _check_structural(self, node_a: Dict, node_b: Dict) -> Optional[Relationship]:
        """Check for structural relationships (e.g., same module, package)."""
        path_a = Path(node_a.get('file_path', ''))
        path_b = Path(node_b.get('file_path', ''))
        
        if not path_a or not path_b:
            return None
            
        # Check if in same package/module (share parent directory)
        if path_a.parent == path_b.parent:
            # Already covered by co-location
            return None
            
        # Check if in related packages (e.g., src/module and tests/module)
        parts_a = path_a.parts
        parts_b = path_b.parts
        
        # Find common structure patterns
        if self._check_test_relationship(parts_a, parts_b):
            return Relationship(
                type=RelationType.STRUCTURAL,
                strength=0.8,
                confidence=0.9,
                metadata={'structure_type': 'test_relationship'}
            )
            
        return None
        
    def _get_language_from_extension(self, ext: str) -> Optional[str]:
        """Get language from file extension."""
        ext_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.jsx': 'javascript',
            '.ts': 'javascript',
            '.tsx': 'javascript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.cc': 'cpp',
            '.cxx': 'cpp',
            '.h': 'cpp',
            '.hpp': 'cpp',
        }
        return ext_map.get(ext)
        
    def _get_timestamp(self, node: Dict) -> Optional[datetime]:
        """Extract timestamp from node metadata."""
        # Try various timestamp fields
        for field in ['created_at', 'modified_at', 'timestamp', 'processed_at']:
            value = node.get(field) or node.get('metadata', {}).get(field)
            if value:
                try:
                    if isinstance(value, str):
                        return datetime.fromisoformat(value.replace('Z', '+00:00'))
                    elif isinstance(value, (int, float)):
                        return datetime.fromtimestamp(value)
                except:
                    continue
        return None
        
    def _contains_reference(self, content_a: str, content_b: str) -> bool:
        """Check if content_a references elements from content_b."""
        # Extract potential identifiers from content_b
        # This is a simplified version - could use AST parsing for accuracy
        identifier_pattern = re.compile(r'\b(?:class|def|function|const|let|var)\s+(\w+)')
        
        identifiers_b = set()
        for match in identifier_pattern.finditer(content_b):
            identifiers_b.add(match.group(1))
            
        # Check if any identifiers appear in content_a
        for identifier in identifiers_b:
            if re.search(r'\b' + identifier + r'\b', content_a):
                return True
                
        return False
        
    def _check_test_relationship(self, parts_a: Tuple, parts_b: Tuple) -> bool:
        """Check if files have a test relationship."""
        # Common test directory patterns
        test_dirs = {'test', 'tests', 'spec', 'specs', '__tests__'}
        
        # Check if one is in test dir and other isn't
        has_test_a = any(part in test_dirs for part in parts_a)
        has_test_b = any(part in test_dirs for part in parts_b)
        
        if has_test_a != has_test_b:
            # Check if they have similar names
            name_a = parts_a[-1].replace('test_', '').replace('_test', '').replace('.test', '')
            name_b = parts_b[-1].replace('test_', '').replace('_test', '').replace('.test', '')
            
            if name_a in name_b or name_b in name_a:
                return True
                
        return False