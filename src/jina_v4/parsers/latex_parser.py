"""
LaTeX Parser for Jina v4

Handles LaTeX documents with special focus on academic papers, extracting
equations, citations, theorems, and creating theory-practice bridges.
"""

import re
import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)


class LaTeXParser:
    """Parse LaTeX files preserving mathematical and academic structure."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize LaTeX parser with configuration."""
        self.config = config or {}
        self.bibliography: Dict[str, Any] = {}
        self.labels: Dict[str, Dict[str, str]] = {}
        self.equations: List[Dict[str, Any]] = []
        self.theorems: List[Dict[str, Any]] = []
        self.algorithms: List[Dict[str, Any]] = []
        self.figures: List[Dict[str, Any]] = []
        self.sections: List[Dict[str, Any]] = []
        
    def parse(self, file_path: Path) -> Dict[str, Any]:
        """
        Parse LaTeX file and extract structure.
        
        Returns:
            Dictionary containing parsed content, equations, citations, and bridges
        """
        try:
            content = file_path.read_text(encoding='utf-8')
            
            # Detect document type
            doc_type = self._detect_document_type(content, file_path)
            
            # Extract document structure
            structure = self._extract_structure(content, doc_type)
            
            # Extract mathematical content
            math_content = self._extract_mathematical_content(content)
            
            # Extract citations and bibliography
            citations = self._extract_citations(content)
            
            # Extract cross-references
            self._extract_labels_and_refs(content)
            
            # Detect bridges to implementations
            bridges = self._detect_bridges(structure, math_content, doc_type)
            
            # Generate structured text
            structured_text = self._generate_structured_text(
                content, structure, math_content, citations
            )
            
            return {
                'content': structured_text,
                'file_type': 'latex',
                'doc_type': doc_type,
                'structure': structure,
                'math_content': math_content,
                'citations': citations,
                'bridges': bridges,
                'metadata': {
                    'equation_count': len(self.equations),
                    'theorem_count': len(self.theorems),
                    'algorithm_count': len(self.algorithms),
                    'citation_count': len(citations),
                    'labels': self.labels,
                    'packages': self._extract_packages(content)
                }
            }
            
        except Exception as e:
            logger.error(f"Error parsing LaTeX file {file_path}: {e}")
            return {
                'content': file_path.read_text(encoding='utf-8'),
                'file_type': 'latex',
                'doc_type': 'unknown',
                'structure': {},
                'bridges': [],
                'metadata': {'error': str(e)}
            }
    
    def _detect_document_type(self, content: str, file_path: Path) -> str:
        """Detect the type of LaTeX document."""
        # Check document class
        doc_class_match = re.search(r'\\documentclass(?:\[.*?\])?\{(\w+)\}', content)
        if doc_class_match:
            doc_class = doc_class_match.group(1)
            if doc_class in ['article', 'paper']:
                return 'research_paper'
            elif doc_class == 'beamer':
                return 'presentation'
            elif doc_class in ['book', 'report']:
                return 'book'
            elif doc_class == 'thesis':
                return 'thesis'
        
        # Check filename patterns
        filename = file_path.name.lower()
        if 'paper' in filename or 'article' in filename:
            return 'research_paper'
        elif 'slides' in filename or 'presentation' in filename:
            return 'presentation'
        elif 'thesis' in filename:
            return 'thesis'
        
        return 'general_document'
    
    def _extract_structure(self, content: str, doc_type: str) -> Dict[str, Any]:
        """Extract document structure."""
        structure = {
            'title': self._extract_title(content),
            'authors': self._extract_authors(content),
            'abstract': self._extract_abstract(content),
            'sections': self._extract_sections(content),
            'document_class': self._extract_document_class(content)
        }
        
        # Extract specific elements based on document type
        if doc_type == 'research_paper':
            structure['keywords'] = self._extract_keywords(content)
            structure['contributions'] = self._extract_contributions(content)
        
        return structure
    
    def _extract_mathematical_content(self, content: str) -> Dict[str, Any]:
        """Extract equations, theorems, and other mathematical content."""
        math_content = {
            'equations': self._extract_equations(content),
            'theorems': self._extract_theorems(content),
            'algorithms': self._extract_algorithms(content),
            'definitions': self._extract_definitions(content),
            'lemmas': self._extract_lemmas(content),
            'proofs': self._extract_proofs(content)
        }
        
        return math_content
    
    def _extract_equations(self, content: str) -> List[Dict[str, Any]]:
        """Extract all equations from the document."""
        equations = []
        
        # Inline math: $...$
        inline_pattern = r'\$([^\$]+)\$'
        for match in re.finditer(inline_pattern, content):
            equations.append({
                'type': 'inline',
                'content': match.group(1),
                'position': match.start()
            })
        
        # Display math: \[...\] or $$...$$
        display_patterns = [
            (r'\\\[(.*?)\\\]', 'display'),
            (r'\$\$(.*?)\$\$', 'display'),
            (r'\\begin\{equation\}(.*?)\\end\{equation\}', 'equation'),
            (r'\\begin\{equation\*\}(.*?)\\end\{equation\*\}', 'equation*'),
            (r'\\begin\{align\}(.*?)\\end\{align\}', 'align'),
            (r'\\begin\{align\*\}(.*?)\\end\{align\*\}', 'align*')
        ]
        
        for pattern, eq_type in display_patterns:
            for match in re.finditer(pattern, content, re.DOTALL):
                eq_content = match.group(1).strip()
                
                # Check for label
                label_match = re.search(r'\\label\{([^}]+)\}', eq_content)
                label = label_match.group(1) if label_match else None
                
                # Clean equation content
                eq_content = re.sub(r'\\label\{[^}]+\}', '', eq_content).strip()
                
                equations.append({
                    'type': eq_type,
                    'content': eq_content,
                    'label': label,
                    'position': match.start()
                })
                
                # Store for cross-referencing
                if label:
                    self.labels[label] = {
                        'type': 'equation',
                        'content': eq_content[:50] + '...' if len(eq_content) > 50 else eq_content
                    }
        
        self.equations = equations
        return equations
    
    def _extract_theorems(self, content: str) -> List[Dict[str, Any]]:
        """Extract theorems, propositions, and corollaries."""
        theorems = []
        
        theorem_envs = [
            'theorem', 'proposition', 'lemma', 'corollary', 
            'definition', 'remark', 'example'
        ]
        
        for env in theorem_envs:
            pattern = rf'\\begin\{{{env}\}}(?:\[([^\]]*)\])?(.*?)\\end\{{{env}\}}'
            for match in re.finditer(pattern, content, re.DOTALL):
                title = match.group(1) if match.group(1) else ''
                theorem_content = match.group(2).strip()
                
                # Extract label if present
                label_match = re.search(r'\\label\{([^}]+)\}', theorem_content)
                label = label_match.group(1) if label_match else None
                
                theorems.append({
                    'type': env,
                    'title': title,
                    'content': theorem_content,
                    'label': label,
                    'position': match.start()
                })
                
                if label:
                    self.labels[label] = {
                        'type': env,
                        'title': title or env.capitalize()
                    }
        
        self.theorems = theorems
        return theorems
    
    def _extract_algorithms(self, content: str) -> List[Dict[str, Any]]:
        """Extract algorithm blocks."""
        algorithms = []
        
        # Algorithm environment
        pattern = r'\\begin\{algorithm\}(?:\[.*?\])?(.*?)\\end\{algorithm\}'
        for match in re.finditer(pattern, content, re.DOTALL):
            algo_content = match.group(1)
            
            # Extract caption
            caption_match = re.search(r'\\caption\{([^}]+)\}', algo_content)
            caption = caption_match.group(1) if caption_match else ''
            
            # Extract label
            label_match = re.search(r'\\label\{([^}]+)\}', algo_content)
            label = label_match.group(1) if label_match else None
            
            # Extract algorithmic content
            algo_body = ''
            algo_match = re.search(r'\\begin\{algorithmic\}(.*?)\\end\{algorithmic\}', 
                                 algo_content, re.DOTALL)
            if algo_match:
                algo_body = algo_match.group(1).strip()
            
            algorithms.append({
                'caption': caption,
                'label': label,
                'content': algo_body,
                'position': match.start()
            })
            
            if label:
                self.labels[label] = {
                    'type': 'algorithm',
                    'caption': caption
                }
        
        self.algorithms = algorithms
        return algorithms
    
    def _extract_citations(self, content: str) -> List[Dict[str, Any]]:
        """Extract citations from the document."""
        citations = []
        
        # \cite{key} or \cite{key1,key2}
        cite_pattern = r'\\cite(?:\[[^\]]*\])?\{([^}]+)\}'
        for match in re.finditer(cite_pattern, content):
            keys = [k.strip() for k in match.group(1).split(',')]
            for key in keys:
                citations.append({
                    'key': key,
                    'position': match.start(),
                    'context': content[max(0, match.start()-50):match.end()+50]
                })
        
        # Extract bibliography if present
        bib_pattern = r'\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}(.*?)(?=\\bibitem|\\end\{thebibliography\})'
        for match in re.finditer(bib_pattern, content, re.DOTALL):
            key = match.group(1)
            reference = match.group(2).strip()
            self.bibliography[key] = reference
        
        return citations
    
    def _extract_labels_and_refs(self, content: str) -> None:
        """Extract all labels and references for cross-referencing."""
        # Labels (already partially extracted above)
        label_pattern = r'\\label\{([^}]+)\}'
        for match in re.finditer(label_pattern, content):
            label = match.group(1)
            if label not in self.labels:
                # Try to determine context
                context = content[max(0, match.start()-100):match.start()]
                if '\\section' in context:
                    self.labels[label] = {'type': 'section'}
                elif '\\figure' in context:
                    self.labels[label] = {'type': 'figure'}
                elif '\\table' in context:
                    self.labels[label] = {'type': 'table'}
                else:
                    self.labels[label] = {'type': 'unknown'}
    
    def _detect_bridges(
        self, 
        structure: Dict[str, Any],
        math_content: Dict[str, Any],
        doc_type: str
    ) -> List[Dict[str, Any]]:
        """Detect bridges between LaTeX content and implementations."""
        bridges = []
        
        # Algorithm bridges
        for algo in math_content.get('algorithms', []):
            if algo['caption']:
                # Extract algorithm name
                algo_name = self._extract_algorithm_name(algo['caption'])
                if algo_name:
                    bridges.append({
                        'type': 'algorithm_implementation',
                        'source': f"Algorithm: {algo_name}",
                        'target': self._suggest_implementation_name(algo_name),
                        'target_type': 'function',
                        'confidence': 0.8,
                        'label': algo.get('label')
                    })
        
        # Equation bridges (for key equations)
        for eq in math_content.get('equations', []):
            if eq.get('label'):
                # Check if it's a significant equation (has a label)
                bridges.append({
                    'type': 'equation_implementation',
                    'source': f"Equation: {eq['label']}",
                    'target_type': 'numerical_method',
                    'confidence': 0.6,
                    'content_preview': eq['content'][:50]
                })
        
        # Theorem/proof bridges
        for theorem in math_content.get('theorems', []):
            if theorem['type'] in ['theorem', 'proposition'] and theorem.get('label'):
                bridges.append({
                    'type': 'theorem_verification',
                    'source': f"{theorem['type'].capitalize()}: {theorem.get('title', theorem['label'])}",
                    'target_type': 'test_case',
                    'confidence': 0.7
                })
        
        # Citation bridges (to other papers/code)
        if doc_type == 'research_paper':
            # Look for implementation-related keywords in title/abstract
            title = structure.get('title', '')
            abstract = structure.get('abstract', '')
            
            impl_keywords = ['implementation', 'system', 'framework', 'tool', 'library']
            if any(keyword in title.lower() or keyword in abstract.lower() for keyword in impl_keywords):
                bridges.append({
                    'type': 'paper_implementation',
                    'source': title,
                    'target_type': 'codebase',
                    'confidence': 0.85
                })
        
        return bridges
    
    def _generate_structured_text(
        self,
        content: str,
        structure: Dict[str, Any],
        math_content: Dict[str, Any],
        citations: List[Dict[str, Any]]
    ) -> str:
        """Generate structured text representation."""
        lines = []
        
        # Header
        lines.append("[LaTeX Document Analysis]")
        lines.append(f"Type: {structure.get('document_class', 'unknown')}")
        
        if structure.get('title'):
            lines.append(f"Title: {structure['title']}")
        if structure.get('authors'):
            lines.append(f"Authors: {', '.join(structure['authors'])}")
        
        # Statistics
        lines.append(f"\n=== Statistics ===")
        lines.append(f"Equations: {len(math_content.get('equations', []))}")
        lines.append(f"Theorems/Lemmas: {len(math_content.get('theorems', []))}")
        lines.append(f"Algorithms: {len(math_content.get('algorithms', []))}")
        lines.append(f"Citations: {len(citations)}")
        
        # Abstract
        if structure.get('abstract'):
            lines.append(f"\n=== Abstract ===")
            lines.append(structure['abstract'][:500] + '...' if len(structure['abstract']) > 500 else structure['abstract'])
        
        # Key equations
        if math_content.get('equations'):
            lines.append(f"\n=== Key Equations ===")
            for eq in math_content['equations'][:5]:  # First 5 equations
                if eq.get('label'):
                    lines.append(f"[{eq['label']}] {eq['content'][:100]}")
        
        # Algorithms
        if math_content.get('algorithms'):
            lines.append(f"\n=== Algorithms ===")
            for algo in math_content['algorithms']:
                lines.append(f"- {algo['caption']}")
        
        # Section structure
        if structure.get('sections'):
            lines.append(f"\n=== Document Structure ===")
            for section in structure['sections'][:10]:  # First 10 sections
                indent = "  " * (section['level'] - 1)
                lines.append(f"{indent}{section['number']} {section['title']}")
        
        return '\n'.join(lines)
    
    # Helper methods
    
    def _extract_title(self, content: str) -> str:
        """Extract document title."""
        match = re.search(r'\\title\{([^}]+)\}', content)
        return match.group(1) if match else ''
    
    def _extract_authors(self, content: str) -> List[str]:
        """Extract author names."""
        authors = []
        match = re.search(r'\\author\{([^}]+)\}', content)
        if match:
            # Handle multiple authors separated by \and
            author_text = match.group(1)
            authors = [a.strip() for a in re.split(r'\\and', author_text)]
        return authors
    
    def _extract_abstract(self, content: str) -> str:
        """Extract abstract."""
        match = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', content, re.DOTALL)
        return match.group(1).strip() if match else ''
    
    def _extract_keywords(self, content: str) -> List[str]:
        """Extract keywords if present."""
        keywords = []
        match = re.search(r'\\keywords\{([^}]+)\}', content)
        if match:
            keywords = [k.strip() for k in match.group(1).split(',')]
        return keywords
    
    def _extract_sections(self, content: str) -> List[Dict[str, Any]]:
        """Extract section structure."""
        sections = []
        section_pattern = r'\\(section|subsection|subsubsection|paragraph)\*?\{([^}]+)\}'
        
        section_levels = {
            'section': 1,
            'subsection': 2,
            'subsubsection': 3,
            'paragraph': 4
        }
        
        section_counters = {1: 0, 2: 0, 3: 0, 4: 0}
        
        for match in re.finditer(section_pattern, content):
            level_name = match.group(1)
            title = match.group(2)
            level = section_levels.get(level_name, 1)
            
            # Update counters
            section_counters[level] += 1
            for l in range(level + 1, 5):
                section_counters[l] = 0
            
            # Build section number
            number_parts = []
            for l in range(1, level + 1):
                if section_counters[l] > 0:
                    number_parts.append(str(section_counters[l]))
            number = '.'.join(number_parts)
            
            sections.append({
                'level': level,
                'number': number,
                'title': title,
                'type': level_name
            })
        
        self.sections = sections
        return sections
    
    def _extract_packages(self, content: str) -> List[str]:
        """Extract used packages."""
        packages = []
        package_pattern = r'\\usepackage(?:\[[^\]]*\])?\{([^}]+)\}'
        for match in re.finditer(package_pattern, content):
            packages.extend([p.strip() for p in match.group(1).split(',')])
        return packages
    
    def _extract_document_class(self, content: str) -> str:
        """Extract document class."""
        match = re.search(r'\\documentclass(?:\[[^\]]*\])?\{([^}]+)\}', content)
        return match.group(1) if match else 'article'
    
    def _extract_definitions(self, content: str) -> List[Dict[str, Any]]:
        """Extract definitions."""
        return self._extract_theorem_like(content, 'definition')
    
    def _extract_lemmas(self, content: str) -> List[Dict[str, Any]]:
        """Extract lemmas."""
        return self._extract_theorem_like(content, 'lemma')
    
    def _extract_proofs(self, content: str) -> List[Dict[str, Any]]:
        """Extract proofs."""
        proofs = []
        pattern = r'\\begin\{proof\}(?:\[([^\]]*)\])?(.*?)\\end\{proof\}'
        for match in re.finditer(pattern, content, re.DOTALL):
            proofs.append({
                'type': 'proof',
                'title': match.group(1) if match.group(1) else 'Proof',
                'content': match.group(2).strip(),
                'position': match.start()
            })
        return proofs
    
    def _extract_theorem_like(self, content: str, env_type: str) -> List[Dict[str, Any]]:
        """Generic extractor for theorem-like environments."""
        items = []
        pattern = rf'\\begin\{{{env_type}\}}(?:\[([^\]]*)\])?(.*?)\\end\{{{env_type}\}}'
        for match in re.finditer(pattern, content, re.DOTALL):
            items.append({
                'type': env_type,
                'title': match.group(1) if match.group(1) else '',
                'content': match.group(2).strip(),
                'position': match.start()
            })
        return items
    
    def _extract_contributions(self, content: str) -> List[str]:
        """Extract main contributions if listed."""
        contributions = []
        
        # Look for itemize/enumerate after "contributions" keyword
        contrib_section = re.search(
            r'contributions?:?\s*\\begin\{(itemize|enumerate)\}(.*?)\\end\{\1\}',
            content, re.IGNORECASE | re.DOTALL
        )
        
        if contrib_section:
            items = re.findall(r'\\item\s*(.+?)(?=\\item|\\end)', contrib_section.group(2), re.DOTALL)
            contributions = [item.strip() for item in items]
        
        return contributions
    
    def _extract_algorithm_name(self, caption: str) -> Optional[str]:
        """Extract algorithm name from caption."""
        # Common patterns: "Algorithm for X", "X Algorithm", "The X Algorithm"
        patterns = [
            r'(?:Algorithm for|The)\s+(\w+(?:\s+\w+)*)',
            r'(\w+(?:\s+\w+)*)\s+Algorithm',
            r'Algorithm:\s*(\w+(?:\s+\w+)*)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, caption, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # If no pattern matches, use first few words
        words = caption.split()[:3]
        return ' '.join(words) if words else None
    
    def _suggest_implementation_name(self, algo_name: str) -> str:
        """Suggest a likely implementation function name."""
        # Convert to snake_case
        name = algo_name.lower()
        name = re.sub(r'\s+', '_', name)
        name = re.sub(r'[^\w_]', '', name)
        return name