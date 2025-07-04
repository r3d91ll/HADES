"""
Theory-Practice Bridge Detector

Automatically discovers bridges between theoretical concepts and practical implementations.
"""

import re
import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import ast

from src.types.hades.relationships_v2 import (
    TheoryPracticeBridge, SourceTarget, HadesRelationshipsV2
)

logger = logging.getLogger(__name__)


class BridgeDetector:
    """Detects theory-practice bridges in code and documentation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize bridge detector with configuration."""
        self.config = config
        self.confidence_thresholds = config.get('confidence_thresholds', {
            'citation_match': 0.9,
            'algorithm_name': 0.8,
            'symbol_match': 0.85,
            'fuzzy_match': 0.6
        })
        
        # Common algorithm name patterns
        self.algorithm_patterns = [
            r'pathrag',
            r'isne',
            r'late[\s_-]?chunk',
            r'attention[\s_-]?pool',
            r'skip[\s_-]?gram',
            r'random[\s_-]?walk',
            r'flow[\s_-]?propag',
            r'multi[\s_-]?hop'
        ]
        
        # Citation patterns
        self.citation_patterns = [
            r'#\s*Based on:?\s*(.+)',
            r'#\s*Reference:?\s*(.+)',
            r'#\s*See:?\s*(.+)',
            r'#\s*From:?\s*(.+)',
            r'#\s*\[(\d+)\]',
            r'#\s*\(([A-Z][a-z]+(?:\s+et\s+al\.)?),?\s*(\d{4})\)'
        ]
    
    def detect_bridges_in_code(
        self,
        code_path: Path,
        ast_analysis: Optional[Dict[str, Any]] = None
    ) -> List[TheoryPracticeBridge]:
        """Detect theory-practice bridges in Python code."""
        bridges = []
        
        try:
            content = code_path.read_text(encoding='utf-8')
            
            # 1. Check for citations in comments
            bridges.extend(self._detect_citation_bridges(code_path, content))
            
            # 2. Check for algorithm name matches
            if ast_analysis and 'symbols' in ast_analysis:
                bridges.extend(self._detect_algorithm_bridges(
                    code_path, ast_analysis['symbols']
                ))
            
            # 3. Check docstrings for references
            bridges.extend(self._detect_docstring_bridges(code_path, content))
            
            # 4. Check import statements for theory modules
            bridges.extend(self._detect_import_bridges(code_path, content))
            
        except Exception as e:
            logger.error(f"Error detecting bridges in {code_path}: {e}")
        
        return bridges
    
    def detect_bridges_in_pdf(
        self,
        pdf_path: Path,
        pdf_metadata: Dict[str, Any]
    ) -> List[TheoryPracticeBridge]:
        """Detect theory-practice bridges in PDF documents."""
        bridges = []
        
        try:
            # 1. Look for code references in PDF
            if 'sections' in pdf_metadata:
                for section in pdf_metadata['sections']:
                    # Check for implementation sections
                    if re.search(r'implement|code|algorithm|listing', 
                               section['text'], re.IGNORECASE):
                        bridges.extend(self._create_implementation_bridges(
                            pdf_path, section
                        ))
            
            # 2. Look for algorithm descriptions
            if 'algorithms' in pdf_metadata:
                for algo in pdf_metadata['algorithms']:
                    bridges.extend(self._match_algorithm_to_code(
                        pdf_path, algo
                    ))
            
        except Exception as e:
            logger.error(f"Error detecting bridges in PDF {pdf_path}: {e}")
        
        return bridges
    
    def detect_bridges_in_markdown(
        self,
        md_path: Path,
        md_metadata: Dict[str, Any]
    ) -> List[TheoryPracticeBridge]:
        """Detect theory-practice bridges in Markdown documentation."""
        bridges = []
        
        try:
            # 1. Check API references
            if 'api_refs' in md_metadata:
                for api_ref in md_metadata['api_refs']:
                    bridges.extend(self._create_api_bridges(
                        md_path, api_ref
                    ))
            
            # 2. Check code block examples
            if 'code_blocks' in md_metadata:
                for code_block in md_metadata['code_blocks']:
                    if code_block['language'] in ['python', 'py']:
                        bridges.extend(self._analyze_code_example(
                            md_path, code_block
                        ))
            
            # 3. Check cross-references
            if 'cross_refs' in md_metadata:
                for cross_ref in md_metadata['cross_refs']:
                    bridges.extend(self._create_cross_ref_bridges(
                        md_path, cross_ref
                    ))
            
        except Exception as e:
            logger.error(f"Error detecting bridges in Markdown {md_path}: {e}")
        
        return bridges
    
    def _detect_citation_bridges(
        self,
        code_path: Path,
        content: str
    ) -> List[TheoryPracticeBridge]:
        """Detect citations in code comments."""
        bridges = []
        
        for line_num, line in enumerate(content.split('\n'), 1):
            for pattern in self.citation_patterns:
                match = re.search(pattern, line)
                if match:
                    # Try to resolve citation to actual paper
                    citation = match.group(1) if match.lastindex else match.group(0)
                    paper_path = self._resolve_citation(citation)
                    
                    if paper_path:
                        bridge = TheoryPracticeBridge(
                            source=SourceTarget(
                                type="code",
                                path=str(code_path),
                                section=None,
                                symbol=None,
                                lines=[line_num, line_num]
                            ),
                            target=SourceTarget(
                                type="research_paper",
                                path=paper_path,
                                section=None,
                                symbol=None,
                                lines=None
                            ),
                            relationship="cites",
                            confidence=self.confidence_thresholds['citation_match'],
                            bidirectional=False,
                            notes=f"Citation found: {citation}",
                            metadata=None
                        )
                        bridges.append(bridge)
        
        return bridges
    
    def _detect_algorithm_bridges(
        self,
        code_path: Path,
        symbols: Dict[str, Any]
    ) -> List[TheoryPracticeBridge]:
        """Detect algorithm implementations based on symbol names."""
        bridges = []
        
        for symbol_type in ['classes', 'functions']:
            for symbol in symbols.get(symbol_type, []):
                symbol_name_lower = symbol['name'].lower()
                
                for algo_pattern in self.algorithm_patterns:
                    if re.search(algo_pattern, symbol_name_lower):
                        # Try to find corresponding paper
                        paper_path = self._find_algorithm_paper(algo_pattern)
                        
                        if paper_path:
                            bridge = TheoryPracticeBridge(
                                source=SourceTarget(
                                    type="research_paper",
                                    path=paper_path,
                                    section=f"Algorithm: {algo_pattern}",
                                    symbol=None,
                                    lines=None
                                ),
                                target=SourceTarget(
                                    type="code",
                                    path=str(code_path),
                                    section=None,
                                    symbol=symbol['name'],
                                    lines=symbol.get('line_range', [])
                                ),
                                relationship="algorithm_of",
                                confidence=self.confidence_thresholds['algorithm_name'],
                                bidirectional=False,
                                notes=f"Algorithm pattern match: {algo_pattern}",
                                metadata=None
                            )
                            bridges.append(bridge)
        
        return bridges
    
    def _detect_docstring_bridges(
        self,
        code_path: Path,
        content: str
    ) -> List[TheoryPracticeBridge]:
        """Detect references in docstrings."""
        bridges = []
        
        # Parse AST to find docstrings
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    docstring = ast.get_docstring(node)
                    if docstring:
                        # Look for references in docstring
                        ref_patterns = [
                            r'See:\s*(.+)',
                            r'Ref:\s*(.+)',
                            r'Based on:\s*(.+)',
                            r'Paper:\s*(.+)'
                        ]
                        
                        for pattern in ref_patterns:
                            match = re.search(pattern, docstring, re.IGNORECASE)
                            if match:
                                ref = match.group(1).strip()
                                
                                # Create bridge
                                bridge = TheoryPracticeBridge(
                                    source=SourceTarget(
                                        type="code",
                                        path=str(code_path),
                                        section=None,
                                        symbol=node.name,
                                        lines=[node.lineno, node.end_lineno]
                                    ),
                                    target=SourceTarget(
                                        type="documentation",
                                        path=ref,
                                        section=None,
                                        symbol=None,
                                        lines=None
                                    ),
                                    relationship="references",
                                    confidence=0.7,
                                    bidirectional=False,
                                    notes=f"Reference in docstring",
                                    metadata=None
                                )
                                bridges.append(bridge)
        
        except Exception as e:
            logger.warning(f"Error parsing AST for {code_path}: {e}")
        
        return bridges
    
    def _detect_import_bridges(
        self,
        code_path: Path,
        content: str
    ) -> List[TheoryPracticeBridge]:
        """Detect imports that suggest theory-practice relationships."""
        bridges = []
        
        # Look for imports from theory/research modules
        theory_import_patterns = [
            r'from\s+research\s+import',
            r'from\s+papers\s+import',
            r'from\s+theory\s+import',
            r'import\s+algorithms\.'
        ]
        
        for pattern in theory_import_patterns:
            if re.search(pattern, content):
                # This file likely implements theoretical concepts
                bridge = TheoryPracticeBridge(
                    source=SourceTarget(
                        type="code",
                        path=str(code_path),
                        section=None,
                        symbol=None,
                        lines=None
                    ),
                    target=SourceTarget(
                        type="research_paper",
                        path="research/",  # Generic, to be resolved
                        section=None,
                        symbol=None,
                        lines=None
                    ),
                    relationship="based_on",
                    confidence=0.6,
                    bidirectional=False,
                    notes="Imports suggest theory implementation",
                    metadata=None
                )
                bridges.append(bridge)
        
        return bridges
    
    def _create_implementation_bridges(
        self,
        pdf_path: Path,
        section: Dict[str, Any]
    ) -> List[TheoryPracticeBridge]:
        """Create bridges for implementation sections in papers."""
        bridges = []
        
        # Look for code references in implementation sections
        code_patterns = [
            r'(?:class|def|function)\s+(\w+)',
            r'Algorithm\s+\d+:\s*(\w+)',
            r'Listing\s+\d+:\s*(\w+)'
        ]
        
        section_text = section.get('text', '')
        for pattern in code_patterns:
            matches = re.findall(pattern, section_text)
            for match in matches:
                # Try to find corresponding code
                code_path = self._find_code_implementation(match)
                
                if code_path:
                    bridge = TheoryPracticeBridge(
                        source=SourceTarget(
                            type="research_paper",
                            path=str(pdf_path),
                            section=section_text[:50],
                            symbol=None,
                            lines=None
                        ),
                        target=SourceTarget(
                            type="code",
                            path=code_path,
                            section=None,
                            symbol=match,
                            lines=None
                        ),
                        relationship="implements",
                        confidence=0.7,
                        bidirectional=False,
                        notes=f"Implementation reference in paper",
                        metadata=None
                    )
                    bridges.append(bridge)
        
        return bridges
    
    def _match_algorithm_to_code(
        self,
        pdf_path: Path,
        algorithm: Dict[str, Any]
    ) -> List[TheoryPracticeBridge]:
        """Match algorithm descriptions to code implementations."""
        bridges = []
        
        algo_title = algorithm.get('title', '')
        
        # Extract algorithm name
        algo_name_match = re.search(r'Algorithm\s+\d+:\s*(.+)', algo_title)
        if algo_name_match:
            algo_name = algo_name_match.group(1).strip()
            
            # Find matching code
            code_path = self._find_code_implementation(algo_name)
            
            if code_path:
                bridge = TheoryPracticeBridge(
                    source=SourceTarget(
                        type="research_paper",
                        path=str(pdf_path),
                        section=algo_title,
                        symbol=None,
                        lines=None
                    ),
                    target=SourceTarget(
                        type="code",
                        path=code_path,
                        section=None,
                        symbol=None,
                        lines=None
                    ),
                    relationship="algorithm_of",
                    confidence=0.85,
                    bidirectional=False,
                    notes=f"Algorithm match: {algo_name}",
                    metadata=None
                )
                bridges.append(bridge)
        
        return bridges
    
    def _create_api_bridges(
        self,
        md_path: Path,
        api_ref: Dict[str, Any]
    ) -> List[TheoryPracticeBridge]:
        """Create bridges for API references in documentation."""
        bridges = []
        
        api_name = api_ref.get('name', '')
        
        # Find corresponding code
        code_path = self._find_code_for_api(api_name)
        
        if code_path:
            bridge = TheoryPracticeBridge(
                source=SourceTarget(
                    type="code",
                    path=code_path,
                    section=None,
                    symbol=api_name,
                    lines=None
                ),
                target=SourceTarget(
                    type="documentation",
                    path=str(md_path),
                    section=api_ref.get('context', ''),
                    symbol=None,
                    lines=None
                ),
                relationship="documented_in",
                confidence=0.9,
                bidirectional=True,
                notes=f"API documentation",
                metadata=None
            )
            bridges.append(bridge)
        
        return bridges
    
    def _analyze_code_example(
        self,
        md_path: Path,
        code_block: Dict[str, Any]
    ) -> List[TheoryPracticeBridge]:
        """Analyze code examples in documentation."""
        bridges = []
        
        code_content = code_block.get('content', '')
        
        # Look for imports and class/function usage
        import_pattern = r'from\s+(\S+)\s+import\s+(\w+)'
        usage_pattern = r'(\w+)\s*\('
        
        imports = re.findall(import_pattern, code_content)
        usages = re.findall(usage_pattern, code_content)
        
        # Create bridges for demonstrated usage
        for module, symbol in imports:
            if module.startswith('src.'):
                bridge = TheoryPracticeBridge(
                    source=SourceTarget(
                        type="code",
                        path=module.replace('.', '/') + '.py',
                        section=None,
                        symbol=symbol,
                        lines=None
                    ),
                    target=SourceTarget(
                        type="documentation",
                        path=str(md_path),
                        section=None,
                        symbol=None,
                        lines=None
                    ),
                    relationship="example_in",
                    confidence=0.8,
                    bidirectional=False,
                    notes="Code example in documentation",
                    metadata=None
                )
                bridges.append(bridge)
        
        return bridges
    
    def _create_cross_ref_bridges(
        self,
        md_path: Path,
        cross_ref: Dict[str, Any]
    ) -> List[TheoryPracticeBridge]:
        """Create bridges for cross-references between documents."""
        bridges = []
        
        target_url = cross_ref.get('url', '')
        
        # Only process local references
        if not target_url.startswith('http'):
            bridge = TheoryPracticeBridge(
                source=SourceTarget(
                    type="documentation",
                    path=str(md_path),
                    section=cross_ref.get('context', ''),
                    symbol=None,
                    lines=None
                ),
                target=SourceTarget(
                    type="documentation",
                    path=target_url,
                    section=None,
                    symbol=None,
                    lines=None
                ),
                relationship="references",
                confidence=0.9,
                bidirectional=False,
                notes=f"Cross-reference: {cross_ref.get('text', '')}",
                metadata=None
            )
            bridges.append(bridge)
        
        return bridges
    
    def _resolve_citation(self, citation: str) -> Optional[str]:
        """Resolve a citation to a paper path."""
        # TODO: Implement citation resolution logic
        # For now, return None
        return None
    
    def _find_algorithm_paper(self, algorithm: str) -> Optional[str]:
        """Find paper containing algorithm description."""
        # TODO: Implement paper search logic
        # For now, return common papers
        algo_paper_map = {
            'pathrag': 'research/pathrag.pdf',
            'isne': 'research/isne_paper.pdf',
            'late_chunk': 'research/jina-embeddings-v4.pdf'
        }
        
        for key, paper in algo_paper_map.items():
            if key in algorithm:
                return paper
        
        return None
    
    def _find_code_implementation(self, name: str) -> Optional[str]:
        """Find code file implementing given name."""
        # TODO: Implement code search logic
        # For now, use heuristics
        name_lower = name.lower()
        
        if 'pathrag' in name_lower:
            return 'src/pathrag/pathrag_rag_strategy.py'
        elif 'isne' in name_lower:
            return 'src/isne/models/isne_model.py'
        elif 'jina' in name_lower or 'chunk' in name_lower:
            return 'src/jina_v4/jina_processor.py'
        
        return None
    
    def _find_code_for_api(self, api_name: str) -> Optional[str]:
        """Find code file for API name."""
        # TODO: Implement API to code mapping
        # For now, use simple heuristics
        parts = api_name.split('.')
        
        if len(parts) > 1:
            # Convert module.Class to path
            module_path = '/'.join(parts[:-1])
            return f"src/{module_path}.py"
        
        return None