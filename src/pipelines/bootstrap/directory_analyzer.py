"""
Directory Analyzer for Bootstrap Pipeline

Analyzes directory structure to extract semantic relationships and build initial graph structure.
"""

import logging
import ast
import json
import yaml
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional, Generator
from collections import defaultdict
import re

from .config import BootstrapConfig, EdgeDiscoveryMethod
from src.storage.incremental.sequential_isne_types import FileType, EdgeType, classify_file_type

logger = logging.getLogger(__name__)


class DirectoryAnalyzer:
    """
    Analyzes directory structure for graph bootstrap.
    
    Extracts relationships from:
    - Directory hierarchy and co-location
    - Import statements and file references
    - Configuration file dependencies
    - Documentation-code proximity
    """
    
    def __init__(self, config: BootstrapConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Track discovered files and relationships
        self.discovered_files: Dict[str, Dict[str, Any]] = {}
        self.relationships: List[Tuple[str, str, str, float, Dict[str, Any]]] = []
        
        # Pattern matching for imports and references
        self._setup_patterns()
        
        self.logger.info("Initialized DirectoryAnalyzer")
    
    def _setup_patterns(self) -> None:
        """Setup regex patterns for analyzing file content."""
        # Python import patterns
        self.python_import_patterns = [
            re.compile(r'^\s*import\s+([a-zA-Z_][a-zA-Z0-9_.]*)', re.MULTILINE),
            re.compile(r'^\s*from\s+([a-zA-Z_][a-zA-Z0-9_.]*)\s+import', re.MULTILINE),
        ]
        
        # JavaScript/TypeScript import patterns
        self.js_import_patterns = [
            re.compile(r'import\s+.*?\s+from\s+[\'"]([^\'"]+)[\'"]', re.MULTILINE),
            re.compile(r'require\s*\(\s*[\'"]([^\'"]+)[\'"]\s*\)', re.MULTILINE),
        ]
        
        # File reference patterns (generic)
        self.file_reference_patterns = [
            re.compile(r'[\'"]([^\'"\s]+\.(py|js|ts|md|json|yaml|yml))[\'"]'),
            re.compile(r'(?:src|include|import)\s*[\'"]([^\'"\s]+)[\'"]'),
        ]
        
        # Configuration dependency patterns
        self.config_dependency_patterns = [
            re.compile(r'[\'"]?(\w+)[\'"]?\s*:\s*[\'"]([^\'"\s]+\.(py|js|ts|json|yaml))[\'"]'),
        ]
    
    def analyze_directory(self, root_path: Path) -> Tuple[Dict[str, Dict[str, Any]], List[Tuple[str, str, str, float, Dict[str, Any]]]]:
        """
        Analyze directory structure and extract files and relationships.
        
        Args:
            root_path: Root directory to analyze
            
        Returns:
            Tuple of (discovered_files, relationships)
        """
        self.logger.info(f"Analyzing directory structure: {root_path}")
        
        # Phase 1: Discover and classify files
        self._discover_files(root_path)
        
        # Phase 2: Extract relationships using enabled methods
        if EdgeDiscoveryMethod.DIRECTORY_STRUCTURE in self.config.edge_discovery_methods:
            self._analyze_directory_structure()
        
        if EdgeDiscoveryMethod.IMPORT_ANALYSIS in self.config.edge_discovery_methods:
            self._analyze_imports()
        
        if EdgeDiscoveryMethod.FILE_REFERENCES in self.config.edge_discovery_methods:
            self._analyze_file_references()
        
        if EdgeDiscoveryMethod.CO_LOCATION in self.config.edge_discovery_methods:
            self._analyze_co_location()
        
        self.logger.info(f"Discovered {len(self.discovered_files)} files and {len(self.relationships)} relationships")
        
        return self.discovered_files, self.relationships
    
    def _discover_files(self, root_path: Path) -> None:
        """Discover and classify files in the directory."""
        self.logger.info("Discovering files...")
        
        for file_path in root_path.rglob('*'):
            if not file_path.is_file():
                continue
            
            # Apply filters
            if not self._should_include_file(file_path):
                continue
            
            # Check file size
            try:
                if file_path.stat().st_size > self.config.max_file_size:
                    self.logger.debug(f"Skipping large file: {file_path}")
                    continue
            except OSError:
                continue
            
            # Classify file and extract metadata
            file_info = self._extract_file_info(file_path, root_path)
            if file_info:
                self.discovered_files[str(file_path)] = file_info
        
        self.logger.info(f"Discovered {len(self.discovered_files)} files")
    
    def _should_include_file(self, file_path: Path) -> bool:
        """Check if file should be included based on filters."""
        # Check exclude patterns
        for pattern in self.config.exclude_patterns:
            if file_path.match(pattern):
                return False
        
        # Check extension filter
        if file_path.suffix.lower() not in self.config.include_extensions:
            return False
        
        # Check file type filter
        file_type = classify_file_type(file_path)
        if self.config.file_type_filter == "code_only" and file_type != FileType.CODE:
            return False
        elif self.config.file_type_filter == "docs_only" and file_type != FileType.DOCUMENTATION:
            return False
        elif self.config.file_type_filter == "config_only" and file_type != FileType.CONFIG:
            return False
        elif self.config.file_type_filter == "code_and_docs" and file_type not in [FileType.CODE, FileType.DOCUMENTATION]:
            return False
        
        return True
    
    def _extract_file_info(self, file_path: Path, root_path: Path) -> Optional[Dict[str, Any]]:
        """Extract metadata and basic information about a file."""
        try:
            stat = file_path.stat()
            relative_path = file_path.relative_to(root_path)
            
            # Basic file information
            file_info = {
                'file_path': str(file_path),
                'relative_path': str(relative_path),
                'file_name': file_path.name,
                'directory': str(file_path.parent),
                'relative_directory': str(relative_path.parent),
                'extension': file_path.suffix.lower(),
                'size': stat.st_size,
                'modified_time': stat.st_mtime,
                'directory_depth': len(relative_path.parts) - 1,
                'file_type': classify_file_type(file_path),
            }
            
            # Try to read content for analysis
            try:
                content = self._read_file_content(file_path)
                if content:
                    file_info.update({
                        'content_hash': hash(content),
                        'content_length': len(content),
                        'line_count': content.count('\n') + 1,
                        'has_content': True
                    })
                    
                    # Extract content-specific metadata
                    content_metadata = self._extract_content_metadata(file_path, content)
                    file_info.update(content_metadata)
                else:
                    file_info['has_content'] = False
                    
            except Exception as e:
                self.logger.debug(f"Could not read content for {file_path}: {e}")
                file_info['has_content'] = False
            
            return file_info
            
        except Exception as e:
            self.logger.warning(f"Failed to extract info for {file_path}: {e}")
            return None
    
    def _read_file_content(self, file_path: Path) -> Optional[str]:
        """Safely read file content."""
        try:
            # Try UTF-8 first
            return file_path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            try:
                # Fallback to latin-1
                return file_path.read_text(encoding='latin-1')
            except Exception:
                return None
        except Exception:
            return None
    
    def _extract_content_metadata(self, file_path: Path, content: str) -> Dict[str, Any]:
        """Extract metadata specific to file content."""
        metadata = {}
        file_type = classify_file_type(file_path)
        
        if file_type == FileType.CODE:
            metadata.update(self._analyze_code_content(file_path, content))
        elif file_type == FileType.DOCUMENTATION:
            metadata.update(self._analyze_doc_content(content))
        elif file_type == FileType.CONFIG:
            metadata.update(self._analyze_config_content(file_path, content))
        
        return metadata
    
    def _analyze_code_content(self, file_path: Path, content: str) -> Dict[str, Any]:
        """Analyze code file content."""
        metadata = {}
        
        # Count lines of code (excluding empty lines and comments)
        lines = content.split('\n')
        code_lines = 0
        comment_lines = 0
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            elif stripped.startswith('#') or stripped.startswith('//'):
                comment_lines += 1
            else:
                code_lines += 1
        
        metadata.update({
            'lines_of_code': code_lines,
            'comment_lines': comment_lines,
            'complexity_score': min(code_lines / 100.0, 1.0)  # Simple complexity metric
        })
        
        # Extract imports and functions (Python-specific for now)
        if file_path.suffix == '.py':
            try:
                tree = ast.parse(content)
                
                imports = []
                functions = []
                classes = []
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        imports.extend([alias.name for alias in node.names])
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            imports.append(node.module)
                    elif isinstance(node, ast.FunctionDef):
                        functions.append({
                            'name': node.name,
                            'line_number': node.lineno,
                            'args': [arg.arg for arg in node.args.args]
                        })
                    elif isinstance(node, ast.ClassDef):
                        classes.append({
                            'name': node.name,
                            'line_number': node.lineno,
                            'bases': [base.id if isinstance(base, ast.Name) else str(base) for base in node.bases]
                        })
                
                metadata.update({
                    'imports': imports,
                    'functions': functions,
                    'classes': classes
                })
                
            except SyntaxError:
                # File has syntax errors, skip AST analysis
                pass
        
        return metadata
    
    def _analyze_doc_content(self, content: str) -> Dict[str, Any]:
        """Analyze documentation content."""
        lines = content.split('\n')
        
        # Count words and headings
        word_count = len(content.split())
        
        # Count markdown headings
        headings = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('#'):
                level = len(stripped) - len(stripped.lstrip('#'))
                headings.append({'level': level, 'text': stripped.lstrip('#').strip()})
        
        # Extract links (markdown format)
        link_pattern = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
        links = link_pattern.findall(content)
        
        return {
            'word_count': word_count,
            'headings': headings,
            'links': [{'text': text, 'url': url} for text, url in links],
            'readability_score': min(word_count / 1000.0, 1.0)  # Simple readability metric
        }
    
    def _analyze_config_content(self, file_path: Path, content: str) -> Dict[str, Any]:
        """Analyze configuration file content."""
        metadata = {}
        
        try:
            if file_path.suffix in ['.json']:
                parsed = json.loads(content)
                metadata['parsed_config'] = parsed
                metadata['config_keys'] = list(parsed.keys()) if isinstance(parsed, dict) else []
            elif file_path.suffix in ['.yaml', '.yml']:
                parsed = yaml.safe_load(content)
                metadata['parsed_config'] = parsed
                metadata['config_keys'] = list(parsed.keys()) if isinstance(parsed, dict) else []
        except Exception:
            # Couldn't parse config file
            pass
        
        return metadata
    
    def _analyze_directory_structure(self) -> None:
        """Analyze directory hierarchy relationships."""
        self.logger.info("Analyzing directory structure...")
        
        # Group files by directory
        directories = defaultdict(list)
        for file_path, file_info in self.discovered_files.items():
            directories[file_info['directory']].append(file_path)
        
        # Create parent-child directory relationships
        for directory, files in directories.items():
            parent_dir = str(Path(directory).parent)
            if parent_dir in directories:
                # Add directory hierarchy edge
                for parent_file in directories[parent_dir]:
                    for child_file in files:
                        if parent_file != child_file:
                            self.relationships.append((
                                parent_file,
                                child_file,
                                EdgeType.DIRECTORY_HIERARCHY.value,
                                self.config.directory_hierarchy_weight,
                                {'source': 'directory_structure', 'type': 'parent_child'}
                            ))
    
    def _analyze_imports(self) -> None:
        """Analyze import statements and dependencies."""
        self.logger.info("Analyzing import statements...")
        
        for file_path, file_info in self.discovered_files.items():
            if not file_info.get('has_content'):
                continue
            
            # Get imports from metadata if available
            imports = file_info.get('imports', [])
            
            # Find corresponding files for imports
            for import_name in imports:
                target_files = self._resolve_import(import_name, file_path)
                for target_file in target_files:
                    if target_file in self.discovered_files:
                        self.relationships.append((
                            file_path,
                            target_file,
                            EdgeType.CODE_IMPORTS.value,
                            self.config.import_weight,
                            {'source': 'import_analysis', 'import_name': import_name}
                        ))
    
    def _analyze_file_references(self) -> None:
        """Analyze explicit file references in content."""
        self.logger.info("Analyzing file references...")
        
        for file_path, file_info in self.discovered_files.items():
            if not file_info.get('has_content'):
                continue
            
            try:
                content = self._read_file_content(Path(file_path))
                if not content:
                    continue
                
                # Look for file references
                for pattern in self.file_reference_patterns:
                    matches = pattern.findall(content)
                    for match in matches:
                        referenced_file = match if isinstance(match, str) else match[0]
                        target_files = self._resolve_file_reference(referenced_file, file_path)
                        
                        for target_file in target_files:
                            if target_file in self.discovered_files:
                                self.relationships.append((
                                    file_path,
                                    target_file,
                                    EdgeType.DOC_REFERENCES.value,
                                    0.7,  # Medium confidence for file references
                                    {'source': 'file_reference', 'referenced_path': referenced_file}
                                ))
                                
            except Exception as e:
                self.logger.debug(f"Error analyzing file references in {file_path}: {e}")
    
    def _analyze_co_location(self) -> None:
        """Analyze file co-location relationships."""
        self.logger.info("Analyzing co-location relationships...")
        
        # Group files by directory
        directories = defaultdict(list)
        for file_path, file_info in self.discovered_files.items():
            directories[file_info['relative_directory']].append(file_path)
        
        # Create co-location edges within directories
        for directory, files in directories.items():
            if len(files) > 1:
                for i, file1 in enumerate(files):
                    for file2 in files[i+1:]:
                        # Check if different modalities (cross-modal)
                        file1_type = self.discovered_files[file1]['file_type']
                        file2_type = self.discovered_files[file2]['file_type']
                        
                        if file1_type != file2_type:
                            # Cross-modal co-location (stronger signal)
                            edge_type = self._get_cross_modal_edge_type(file1_type, file2_type)
                            weight = self.config.co_location_weight * 1.2  # Boost cross-modal
                        else:
                            # Same modality co-location
                            edge_type = EdgeType.DIRECTORY_COLOCATION.value
                            weight = self.config.co_location_weight
                        
                        self.relationships.append((
                            file1,
                            file2,
                            edge_type,
                            weight,
                            {'source': 'co_location', 'directory': directory}
                        ))
    
    def _resolve_import(self, import_name: str, source_file: str) -> List[str]:
        """Resolve import statement to actual file paths."""
        candidates = []
        source_path = Path(source_file)
        
        # Convert import name to potential file paths
        import_parts = import_name.split('.')
        
        # Try different resolution strategies
        for file_path in self.discovered_files:
            file_parts = Path(file_path).stem.split('_')
            
            # Check if file name matches import
            if any(part in import_parts for part in file_parts):
                candidates.append(file_path)
            
            # Check if directory structure matches
            path_parts = Path(file_path).parts
            if any(part in import_parts for part in path_parts):
                candidates.append(file_path)
        
        return candidates
    
    def _resolve_file_reference(self, referenced_file: str, source_file: str) -> List[str]:
        """Resolve file reference to actual file paths."""
        candidates = []
        source_path = Path(source_file)
        
        # Try relative to source file
        relative_path = source_path.parent / referenced_file
        if str(relative_path) in self.discovered_files:
            candidates.append(str(relative_path))
        
        # Try exact match
        for file_path in self.discovered_files:
            if Path(file_path).name == referenced_file:
                candidates.append(file_path)
        
        return candidates
    
    def _get_cross_modal_edge_type(self, type1: str, type2: str) -> str:
        """Get appropriate cross-modal edge type."""
        type_mapping = {
            (FileType.CODE, FileType.DOCUMENTATION): EdgeType.CODE_TO_DOC.value,
            (FileType.DOCUMENTATION, FileType.CODE): EdgeType.DOC_TO_CODE.value,
            (FileType.CODE, FileType.CONFIG): EdgeType.CODE_TO_CONFIG.value,
            (FileType.CONFIG, FileType.CODE): EdgeType.CONFIG_TO_CODE.value,
            (FileType.DOCUMENTATION, FileType.CONFIG): EdgeType.DOC_TO_CONFIG.value,
        }
        
        return type_mapping.get((type1, type2), EdgeType.SEMANTIC_SIMILARITY.value)