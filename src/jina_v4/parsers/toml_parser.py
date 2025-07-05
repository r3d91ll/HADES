"""
TOML Parser for Jina v4

Handles TOML configuration files with section hierarchy preservation and
bridge detection for Rust/Python project configurations.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import toml  # type: ignore[import-untyped]
import re
from collections import defaultdict

logger = logging.getLogger(__name__)


class TOMLParser:
    """Parse TOML files preserving hierarchical structure and detecting bridges."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize TOML parser with configuration."""
        self.config = config or {}
        self.purpose_patterns = {
            'cargo': ['package', 'dependencies', 'dev-dependencies', 'build-dependencies'],
            'pyproject': ['project', 'tool.poetry', 'tool.setuptools', 'build-system'],
            'config': ['server', 'database', 'logging', 'features'],
            'workspace': ['workspace', 'members', 'exclude']
        }
        
    def parse(self, file_path: Path) -> Dict[str, Any]:
        """
        Parse TOML file and extract structure.
        
        Returns:
            Dictionary containing parsed content, hierarchy, and bridges
        """
        try:
            # Load TOML content
            with open(file_path, 'r', encoding='utf-8') as f:
                toml_data = toml.load(f)
            
            # Also read raw content for structure analysis
            raw_content = file_path.read_text(encoding='utf-8')
            
            # Detect TOML purpose
            purpose = self._detect_toml_purpose(toml_data, file_path)
            
            # Extract hierarchical structure
            hierarchy = self._extract_hierarchy(toml_data, purpose)
            
            # Extract metadata based on purpose
            metadata = self._extract_metadata(toml_data, purpose)
            
            # Detect bridges based on content
            bridges = self._detect_bridges(toml_data, purpose, file_path)
            
            # Generate structured text representation
            structured_text = self._generate_structured_text(
                toml_data, purpose, hierarchy, metadata
            )
            
            return {
                'content': structured_text,
                'file_type': 'toml',
                'purpose': purpose,
                'hierarchy': hierarchy,
                'metadata': metadata,
                'bridges': bridges,
                'raw_data': toml_data
            }
            
        except Exception as e:
            logger.error(f"Error parsing TOML file {file_path}: {e}")
            return {
                'content': file_path.read_text(encoding='utf-8'),
                'file_type': 'toml',
                'purpose': 'unknown',
                'error': str(e)
            }
    
    def _detect_toml_purpose(self, data: Dict[str, Any], file_path: Path) -> str:
        """Detect the purpose/type of TOML file."""
        filename = file_path.name.lower()
        
        # Check filename patterns
        if filename == 'cargo.toml':
            return 'cargo_manifest'
        elif filename == 'pyproject.toml':
            return 'python_project'
        elif filename == 'rustfmt.toml':
            return 'rust_formatter'
        elif filename == 'config.toml' or filename.endswith('.config.toml'):
            return 'application_config'
        elif 'workspace' in filename:
            return 'workspace_config'
        
        # Check content patterns
        top_keys = set(data.keys())
        
        if 'package' in top_keys and 'dependencies' in top_keys:
            return 'cargo_manifest'
        elif 'project' in top_keys or 'tool' in top_keys:
            return 'python_project'
        elif 'workspace' in top_keys:
            return 'workspace_config'
        elif any(key in top_keys for key in ['server', 'database', 'api', 'app']):
            return 'application_config'
        
        return 'generic_config'
    
    def _extract_hierarchy(self, data: Dict[str, Any], purpose: str) -> Dict[str, Any]:
        """Extract hierarchical structure from TOML data."""
        hierarchy: Dict[str, Any] = {
            'sections': [],
            'depth': 0,
            'total_keys': 0
        }
        
        def traverse(obj: Any, path: Optional[List[str]] = None, depth: int = 0) -> None:
            if path is None:
                path = []
            
            hierarchy['depth'] = max(hierarchy['depth'], depth)
            
            if isinstance(obj, dict):
                for key, value in obj.items():
                    current_path = path + [key]
                    section_info = {
                        'path': '.'.join(current_path),
                        'depth': depth,
                        'type': type(value).__name__
                    }
                    
                    if isinstance(value, dict):
                        # Count nested keys
                        section_info['key_count'] = len(value)
                        hierarchy['sections'].append(section_info)
                        traverse(value, current_path, depth + 1)
                    elif isinstance(value, list):
                        section_info['item_count'] = len(value)
                        hierarchy['sections'].append(section_info)
                    else:
                        hierarchy['total_keys'] += 1
        
        traverse(data)
        return hierarchy
    
    def _extract_metadata(self, data: Dict[str, Any], purpose: str) -> Dict[str, Any]:
        """Extract purpose-specific metadata."""
        metadata: Dict[str, Any] = {
            'purpose': purpose,
            'top_level_sections': list(data.keys())
        }
        
        if purpose == 'cargo_manifest':
            # Extract Rust project metadata
            if 'package' in data:
                package = data['package']
                metadata.update({
                    'project_name': package.get('name', ''),
                    'version': package.get('version', ''),
                    'authors': package.get('authors', []),
                    'edition': package.get('edition', ''),
                    'description': package.get('description', '')
                })
            
            # Count dependencies
            dep_types = ['dependencies', 'dev-dependencies', 'build-dependencies']
            for dep_type in dep_types:
                if dep_type in data:
                    metadata[f'{dep_type}_count'] = len(data[dep_type])
            
            # Extract features
            if 'features' in data:
                metadata['features'] = list(data['features'].keys())
                
        elif purpose == 'python_project':
            # Extract Python project metadata
            if 'project' in data:
                project = data['project']
                metadata.update({
                    'project_name': project.get('name', ''),
                    'version': project.get('version', ''),
                    'authors': project.get('authors', []),
                    'description': project.get('description', ''),
                    'requires_python': project.get('requires-python', '')
                })
                
                if 'dependencies' in project:
                    metadata['dependencies_count'] = len(project['dependencies'])
            
            # Check for Poetry configuration
            if 'tool' in data and 'poetry' in data['tool']:
                poetry = data['tool']['poetry']
                if 'name' in poetry:
                    metadata['project_name'] = poetry['name']
                if 'dependencies' in poetry:
                    metadata['poetry_dependencies'] = len(poetry['dependencies'])
                    
        elif purpose == 'application_config':
            # Extract configuration sections
            config_sections = []
            for key in data:
                if isinstance(data[key], dict):
                    config_sections.append(key)
            metadata['config_sections'] = config_sections
            
            # Look for common patterns
            if 'server' in data:
                server = data['server']
                metadata['server_config'] = {
                    'host': server.get('host', ''),
                    'port': server.get('port', ''),
                    'workers': server.get('workers', '')
                }
                
        return metadata
    
    def _detect_bridges(
        self, 
        data: Dict[str, Any], 
        purpose: str,
        file_path: Path
    ) -> List[Dict[str, Any]]:
        """Detect bridges between TOML configuration and code."""
        bridges = []
        
        if purpose == 'cargo_manifest':
            # Cargo.toml bridges to Rust code
            if 'package' in data:
                package_name = data['package'].get('name', '')
                if package_name:
                    bridges.append({
                        'type': 'package_definition',
                        'source': 'Cargo.toml#package.name',
                        'target': f"src/lib.rs or src/main.rs",
                        'target_type': 'rust_module',
                        'confidence': 0.9
                    })
            
            # Dependencies bridge to imports
            if 'dependencies' in data:
                for dep_name, dep_info in data['dependencies'].items():
                    bridges.append({
                        'type': 'dependency_usage',
                        'source': f"Cargo.toml#dependencies.{dep_name}",
                        'target': f"use {dep_name}",
                        'target_type': 'rust_import',
                        'confidence': 0.8,
                        'version': dep_info if isinstance(dep_info, str) else dep_info.get('version', '')
                    })
            
            # Features bridge to conditional compilation
            if 'features' in data:
                for feature_name, feature_deps in data['features'].items():
                    bridges.append({
                        'type': 'feature_flag',
                        'source': f"Cargo.toml#features.{feature_name}",
                        'target': f'#[cfg(feature = "{feature_name}")]',
                        'target_type': 'rust_attribute',
                        'confidence': 0.85
                    })
                    
        elif purpose == 'python_project':
            # pyproject.toml bridges to Python code
            project_name = ''
            if 'project' in data:
                project_name = data['project'].get('name', '')
            elif 'tool' in data and 'poetry' in data['tool']:
                project_name = data['tool']['poetry'].get('name', '')
                
            if project_name:
                bridges.append({
                    'type': 'package_definition',
                    'source': 'pyproject.toml#project.name',
                    'target': f"{project_name}/__init__.py",
                    'target_type': 'python_module',
                    'confidence': 0.9
                })
            
            # Dependencies bridge to imports
            dependencies = []
            if 'project' in data and 'dependencies' in data['project']:
                dependencies = data['project']['dependencies']
            elif 'tool' in data and 'poetry' in data['tool'] and 'dependencies' in data['tool']['poetry']:
                dependencies = data['tool']['poetry']['dependencies']
                
            for dep in dependencies:
                if isinstance(dep, str):
                    # Simple dependency string
                    dep_name = dep.split('[')[0].split('>')[0].split('<')[0].split('=')[0].strip()
                    bridges.append({
                        'type': 'dependency_import',
                        'source': f"pyproject.toml#dependencies",
                        'target': f"import {dep_name}",
                        'target_type': 'python_import',
                        'confidence': 0.7
                    })
                    
        elif purpose == 'application_config':
            # Configuration bridges to code usage
            def extract_config_bridges(obj: Any, path: Optional[List[str]] = None) -> None:
                if path is None:
                    path = []
                    
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        current_path = path + [key]
                        path_str = '.'.join(current_path)
                        
                        # Look for patterns that suggest code usage
                        if any(pattern in key.lower() for pattern in ['class', 'handler', 'module', 'function']):
                            if isinstance(value, str) and '.' in value:
                                # Likely a module path
                                bridges.append({
                                    'type': 'config_reference',
                                    'source': f"{file_path.name}#{path_str}",
                                    'target': value,
                                    'target_type': 'module_reference',
                                    'confidence': 0.75
                                })
                        
                        # Recursively process nested objects
                        if isinstance(value, dict):
                            extract_config_bridges(value, current_path)
            
            extract_config_bridges(data)
        
        return bridges
    
    def _generate_structured_text(
        self,
        data: Dict[str, Any],
        purpose: str,
        hierarchy: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> str:
        """Generate structured text representation of TOML."""
        lines = []
        
        # Header
        lines.append(f"[TOML Configuration Analysis]")
        lines.append(f"Type: {purpose.replace('_', ' ').title()}")
        
        # Metadata summary
        if metadata.get('project_name'):
            lines.append(f"Project: {metadata['project_name']}")
        if metadata.get('version'):
            lines.append(f"Version: {metadata['version']}")
            
        # Statistics
        lines.append(f"\n=== Structure ===")
        lines.append(f"Top-level sections: {len(metadata.get('top_level_sections', []))}")
        lines.append(f"Maximum depth: {hierarchy['depth']}")
        lines.append(f"Total keys: {hierarchy['total_keys']}")
        
        # Purpose-specific summary
        if purpose == 'cargo_manifest':
            lines.append(f"\n=== Rust Project ===")
            if metadata.get('edition'):
                lines.append(f"Edition: {metadata['edition']}")
            for dep_type in ['dependencies', 'dev-dependencies', 'build-dependencies']:
                count_key = f'{dep_type}_count'
                if count_key in metadata:
                    lines.append(f"{dep_type.replace('-', ' ').title()}: {metadata[count_key]}")
            if metadata.get('features'):
                lines.append(f"Features: {', '.join(metadata['features'][:5])}")
                if len(metadata['features']) > 5:
                    lines.append(f"  ... and {len(metadata['features']) - 5} more")
                    
        elif purpose == 'python_project':
            lines.append(f"\n=== Python Project ===")
            if metadata.get('requires_python'):
                lines.append(f"Python: {metadata['requires_python']}")
            if metadata.get('dependencies_count'):
                lines.append(f"Dependencies: {metadata['dependencies_count']}")
            if metadata.get('poetry_dependencies'):
                lines.append(f"Poetry dependencies: {metadata['poetry_dependencies']}")
                
        elif purpose == 'application_config':
            lines.append(f"\n=== Application Configuration ===")
            if metadata.get('config_sections'):
                lines.append(f"Sections: {', '.join(metadata['config_sections'])}")
            if metadata.get('server_config'):
                server = metadata['server_config']
                if server.get('host') and server.get('port'):
                    lines.append(f"Server: {server['host']}:{server['port']}")
        
        # Section hierarchy
        lines.append(f"\n=== Section Hierarchy ===")
        # Group sections by depth
        sections_by_depth = defaultdict(list)
        for section in hierarchy['sections'][:20]:  # First 20 sections
            sections_by_depth[section['depth']].append(section)
        
        for depth in sorted(sections_by_depth.keys()):
            indent = "  " * depth
            for section in sections_by_depth[depth]:
                lines.append(f"{indent}{section['path'].split('.')[-1]} ({section['type']})")
        
        # Add raw TOML preview
        lines.append(f"\n=== Raw Content Preview ===")
        lines.extend(self._format_toml_preview(data, max_lines=30))
        
        return '\n'.join(lines)
    
    def _format_toml_preview(self, data: Dict[str, Any], max_lines: int = 30) -> List[str]:
        """Format a preview of TOML content."""
        lines = []
        line_count = 0
        
        def format_value(value: Any) -> str:
            """Format a value for display."""
            if isinstance(value, str):
                return f'"{value}"' if len(value) < 50 else f'"{value[:47]}..."'
            elif isinstance(value, list):
                return f"[...{len(value)} items...]"
            elif isinstance(value, dict):
                return f"{{...{len(value)} keys...}}"
            else:
                return str(value)
        
        def format_section(obj: Any, path: Optional[List[str]] = None, indent: int = 0) -> None:
            nonlocal line_count
            if line_count >= max_lines:
                return
                
            if path is None:
                path = []
                
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if line_count >= max_lines:
                        break
                        
                    current_path = path + [key]
                    
                    if isinstance(value, dict):
                        # Section header
                        if path:
                            lines.append(f"\n[{'.'.join(current_path)}]")
                        else:
                            lines.append(f"\n[{key}]")
                        line_count += 2
                        format_section(value, current_path, indent)
                    else:
                        # Key-value pair
                        indent_str = ""
                        lines.append(f"{indent_str}{key} = {format_value(value)}")
                        line_count += 1
        
        format_section(data)
        
        if line_count >= max_lines:
            lines.append("... (truncated)")
            
        return lines