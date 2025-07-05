"""
YAML/JSON Parser for Jina v4

Handles configuration files, API specifications, and data files with
schema awareness and reference resolution.
"""

import json
import yaml
import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import re

logger = logging.getLogger(__name__)


class YAMLJSONParser:
    """Parse YAML/JSON files with structure preservation and bridge detection."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize parser with configuration."""
        self.config = config or {}
        self.references: Dict[str, Any] = {}
        self.detected_schemas: List[str] = []
        
    def parse(self, file_path: Path) -> Dict[str, Any]:
        """
        Parse YAML/JSON file and extract structure.
        
        Returns:
            Dictionary containing parsed content, structure, and detected bridges
        """
        try:
            # Load content
            content_str = file_path.read_text(encoding='utf-8')
            
            # Parse based on extension
            if file_path.suffix.lower() in ['.yaml', '.yml']:
                data = yaml.safe_load(content_str)
                file_type = 'yaml'
            else:
                data = json.loads(content_str)
                file_type = 'json'
            
            # Detect document purpose
            purpose = self._detect_purpose(data, file_path)
            
            # Extract structure based on purpose
            structure = self._extract_structure(data, purpose)
            
            # Resolve references
            self._resolve_references(data, file_path.parent)
            
            # Detect bridges
            bridges = self._detect_bridges(data, structure, purpose)
            
            # Generate structured text
            structured_text = self._generate_structured_text(data, structure, purpose)
            
            return {
                'content': structured_text,
                'file_type': file_type,
                'purpose': purpose,
                'structure': structure,
                'bridges': bridges,
                'metadata': {
                    'references': self.references,
                    'schemas': self.detected_schemas,
                    'key_count': self._count_keys(data),
                    'max_depth': self._calculate_depth(data),
                    'has_arrays': self._has_arrays(data)
                }
            }
            
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
            return {
                'content': file_path.read_text(encoding='utf-8'),
                'file_type': file_path.suffix[1:],
                'purpose': 'unknown',
                'structure': {},
                'bridges': [],
                'metadata': {'error': str(e)}
            }
    
    def _detect_purpose(self, data: Union[Dict, List], file_path: Path) -> str:
        """Detect the purpose of the YAML/JSON file."""
        filename = file_path.name.lower()
        
        # Check filename patterns
        if filename == 'package.json':
            return 'npm_package'
        elif filename == 'composer.json':
            return 'composer_package'
        elif filename in ['swagger.yaml', 'swagger.json', 'openapi.yaml', 'openapi.json']:
            return 'openapi_spec'
        elif filename == 'docker-compose.yml' or filename.endswith('compose.yaml'):
            return 'docker_compose'
        elif filename in ['.gitlab-ci.yml', '.github/workflows/*.yml']:
            return 'ci_config'
        elif filename == 'environment.yml' or filename == 'environment.yaml':
            return 'conda_env'
        elif filename.endswith('.schema.json'):
            return 'json_schema'
        
        # Check content patterns
        if isinstance(data, dict):
            # OpenAPI/Swagger detection
            if 'openapi' in data or 'swagger' in data:
                return 'openapi_spec'
            
            # AsyncAPI detection
            elif 'asyncapi' in data:
                return 'asyncapi_spec'
            
            # JSON Schema detection
            elif '$schema' in data or ('type' in data and 'properties' in data):
                return 'json_schema'
            
            # Kubernetes manifests
            elif 'apiVersion' in data and 'kind' in data:
                return f"k8s_{data.get('kind', 'resource').lower()}"
            
            # Ansible playbooks
            elif any(key in data for key in ['hosts', 'tasks', 'playbook']):
                return 'ansible_playbook'
            
            # CloudFormation templates
            elif 'AWSTemplateFormatVersion' in data:
                return 'cloudformation'
            
            # Generic config detection
            elif any(key in str(data).lower() for key in ['config', 'settings', 'options']):
                return 'config_file'
        
        return 'data_file'
    
    def _extract_structure(self, data: Any, purpose: str) -> Dict[str, Any]:
        """Extract structure based on file purpose."""
        if purpose == 'npm_package':
            return self._extract_npm_structure(data)
        elif purpose == 'openapi_spec':
            return self._extract_openapi_structure(data)
        elif purpose == 'docker_compose':
            return self._extract_compose_structure(data)
        elif purpose.startswith('k8s_'):
            return self._extract_k8s_structure(data)
        elif purpose == 'json_schema':
            return self._extract_schema_structure(data)
        else:
            return self._extract_generic_structure(data)
    
    def _extract_npm_structure(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract NPM package.json structure."""
        structure = {
            'package_info': {
                'name': data.get('name'),
                'version': data.get('version'),
                'description': data.get('description'),
                'main': data.get('main'),
                'module': data.get('module'),
                'types': data.get('types')
            },
            'scripts': data.get('scripts', {}),
            'dependencies': {
                'prod': data.get('dependencies', {}),
                'dev': data.get('devDependencies', {}),
                'peer': data.get('peerDependencies', {}),
                'optional': data.get('optionalDependencies', {})
            },
            'exports': data.get('exports', {}),
            'bin': data.get('bin', {}),
            'engines': data.get('engines', {})
        }
        return structure
    
    def _extract_openapi_structure(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract OpenAPI/Swagger structure."""
        structure = {
            'version': data.get('openapi') or data.get('swagger'),
            'info': data.get('info', {}),
            'servers': data.get('servers', []),
            'paths': {},
            'components': data.get('components', {}),
            'tags': data.get('tags', [])
        }
        
        # Extract path operations
        paths = data.get('paths', {})
        for path, path_item in paths.items():
            if isinstance(path_item, dict):
                structure['paths'][path] = {
                    'operations': []
                }
                for method in ['get', 'post', 'put', 'delete', 'patch', 'options', 'head']:
                    if method in path_item:
                        op = path_item[method]
                        structure['paths'][path]['operations'].append({
                            'method': method.upper(),
                            'operationId': op.get('operationId'),
                            'summary': op.get('summary'),
                            'tags': op.get('tags', []),
                            'parameters': len(op.get('parameters', [])),
                            'requestBody': 'requestBody' in op,
                            'responses': list(op.get('responses', {}).keys())
                        })
        
        return structure
    
    def _extract_compose_structure(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract Docker Compose structure."""
        structure = {
            'version': data.get('version'),
            'services': {},
            'networks': data.get('networks', {}),
            'volumes': data.get('volumes', {}),
            'configs': data.get('configs', {}),
            'secrets': data.get('secrets', {})
        }
        
        # Extract service details
        services = data.get('services', {})
        for name, service in services.items():
            if isinstance(service, dict):
                structure['services'][name] = {
                    'image': service.get('image'),
                    'build': service.get('build'),
                    'ports': service.get('ports', []),
                    'environment': list(service.get('environment', {}).keys()) if isinstance(service.get('environment'), dict) else [],
                    'depends_on': service.get('depends_on', []),
                    'volumes': service.get('volumes', []),
                    'networks': service.get('networks', [])
                }
        
        return structure
    
    def _extract_k8s_structure(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract Kubernetes manifest structure."""
        structure = {
            'apiVersion': data.get('apiVersion'),
            'kind': data.get('kind'),
            'metadata': {
                'name': data.get('metadata', {}).get('name'),
                'namespace': data.get('metadata', {}).get('namespace'),
                'labels': data.get('metadata', {}).get('labels', {}),
                'annotations': data.get('metadata', {}).get('annotations', {})
            },
            'spec': {}
        }
        
        # Extract spec based on kind
        kind = data.get('kind', '').lower()
        spec = data.get('spec', {})
        
        if kind == 'deployment':
            structure['spec'] = {
                'replicas': spec.get('replicas'),
                'selector': spec.get('selector'),
                'template': {
                    'containers': len(spec.get('template', {}).get('spec', {}).get('containers', []))
                }
            }
        elif kind == 'service':
            structure['spec'] = {
                'type': spec.get('type'),
                'ports': spec.get('ports', []),
                'selector': spec.get('selector', {})
            }
        elif kind == 'configmap':
            structure['spec'] = {
                'data_keys': list(data.get('data', {}).keys())
            }
        
        return structure
    
    def _extract_schema_structure(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract JSON Schema structure."""
        structure = {
            '$schema': data.get('$schema'),
            'type': data.get('type'),
            'title': data.get('title'),
            'properties': {},
            'required': data.get('required', []),
            'definitions': {}
        }
        
        # Extract property information
        props = data.get('properties', {})
        for prop_name, prop_schema in props.items():
            if isinstance(prop_schema, dict):
                structure['properties'][prop_name] = {
                    'type': prop_schema.get('type'),
                    'format': prop_schema.get('format'),
                    'description': prop_schema.get('description'),
                    '$ref': prop_schema.get('$ref')
                }
        
        # Extract definitions
        defs = data.get('definitions', {})
        for def_name in defs:
            structure['definitions'][def_name] = {
                'type': defs[def_name].get('type') if isinstance(defs[def_name], dict) else None
            }
        
        return structure
    
    def _extract_generic_structure(self, data: Any) -> Dict[str, Any]:
        """Extract generic structure from YAML/JSON."""
        structure = {
            'hierarchy': self._build_hierarchy(data),
            'data_types': self._analyze_data_types(data),
            'patterns': self._detect_patterns(data),
            'statistics': {
                'total_keys': self._count_keys(data),
                'max_depth': self._calculate_depth(data),
                'array_count': self._count_arrays(data),
                'null_count': self._count_nulls(data)
            }
        }
        return structure
    
    def _resolve_references(self, data: Any, base_path: Path) -> None:
        """Resolve $ref and other references in the document."""
        if isinstance(data, dict):
            for key, value in data.items():
                if key == '$ref':
                    ref_path = value
                    if ref_path.startswith('#/'):
                        # Internal reference
                        self.references[ref_path] = 'internal'
                    elif ref_path.startswith('http'):
                        # External URL reference
                        self.references[ref_path] = 'external_url'
                    else:
                        # File reference
                        self.references[ref_path] = 'file'
                        # Could attempt to load the referenced file here
                elif isinstance(value, (dict, list)):
                    self._resolve_references(value, base_path)
        elif isinstance(data, list):
            for item in data:
                self._resolve_references(item, base_path)
    
    def _detect_bridges(self, data: Any, structure: Dict[str, Any], purpose: str) -> List[Dict[str, Any]]:
        """Detect theory-practice bridges based on file purpose."""
        bridges = []
        
        if purpose == 'npm_package':
            # Scripts bridge to commands
            for script_name, command in structure.get('scripts', {}).items():
                bridges.append({
                    'type': 'npm_script',
                    'source': f"npm run {script_name}",
                    'target': command,
                    'target_type': 'shell_command',
                    'confidence': 0.95
                })
            
            # Dependencies bridge to imports
            for dep_type in ['prod', 'dev']:
                for dep_name in structure['dependencies'].get(dep_type, {}):
                    bridges.append({
                        'type': 'npm_dependency',
                        'source': dep_name,
                        'target_type': 'js_import',
                        'confidence': 0.9
                    })
        
        elif purpose == 'openapi_spec':
            # Operations bridge to implementations
            for path, path_data in structure.get('paths', {}).items():
                for op in path_data.get('operations', []):
                    if op.get('operationId'):
                        bridges.append({
                            'type': 'api_operation',
                            'source': f"{op['method']} {path}",
                            'target': op['operationId'],
                            'target_type': 'function',
                            'confidence': 0.85
                        })
        
        elif purpose == 'docker_compose':
            # Services bridge to containers/images
            for service_name, service_data in structure.get('services', {}).items():
                if service_data.get('image'):
                    bridges.append({
                        'type': 'docker_service',
                        'source': service_name,
                        'target': service_data['image'],
                        'target_type': 'docker_image',
                        'confidence': 0.9
                    })
        
        elif purpose.startswith('k8s_'):
            # Kubernetes resources bridge to implementations
            metadata = structure.get('metadata', {})
            if metadata.get('name'):
                bridges.append({
                    'type': 'k8s_resource',
                    'source': f"{structure['kind']}/{metadata['name']}",
                    'target_type': 'k8s_controller',
                    'confidence': 0.8
                })
        
        return bridges
    
    def _generate_structured_text(self, data: Any, structure: Dict[str, Any], purpose: str) -> str:
        """Generate structured text representation."""
        lines = []
        
        # Add metadata header
        lines.append(f"[{purpose.upper()} Structure Analysis]")
        
        if purpose == 'npm_package':
            lines.append(f"Package: {structure['package_info'].get('name')} v{structure['package_info'].get('version')}")
            lines.append(f"Scripts: {len(structure.get('scripts', {}))}")
            lines.append(f"Dependencies: {sum(len(deps) for deps in structure['dependencies'].values())}")
        elif purpose == 'openapi_spec':
            lines.append(f"API Version: {structure.get('version')}")
            lines.append(f"Paths: {len(structure.get('paths', {}))}")
            total_ops = sum(len(p.get('operations', [])) for p in structure.get('paths', {}).values())
            lines.append(f"Operations: {total_ops}")
        elif purpose == 'docker_compose':
            lines.append(f"Version: {structure.get('version')}")
            lines.append(f"Services: {len(structure.get('services', {}))}")
        
        lines.append("")
        lines.append("=== Content ===")
        
        # Add formatted content
        if isinstance(data, dict):
            lines.extend(self._dict_to_text(data))
        elif isinstance(data, list):
            lines.extend(self._list_to_text(data))
        else:
            lines.append(str(data))
        
        return '\n'.join(lines)
    
    def _dict_to_text(self, data: Dict[str, Any], indent: int = 0) -> List[str]:
        """Convert dictionary to indented text."""
        lines = []
        prefix = "  " * indent
        
        for key, value in data.items():
            if isinstance(value, dict):
                lines.append(f"{prefix}{key}:")
                lines.extend(self._dict_to_text(value, indent + 1))
            elif isinstance(value, list):
                lines.append(f"{prefix}{key}:")
                lines.extend(self._list_to_text(value, indent + 1))
            else:
                lines.append(f"{prefix}{key}: {value}")
        
        return lines
    
    def _list_to_text(self, data: List[Any], indent: int = 0) -> List[str]:
        """Convert list to indented text."""
        lines = []
        prefix = "  " * indent
        
        for i, item in enumerate(data):
            if isinstance(item, dict):
                lines.append(f"{prefix}- ")
                lines.extend(self._dict_to_text(item, indent + 1))
            elif isinstance(item, list):
                lines.append(f"{prefix}- ")
                lines.extend(self._list_to_text(item, indent + 1))
            else:
                lines.append(f"{prefix}- {item}")
        
        return lines
    
    def _build_hierarchy(self, data: Any, path: str = '$') -> Dict[str, Any]:
        """Build hierarchical representation of data."""
        if isinstance(data, dict):
            return {
                'type': 'object',
                'path': path,
                'keys': list(data.keys()),
                'children': {
                    key: self._build_hierarchy(value, f"{path}.{key}")
                    for key, value in data.items()
                }
            }
        elif isinstance(data, list):
            return {
                'type': 'array',
                'path': path,
                'length': len(data),
                'items': [
                    self._build_hierarchy(item, f"{path}[{i}]")
                    for i, item in enumerate(data[:3])  # Limit to first 3 items
                ]
            }
        else:
            return {
                'type': type(data).__name__,
                'path': path,
                'value': str(data)[:50] if data is not None else None
            }
    
    def _analyze_data_types(self, data: Any) -> Dict[str, int]:
        """Analyze data types in the structure."""
        type_counts: Dict[str, int] = {}
        
        def count_types(obj: Any) -> None:
            if isinstance(obj, dict):
                type_counts['object'] = type_counts.get('object', 0) + 1
                for value in obj.values():
                    count_types(value)
            elif isinstance(obj, list):
                type_counts['array'] = type_counts.get('array', 0) + 1
                for item in obj:
                    count_types(item)
            elif isinstance(obj, str):
                type_counts['string'] = type_counts.get('string', 0) + 1
            elif isinstance(obj, (int, float)):
                type_counts['number'] = type_counts.get('number', 0) + 1
            elif isinstance(obj, bool):
                type_counts['boolean'] = type_counts.get('boolean', 0) + 1
            elif obj is None:
                type_counts['null'] = type_counts.get('null', 0) + 1
        
        count_types(data)
        return type_counts
    
    def _detect_patterns(self, data: Any) -> Dict[str, Any]:
        """Detect common patterns in the data."""
        patterns: Dict[str, List[str]] = {
            'env_vars': [],
            'urls': [],
            'file_paths': [],
            'version_strings': []
        }
        
        def find_patterns(obj: Any, key: Optional[str] = None) -> None:
            if isinstance(obj, str):
                # Environment variables
                if obj.startswith('$') or obj.startswith('${'):
                    patterns['env_vars'].append(obj)
                # URLs
                elif obj.startswith(('http://', 'https://')):
                    patterns['urls'].append(obj)
                # File paths
                elif '/' in obj and (obj.startswith('/') or obj.startswith('./')):
                    patterns['file_paths'].append(obj)
                # Version strings
                elif re.match(r'^\d+\.\d+(\.\d+)?', obj):
                    patterns['version_strings'].append(obj)
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    find_patterns(v, k)
            elif isinstance(obj, list):
                for item in obj:
                    find_patterns(item)
        
        find_patterns(data)
        return patterns
    
    def _count_keys(self, data: Any) -> int:
        """Count total number of keys in nested structure."""
        count = 0
        if isinstance(data, dict):
            count += len(data)
            for value in data.values():
                count += self._count_keys(value)
        elif isinstance(data, list):
            for item in data:
                count += self._count_keys(item)
        return count
    
    def _calculate_depth(self, data: Any, current_depth: int = 0) -> int:
        """Calculate maximum depth of nested structure."""
        if isinstance(data, dict):
            if not data:
                return current_depth
            return max(self._calculate_depth(v, current_depth + 1) for v in data.values())
        elif isinstance(data, list):
            if not data:
                return current_depth
            return max(self._calculate_depth(item, current_depth + 1) for item in data)
        else:
            return current_depth
    
    def _has_arrays(self, data: Any) -> bool:
        """Check if structure contains arrays."""
        if isinstance(data, list):
            return True
        elif isinstance(data, dict):
            return any(self._has_arrays(v) for v in data.values())
        return False
    
    def _count_arrays(self, data: Any) -> int:
        """Count number of arrays in structure."""
        count = 0
        if isinstance(data, list):
            count += 1
            for item in data:
                count += self._count_arrays(item)
        elif isinstance(data, dict):
            for value in data.values():
                count += self._count_arrays(value)
        return count
    
    def _count_nulls(self, data: Any) -> int:
        """Count null values in structure."""
        count = 0
        if data is None:
            count += 1
        elif isinstance(data, dict):
            for value in data.values():
                count += self._count_nulls(value)
        elif isinstance(data, list):
            for item in data:
                count += self._count_nulls(item)
        return count