"""
XML Structure Parser for Jina v4

Extracts hierarchical structure, schemas, and semantic relationships from XML files.
"""

import xml.etree.ElementTree as ET
import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import re

logger = logging.getLogger(__name__)


class XMLStructureParser:
    """Parse XML files preserving structure for theory-practice bridges."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize XML parser with configuration."""
        self.config = config or {}
        self.namespace_map: Dict[str, str] = {}
        self.schema_references: List[str] = []
        self.element_stats: Dict[str, int] = {}
        
    def parse(self, file_path: Path) -> Dict[str, Any]:
        """
        Parse XML file and extract structure.
        
        Returns:
            Dictionary containing parsed content, structure, and detected bridges
        """
        try:
            # Parse XML
            tree = ET.parse(str(file_path))
            root = tree.getroot()
            
            # Extract namespaces
            self._extract_namespaces(root)
            
            # Detect document type and purpose
            doc_type = self._detect_document_type(root, file_path)
            
            # Extract structure based on type
            structure = self._extract_structure(root, doc_type)
            
            # Detect potential bridges
            bridges = self._detect_bridges(structure, doc_type)
            
            # Generate structured text representation
            content = self._generate_structured_text(root, structure)
            
            return {
                'content': content,
                'file_type': 'xml',
                'doc_type': doc_type,
                'structure': structure,
                'bridges': bridges,
                'metadata': {
                    'namespaces': self.namespace_map,
                    'schemas': self.schema_references,
                    'root_element': root.tag,
                    'element_count': len(self.element_stats),
                    'max_depth': self._calculate_depth(root)
                }
            }
            
        except Exception as e:
            logger.error(f"Error parsing XML file {file_path}: {e}")
            # Return basic extraction
            return {
                'content': file_path.read_text(encoding='utf-8'),
                'file_type': 'xml',
                'doc_type': 'unknown',
                'structure': {},
                'bridges': [],
                'metadata': {'error': str(e)}
            }
    
    def _extract_namespaces(self, root: ET.Element) -> None:
        """Extract XML namespaces from the document."""
        # Get namespaces from root attributes
        for key, value in root.attrib.items():
            if key.startswith('xmlns'):
                if ':' in key:
                    prefix = key.split(':')[1]
                    self.namespace_map[prefix] = value
                else:
                    self.namespace_map['default'] = value
        
        # Look for schema references
        for attr, value in root.attrib.items():
            if 'schemaLocation' in attr or attr.endswith('noNamespaceSchemaLocation'):
                self.schema_references.append(value)
    
    def _detect_document_type(self, root: ET.Element, file_path: Path) -> str:
        """Detect the type of XML document."""
        root_tag = root.tag.split('}')[-1] if '}' in root.tag else root.tag
        filename = file_path.name.lower()
        
        # Common XML document types
        if 'pom.xml' in filename:
            return 'maven_pom'
        elif filename.endswith('.csproj') or filename.endswith('.vbproj'):
            return 'msbuild_project'
        elif root_tag == 'project' and 'ant' in str(root.attrib):
            return 'ant_build'
        elif root_tag in ['web-app', 'web-fragment']:
            return 'web_xml'
        elif 'spring' in str(self.namespace_map.values()):
            return 'spring_config'
        elif root_tag == 'configuration':
            return 'config_xml'
        elif root_tag in ['definitions', 'types'] and 'wsdl' in str(self.namespace_map.values()):
            return 'wsdl'
        elif root_tag == 'xsd:schema' or root_tag == 'schema':
            return 'xml_schema'
        elif root_tag == 'rss' or root_tag == 'feed':
            return 'rss_feed'
        elif root_tag == 'svg':
            return 'svg_image'
        else:
            return 'generic_xml'
    
    def _extract_structure(self, root: ET.Element, doc_type: str) -> Dict[str, Any]:
        """Extract structure based on document type."""
        if doc_type == 'maven_pom':
            return self._extract_maven_structure(root)
        elif doc_type == 'spring_config':
            return self._extract_spring_structure(root)
        elif doc_type == 'xml_schema':
            return self._extract_schema_structure(root)
        elif doc_type == 'wsdl':
            return self._extract_wsdl_structure(root)
        else:
            return self._extract_generic_structure(root)
    
    def _extract_maven_structure(self, root: ET.Element) -> Dict[str, Any]:
        """Extract Maven POM structure."""
        structure: Dict[str, Any] = {
            'artifact': {},
            'dependencies': [],
            'build': {},
            'properties': {}
        }
        
        # Extract artifact info
        for field in ['groupId', 'artifactId', 'version', 'packaging']:
            elem = root.find(field)
            if elem is not None:
                structure['artifact'][field] = elem.text
        
        # Extract dependencies
        deps = root.find('.//dependencies')
        if deps is not None:
            for dep in deps.findall('dependency'):
                dep_info: Dict[str, Optional[str]] = {}
                for field in ['groupId', 'artifactId', 'version', 'scope']:
                    elem = dep.find(field)
                    if elem is not None:
                        dep_info[field] = elem.text
                structure['dependencies'].append(dep_info)
        
        # Extract build configuration
        build = root.find('build')
        if build is not None:
            plugins = build.find('plugins')
            if plugins is not None:
                structure['build']['plugins'] = []
                for plugin in plugins.findall('plugin'):
                    plugin_info = {
                        'groupId': plugin.findtext('groupId', ''),
                        'artifactId': plugin.findtext('artifactId', '')
                    }
                    structure['build']['plugins'].append(plugin_info)
        
        return structure
    
    def _extract_spring_structure(self, root: ET.Element) -> Dict[str, Any]:
        """Extract Spring configuration structure."""
        structure: Dict[str, Any] = {
            'beans': [],
            'imports': [],
            'profiles': [],
            'properties': {}
        }
        
        # Extract bean definitions
        for bean in root.findall('.//{http://www.springframework.org/schema/beans}bean'):
            bean_info: Dict[str, Any] = {
                'id': bean.get('id'),
                'class': bean.get('class'),
                'scope': bean.get('scope', 'singleton'),
                'properties': []
            }
            
            # Extract properties
            for prop in bean.findall('.//{http://www.springframework.org/schema/beans}property'):
                bean_info['properties'].append({
                    'name': prop.get('name'),
                    'value': prop.get('value'),
                    'ref': prop.get('ref')
                })
            
            structure['beans'].append(bean_info)
        
        # Extract imports
        for imp in root.findall('.//{http://www.springframework.org/schema/beans}import'):
            structure['imports'].append(imp.get('resource'))
        
        return structure
    
    def _extract_schema_structure(self, root: ET.Element) -> Dict[str, Any]:
        """Extract XML Schema structure."""
        structure: Dict[str, Any] = {
            'target_namespace': root.get('targetNamespace'),
            'elements': [],
            'complex_types': [],
            'simple_types': [],
            'imports': []
        }
        
        # Extract element definitions
        for elem in root.findall('.//{http://www.w3.org/2001/XMLSchema}element'):
            structure['elements'].append({
                'name': elem.get('name'),
                'type': elem.get('type'),
                'minOccurs': elem.get('minOccurs', '1'),
                'maxOccurs': elem.get('maxOccurs', '1')
            })
        
        # Extract complex types
        for ct in root.findall('.//{http://www.w3.org/2001/XMLSchema}complexType'):
            structure['complex_types'].append({
                'name': ct.get('name'),
                'base': ct.get('base')
            })
        
        return structure
    
    def _extract_wsdl_structure(self, root: ET.Element) -> Dict[str, Any]:
        """Extract WSDL structure."""
        structure: Dict[str, Any] = {
            'services': [],
            'operations': [],
            'messages': [],
            'types': []
        }
        
        # Extract service definitions
        for service in root.findall('.//{http://schemas.xmlsoap.org/wsdl/}service'):
            service_info: Dict[str, Any] = {
                'name': service.get('name'),
                'ports': []
            }
            
            for port in service.findall('.//{http://schemas.xmlsoap.org/wsdl/}port'):
                service_info['ports'].append({
                    'name': port.get('name'),
                    'binding': port.get('binding')
                })
            
            structure['services'].append(service_info)
        
        # Extract operations
        for op in root.findall('.//{http://schemas.xmlsoap.org/wsdl/}operation'):
            structure['operations'].append({
                'name': op.get('name'),
                'input': op.findtext('.//{http://schemas.xmlsoap.org/wsdl/}input/@message', ''),
                'output': op.findtext('.//{http://schemas.xmlsoap.org/wsdl/}output/@message', '')
            })
        
        return structure
    
    def _extract_generic_structure(self, root: ET.Element) -> Dict[str, Any]:
        """Extract generic XML structure."""
        structure = {
            'hierarchy': self._build_hierarchy(root),
            'attributes': self._collect_attributes(root),
            'text_content': self._collect_text_nodes(root),
            'patterns': self._detect_patterns(root)
        }
        return structure
    
    def _build_hierarchy(self, element: ET.Element, path: str = '') -> Dict[str, Any]:
        """Build hierarchical representation of XML."""
        current_path = f"{path}/{element.tag}" if path else element.tag
        
        hierarchy: Dict[str, Any] = {
            'tag': element.tag,
            'attributes': dict(element.attrib),
            'text': element.text.strip() if element.text else None,
            'children': {},
            'path': current_path
        }
        
        # Track element statistics
        self.element_stats[element.tag] = self.element_stats.get(element.tag, 0) + 1
        
        # Process children
        for child in element:
            child_key = child.tag
            if child_key in hierarchy['children']:
                # Handle multiple children with same tag
                if not isinstance(hierarchy['children'][child_key], list):
                    hierarchy['children'][child_key] = [hierarchy['children'][child_key]]
                hierarchy['children'][child_key].append(
                    self._build_hierarchy(child, current_path)
                )
            else:
                hierarchy['children'][child_key] = self._build_hierarchy(child, current_path)
        
        return hierarchy
    
    def _collect_attributes(self, root: ET.Element) -> Dict[str, List[str]]:
        """Collect all unique attributes in the document."""
        attributes: Dict[str, List[str]] = {}
        
        for elem in root.iter():
            for attr, value in elem.attrib.items():
                if attr not in attributes:
                    attributes[attr] = []
                if value and value not in attributes[attr]:
                    attributes[attr].append(value)
        
        return attributes
    
    def _collect_text_nodes(self, root: ET.Element) -> List[Dict[str, str]]:
        """Collect text content from elements."""
        text_nodes = []
        
        for elem in root.iter():
            if elem.text and elem.text.strip():
                text_nodes.append({
                    'path': self._get_element_path(elem, root),
                    'tag': elem.tag,
                    'text': elem.text.strip()
                })
        
        return text_nodes
    
    def _detect_patterns(self, root: ET.Element) -> Dict[str, Any]:
        """Detect common patterns in XML structure."""
        patterns: Dict[str, List[Any]] = {
            'repeated_structures': [],
            'id_references': [],
            'config_patterns': []
        }
        
        # Detect repeated element patterns
        for tag, count in self.element_stats.items():
            if count > 3:  # Arbitrary threshold
                patterns['repeated_structures'].append({
                    'tag': tag,
                    'count': count
                })
        
        # Detect ID/IDREF patterns
        for elem in root.iter():
            if 'id' in elem.attrib:
                patterns['id_references'].append({
                    'element': elem.tag,
                    'id': elem.get('id')
                })
        
        return patterns
    
    def _detect_bridges(self, structure: Dict[str, Any], doc_type: str) -> List[Dict[str, Any]]:
        """Detect potential theory-practice bridges in XML."""
        bridges = []
        
        if doc_type == 'maven_pom':
            # Dependencies bridge to code imports
            for dep in structure.get('dependencies', []):
                bridges.append({
                    'type': 'dependency_import',
                    'source': f"{dep.get('groupId')}.{dep.get('artifactId')}",
                    'target_type': 'java_import',
                    'confidence': 0.9
                })
        
        elif doc_type == 'spring_config':
            # Bean definitions bridge to classes
            for bean in structure.get('beans', []):
                if bean.get('class'):
                    bridges.append({
                        'type': 'bean_class',
                        'source': bean.get('id', bean.get('class')),
                        'target': bean.get('class'),
                        'target_type': 'java_class',
                        'confidence': 0.95
                    })
        
        elif doc_type == 'wsdl':
            # Operations bridge to service methods
            for op in structure.get('operations', []):
                bridges.append({
                    'type': 'wsdl_operation',
                    'source': op.get('name'),
                    'target_type': 'service_method',
                    'confidence': 0.85
                })
        
        elif doc_type == 'xml_schema':
            # Complex types bridge to data classes
            for ct in structure.get('complex_types', []):
                if ct.get('name'):
                    bridges.append({
                        'type': 'schema_type',
                        'source': ct.get('name'),
                        'target_type': 'data_class',
                        'confidence': 0.8
                    })
        
        return bridges
    
    def _generate_structured_text(self, root: ET.Element, structure: Dict[str, Any]) -> str:
        """Generate structured text representation of XML."""
        lines = []
        
        # Add structure summary
        lines.append("[XML Structure Analysis]")
        lines.append(f"Root element: {root.tag}")
        lines.append(f"Namespaces: {len(self.namespace_map)}")
        lines.append(f"Total elements: {sum(self.element_stats.values())}")
        lines.append(f"Unique elements: {len(self.element_stats)}")
        lines.append("")
        
        # Add hierarchical content
        lines.append("=== Content ===")
        lines.extend(self._element_to_text(root))
        
        return '\n'.join(lines)
    
    def _element_to_text(self, elem: ET.Element, level: int = 0) -> List[str]:
        """Convert element to indented text representation."""
        lines = []
        indent = "  " * level
        
        # Element with attributes
        attrs = ' '.join(f'{k}="{v}"' for k, v in elem.attrib.items())
        if attrs:
            lines.append(f"{indent}<{elem.tag} {attrs}>")
        else:
            lines.append(f"{indent}<{elem.tag}>")
        
        # Text content
        if elem.text and elem.text.strip():
            lines.append(f"{indent}  {elem.text.strip()}")
        
        # Child elements
        for child in elem:
            lines.extend(self._element_to_text(child, level + 1))
        
        return lines
    
    def _get_element_path(self, elem: ET.Element, root: ET.Element) -> str:
        """Get XPath-like path to element."""
        path_parts = []
        current: Optional[ET.Element] = elem
        
        while current is not None and current != root:
            parent = self._find_parent(root, current)
            if parent is not None:
                # Count position among siblings with same tag
                position = 1
                for sibling in parent:
                    if sibling == current:
                        break
                    if sibling.tag == current.tag:
                        position += 1
                path_parts.append(f"{current.tag}[{position}]")
            else:
                path_parts.append(current.tag)
            current = parent
        
        path_parts.reverse()
        return '/' + '/'.join(path_parts)
    
    def _find_parent(self, root: ET.Element, child: ET.Element) -> Optional[ET.Element]:
        """Find parent of an element."""
        for parent in root.iter():
            if child in parent:
                return parent
        return None
    
    def _calculate_depth(self, elem: ET.Element, current_depth: int = 0) -> int:
        """Calculate maximum depth of XML tree."""
        if not list(elem):  # No children
            return current_depth
        
        max_child_depth = current_depth
        for child in elem:
            child_depth = self._calculate_depth(child, current_depth + 1)
            max_child_depth = max(max_child_depth, child_depth)
        
        return max_child_depth