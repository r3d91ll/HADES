"""
Configuration Router for Jina v4

Provides unified detection and routing for configuration file processing,
ensuring the appropriate parser is used based on file type and content.
"""

import logging
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import re

logger = logging.getLogger(__name__)


class ConfigurationRouter:
    """Route configuration files to appropriate parsers based on type and content."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize configuration router."""
        self.config = config or {}
        
        # Import parsers lazily to avoid circular imports
        self._parsers: Dict[str, Any] = {}
        
        # Define file extension mappings
        self.extension_map = {
            '.xml': 'xml',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.json': 'json',
            '.toml': 'toml',
            '.ini': 'ini',
            '.cfg': 'ini',
            '.conf': 'conf',
            '.properties': 'properties'
        }
        
        # Define content patterns for format detection
        self.content_patterns = {
            'xml': re.compile(r'^\s*<\?xml\s+version', re.IGNORECASE),
            'json': re.compile(r'^\s*[\{\[]'),
            'yaml': re.compile(r'^---\s*$', re.MULTILINE),
            'toml': re.compile(r'^\s*\[[\w\s]+\]\s*$', re.MULTILINE),
            'ini': re.compile(r'^\s*\[[\w\s]+\]\s*$', re.MULTILINE),
            'properties': re.compile(r'^\s*[\w\.]+=', re.MULTILINE)
        }
        
    def route(self, file_path: Path) -> Dict[str, Any]:
        """
        Route configuration file to appropriate parser.
        
        Args:
            file_path: Path to configuration file
            
        Returns:
            Parsed configuration with metadata and bridges
        """
        try:
            # Detect configuration format
            format_type = self._detect_format(file_path)
            
            # Get appropriate parser
            parser = self._get_parser(format_type)
            
            if parser:
                logger.info(f"Routing {file_path} to {format_type} parser")
                result: Dict[str, Any] = parser.parse(file_path)
                
                # Add routing metadata
                result['routing'] = {
                    'detected_format': format_type,
                    'confidence': result.get('confidence', 1.0),
                    'fallback_used': False
                }
                
                return result
            else:
                # Fallback to basic text extraction
                logger.warning(f"No parser available for {format_type}, using fallback")
                return self._fallback_parse(file_path, format_type)
                
        except Exception as e:
            logger.error(f"Error routing configuration file {file_path}: {e}")
            return self._error_response(file_path, str(e))
    
    def _detect_format(self, file_path: Path) -> str:
        """
        Detect configuration file format from extension and content.
        
        Returns:
            Detected format type
        """
        # First try extension-based detection
        suffix = file_path.suffix.lower()
        if suffix in self.extension_map:
            format_type = self.extension_map[suffix]
            
            # For ambiguous extensions, verify with content
            if format_type in ['ini', 'conf']:
                format_type = self._verify_format_from_content(file_path, format_type)
                
            return format_type
        
        # If no extension match, try content-based detection
        return self._detect_format_from_content(file_path)
    
    def _verify_format_from_content(self, file_path: Path, suspected_format: str) -> str:
        """Verify suspected format by examining content."""
        try:
            # Read first few lines
            with open(file_path, 'r', encoding='utf-8') as f:
                content_preview = f.read(1024)  # Read first 1KB
                
            # Check if it might be TOML (has more complex syntax than INI)
            if suspected_format == 'ini':
                # TOML indicators
                toml_indicators = [
                    '[[',  # Array of tables
                    '"""',  # Multi-line strings
                    "'''",  # Multi-line strings
                    ' = {',  # Inline tables
                    ' = [',  # Arrays
                    'true', 'false',  # Boolean values
                    re.compile(r'\d{4}-\d{2}-\d{2}')  # Dates
                ]
                
                for indicator in toml_indicators:
                    if isinstance(indicator, str):
                        if indicator in content_preview:
                            return 'toml'
                    elif isinstance(indicator, re.Pattern):  # regex Pattern
                        if indicator.search(content_preview):
                            return 'toml'
                            
            return suspected_format
            
        except Exception as e:
            logger.warning(f"Could not verify format from content: {e}")
            return suspected_format
    
    def _detect_format_from_content(self, file_path: Path) -> str:
        """Detect format purely from file content."""
        try:
            # Read file content
            content = file_path.read_text(encoding='utf-8')
            
            # Try each pattern
            for format_type, pattern in self.content_patterns.items():
                if pattern.search(content):
                    return format_type
                    
            # Check for specific content structures
            content_lower = content.lower()
            
            # nginx.conf style
            if 'server {' in content or 'location ' in content:
                return 'nginx'
                
            # Apache style
            if '<virtualhost' in content_lower or '<directory' in content_lower:
                return 'apache'
                
            # Shell-style config
            if content.startswith('#!/') or 'export ' in content:
                return 'shell'
                
            # Default to generic config
            return 'generic'
            
        except Exception as e:
            logger.error(f"Error detecting format from content: {e}")
            return 'unknown'
    
    def _get_parser(self, format_type: str) -> Optional[Any]:
        """Get parser instance for format type."""
        # Lazy load parsers
        if format_type not in self._parsers:
            try:
                if format_type == 'xml':
                    from .xml_parser import XMLStructureParser
                    self._parsers['xml'] = XMLStructureParser(self.config)
                    
                elif format_type in ['yaml', 'json']:
                    from .yaml_json_parser import YAMLJSONParser
                    self._parsers['yaml'] = YAMLJSONParser(self.config)
                    self._parsers['json'] = self._parsers['yaml']  # Same parser
                    
                elif format_type == 'toml':
                    from .toml_parser import TOMLParser
                    self._parsers['toml'] = TOMLParser(self.config)
                    
                elif format_type == 'ini':
                    # Use basic INI parser (to be implemented)
                    logger.info(f"INI parser not yet implemented")
                    return None
                    
                else:
                    # No specific parser available
                    return None
                    
            except ImportError as e:
                logger.error(f"Failed to import parser for {format_type}: {e}")
                return None
                
        return self._parsers.get(format_type)
    
    def _fallback_parse(self, file_path: Path, format_type: str) -> Dict[str, Any]:
        """Fallback parsing for unsupported formats."""
        try:
            content = file_path.read_text(encoding='utf-8')
            
            # Basic structure detection
            structure = self._detect_basic_structure(content, format_type)
            
            # Basic bridge detection
            bridges = self._detect_basic_bridges(content, file_path, format_type)
            
            return {
                'content': f"[{format_type.upper()} Configuration]\n\n{content}",
                'file_type': format_type,
                'purpose': 'configuration',
                'structure': structure,
                'bridges': bridges,
                'routing': {
                    'detected_format': format_type,
                    'confidence': 0.5,
                    'fallback_used': True
                }
            }
            
        except Exception as e:
            logger.error(f"Fallback parsing failed: {e}")
            return self._error_response(file_path, str(e))
    
    def _detect_basic_structure(self, content: str, format_type: str) -> Dict[str, Any]:
        """Detect basic structure in configuration files."""
        lines = content.split('\n')
        structure: Dict[str, Any] = {
            'sections': [],
            'key_value_pairs': 0,
            'comments': 0,
            'blank_lines': 0
        }
        
        current_section = None
        
        for line in lines:
            stripped = line.strip()
            
            # Count blank lines
            if not stripped:
                structure['blank_lines'] += 1
                continue
                
            # Count comments
            if stripped.startswith('#') or stripped.startswith('//') or stripped.startswith(';'):
                structure['comments'] += 1
                continue
                
            # Detect sections
            if format_type in ['ini', 'toml', 'conf']:
                section_match = re.match(r'^\[([^\]]+)\]$', stripped)
                if section_match:
                    current_section = section_match.group(1)
                    for idx, l in enumerate(lines):
                        if l == line:
                            structure['sections'].append({
                                'name': current_section,
                                'line': idx + 1
                            })
                            break
                    continue
                    
            # Count key-value pairs
            if '=' in stripped or ':' in stripped:
                structure['key_value_pairs'] += 1
                
        return structure
    
    def _detect_basic_bridges(
        self, 
        content: str, 
        file_path: Path,
        format_type: str
    ) -> List[Dict[str, Any]]:
        """Detect basic bridges in configuration files."""
        bridges = []
        
        # Look for file paths
        file_pattern = re.compile(r'["\']?(/[\w\-./]+|[\w\-./]+\.(py|js|java|cpp|go|rs))["\']?')
        for match in file_pattern.finditer(content):
            path = match.group(1)
            bridges.append({
                'type': 'config_file_reference',
                'source': f"{file_path.name}",
                'target': path,
                'target_type': 'file_path',
                'confidence': 0.6
            })
        
        # Look for class/module references
        module_pattern = re.compile(r'["\']?([\w\.]+\.([\w]+))["\']?\s*[=:]')
        for match in module_pattern.finditer(content):
            if '.' in match.group(1):
                bridges.append({
                    'type': 'config_module_reference',
                    'source': f"{file_path.name}",
                    'target': match.group(1),
                    'target_type': 'module_path',
                    'confidence': 0.7
                })
        
        # Look for URLs/endpoints
        url_pattern = re.compile(r'https?://[\w\-._~:/?#[\]@!$&\'()*+,;=]+')
        for match in url_pattern.finditer(content):
            bridges.append({
                'type': 'config_endpoint',
                'source': f"{file_path.name}",
                'target': match.group(0),
                'target_type': 'url',
                'confidence': 0.9
            })
        
        return bridges
    
    def _error_response(self, file_path: Path, error: str) -> Dict[str, Any]:
        """Generate error response."""
        return {
            'content': f"[Configuration File Error]\nFile: {file_path}\nError: {error}",
            'file_type': 'unknown',
            'purpose': 'error',
            'error': error,
            'routing': {
                'detected_format': 'unknown',
                'confidence': 0.0,
                'fallback_used': False
            }
        }
    
    def detect_all_formats(self, directory: Path) -> Dict[str, List[Path]]:
        """
        Scan directory and group configuration files by detected format.
        
        Useful for batch processing and analysis.
        """
        format_groups: Dict[str, List[Path]] = {}
        
        # Common config file patterns
        config_patterns = [
            '*.xml', '*.yaml', '*.yml', '*.json', '*.toml',
            '*.ini', '*.cfg', '*.conf', '*.config', '*.properties',
            'config.*', 'settings.*', 'configuration.*'
        ]
        
        for pattern in config_patterns:
            for file_path in directory.rglob(pattern):
                if file_path.is_file():
                    format_type = self._detect_format(file_path)
                    
                    if format_type not in format_groups:
                        format_groups[format_type] = []
                    format_groups[format_type].append(file_path)
        
        return format_groups