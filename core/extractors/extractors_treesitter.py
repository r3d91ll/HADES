"""
Tree-sitter Symbol Extractor
============================

Extracts symbol tables and structural information from code files using Tree-sitter.
This module is specifically designed for the GitHub pipeline to provide rich metadata
for the Jina v4 coding LoRA embeddings.

In Actor-Network Theory terms, this extractor serves as a translation device between
code syntax (tree structures) and semantic understanding (symbol tables). It transforms
the implicit knowledge embedded in code structure into explicit metadata that can
enhance embedding quality.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

try:
    from tree_sitter import Language, Parser

    # Try to import language modules directly (tree-sitter 0.25+ compatible)
    try:
        import tree_sitter_python
        import tree_sitter_javascript
        import tree_sitter_go
        TREE_SITTER_AVAILABLE = True
        TREE_SITTER_VERSION = "0.25.2"
        USE_DIRECT_IMPORTS = True
    except ImportError:
        # Fallback to tree-sitter-languages (older API)
        try:
            from tree_sitter_languages import get_language
            TREE_SITTER_AVAILABLE = True
            TREE_SITTER_VERSION = "legacy"
            USE_DIRECT_IMPORTS = False
        except ImportError as e:
            TREE_SITTER_AVAILABLE = False
            TREE_SITTER_VERSION = "unavailable"
            USE_DIRECT_IMPORTS = False
            logging.warning(f"Tree-sitter languages not available: {e}")
except ImportError as e:
    TREE_SITTER_AVAILABLE = False
    TREE_SITTER_VERSION = "unavailable"
    USE_DIRECT_IMPORTS = False
    logging.warning(f"Tree-sitter not available: {e}")

logger = logging.getLogger(__name__)


class TreeSitterExtractor:
    """
    Extract symbol tables and code structure using Tree-sitter parsers.
    
    This extractor identifies and categorizes code symbols (functions, classes,
    variables, imports) to provide rich metadata for embedding generation.
    The symbol table acts as a 'boundary object' in ANT terms - maintaining
    coherence across different analytical contexts.
    """

    # Language mapping to Tree-sitter language names
    LANGUAGE_MAP = {
        '.py': 'python',
        '.js': 'javascript',
        '.jsx': 'javascript',
        '.ts': 'typescript',
        '.tsx': 'typescript',
        '.java': 'java',
        '.go': 'go',
        '.rs': 'rust',
        '.cpp': 'cpp',
        '.cc': 'cpp',
        '.cxx': 'cpp',
        '.c': 'c',
        '.h': 'c',
        '.hpp': 'cpp',
        '.cs': 'c_sharp',
        '.rb': 'ruby',
        '.php': 'php',
        '.swift': 'swift',
        '.kt': 'kotlin',
        '.scala': 'scala',
        '.r': 'r',
        '.lua': 'lua',
        '.jl': 'julia',
        '.m': 'objc',
        '.mm': 'objc',
        '.sh': 'bash',
        '.bash': 'bash',
        '.zsh': 'bash',
        '.yaml': 'yaml',
        '.yml': 'yaml',
        '.json': 'json',
        '.xml': 'xml',
        '.html': 'html',
        '.css': 'css',
        '.scss': 'scss',
        '.sql': 'sql',
        '.md': 'markdown',
        '.markdown': 'markdown',
        '.toml': 'toml',
        '.ini': 'ini',
        '.cfg': 'ini',
        '.conf': 'ini',
        '.properties': 'ini',
        '.env': 'ini'
    }

    def __init__(self):
        """Initialize the Tree-sitter extractor."""
        self.parsers: Dict[str, Any] = {}

        if not TREE_SITTER_AVAILABLE:
            logger.warning("Tree-sitter not available - symbol extraction will be skipped")
            return

        # Pre-load common language parsers when available
        for lang in ['python', 'javascript', 'typescript', 'java', 'go', 'rust', 'cpp', 'c']:
            parser = self._create_parser(lang)
            if parser is not None:
                self.parsers[lang] = parser
                logger.debug("Loaded Tree-sitter parser for %s", lang)

    def extract_symbols(self, file_path: Union[str, Path], content: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract symbol table and structural information from a code file.
        
        Args:
            file_path: Path to the code file
            content: Optional file content (if already loaded)
            
        Returns:
            Dictionary containing:
            - symbols: Categorized symbols (functions, classes, imports, etc.)
            - metrics: Code complexity metrics
            - structure: Nested structure information
            - language: Detected programming language
        """
        if not TREE_SITTER_AVAILABLE:
            return self._empty_result()

        file_path = Path(file_path)

        # Detect language from file extension
        language = self._detect_language(file_path)
        if not language:
            logger.debug(f"No Tree-sitter parser for {file_path.suffix}")
            return self._empty_result()

        # Load content if not provided
        if content is None:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
            except Exception as e:
                logger.error(f"Failed to read file {file_path}: {e}")
                return self._empty_result()

        # Get or load parser
        parser = self._get_parser(language)
        if not parser:
            return self._empty_result()

        try:
            # Parse the code
            tree = parser.parse(bytes(content, 'utf-8'))

            # Extract symbols based on language
            symbols = self._extract_language_symbols(tree, content, language)

            # Calculate metrics
            metrics = self._calculate_metrics(tree, content)

            # Extract structure
            structure = self._extract_structure(tree, content, language)

            return {
                'symbols': symbols,
                'metrics': metrics,
                'structure': structure,
                'language': language,
                'tree_sitter_version': TREE_SITTER_VERSION,
            }

        except Exception as e:
            logger.error(f"Failed to parse {file_path} with Tree-sitter: {e}")
            return self._empty_result()

    def _detect_language(self, file_path: Path) -> Optional[str]:
        """Detect programming language from file extension."""
        suffix = file_path.suffix.lower()
        return self.LANGUAGE_MAP.get(suffix)

    def _get_parser(self, language: str):
        """Get or create parser for a language."""
        if language in self.parsers:
            return self.parsers[language]

        parser = self._create_parser(language)
        if parser is not None:
            self.parsers[language] = parser
        return parser

    def _create_parser(self, language: str) -> Optional[Parser]:
        """Instantiate a parser for a given language if available."""

        try:
            if USE_DIRECT_IMPORTS:
                # Use direct imports (tree-sitter 0.25+ API)
                lang_module_map = {
                    'python': 'tree_sitter_python',
                    'javascript': 'tree_sitter_javascript',
                    'typescript': 'tree_sitter_javascript',  # TypeScript uses JS parser
                    'go': 'tree_sitter_go',
                }
                module_name = lang_module_map.get(language)
                if not module_name:
                    return None

                import importlib
                try:
                    lang_module = importlib.import_module(module_name)
                    lang_obj = Language(lang_module.language())
                except (ImportError, AttributeError) as e:
                    logger.debug("Tree-sitter language '%s' unavailable: %s", language, e)
                    return None
            else:
                # Use tree-sitter-languages (legacy API)
                lang_obj = get_language(language)
        except Exception as exc:  # pragma: no cover - depends on optional languages
            logger.debug("Tree-sitter language '%s' unavailable: %s", language, exc)
            return None

        parser = Parser()
        try:
            parser.language = lang_obj
        except AttributeError:
            # Fallback for older API
            parser.set_language(lang_obj)
        return parser

    def _extract_language_symbols(self, tree, content: str, language: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract symbols specific to each programming language.
        
        Returns a dictionary with categories:
        - functions: Function/method definitions
        - classes: Class/type definitions
        - imports: Import statements
        - variables: Global/constant variables
        - exports: Exported symbols (for JS/TS)
        """
        symbols: Dict[str, List[Dict[str, Any]]] = {
            'functions': [],
            'classes': [],
            'imports': [],
            'variables': [],
            'exports': [],
            'interfaces': [],
            'enums': [],
            'types': []
        }

        # Language-specific extractors
        if language == 'python':
            symbols = self._extract_python_symbols(tree.root_node, content)
        elif language in ['javascript', 'typescript']:
            symbols = self._extract_javascript_symbols(tree.root_node, content)
        elif language == 'java':
            symbols = self._extract_java_symbols(tree.root_node, content)
        elif language == 'go':
            symbols = self._extract_go_symbols(tree.root_node, content)
        elif language == 'rust':
            symbols = self._extract_rust_symbols(tree.root_node, content)
        elif language in ['c', 'cpp']:
            symbols = self._extract_c_symbols(tree.root_node, content)
        elif language in ['json', 'yaml', 'xml', 'toml']:
            # For config files, provide minimal structural metadata
            # Let Jina v4's coding LoRA handle semantic understanding
            symbols = self._extract_config_metadata(tree.root_node, content, language)
        else:
            # Generic extraction for other languages
            symbols = self._extract_generic_symbols(tree.root_node, content)

        return symbols

    def _extract_python_symbols(self, node, content: str) -> Dict[str, List[Dict[str, Any]]]:
        """Extract symbols from Python code."""
        symbols: Dict[str, List[Dict[str, Any]]] = {
            'functions': [],
            'classes': [],
            'imports': [],
            'variables': [],
            'decorators': []
        }

        def traverse(node, scope='module'):
            if node.type == 'function_definition':
                name_node = node.child_by_field_name('name')
                if name_node:
                    func_info = {
                        'name': content[name_node.start_byte:name_node.end_byte],
                        'line': name_node.start_point[0] + 1,
                        'scope': scope,
                        'parameters': self._extract_parameters(node, content),
                        'decorators': self._extract_decorators(node, content),
                        'docstring': self._extract_docstring(node, content)
                    }
                    symbols['functions'].append(func_info)

            elif node.type == 'class_definition':
                name_node = node.child_by_field_name('name')
                if name_node:
                    class_name = content[name_node.start_byte:name_node.end_byte]
                    class_info = {
                        'name': class_name,
                        'line': name_node.start_point[0] + 1,
                        'scope': scope,
                        'bases': self._extract_bases(node, content),
                        'decorators': self._extract_decorators(node, content),
                        'docstring': self._extract_docstring(node, content)
                    }
                    symbols['classes'].append(class_info)
                    # Traverse class body with updated scope
                    for child in node.children:
                        traverse(child, f"{scope}.{class_name}")
                    return  # Don't traverse children again

            elif node.type in ['import_statement', 'import_from_statement']:
                import_info = {
                    'statement': content[node.start_byte:node.end_byte].strip(),
                    'line': node.start_point[0] + 1,
                    'type': 'from_import' if node.type == 'import_from_statement' else 'import'
                }
                symbols['imports'].append(import_info)

            elif node.type == 'assignment' and scope == 'module':
                # Global variable assignments
                left = node.child_by_field_name('left')
                if left and left.type == 'identifier':
                    var_name = content[left.start_byte:left.end_byte]
                    if var_name.isupper():  # Likely a constant
                        symbols['variables'].append({
                            'name': var_name,
                            'line': left.start_point[0] + 1,
                            'type': 'constant',
                            'scope': scope
                        })

            # Traverse children
            for child in node.children:
                traverse(child, scope)

        traverse(node)
        return symbols

    def _extract_javascript_symbols(self, node, content: str) -> Dict[str, List[Dict[str, Any]]]:
        """Extract symbols from JavaScript/TypeScript code."""
        symbols: Dict[str, List[Dict[str, Any]]] = {
            'functions': [],
            'classes': [],
            'imports': [],
            'exports': [],
            'variables': [],
            'interfaces': [],
            'types': []
        }

        def traverse(node):
            if node.type in ['function_declaration', 'function_expression', 'arrow_function']:
                name_node = node.child_by_field_name('name')
                if name_node:
                    # Check for generator by looking for '*' token
                    # Rely on the node's generator attribute for JS/TS
                    is_generator = bool(getattr(node, 'generator', False))

                    symbols['functions'].append({
                        'name': content[name_node.start_byte:name_node.end_byte],
                        'line': name_node.start_point[0] + 1,
                        'async': self._has_child_type(node, 'async'),
                        'generator': is_generator
                    })

            elif node.type == 'class_declaration':
                name_node = node.child_by_field_name('name')
                if name_node:
                    symbols['classes'].append({
                        'name': content[name_node.start_byte:name_node.end_byte],
                        'line': name_node.start_point[0] + 1,
                        'extends': self._get_extends(node, content)
                    })

            elif node.type in ['import_statement', 'import_specifier']:
                symbols['imports'].append({
                    'statement': content[node.start_byte:node.end_byte].strip(),
                    'line': node.start_point[0] + 1
                })

            elif node.type in ['export_statement', 'export_specifier']:
                symbols['exports'].append({
                    'statement': content[node.start_byte:node.end_byte].strip(),
                    'line': node.start_point[0] + 1
                })

            elif node.type == 'interface_declaration':
                name_node = node.child_by_field_name('name')
                if name_node:
                    symbols['interfaces'].append({
                        'name': content[name_node.start_byte:name_node.end_byte],
                        'line': name_node.start_point[0] + 1
                    })

            elif node.type == 'type_alias_declaration':
                name_node = node.child_by_field_name('name')
                if name_node:
                    symbols['types'].append({
                        'name': content[name_node.start_byte:name_node.end_byte],
                        'line': name_node.start_point[0] + 1
                    })

            for child in node.children:
                traverse(child)

        traverse(node)
        return symbols

    def _extract_java_symbols(self, node, content: str) -> Dict[str, List[Dict[str, Any]]]:
        """Extract symbols from Java code."""
        symbols: Dict[str, List[Dict[str, Any]]] = {
            'functions': [],
            'classes': [],
            'imports': [],
            'interfaces': [],
            'enums': [],
            'annotations': []
        }

        def traverse(node):
            if node.type == 'method_declaration':
                name_node = node.child_by_field_name('name')
                if name_node:
                    symbols['functions'].append({
                        'name': content[name_node.start_byte:name_node.end_byte],
                        'line': name_node.start_point[0] + 1,
                        'modifiers': self._extract_modifiers(node, content)
                    })

            elif node.type == 'class_declaration':
                name_node = node.child_by_field_name('name')
                if name_node:
                    symbols['classes'].append({
                        'name': content[name_node.start_byte:name_node.end_byte],
                        'line': name_node.start_point[0] + 1,
                        'modifiers': self._extract_modifiers(node, content)
                    })

            elif node.type == 'import_declaration':
                symbols['imports'].append({
                    'statement': content[node.start_byte:node.end_byte].strip(),
                    'line': node.start_point[0] + 1
                })

            elif node.type == 'interface_declaration':
                name_node = node.child_by_field_name('name')
                if name_node:
                    symbols['interfaces'].append({
                        'name': content[name_node.start_byte:name_node.end_byte],
                        'line': name_node.start_point[0] + 1
                    })

            elif node.type == 'enum_declaration':
                name_node = node.child_by_field_name('name')
                if name_node:
                    symbols['enums'].append({
                        'name': content[name_node.start_byte:name_node.end_byte],
                        'line': name_node.start_point[0] + 1
                    })

            for child in node.children:
                traverse(child)

        traverse(node)
        return symbols

    def _extract_go_symbols(self, node, content: str) -> Dict[str, List[Dict[str, Any]]]:
        """Extract symbols from Go code."""
        symbols: Dict[str, List[Dict[str, Any]]] = {
            'functions': [],
            'types': [],
            'imports': [],
            'interfaces': [],
            'constants': [],
            'variables': []
        }

        def traverse(node):
            if node.type == 'function_declaration':
                name_node = node.child_by_field_name('name')
                if name_node:
                    symbols['functions'].append({
                        'name': content[name_node.start_byte:name_node.end_byte],
                        'line': name_node.start_point[0] + 1,
                        'receiver': self._extract_receiver(node, content)
                    })

            elif node.type == 'type_declaration':
                for spec in node.children:
                    if spec.type == 'type_spec':
                        name_node = spec.child_by_field_name('name')
                        if name_node:
                            symbols['types'].append({
                                'name': content[name_node.start_byte:name_node.end_byte],
                                'line': name_node.start_point[0] + 1
                            })

            elif node.type == 'import_declaration':
                symbols['imports'].append({
                    'statement': content[node.start_byte:node.end_byte].strip(),
                    'line': node.start_point[0] + 1
                })

            elif node.type == 'const_declaration':
                for spec in node.children:
                    if spec.type == 'const_spec':
                        name_node = spec.child_by_field_name('name')
                        if name_node:
                            symbols['constants'].append({
                                'name': content[name_node.start_byte:name_node.end_byte],
                                'line': name_node.start_point[0] + 1
                            })

            for child in node.children:
                traverse(child)

        traverse(node)
        return symbols

    def _extract_rust_symbols(self, node, content: str) -> Dict[str, List[Dict[str, Any]]]:
        """Extract symbols from Rust code."""
        symbols: Dict[str, List[Dict[str, Any]]] = {
            'functions': [],
            'structs': [],
            'enums': [],
            'traits': [],
            'imports': [],
            'types': [],
            'macros': []
        }

        def traverse(node):
            if node.type == 'function_item':
                name_node = node.child_by_field_name('name')
                if name_node:
                    symbols['functions'].append({
                        'name': content[name_node.start_byte:name_node.end_byte],
                        'line': name_node.start_point[0] + 1,
                        'async': self._has_child_type(node, 'async')
                    })

            elif node.type == 'struct_item':
                name_node = node.child_by_field_name('name')
                if name_node:
                    symbols['structs'].append({
                        'name': content[name_node.start_byte:name_node.end_byte],
                        'line': name_node.start_point[0] + 1
                    })

            elif node.type == 'enum_item':
                name_node = node.child_by_field_name('name')
                if name_node:
                    symbols['enums'].append({
                        'name': content[name_node.start_byte:name_node.end_byte],
                        'line': name_node.start_point[0] + 1
                    })

            elif node.type == 'trait_item':
                name_node = node.child_by_field_name('name')
                if name_node:
                    symbols['traits'].append({
                        'name': content[name_node.start_byte:name_node.end_byte],
                        'line': name_node.start_point[0] + 1
                    })

            elif node.type == 'use_declaration':
                symbols['imports'].append({
                    'statement': content[node.start_byte:node.end_byte].strip(),
                    'line': node.start_point[0] + 1
                })

            for child in node.children:
                traverse(child)

        traverse(node)
        return symbols

    def _extract_c_symbols(self, node, content: str) -> Dict[str, List[Dict[str, Any]]]:
        """Extract symbols from C/C++ code."""
        symbols: Dict[str, List[Dict[str, Any]]] = {
            'functions': [],
            'structs': [],
            'classes': [],
            'includes': [],
            'defines': [],
            'typedefs': []
        }

        def traverse(node):
            if node.type == 'function_definition':
                # Look for the declarator which contains the name
                declarator = node.child_by_field_name('declarator')
                if declarator:
                    name = self._extract_function_name(declarator, content)
                    if name:
                        symbols['functions'].append({
                            'name': name,
                            'line': node.start_point[0] + 1
                        })

            elif node.type == 'struct_specifier':
                name_node = node.child_by_field_name('name')
                if name_node:
                    symbols['structs'].append({
                        'name': content[name_node.start_byte:name_node.end_byte],
                        'line': name_node.start_point[0] + 1
                    })

            elif node.type == 'class_specifier':
                name_node = node.child_by_field_name('name')
                if name_node:
                    symbols['classes'].append({
                        'name': content[name_node.start_byte:name_node.end_byte],
                        'line': name_node.start_point[0] + 1
                    })

            elif node.type == 'preproc_include':
                symbols['includes'].append({
                    'statement': content[node.start_byte:node.end_byte].strip(),
                    'line': node.start_point[0] + 1
                })

            elif node.type == 'preproc_def':
                name_node = node.child_by_field_name('name')
                if name_node:
                    symbols['defines'].append({
                        'name': content[name_node.start_byte:name_node.end_byte],
                        'line': name_node.start_point[0] + 1
                    })

            for child in node.children:
                traverse(child)

        traverse(node)
        return symbols

    def _extract_config_metadata(self, node, content: str, language: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract minimal structural metadata from config files.
        
        We don't interpret semantic meaning - that's Jina v4's job.
        We just provide basic structural information for context.
        """
        structure_info: List[Dict[str, Any]] = []
        symbols: Dict[str, Any] = {
            'config_type': language,
            'structure_info': structure_info
        }

        # Count basic structural elements without interpreting them
        key_count = 0
        max_depth = 0

        def traverse(node, depth=0):
            nonlocal key_count, max_depth
            max_depth = max(max_depth, depth)

            # Count keys/properties without interpreting their meaning
            if language == 'json':
                if node.type in ['pair', 'property']:
                    key_count += 1
            elif language == 'yaml':
                if node.type in ['block_mapping_pair', 'flow_pair']:
                    key_count += 1
            elif language == 'xml':
                if node.type in ['element', 'start_tag']:
                    key_count += 1
            elif language == 'toml':
                if node.type in ['pair', 'table']:
                    key_count += 1

            for child in node.children:
                traverse(child, depth + 1)

        traverse(node)

        # Provide minimal metadata
        structure_info.append({
            'type': 'config_metadata',
            'format': language,
            'key_count': key_count,
            'max_nesting_depth': max_depth,
            'is_valid_syntax': True  # If we got here, it parsed successfully
        })

        return symbols

    def _extract_generic_symbols(self, node, content: str) -> Dict[str, List[Dict[str, Any]]]:
        """Generic symbol extraction for unsupported languages."""
        symbols: Dict[str, List[Dict[str, Any]]] = {
            'identifiers': [],
            'literals': []
        }

        def traverse(node):
            if node.type == 'identifier':
                text = content[node.start_byte:node.end_byte]
                if len(text) > 2:  # Skip short identifiers
                    symbols['identifiers'].append({
                        'name': text,
                        'line': node.start_point[0] + 1
                    })

            for child in node.children:
                traverse(child)

        traverse(node)

        # Deduplicate identifiers
        seen: Set[str] = set()
        unique_identifiers: List[Dict[str, Any]] = []
        for ident in symbols['identifiers']:
            if ident['name'] not in seen:
                seen.add(ident['name'])
                unique_identifiers.append(ident)
        symbols['identifiers'] = unique_identifiers[:100]  # Limit to top 100

        return symbols

    def _calculate_metrics(self, tree, content: str) -> Dict[str, Any]:
        """
        Calculate code complexity metrics.
        
        Returns metrics including:
        - lines_of_code: Total lines
        - complexity: Cyclomatic complexity estimate
        - depth: Maximum nesting depth
        - node_count: Total AST nodes
        """
        lines = content.count('\n') + 1

        # Count nodes and calculate depth and complexity
        node_count = 0
        max_depth = 0
        complexity = 1  # Start with 1 for the function/file itself

        # Define control flow node types per language
        control_flow_nodes = {
            'python': ['if_statement', 'elif_clause', 'else_clause', 'for_statement',
                      'while_statement', 'try_statement', 'except_clause', 'with_statement'],
            'javascript': ['if_statement', 'else_clause', 'for_statement', 'for_in_statement',
                          'while_statement', 'do_statement', 'switch_statement', 'case_statement',
                          'catch_clause', 'ternary_expression'],
            'typescript': ['if_statement', 'else_clause', 'for_statement', 'for_in_statement',
                          'while_statement', 'do_statement', 'switch_statement', 'case_statement',
                          'catch_clause', 'ternary_expression'],
            'java': ['if_statement', 'else_clause', 'for_statement', 'enhanced_for_statement',
                    'while_statement', 'do_statement', 'switch_statement', 'case_statement',
                    'catch_clause', 'ternary_expression'],
            'c': ['if_statement', 'else_clause', 'for_statement', 'while_statement',
                 'do_statement', 'switch_statement', 'case_statement'],
            'cpp': ['if_statement', 'else_clause', 'for_statement', 'while_statement',
                   'do_statement', 'switch_statement', 'case_statement', 'catch_clause'],
            'go': ['if_statement', 'else_clause', 'for_statement', 'switch_statement',
                  'case_clause', 'type_switch_statement'],
            'rust': ['if_expression', 'else_clause', 'for_expression', 'while_expression',
                    'loop_expression', 'match_expression', 'match_arm']
        }

        # Get language-specific control flow nodes or use a default set
        language = getattr(tree, 'language', 'python')
        cf_nodes = control_flow_nodes.get(language, control_flow_nodes['python'])

        def traverse(node, depth=0, in_string_or_comment=False):
            nonlocal node_count, max_depth, complexity

            # Skip nodes inside strings and comments
            if node.type in ['string', 'string_literal', 'comment', 'block_comment', 'line_comment']:
                in_string_or_comment = True

            if not in_string_or_comment:
                node_count += 1
                max_depth = max(max_depth, depth)

                # Increment complexity for control flow nodes
                if node.type in cf_nodes:
                    complexity += 1

            for child in node.children:
                traverse(child, depth + 1, in_string_or_comment)

        traverse(tree.root_node)

        return {
            'lines_of_code': lines,
            'complexity': complexity,
            'max_depth': max_depth,
            'node_count': node_count,
            'avg_depth': max_depth / max(node_count, 1)
        }

    def _extract_structure(self, tree, content: str, language: str) -> Dict[str, Any]:
        """
        Extract high-level structural information.
        
        Returns nested structure showing code organization.
        """
        children: List[Dict[str, Any]] = []
        structure: Dict[str, Any] = {
            'type': 'module',
            'language': language,
            'children': children
        }

        # Extract top-level structure
        for child in tree.root_node.children:
            if child.type in ['function_definition', 'function_declaration', 'function_item']:
                children.append({
                    'type': 'function',
                    'line': child.start_point[0] + 1,
                    'end_line': child.end_point[0] + 1
                })
            elif child.type in ['class_definition', 'class_declaration', 'class_specifier']:
                children.append({
                    'type': 'class',
                    'line': child.start_point[0] + 1,
                    'end_line': child.end_point[0] + 1
                })

        return structure

    # Helper methods
    def _extract_parameters(self, node, content: str) -> List[str]:
        """Extract function parameters."""
        params = []
        param_list = node.child_by_field_name('parameters')
        if param_list:
            for child in param_list.children:
                if child.type in ['identifier', 'typed_parameter', 'simple_parameter']:
                    param_text = content[child.start_byte:child.end_byte].strip()
                    if param_text and param_text not in ['(', ')', ',']:
                        params.append(param_text)
        return params

    def _extract_decorators(self, node, content: str) -> List[str]:
        """Extract Python decorators."""
        decorators = []
        for child in node.children:
            if child.type == 'decorator':
                dec_text = content[child.start_byte:child.end_byte].strip()
                decorators.append(dec_text)
        return decorators

    def _extract_docstring(self, node, content: str) -> Optional[str]:
        """Extract Python docstring."""
        body = node.child_by_field_name('body')
        if body and body.children:
            first_stmt = body.children[0]
            if first_stmt.type == 'expression_statement':
                for child in first_stmt.children:
                    if child.type == 'string':
                        docstring = content[child.start_byte:child.end_byte]

                        # Handle raw string prefixes (r, R, f, F, fr, rf, etc.)
                        prefix_len = 0
                        if docstring.lower().startswith(('r"""', "r'''", 'f"""', "f'''",
                                                        'fr"""', "fr'''", 'rf"""', "rf'''")):
                            # Handle compound prefixes (fr, rf)
                            if docstring[:2].lower() in ('fr', 'rf'):
                                prefix_len = 2
                            else:
                                prefix_len = 1
                        elif docstring.lower().startswith(('r"', "r'", 'f"', "f'")):
                            prefix_len = 1

                        # Strip prefix and quotes
                        if prefix_len > 0:
                            docstring = docstring[prefix_len:]

                        # Clean up quotes
                        if docstring.startswith('"""') or docstring.startswith("'''"):
                            docstring = docstring[3:-3]
                        elif docstring.startswith('"') or docstring.startswith("'"):
                            docstring = docstring[1:-1]

                        return docstring.strip()
        return None

    def _extract_bases(self, node, content: str) -> List[str]:
        """Extract base classes."""
        bases = []
        superclasses = node.child_by_field_name('superclasses')
        if superclasses:
            for child in superclasses.children:
                if child.type == 'identifier':
                    bases.append(content[child.start_byte:child.end_byte])
        return bases

    def _has_child_type(self, node, child_type: str) -> bool:
        """Check if node has a child of specific type."""
        for child in node.children:
            if child.type == child_type:
                return True
        return False

    def _get_extends(self, node, content: str) -> Optional[str]:
        """Get extended class for JavaScript/TypeScript."""
        heritage = node.child_by_field_name('heritage')
        if heritage:
            for child in heritage.children:
                if child.type == 'extends_clause':
                    return content[child.start_byte:child.end_byte].replace('extends', '').strip()
        return None

    def _extract_modifiers(self, node, content: str) -> List[str]:
        """Extract Java modifiers."""
        modifiers = []
        mod_node = node.child_by_field_name('modifiers')
        if mod_node:
            for child in mod_node.children:
                if child.type == 'modifier':
                    modifiers.append(content[child.start_byte:child.end_byte])
        return modifiers

    def _extract_receiver(self, node, content: str) -> Optional[str]:
        """Extract Go method receiver."""
        params = node.child_by_field_name('parameters')
        if params and params.children:
            first_param = params.children[0]
            if first_param.type == 'parameter_list':
                return content[first_param.start_byte:first_param.end_byte]
        return None

    def _extract_function_name(self, declarator, content: str) -> Optional[str]:
        """Extract function name from C/C++ declarator."""
        if declarator.type == 'function_declarator':
            for child in declarator.children:
                if child.type == 'identifier':
                    return content[child.start_byte:child.end_byte]
                elif child.type == 'field_identifier':
                    return content[child.start_byte:child.end_byte]
        elif declarator.type == 'pointer_declarator':
            # Recursive call for pointer functions
            for child in declarator.children:
                name = self._extract_function_name(child, content)
                if name:
                    return name
        elif declarator.type == 'identifier':
            return content[declarator.start_byte:declarator.end_byte]
        return None

    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result structure."""
        return {
            'symbols': {
                'functions': [],
                'classes': [],
                'imports': [],
                'variables': [],
                'exports': []
            },
            'metrics': {
                'lines_of_code': 0,
                'complexity': 0,
                'max_depth': 0,
                'node_count': 0
            },
            'structure': {},
            'language': None
        }

    def generate_symbol_hash(self, symbols: Dict[str, List]) -> str:
        """
        Generate a hash of the symbol table for comparison.
        
        This creates a stable hash that can be used to detect changes
        in code structure without comparing full content.
        """
        # Create a stable string representation
        symbol_str = ""
        for category in sorted(symbols.keys()):
            symbol_str += f"{category}:"
            items = symbols[category]
            if items and isinstance(items[0], dict):
                names = sorted([item.get('name', '') for item in items if item.get('name')])
                symbol_str += ','.join(names)
            symbol_str += ';'

        return hashlib.sha256(symbol_str.encode()).hexdigest()
