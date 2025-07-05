"""
AST-based code analyzer for Jina v4 processor.

This module provides AST analysis capabilities for code files, extracting
semantic information that enhances Jina v4's understanding of code structure.
"""

import ast
import logging
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CodeSymbol:
    """Represents a code symbol extracted from AST."""
    name: str
    type: str  # 'class', 'function', 'method', 'variable', 'import'
    line_start: int
    line_end: int
    docstring: Optional[str] = None
    signature: Optional[str] = None
    decorators: List[str] = field(default_factory=list)
    parent: Optional[str] = None
    children: List[str] = field(default_factory=list)
    dependencies: Set[str] = field(default_factory=set)


class ASTAnalyzer:
    """
    Analyzes Python code using AST to extract semantic information.
    
    This analyzer is specifically designed to work with Jina v4's processing
    pipeline, providing rich metadata about code structure that can be used
    for better embeddings and keyword extraction.
    """
    
    def __init__(self) -> None:
        """Initialize the AST analyzer."""
        self.symbols: Dict[str, CodeSymbol] = {}
        self.imports: Dict[str, str] = {}  # name -> module mapping
        self.call_graph: Dict[str, Set[str]] = {}  # function -> called functions
        
    def analyze_code(self, code: str, filename: str = "unknown.py") -> Dict[str, Any]:
        """
        Analyze Python code and extract semantic information.
        
        Args:
            code: Python source code
            filename: Name of the file being analyzed
            
        Returns:
            Dictionary containing:
            - symbols: Map of symbol names to CodeSymbol objects
            - imports: Import relationships
            - call_graph: Function call relationships
            - structure: Hierarchical code structure
            - keywords: Extracted technical keywords
        """
        try:
            tree = ast.parse(code, filename=filename)
            
            # Reset state
            self.symbols.clear()
            self.imports.clear()
            self.call_graph.clear()
            
            # Analyze the AST
            self._analyze_node(tree)
            
            # Build hierarchical structure
            structure = self._build_hierarchical_structure()
            
            # Extract keywords
            keywords = self._extract_keywords()
            
            return {
                "symbols": {name: self._symbol_to_dict(sym) 
                           for name, sym in self.symbols.items()},
                "imports": self.imports,
                "call_graph": {k: list(v) for k, v in self.call_graph.items()},
                "structure": structure,
                "keywords": keywords,
                "stats": {
                    "total_symbols": len(self.symbols),
                    "classes": sum(1 for s in self.symbols.values() if s.type == "class"),
                    "functions": sum(1 for s in self.symbols.values() if s.type in ["function", "method"]),
                    "imports": len(self.imports)
                }
            }
            
        except SyntaxError as e:
            logger.warning(f"Syntax error in {filename}: {e}")
            return self._empty_analysis()
        except Exception as e:
            logger.error(f"Error analyzing {filename}: {e}")
            return self._empty_analysis()
    
    def extract_symbol_chunks(self, code: str, max_chunk_size: int = 512) -> List[Dict[str, Any]]:
        """
        Extract code chunks based on symbol boundaries.
        
        Instead of arbitrary line-based chunking, this creates chunks that
        respect code structure (complete functions, classes, etc.).
        
        Args:
            code: Python source code
            max_chunk_size: Maximum size of each chunk in lines
            
        Returns:
            List of chunks with symbol information
        """
        analysis = self.analyze_code(code)
        if not analysis["symbols"]:
            return []
        
        lines = code.split('\n')
        chunks = []
        
        # Sort symbols by line number
        sorted_symbols = sorted(
            analysis["symbols"].items(),
            key=lambda x: x[1]["line_start"]
        )
        
        for symbol_name, symbol_info in sorted_symbols:
            # Skip small symbols (variables, imports)
            if symbol_info["type"] not in ["class", "function", "method"]:
                continue
                
            start = symbol_info["line_start"] - 1  # Convert to 0-based
            end = symbol_info["line_end"]
            
            # Extract symbol code
            symbol_code = '\n'.join(lines[start:end])
            
            # If symbol is too large, we need to split it
            # (This is rare but handles very large classes/functions)
            if end - start > max_chunk_size:
                # For now, we'll include the whole symbol
                # In production, you might want smarter splitting
                logger.warning(f"Large symbol {symbol_name}: {end - start} lines")
            
            chunk = {
                "content": symbol_code,
                "symbol": symbol_name,
                "type": symbol_info["type"],
                "line_range": (symbol_info["line_start"], symbol_info["line_end"]),
                "docstring": symbol_info.get("docstring"),
                "dependencies": list(symbol_info.get("dependencies", [])),
                "parent": symbol_info.get("parent"),
                "metadata": {
                    "chunk_type": "code_symbol",
                    "language": "python"
                }
            }
            
            chunks.append(chunk)
        
        return chunks
    
    def _analyze_node(self, node: ast.AST, parent: Optional[str] = None) -> None:
        """Recursively analyze AST nodes."""
        if isinstance(node, ast.ClassDef):
            self._analyze_class(node, parent)
        elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            self._analyze_function(node, parent)
        elif isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
            self._analyze_import(node)
        
        # Recurse through child nodes
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                # These are handled specially above
                continue
            self._analyze_node(child, parent)
    
    def _analyze_class(self, node: ast.ClassDef, parent: Optional[str] = None) -> None:
        """Analyze a class definition."""
        symbol = CodeSymbol(
            name=node.name,
            type="class",
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            docstring=ast.get_docstring(node),
            parent=parent,
            decorators=[self._get_decorator_name(d) for d in node.decorator_list]
        )
        
        # Store symbol
        full_name = f"{parent}.{node.name}" if parent else node.name
        self.symbols[full_name] = symbol
        
        # Analyze methods
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._analyze_function(item, full_name)
    
    def _analyze_function(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], parent: Optional[str] = None) -> None:
        """Analyze a function/method definition."""
        # Get function signature
        args = []
        for arg in node.args.args:
            args.append(arg.arg)
        signature = f"({', '.join(args)})"
        
        symbol = CodeSymbol(
            name=node.name,
            type="method" if parent and self.symbols.get(parent, CodeSymbol("", "", 0, 0)).type == "class" else "function",
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            docstring=ast.get_docstring(node),
            signature=signature,
            parent=parent,
            decorators=[self._get_decorator_name(d) for d in node.decorator_list]
        )
        
        # Store symbol
        full_name = f"{parent}.{node.name}" if parent else node.name
        self.symbols[full_name] = symbol
        
        # Analyze function body for dependencies
        dependencies = self._extract_dependencies(node)
        symbol.dependencies = dependencies
        
        # Update call graph
        if full_name not in self.call_graph:
            self.call_graph[full_name] = set()
        self.call_graph[full_name].update(dependencies)
    
    def _analyze_import(self, node: Union[ast.Import, ast.ImportFrom]) -> None:
        """Analyze import statements."""
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name
                self.imports[name] = alias.name
        else:  # ImportFrom
            module = node.module or ""
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name
                self.imports[name] = f"{module}.{alias.name}"
    
    def _extract_dependencies(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> Set[str]:
        """Extract function calls from a function body."""
        dependencies = set()
        
        class CallVisitor(ast.NodeVisitor):
            def visit_Call(self, node: ast.Call) -> None:
                if isinstance(node.func, ast.Name):
                    dependencies.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    # Handle method calls like obj.method()
                    if isinstance(node.func.value, ast.Name):
                        dependencies.add(f"{node.func.value.id}.{node.func.attr}")
                self.generic_visit(node)
        
        visitor = CallVisitor()
        visitor.visit(node)
        
        return dependencies
    
    def _build_hierarchical_structure(self) -> Dict[str, Any]:
        """Build a hierarchical representation of the code structure."""
        structure: Dict[str, Any] = {}
        
        for full_name, symbol in self.symbols.items():
            if not symbol.parent:  # Top-level symbol
                structure[symbol.name] = {
                    "type": symbol.type,
                    "docstring": symbol.docstring,
                    "children": {}
                }
        
        # Add children
        for full_name, symbol in self.symbols.items():
            if symbol.parent:
                parent_parts = symbol.parent.split('.')
                current = structure
                
                for part in parent_parts:
                    if part in current:
                        current = current[part]["children"]
                
                if current is not None:
                    current[symbol.name] = {
                        "type": symbol.type,
                        "docstring": symbol.docstring,
                        "children": {}
                    }
        
        return structure
    
    def _extract_keywords(self) -> Dict[str, List[str]]:
        """Extract technical keywords from the analyzed code."""
        keywords: Dict[str, List[str]] = {
            "symbols": [],
            "patterns": [],
            "libraries": [],
            "concepts": []
        }
        
        # Symbol names (classes and functions)
        for name, symbol in self.symbols.items():
            if symbol.type in ["class", "function"]:
                keywords["symbols"].append(name)
        
        # Common patterns from decorators
        decorator_patterns = set()
        for symbol in self.symbols.values():
            decorator_patterns.update(symbol.decorators)
        keywords["patterns"] = list(decorator_patterns)
        
        # Libraries from imports
        keywords["libraries"] = list(set(self.imports.values()))
        
        # Concepts from docstrings (simple extraction)
        # In production, you'd want more sophisticated NLP here
        for symbol in self.symbols.values():
            if symbol.docstring:
                # Extract capitalized words as potential concepts
                words = symbol.docstring.split()
                concepts = [w for w in words if w[0].isupper() and len(w) > 3]
                keywords["concepts"].extend(concepts[:3])  # Limit per docstring
        
        keywords["concepts"] = list(set(keywords["concepts"]))
        
        return keywords
    
    def _get_decorator_name(self, decorator: ast.AST) -> str:
        """Extract decorator name from AST node."""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name):
            return decorator.func.id
        elif isinstance(decorator, ast.Attribute):
            return decorator.attr
        return "unknown"
    
    def _symbol_to_dict(self, symbol: CodeSymbol) -> Dict[str, Any]:
        """Convert CodeSymbol to dictionary."""
        return {
            "name": symbol.name,
            "type": symbol.type,
            "line_start": symbol.line_start,
            "line_end": symbol.line_end,
            "docstring": symbol.docstring,
            "signature": symbol.signature,
            "decorators": symbol.decorators,
            "parent": symbol.parent,
            "children": symbol.children,
            "dependencies": list(symbol.dependencies)
        }
    
    def _empty_analysis(self) -> Dict[str, Any]:
        """Return empty analysis structure."""
        return {
            "symbols": {},
            "imports": {},
            "call_graph": {},
            "structure": {},
            "keywords": {"symbols": [], "patterns": [], "libraries": [], "concepts": []},
            "stats": {
                "total_symbols": 0,
                "classes": 0,
                "functions": 0,
                "imports": 0
            }
        }