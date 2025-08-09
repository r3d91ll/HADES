#!/usr/bin/env python3
"""
Tree-sitter based symbol extraction for multiple languages.
Fixed version that works with tree-sitter-languages.

Theory Connection:
Symbols are the boundary objects between theory and practice. 
Imports create the dependency graph that shows knowledge flow.
"""

import hashlib
import re
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

# Tree-sitter imports
try:
    from tree_sitter_languages import get_language, get_parser
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    print("Warning: tree-sitter not installed. Install with:")
    print("pip install tree-sitter tree-sitter-languages")

logger = logging.getLogger(__name__)


class TreeSitterSymbolExtractor:
    """
    Extract symbols (functions, classes, imports) from code using tree-sitter.
    Preserves docstrings and nearby comments for semantic understanding.
    """
    
    # Token limits for different parts
    MAX_DOCSTRING_CHARS = 2000
    MAX_COMMENTS_CHARS = 2000  
    MAX_BODY_CHARS = 4000
    MAX_TOTAL_CHARS = 6000
    
    def __init__(self):
        """Initialize parsers for supported languages."""
        if not TREE_SITTER_AVAILABLE:
            raise ImportError("tree-sitter libraries not installed")
        
        self.parsers = {}
        self._init_parsers()
    
    def _init_parsers(self):
        """Initialize language parsers using tree-sitter-languages."""
        # Languages available in tree-sitter-languages
        supported_langs = [
            'c', 'cpp', 'python', 'javascript', 'typescript',
            'go', 'rust', 'java', 'ruby', 'php', 'c_sharp'
        ]
        
        for lang in supported_langs:
            try:
                parser = get_parser(lang)
                # Map c_sharp to csharp for our interface
                if lang == 'c_sharp':
                    self.parsers['csharp'] = parser
                else:
                    self.parsers[lang] = parser
                logger.debug(f"Initialized {lang} parser")
            except Exception as e:
                logger.debug(f"Parser {lang} not available: {e}")
    
    def extract_symbols(self, content: str, language: str) -> List[Dict]:
        """
        Extract all symbols from code.
        
        Args:
            content: Source code content
            language: Programming language
            
        Returns:
            List of symbol dictionaries
        """
        if language not in self.parsers:
            logger.debug(f"No parser for language: {language}")
            return []
        
        parser = self.parsers[language]
        tree = parser.parse(bytes(content, 'utf-8'))
        
        symbols = []
        
        # Extract different symbol types
        imports = self._extract_imports(tree.root_node, content, language)
        functions = self._extract_functions(tree.root_node, content, language)
        classes = self._extract_classes(tree.root_node, content, language)
        
        # Add all symbols
        symbols.extend(imports)
        symbols.extend(functions)
        symbols.extend(classes)
        
        return symbols
    
    def _extract_imports(self, root_node, content: str, language: str) -> List[Dict]:
        """Extract import statements."""
        imports = []
        
        # Define import node types per language
        import_types = {
            'python': ['import_statement', 'import_from_statement'],
            'javascript': ['import_statement'],
            'typescript': ['import_statement'],
            'java': ['import_declaration'],
            'go': ['import_declaration', 'import_spec'],
            'c': ['preproc_include'],
            'cpp': ['preproc_include'],
            'rust': ['use_declaration'],
            'ruby': ['require', 'require_relative'],
            'php': ['namespace_use_declaration', 'require_expression', 'include_expression']
        }
        
        if language not in import_types:
            return []
        
        # Traverse tree to find import nodes
        def traverse(node):
            if node.type in import_types.get(language, []):
                import_text = content[node.start_byte:node.end_byte]
                imported_name = self._parse_import_name(import_text, language)
                
                imports.append({
                    'type': 'import',
                    'name': imported_name,
                    'full_statement': import_text.strip(),
                    'start_byte': node.start_byte,
                    'end_byte': node.end_byte,
                    'start_line': node.start_point[0] + 1,
                    'end_line': node.end_point[0] + 1,
                })
            
            for child in node.children:
                traverse(child)
        
        traverse(root_node)
        return imports
    
    def _extract_functions(self, root_node, content: str, language: str) -> List[Dict]:
        """Extract function definitions."""
        functions = []
        
        # Define function node types per language
        function_node_types = {
            'python': ['function_definition'],
            'javascript': ['function_declaration', 'arrow_function', 'function'],
            'typescript': ['function_declaration', 'arrow_function', 'function'],
            'java': ['method_declaration'],
            'go': ['function_declaration', 'method_declaration'],
            'c': ['function_definition'],
            'cpp': ['function_definition'],
            'rust': ['function_item'],
            'ruby': ['method'],
            'php': ['function_definition', 'method_declaration']
        }
        
        if language not in function_node_types:
            return []
        
        def traverse(node):
            if node.type in function_node_types.get(language, []):
                # Extract function name
                func_name = self._get_function_name(node, content, language)
                
                if func_name:
                    # Extract signature
                    signature = self._get_function_signature(node, content, language)
                    
                    # Extract docstring
                    docstring = self._extract_docstring(node, content, language)
                    
                    # Extract body (limited)
                    body_text = content[node.start_byte:node.end_byte]
                    body_text = self._clean_and_limit_text(body_text, self.MAX_BODY_CHARS)
                    
                    # Extract nearby comments
                    comments = self._extract_nearby_comments(node, content)
                    
                    # Build embedding text
                    embedding_text = self._build_embedding_text(
                        name=func_name,
                        signature=signature,
                        docstring=docstring,
                        comments=comments,
                        body=body_text
                    )
                    
                    functions.append({
                        'type': 'function',
                        'name': func_name,
                        'signature': signature,
                        'docstring': docstring,
                        'comments_near': comments,
                        'body_text': body_text[:1000],
                        'embedding_text': embedding_text,
                        'start_byte': node.start_byte,
                        'end_byte': node.end_byte,
                        'start_line': node.start_point[0] + 1,
                        'end_line': node.end_point[0] + 1,
                        'imports': []
                    })
            
            for child in node.children:
                traverse(child)
        
        traverse(root_node)
        return functions
    
    def _extract_classes(self, root_node, content: str, language: str) -> List[Dict]:
        """Extract class definitions."""
        classes = []
        
        # Define class node types per language
        class_node_types = {
            'python': ['class_definition'],
            'javascript': ['class_declaration'],
            'typescript': ['class_declaration'],
            'java': ['class_declaration'],
            'cpp': ['class_specifier', 'struct_specifier'],
            'rust': ['struct_item', 'impl_item'],
            'ruby': ['class'],
            'php': ['class_declaration'],
            'csharp': ['class_declaration']
        }
        
        if language not in class_node_types:
            return []
        
        def traverse(node):
            if node.type in class_node_types.get(language, []):
                # Extract class name
                class_name = self._get_class_name(node, content, language)
                
                if class_name:
                    # Extract docstring
                    docstring = self._extract_docstring(node, content, language)
                    
                    # Extract body (limited)
                    body_text = content[node.start_byte:node.end_byte]
                    body_text = self._clean_and_limit_text(body_text, self.MAX_BODY_CHARS)
                    
                    # Extract nearby comments
                    comments = self._extract_nearby_comments(node, content)
                    
                    # Build embedding text
                    embedding_text = self._build_embedding_text(
                        name=class_name,
                        signature=f"class {class_name}",
                        docstring=docstring,
                        comments=comments,
                        body=body_text
                    )
                    
                    classes.append({
                        'type': 'class',
                        'name': class_name,
                        'signature': f"class {class_name}",
                        'docstring': docstring,
                        'comments_near': comments,
                        'body_text': body_text[:1000],
                        'embedding_text': embedding_text,
                        'start_byte': node.start_byte,
                        'end_byte': node.end_byte,
                        'start_line': node.start_point[0] + 1,
                        'end_line': node.end_point[0] + 1,
                        'imports': []
                    })
            
            for child in node.children:
                traverse(child)
        
        traverse(root_node)
        return classes
    
    def _get_function_name(self, node, content: str, language: str) -> Optional[str]:
        """Extract function name from node."""
        for child in node.children:
            if child.type == 'identifier':
                return content[child.start_byte:child.end_byte]
            elif child.type == 'property_identifier':
                return content[child.start_byte:child.end_byte]
            elif child.type == 'field_identifier':
                return content[child.start_byte:child.end_byte]
            elif child.type in ['function_declarator', 'method_declarator']:
                # C/C++ style - recurse into declarator
                return self._get_function_name(child, content, language)
        return None
    
    def _get_class_name(self, node, content: str, language: str) -> Optional[str]:
        """Extract class name from node."""
        for child in node.children:
            if child.type in ['identifier', 'type_identifier']:
                return content[child.start_byte:child.end_byte]
        return None
    
    def _get_function_signature(self, node, content: str, language: str) -> str:
        """Extract function signature."""
        # For now, just get the first line
        func_text = content[node.start_byte:node.end_byte]
        first_line = func_text.split('\n')[0]
        if '{' in first_line:
            first_line = first_line[:first_line.index('{')]
        return first_line.strip()
    
    def _parse_import_name(self, import_text: str, language: str) -> str:
        """Parse the actual module/package being imported."""
        if language == 'python':
            match = re.match(r'(?:from\s+)?([\w\.]+)', import_text)
            return match.group(1) if match else import_text
            
        elif language in ['javascript', 'typescript']:
            match = re.search(r'from\s+[\'"](.+?)[\'"]', import_text)
            return match.group(1) if match else import_text
            
        elif language == 'java':
            match = re.match(r'import\s+([\w\.]+)', import_text)
            return match.group(1) if match else import_text
            
        elif language == 'go':
            match = re.search(r'"(.+?)"', import_text)
            return match.group(1) if match else import_text
            
        elif language in ['c', 'cpp']:
            match = re.search(r'[<"](.+?)[>"]', import_text)
            return match.group(1) if match else import_text
            
        elif language == 'rust':
            match = re.match(r'use\s+([\w:]+)', import_text)
            return match.group(1) if match else import_text
            
        return import_text
    
    def _extract_docstring(self, node, content: str, language: str) -> str:
        """Extract docstring from function/class."""
        if language == 'python':
            # Look for first string in body
            for child in node.children:
                if child.type == 'block':
                    for stmt in child.children:
                        if stmt.type == 'expression_statement':
                            for expr in stmt.children:
                                if expr.type == 'string':
                                    docstring = content[expr.start_byte:expr.end_byte]
                                    # Clean triple quotes
                                    docstring = docstring.strip('"""').strip("'''").strip()
                                    return self._clean_and_limit_text(docstring, self.MAX_DOCSTRING_CHARS)
        
        elif language in ['javascript', 'typescript', 'java', 'c', 'cpp']:
            # Look for comment before node
            start_byte = node.start_byte
            # Search backwards for /** comment
            search_start = max(0, start_byte - 1000)
            preceding = content[search_start:start_byte]
            match = re.search(r'/\*\*(.*?)\*/', preceding, re.DOTALL)
            if match:
                docstring = match.group(1).strip()
                return self._clean_and_limit_text(docstring, self.MAX_DOCSTRING_CHARS)
        
        return ""
    
    def _extract_nearby_comments(self, node, content: str) -> str:
        """Extract comments near a symbol."""
        comments = []
        
        # Get text around the node
        start = max(0, node.start_byte - 500)
        end = min(len(content), node.end_byte + 500)
        surrounding = content[start:end]
        
        # Find single-line comments
        for match in re.finditer(r'(?:^|\n)\s*(?://|#)\s*(.+?)$', surrounding, re.MULTILINE):
            comment = match.group(1).strip()
            # Skip license headers and auto-generated
            if not any(skip in comment.lower() for skip in ['license', 'copyright', 'generated']):
                if len(comment) > 10:  # Skip trivial comments
                    comments.append(comment)
        
        # Find multi-line comments (but not docstrings)
        for match in re.finditer(r'/\*(?!\*)(.*?)\*/', surrounding, re.DOTALL):
            comment = match.group(1).strip()
            if len(comment) < 500:  # Skip huge comment blocks
                comments.append(comment)
        
        # Join and limit
        all_comments = ' '.join(comments[:5])  # Limit to 5 comments
        return self._clean_and_limit_text(all_comments, self.MAX_COMMENTS_CHARS)
    
    def _clean_and_limit_text(self, text: str, max_chars: int) -> str:
        """Clean and limit text to max characters."""
        if not text:
            return ""
        
        # Remove license banners
        text = re.sub(r'(?i)copyright.*?(?:\n\n|\*/)', '', text)
        text = re.sub(r'(?i)licensed under.*?(?:\n\n|\*/)', '', text)
        
        # Remove huge URLs
        text = re.sub(r'https?://[^\s]{100,}', '[URL]', text)
        
        # Remove base64 or hex blobs
        text = re.sub(r'[A-Fa-f0-9]{64,}', '[HEX_BLOB]', text)
        text = re.sub(r'[A-Za-z0-9+/]{50,}={0,2}', '[BASE64_BLOB]', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Limit length
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        
        return text.strip()
    
    def _build_embedding_text(self, name: str, signature: str, 
                              docstring: str, comments: str, body: str) -> str:
        """Build text for embedding with clear sections."""
        parts = []
        
        # Symbol name and signature
        parts.append(f"<SYMBOL> {name} {signature}")
        
        # Docstring
        if docstring:
            parts.append(f"<DOCSTRING>\n{docstring}")
        
        # Nearby comments
        if comments:
            parts.append(f"<NEARBY_COMMENTS>\n{comments}")
        
        # Body (limited)
        if body:
            # Take first ~2000 chars of body
            body_limited = body[:2000]
            parts.append(f"<BODY>\n{body_limited}")
        
        # Join with clear separators
        embedding_text = "\n\n".join(parts)
        
        # Final limit to total budget
        if len(embedding_text) > self.MAX_TOTAL_CHARS:
            embedding_text = embedding_text[:self.MAX_TOTAL_CHARS] + "..."
        
        return embedding_text


def test_extractor():
    """Test the symbol extractor with sample code."""
    extractor = TreeSitterSymbolExtractor()
    
    # Test C code (since word2vec is in C)
    c_code = '''
#include <stdio.h>
#include <math.h>

// Train word2vec model
void train_word2vec(char *train_file, char *output_file, int size, int window) {
    // Initialize vocabulary
    LearnVocabFromTrainFile();
    
    // Train the model
    for (int iter = 0; iter < num_iter; iter++) {
        TrainModel();
    }
    
    // Save the model
    SaveVocab(output_file);
}

int main(int argc, char **argv) {
    train_word2vec(argv[1], argv[2], 100, 5);
    return 0;
}
'''
    
    symbols = extractor.extract_symbols(c_code, 'c')
    
    print(f"Found {len(symbols)} symbols:")
    for symbol in symbols:
        print(f"  - {symbol['type']}: {symbol['name']}")
        if symbol['type'] == 'import':
            print(f"    Full: {symbol['full_statement']}")
        else:
            print(f"    Lines: {symbol['start_line']}-{symbol['end_line']}")
    
    return len(symbols) > 0


if __name__ == "__main__":
    success = test_extractor()
    if success:
        print("\n✅ Tree-sitter extractor is working!")
    else:
        print("\n❌ Tree-sitter extractor failed")