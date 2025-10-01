"""
Code File Extractor
===================

Extracts content from code files for embedding generation.
Integrates Tree-sitter for symbol table extraction to provide rich metadata
for Jina v4 coding LoRA embeddings.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .extractors_base import ExtractionResult, ExtractorBase, ExtractorConfig
from .extractors_treesitter import TreeSitterExtractor

logger = logging.getLogger(__name__)


class CodeExtractor(ExtractorBase):
    """
    Extract content from code files.
    
    This extractor represents the CONVEYANCE dimension for code,
    transforming source code into processable text while preserving
    semantic structure.
    """

    def __init__(self, config: Optional[ExtractorConfig] = None, use_tree_sitter: bool = True):
        """
        Initialize the CodeExtractor instance.

        Sets up a mapping of common file extensions to their single-line comment marker (used by metadata heuristics)
        and optionally initializes Tree-sitter for symbol extraction.

        Args:
            config: Extractor configuration
            use_tree_sitter: Whether to use Tree-sitter for symbol extraction
        """
        super().__init__(config)

        # Common comment patterns for different languages
        self.comment_patterns = {
            '.py': '#',
            '.js': '//',
            '.ts': '//',
            '.java': '//',
            '.c': '//',
            '.cpp': '//',
            '.go': '//',
            '.rs': '//',
            '.rb': '#',
            '.sh': '#',
            '.yaml': '#',
            '.yml': '#'
        }

        # Initialize Tree-sitter extractor
        self.use_tree_sitter = use_tree_sitter
        self.tree_sitter: Optional[TreeSitterExtractor]
        if use_tree_sitter:
            try:
                self.tree_sitter = TreeSitterExtractor()
                logger.info("Initialized CodeExtractor with Tree-sitter support")
            except Exception as e:
                logger.warning(f"Failed to initialize Tree-sitter: {e}")
                self.tree_sitter = None
                self.use_tree_sitter = False
        else:
            self.tree_sitter = None
            logger.info("Initialized CodeExtractor without Tree-sitter")

    def extract(self, file_path: Union[str, Path], **kwargs) -> ExtractionResult:
        """
        Extracts text and basic metadata from a code file for embedding generation.
        
        Reads the file at `file_path` as UTF-8 (errors ignored) and returns a dictionary containing the raw content and simple derived fields:
        - full_text: the file's full raw content
        - text: same as full_text (kept for compatibility)
        - markdown: fenced code block using the file extension as the language tag
        - num_lines: number of lines in the file
        - file_size: file size in bytes
        - file_extension: file suffix (e.g., ".py")
        - extractor: constant string 'code_extractor'
        - metadata: language-aware summary (line_count, char_count, has_docstring, import_count, function_count, class_count)
        
        Parameters:
            file_path (str): Path to the code file to read.
        
        Returns:
            Optional[Dict[str, Any]]: The extraction result dictionary on success; None if the file does not exist or extraction fails.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return ExtractionResult(
                text="",
                metadata={"file_path": str(file_path)},
                error=f"File not found: {file_path}"
            )

        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # Extract Tree-sitter symbols if available
            tree_sitter_data = {}
            if self.use_tree_sitter and self.tree_sitter:
                try:
                    tree_sitter_data = self.tree_sitter.extract_symbols(str(file_path), content)
                    logger.debug(f"Extracted Tree-sitter symbols for {file_path}")
                except Exception as e:
                    logger.warning(f"Tree-sitter extraction failed for {file_path}: {e}")

            # Build metadata
            metadata = self._extract_metadata(content, file_path)
            metadata['file_extension'] = file_path.suffix
            metadata['file_size'] = file_path.stat().st_size
            metadata['num_lines'] = len(content.splitlines())
            metadata['extractor'] = 'code_extractor_with_tree_sitter' if tree_sitter_data else 'code_extractor'

            # Add Tree-sitter data to metadata if available
            if tree_sitter_data:
                metadata['symbols'] = tree_sitter_data.get('symbols', {})
                metadata['code_metrics'] = tree_sitter_data.get('metrics', {})
                metadata['code_structure'] = tree_sitter_data.get('structure', {})
                metadata['language'] = tree_sitter_data.get('language')

                # Generate symbol hash for comparison
                if self.tree_sitter and tree_sitter_data.get('symbols'):
                    metadata['symbol_hash'] = self.tree_sitter.generate_symbol_hash(tree_sitter_data['symbols'])

            # Merge Tree-sitter metrics into metadata if available
            if tree_sitter_data.get('metrics'):
                metadata.update(tree_sitter_data['metrics'])

            # Create code block
            code_block = {
                'language': file_path.suffix[1:] if file_path.suffix else 'text',
                'code': content,
                'markdown': f"```{file_path.suffix[1:]}\n{content}\n```"
            }

            return ExtractionResult(
                text=content,
                metadata=metadata,
                code_blocks=[code_block]
            )

        except OSError as e:
            logger.exception(f"I/O error extracting {file_path}")
            return ExtractionResult(
                text="",
                metadata={"file_path": str(file_path)},
                error=f"I/O error: {str(e)}"
            )
        except Exception as e:
            logger.exception(f"Unexpected error extracting {file_path}")
            return ExtractionResult(
                text="",
                metadata={"file_path": str(file_path)},
                error=f"Unexpected error: {str(e)}"
            )

    def _extract_metadata(self, content: str, file_path: Path) -> Dict[str, Any]:
        """
        Compute basic metadata for a source code file.
        
        Returns a dictionary of metrics computed from `content`. The returned keys are:
        - line_count: number of lines in the content.
        - char_count: total number of characters.
        - has_docstring: (Python) True if triple-quote docstrings are present.
        - import_count: approximate count of import/require statements.
        - function_count: approximate count of function definitions or arrow functions.
        - class_count: approximate count of class declarations.
        
        Language-specific behavior is inferred from file_path.suffix (lowercased):
        - .py: detects Python docstrings, lines starting with `import`/`from`, `def`, and `class`.
        - .js, .ts, .jsx, .tsx: counts occurrences of `import`/`require`, `function` and `=>`, and lines starting with `class`.
        - .java: counts lines starting with `import` and lines containing `class` with `{`.
        
        The counts are simple heuristics (line-based or substring matches) intended for lightweight metadata and may over- or under-count in complex code constructs.
        """
        lines = content.splitlines()

        metadata = {
            'line_count': len(lines),
            'char_count': len(content),
            'has_docstring': False,
            'import_count': 0,
            'function_count': 0,
            'class_count': 0
        }

        # Language-specific analysis
        ext = file_path.suffix.lower()

        if ext == '.py':
            # Python-specific analysis
            metadata['has_docstring'] = '"""' in content or "'''" in content
            metadata['import_count'] = sum(1 for line in lines if line.strip().startswith(('import ', 'from ')))
            metadata['function_count'] = sum(1 for line in lines if line.strip().startswith('def '))
            metadata['class_count'] = sum(1 for line in lines if line.strip().startswith('class '))

        elif ext in ['.js', '.ts', '.jsx', '.tsx']:
            # JavaScript/TypeScript analysis
            metadata['import_count'] = sum(1 for line in lines if 'import ' in line or 'require(' in line)
            metadata['function_count'] = content.count('function ') + content.count('=>')
            metadata['class_count'] = sum(1 for line in lines if line.strip().startswith('class '))

        elif ext == '.java':
            # Java analysis
            metadata['import_count'] = sum(1 for line in lines if line.strip().startswith('import '))
            metadata['class_count'] = sum(1 for line in lines if 'class ' in line and '{' in line)

        return metadata

    def extract_batch(self,
                     file_paths: List[Union[str, Path]],
                     **kwargs) -> List[ExtractionResult]:
        """
        Extract content from multiple code files.

        Args:
            file_paths: List of file paths
            **kwargs: Additional extraction options

        Returns:
            List of ExtractionResult objects
        """
        results = []
        for file_path in file_paths:
            results.append(self.extract(file_path, **kwargs))
        return results

    @property
    def supported_formats(self) -> List[str]:
        """Get list of supported file formats."""
        return [
            '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.c', '.cpp', '.h', '.hpp',
            '.cs', '.go', '.rs', '.rb', '.php', '.swift', '.kt', '.scala', '.r',
            '.m', '.mm', '.sh', '.bash', '.zsh', '.ps1', '.yaml', '.yml', '.toml',
            '.json', '.xml', '.html', '.css', '.scss', '.sass', '.sql'
        ]
