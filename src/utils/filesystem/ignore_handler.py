"""
Ignore File Handler

Handles multiple ignore file formats (.gitignore, .dockerignore, etc.) to determine
which files should be excluded from graph indexing.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any
import pathspec

logger = logging.getLogger(__name__)


class IgnoreFileHandler:
    """
    Unified handler for multiple ignore file formats.
    
    Supports common ignore files like .gitignore, .dockerignore, .npmignore, etc.
    Uses pathspec library for proper gitignore-style pattern matching.
    """
    
    # Supported ignore files and their pattern types
    IGNORE_FILES = {
        '.gitignore': 'gitwildmatch',
        '.dockerignore': 'gitwildmatch',
        '.npmignore': 'gitwildmatch',
        '.eslintignore': 'gitwildmatch',
        '.prettierignore': 'gitwildmatch',
        '.helmignore': 'gitwildmatch',
        '.gcloudignore': 'gitwildmatch',
        '.vercelignore': 'gitwildmatch',
        '.nowignore': 'gitwildmatch',
        '.slugignore': 'gitwildmatch',
        '.cfignore': 'gitwildmatch',
        '.ebignore': 'gitwildmatch',
        '.fdignore': 'gitwildmatch',
        '.rgignore': 'gitwildmatch',
        '.agignore': 'gitwildmatch',
        '.hgignore': 'gitwildmatch',  # Can also use regex syntax
        '.bzrignore': 'gitwildmatch',
        '.boringignore': 'gitwildmatch',
    }
    
    # Always exclude these, regardless of ignore files
    DEFAULT_EXCLUDES = {
        '.git',  # Git repository data
        '.svn',  # Subversion data
        '.hg',   # Mercurial data
        '.bzr',  # Bazaar data
    }
    
    def __init__(self, repo_root: Path, respect_ignore_files: bool = True):
        """
        Initialize the ignore handler.
        
        Args:
            repo_root: Root directory of the repository/project
            respect_ignore_files: Whether to respect ignore files (can be disabled for debugging)
        """
        self.repo_root = Path(repo_root).resolve()
        self.respect_ignore_files = respect_ignore_files
        self._ignore_specs: Dict[str, pathspec.PathSpec] = {}
        self._combined_spec: Optional[pathspec.PathSpec] = None
        
        if respect_ignore_files:
            self._load_all_ignore_files()
        else:
            # Still respect default excludes even when ignore files are disabled
            self._load_default_excludes()
    
    def _load_default_excludes(self) -> None:
        """Load default exclusion patterns."""
        patterns = [f"{exclude}/" for exclude in self.DEFAULT_EXCLUDES]
        self._ignore_specs['__defaults__'] = pathspec.PathSpec.from_lines('gitwildmatch', patterns)
        self._update_combined_spec()
    
    def _load_all_ignore_files(self) -> None:
        """Load all ignore files found in repo root."""
        # Start with defaults
        self._load_default_excludes()
        
        # Load each ignore file
        for ignore_file, pattern_type in self.IGNORE_FILES.items():
            ignore_path = self.repo_root / ignore_file
            if ignore_path.exists() and ignore_path.is_file():
                try:
                    patterns = self._read_ignore_file(ignore_path)
                    if patterns:
                        spec = pathspec.PathSpec.from_lines(pattern_type, patterns)
                        self._ignore_specs[ignore_file] = spec
                        logger.debug(f"Loaded {len(patterns)} patterns from {ignore_file}")
                except Exception as e:
                    logger.warning(f"Failed to load {ignore_file}: {e}")
        
        self._update_combined_spec()
        logger.info(f"Loaded {len(self._ignore_specs)} ignore files")
    
    def _read_ignore_file(self, path: Path) -> List[str]:
        """Read and parse an ignore file."""
        try:
            content = path.read_text(encoding='utf-8')
            # Filter out empty lines and comments
            patterns = []
            for line in content.splitlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    patterns.append(line)
            return patterns
        except Exception as e:
            logger.error(f"Error reading {path}: {e}")
            return []
    
    def _update_combined_spec(self) -> None:
        """Update the combined pathspec from all loaded specs."""
        all_patterns: List[Any] = []
        for spec in self._ignore_specs.values():
            all_patterns.extend(spec.patterns)
        
        if all_patterns:
            self._combined_spec = pathspec.PathSpec(all_patterns)
        else:
            self._combined_spec = pathspec.PathSpec([])
    
    def should_ignore(self, file_path: Path) -> bool:
        """
        Check if a file should be ignored based on all ignore patterns.
        
        Args:
            file_path: Path to check
            
        Returns:
            True if file should be ignored, False otherwise
        """
        if not self._combined_spec:
            return False
        
        # Convert to relative path from repo root
        try:
            relative_path = file_path.relative_to(self.repo_root)
        except ValueError:
            # Path is outside repo root
            logger.warning(f"Path {file_path} is outside repo root {self.repo_root}")
            return True
        
        # Check if any component is in DEFAULT_EXCLUDES
        for part in relative_path.parts:
            if part in self.DEFAULT_EXCLUDES:
                return True
        
        # Check against combined patterns
        return bool(self._combined_spec.match_file(str(relative_path)))
    
    def get_matching_rules(self, file_path: Path) -> List[Tuple[str, str]]:
        """
        Get all ignore rules that match a given file.
        
        Args:
            file_path: Path to check
            
        Returns:
            List of tuples (ignore_file, pattern) that match
        """
        matches = []
        
        try:
            relative_path = file_path.relative_to(self.repo_root)
        except ValueError:
            return [("__system__", "Path outside repo root")]
        
        # Check each ignore spec separately
        for ignore_file, spec in self._ignore_specs.items():
            if spec.match_file(str(relative_path)):
                # Try to find which specific pattern matched
                for pattern in spec.patterns:
                    if pathspec.PathSpec([pattern]).match_file(str(relative_path)):
                        matches.append((ignore_file, str(pattern)))
                        break
                else:
                    # Couldn't identify specific pattern
                    matches.append((ignore_file, "<complex pattern>"))
        
        return matches
    
    def debug_path(self, file_path: Path) -> Dict[str, Any]:
        """
        Debug why a file might be ignored.
        
        Args:
            file_path: Path to debug
            
        Returns:
            Dictionary with debugging information
        """
        file_path = Path(file_path).resolve()
        
        result: Dict[str, Any] = {
            "file": str(file_path),
            "relative_path": None,
            "ignored": False,
            "matched_by": [],
            "ignore_files_checked": list(self._ignore_specs.keys()),
            "ignore_files_found": [f for f in self._ignore_specs.keys() if f != '__defaults__'],
        }
        
        try:
            relative_path = file_path.relative_to(self.repo_root)
            result["relative_path"] = str(relative_path)
        except ValueError:
            result["ignored"] = True
            result["matched_by"] = [("__system__", "Path outside repo root")]
            return result
        
        # Check if ignored
        if self.should_ignore(file_path):
            result["ignored"] = True
            result["matched_by"] = self.get_matching_rules(file_path)
        
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about loaded ignore files."""
        stats: Dict[str, Any] = {
            "repo_root": str(self.repo_root),
            "respect_ignore_files": self.respect_ignore_files,
            "ignore_files_loaded": len([f for f in self._ignore_specs.keys() if f != '__defaults__']),
            "total_patterns": sum(len(spec.patterns) for spec in self._ignore_specs.values()),
            "ignore_files": {}
        }
        
        for ignore_file, spec in self._ignore_specs.items():
            if ignore_file != '__defaults__':
                stats["ignore_files"][ignore_file] = {
                    "patterns": len(spec.patterns),
                    "path": str(self.repo_root / ignore_file)
                }
        
        return stats
    
    def add_custom_patterns(self, patterns: List[str], name: str = "__custom__") -> None:
        """
        Add custom ignore patterns.
        
        Args:
            patterns: List of gitignore-style patterns
            name: Name for this set of patterns (for debugging)
        """
        if patterns:
            spec = pathspec.PathSpec.from_lines('gitwildmatch', patterns)
            self._ignore_specs[name] = spec
            self._update_combined_spec()
            logger.debug(f"Added {len(patterns)} custom patterns as '{name}'")
    
    def walk_directory(self, 
                      start_path: Optional[Path] = None,
                      include_dirs: bool = True,
                      follow_symlinks: bool = False) -> Any:
        """
        Walk directory tree, respecting ignore patterns.
        
        Args:
            start_path: Starting directory (defaults to repo_root)
            include_dirs: Whether to include directories in results
            follow_symlinks: Whether to follow symbolic links
            
        Yields:
            Paths that are not ignored
        """
        if start_path is None:
            start_path = self.repo_root
        else:
            start_path = Path(start_path).resolve()
        
        # Use os.walk for compatibility
        import os
        for root, dirs, files in os.walk(start_path, followlinks=follow_symlinks):
            root_path = Path(root)
            
            # Filter directories to prevent descending into ignored ones
            dirs_to_remove = []
            for dir_name in dirs:
                dir_path = root_path / dir_name
                if self.should_ignore(dir_path):
                    dirs_to_remove.append(dir_name)
            
            for dir_name in dirs_to_remove:
                dirs.remove(dir_name)
            
            # Yield non-ignored directories if requested
            if include_dirs:
                if not self.should_ignore(root_path):
                    yield root_path
            
            # Yield non-ignored files
            for file_name in files:
                file_path = root_path / file_name
                if not self.should_ignore(file_path):
                    yield file_path