"""
File Metadata Extractor

Extracts comprehensive metadata from files including permissions, ownership,
timestamps, and other filesystem attributes.
"""

import os
import stat
import pwd
import grp
import hashlib
import mimetypes
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
try:
    import magic  # python-magic for better file type detection
except ImportError:
    magic = None  # type: ignore

logger = logging.getLogger(__name__)


class FileMetadataExtractor:
    """
    Extracts comprehensive metadata from files and directories.
    
    Includes permissions, ownership, timestamps, file type detection,
    and special attributes like immutability.
    """
    
    def __init__(self, include_content_hash: bool = True, hash_algorithm: str = 'sha256'):
        """
        Initialize the metadata extractor.
        
        Args:
            include_content_hash: Whether to compute content hashes
            hash_algorithm: Hash algorithm to use ('md5', 'sha1', 'sha256', etc.)
        """
        self.include_content_hash = include_content_hash
        self.hash_algorithm = hash_algorithm
        
        # Initialize magic for file type detection
        self.magic: Optional[Any] = None
        self.magic_available = False
        
        if magic is not None:
            try:
                self.magic = magic.Magic(mime=True)
                self.magic_available = True
            except Exception as e:
                logger.warning(f"python-magic not available: {e}. Falling back to mimetypes.")
    
    def extract_metadata(self, file_path: Path) -> Dict[str, Any]:
        """
        Extract comprehensive metadata from a file or directory.
        
        Args:
            file_path: Path to file or directory
            
        Returns:
            Dictionary containing all extracted metadata
        """
        file_path = Path(file_path).resolve()
        
        if not file_path.exists():
            return {
                "error": "Path does not exist",
                "path": str(file_path)
            }
        
        try:
            stat_info = file_path.stat()
            
            metadata = {
                # Basic info
                "path": str(file_path),
                "name": file_path.name,
                "type": self._get_file_type(file_path),
                "is_file": file_path.is_file(),
                "is_dir": file_path.is_dir(),
                "is_symlink": file_path.is_symlink(),
                
                # Size
                "size": stat_info.st_size,
                "size_human": self._human_readable_size(stat_info.st_size),
                
                # Timestamps
                "created": datetime.fromtimestamp(stat_info.st_ctime).isoformat(),
                "modified": datetime.fromtimestamp(stat_info.st_mtime).isoformat(),
                "accessed": datetime.fromtimestamp(stat_info.st_atime).isoformat(),
                "created_timestamp": stat_info.st_ctime,
                "modified_timestamp": stat_info.st_mtime,
                "accessed_timestamp": stat_info.st_atime,
                
                # Permissions
                "permissions": self._extract_permissions(stat_info),
                
                # Ownership
                "ownership": self._extract_ownership(stat_info),
                
                # Special attributes
                "attributes": self._extract_attributes(file_path, stat_info),
            }
            
            # File-specific metadata
            if file_path.is_file():
                metadata.update(self._extract_file_specific_metadata(file_path))
            
            # Directory-specific metadata
            elif file_path.is_dir():
                metadata.update(self._extract_directory_specific_metadata(file_path))
            
            # Symlink-specific metadata
            if file_path.is_symlink():
                metadata.update(self._extract_symlink_metadata(file_path))
            
            # Git metadata if in a git repo
            git_metadata = self._extract_git_metadata(file_path)
            if git_metadata:
                metadata["git"] = git_metadata
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting metadata for {file_path}: {e}")
            return {
                "error": str(e),
                "path": str(file_path)
            }
    
    def _get_file_type(self, file_path: Path) -> str:
        """Determine file type."""
        if file_path.is_dir():
            return "directory"
        elif file_path.is_symlink():
            return "symlink"
        elif file_path.is_file():
            return "file"
        else:
            return "unknown"
    
    def _human_readable_size(self, size: int) -> str:
        """Convert size to human readable format."""
        size_float = float(size)  # Convert to float for division
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_float < 1024.0:
                return f"{size_float:.2f} {unit}"
            size_float /= 1024.0
        return f"{size_float:.2f} PB"
    
    def _extract_permissions(self, stat_info: os.stat_result) -> Dict[str, Any]:
        """Extract permission information."""
        mode = stat_info.st_mode
        
        return {
            "octal": oct(stat.S_IMODE(mode)),
            "string": stat.filemode(mode),
            "owner": {
                "read": bool(mode & stat.S_IRUSR),
                "write": bool(mode & stat.S_IWUSR),
                "execute": bool(mode & stat.S_IXUSR),
            },
            "group": {
                "read": bool(mode & stat.S_IRGRP),
                "write": bool(mode & stat.S_IWGRP),
                "execute": bool(mode & stat.S_IXGRP),
            },
            "others": {
                "read": bool(mode & stat.S_IROTH),
                "write": bool(mode & stat.S_IWOTH),
                "execute": bool(mode & stat.S_IXOTH),
            },
            "special": {
                "setuid": bool(mode & stat.S_ISUID),
                "setgid": bool(mode & stat.S_ISGID),
                "sticky": bool(mode & stat.S_ISVTX),
            }
        }
    
    def _extract_ownership(self, stat_info: os.stat_result) -> Dict[str, Any]:
        """Extract ownership information."""
        ownership: Dict[str, Any] = {
            "uid": stat_info.st_uid,
            "gid": stat_info.st_gid,
            "user": "",  # Initialize as empty string
            "group": "",  # Initialize as empty string
        }
        
        # Try to get username
        try:
            ownership["user"] = pwd.getpwuid(stat_info.st_uid).pw_name
        except KeyError:
            ownership["user"] = str(stat_info.st_uid)
        
        # Try to get group name
        try:
            ownership["group"] = grp.getgrgid(stat_info.st_gid).gr_name
        except KeyError:
            ownership["group"] = str(stat_info.st_gid)
        
        return ownership
    
    def _extract_attributes(self, file_path: Path, stat_info: os.stat_result) -> Dict[str, Any]:
        """Extract special file attributes."""
        attributes: Dict[str, Any] = {
            "hidden": file_path.name.startswith('.'),
            "executable": os.access(str(file_path), os.X_OK),
            "readable": os.access(str(file_path), os.R_OK),
            "writable": os.access(str(file_path), os.W_OK),
        }
        
        # Check for immutability (Linux-specific)
        immutable_status: Optional[bool] = None
        if hasattr(os, 'O_IMMUTABLE'):
            try:
                # This is platform-specific and may not work everywhere
                import fcntl
                with open(file_path, 'rb') as f:
                    flags = fcntl.fcntl(f.fileno(), fcntl.F_GETFL)
                    immutable_status = bool(flags & os.O_IMMUTABLE)
            except Exception:
                pass  # Keep as None
        
        if immutable_status is not None:
            attributes["immutable"] = immutable_status
        
        return attributes
    
    def _extract_file_specific_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract metadata specific to files."""
        metadata = {}
        
        # File extension
        metadata["extension"] = file_path.suffix.lower()
        metadata["stem"] = file_path.stem
        
        # MIME type
        mime_type: Optional[str] = None
        if self.magic_available and self.magic:
            try:
                mime_type = self.magic.from_file(str(file_path))
            except Exception as e:
                logger.debug(f"Magic failed for {file_path}: {e}")
                mime_type = mimetypes.guess_type(str(file_path))[0]
        else:
            mime_type = mimetypes.guess_type(str(file_path))[0]
        
        metadata["mime_type"] = mime_type if mime_type is not None else ""
        
        # Content hash (optional)
        if self.include_content_hash and file_path.stat().st_size < 100 * 1024 * 1024:  # Skip files > 100MB
            try:
                metadata["content_hash"] = self._compute_file_hash(file_path)
            except Exception as e:
                logger.debug(f"Failed to compute hash for {file_path}: {e}")
                metadata["content_hash"] = ""
        
        # Line count for text files
        if metadata.get("mime_type", "").startswith("text/") or file_path.suffix in {'.py', '.js', '.md', '.txt'}:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    line_count = sum(1 for _ in f)
                    metadata["line_count"] = str(line_count)
            except Exception:
                metadata["line_count"] = "0"
        
        return metadata
    
    def _extract_directory_specific_metadata(self, dir_path: Path) -> Dict[str, Any]:
        """Extract metadata specific to directories."""
        metadata: Dict[str, Any] = {}
        
        try:
            # Count children (non-recursive)
            children = list(dir_path.iterdir())
            metadata["child_count"] = len(children)
            metadata["file_count"] = sum(1 for child in children if child.is_file())
            metadata["dir_count"] = sum(1 for child in children if child.is_dir())
            
            # Check for special directory types
            special_files = {
                ".git": "git_repository",
                "package.json": "node_project",
                "pyproject.toml": "python_project",
                "Cargo.toml": "rust_project",
                "go.mod": "go_project",
                ".venv": "python_venv",
                "node_modules": "node_modules",
            }
            
            metadata["special_type"] = None
            for special_file, dir_type in special_files.items():
                if (dir_path / special_file).exists():
                    metadata["special_type"] = dir_type
                    break
        
        except PermissionError:
            metadata["error"] = "Permission denied"
        
        return metadata
    
    def _extract_symlink_metadata(self, symlink_path: Path) -> Dict[str, Any]:
        """Extract metadata specific to symbolic links."""
        metadata: Dict[str, Any] = {"symlink": {}}
        
        try:
            target = symlink_path.readlink()
            metadata["symlink"]["target"] = str(target)
            metadata["symlink"]["target_exists"] = (symlink_path.parent / target).exists()
            metadata["symlink"]["is_absolute"] = target.is_absolute()
        except Exception as e:
            metadata["symlink"]["error"] = str(e)
        
        return metadata
    
    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute hash of file contents."""
        hash_func = hashlib.new(self.hash_algorithm)
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b''):
                hash_func.update(chunk)
        
        return f"{self.hash_algorithm}:{hash_func.hexdigest()}"
    
    def _extract_git_metadata(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Extract git-related metadata if file is in a git repository."""
        # Walk up to find .git directory
        current = file_path if file_path.is_dir() else file_path.parent
        
        while current != current.parent:
            if (current / '.git').exists():
                return {
                    "is_tracked": None,  # Would need git commands to determine
                    "repo_root": str(current),
                    "relative_path": str(file_path.relative_to(current))
                }
            current = current.parent
        
        return None
    
    def extract_batch(self, file_paths: List[Path]) -> Dict[Path, Dict[str, Any]]:
        """
        Extract metadata for multiple files.
        
        Args:
            file_paths: List of paths to process
            
        Returns:
            Dictionary mapping paths to their metadata
        """
        results = {}
        
        for file_path in file_paths:
            results[file_path] = self.extract_metadata(file_path)
        
        return results