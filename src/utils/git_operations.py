"""
Git operations module for cloning and managing repositories.
"""
import os
import subprocess
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

class GitOperations:
    """
    Class to handle Git operations such as cloning repositories
    and extracting repository information.
    """
    
    def __init__(self, base_dir: str = "/home/todd/ML-Lab/"):
        """
        Initialize GitOperations with base directory for cloning repositories.
        
        Args:
            base_dir: Base directory to clone repositories into
        """
        self.base_dir = Path(base_dir).resolve()
        
    def clone_repository(self, repo_url: str, repo_name: Optional[str] = None) -> Tuple[bool, str, Optional[Path]]:
        """
        Clone a repository to the base directory.
        
        Args:
            repo_url: URL of the GitHub repository
            repo_name: Optional name for the cloned repository folder, defaults to repo name from URL
            
        Returns:
            Tuple containing (success_status, message, repo_path)
        """
        if not repo_url.endswith(".git") and "github.com" in repo_url:
            repo_url = f"{repo_url}.git"
            
        # Extract repo name from URL if not provided
        if not repo_name:
            repo_name = repo_url.split("/")[-1].replace(".git", "")
        
        # Create target directory
        target_dir = self.base_dir / repo_name
        
        if target_dir.exists():
            return False, f"Directory {target_dir} already exists. Choose another name or remove existing directory.", None
        
        try:
            subprocess.run(
                ["git", "clone", repo_url, str(target_dir)],
                check=True,
                capture_output=True,
                text=True
            )
            return True, f"Successfully cloned repository to {target_dir}", target_dir
        except subprocess.CalledProcessError as e:
            return False, f"Failed to clone repository: {e.stderr}", None
    
    @staticmethod
    def get_repo_info(repo_path: Path) -> Dict[str, Any]:
        """
        Get information about a Git repository.
        
        Args:
            repo_path: Path to the Git repository
            
        Returns:
            Dictionary containing repository information
        """
        info = {
            "repo_path": str(repo_path),
            "repo_name": repo_path.name,
            "branches": [],
            "current_branch": "",
            "remote_url": "",
            "commit_count": 0,
            "last_commit": "",
            "contributors": []
        }
        
        try:
            # Get remote URL
            result = subprocess.run(
                ["git", "config", "--get", "remote.origin.url"],
                cwd=repo_path,
                check=True,
                capture_output=True,
                text=True
            )
            info["remote_url"] = result.stdout.strip()
            
            # Get current branch
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=repo_path,
                check=True,
                capture_output=True,
                text=True
            )
            info["current_branch"] = result.stdout.strip()
            
            # Get all branches
            result = subprocess.run(
                ["git", "branch", "-a"],
                cwd=repo_path,
                check=True,
                capture_output=True,
                text=True
            )
            info["branches"] = [
                branch.strip().replace("* ", "") 
                for branch in result.stdout.strip().split("\n")
                if branch.strip()
            ]
            
            # Get commit count
            result = subprocess.run(
                ["git", "rev-list", "--count", "HEAD"],
                cwd=repo_path,
                check=True,
                capture_output=True,
                text=True
            )
            info["commit_count"] = int(result.stdout.strip())
            
            # Get last commit
            result = subprocess.run(
                ["git", "log", "-1", "--pretty=format:%h - %an, %ar : %s"],
                cwd=repo_path,
                check=True,
                capture_output=True,
                text=True
            )
            info["last_commit"] = result.stdout.strip()
            
            # Get contributors
            result = subprocess.run(
                ["git", "shortlog", "-s", "-n", "--all"],
                cwd=repo_path,
                check=True,
                capture_output=True,
                text=True
            )
            info["contributors"] = [
                {
                    "name": line.strip().split("\t")[1],
                    "commits": int(line.strip().split("\t")[0])
                }
                for line in result.stdout.strip().split("\n")
                if line.strip()
            ]
            
            return info
        
        except subprocess.CalledProcessError as e:
            # Return partial info if something fails
            return info
