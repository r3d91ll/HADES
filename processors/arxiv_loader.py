"""
ArXiv Data Loader

Loads papers from the arXiv dataset with support for:
- Diverse sampling across categories and time
- Metadata extraction
- PDF path resolution
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union, Any
from datetime import datetime
import random
from collections import defaultdict


logger = logging.getLogger(__name__)


class ArxivLoader:
    """
    Loads papers from arXiv dataset with intelligent sampling.
    
    Features:
    - Category-balanced sampling
    - Time-based filtering
    - Metadata extraction
    - Missing file handling
    
    Example:
        loader = ArxivLoader(base_path="/mnt/data/arxiv_data")
        papers = loader.load_papers(
            num_papers=2000,
            categories=["cs.AI", "cs.LG", "math.ST"],
            sampling_strategy="diverse"
        )
    """
    
    def __init__(self, base_path: Union[str, Path]):
        """
        Initialize ArXiv loader.
        
        Args:
            base_path: Base directory containing arxiv_data/pdf and arxiv_data/metadata
        """
        self.base_path = Path(base_path)
        
        # Validate base path exists
        if not self.base_path.exists():
            raise ValueError(f"Base path does not exist: {self.base_path}")
        
        self.pdf_dir = self.base_path / "pdf"
        self.metadata_dir = self.base_path / "metadata"
        
        if not self.pdf_dir.exists():
            raise ValueError(f"PDF directory not found: {self.pdf_dir}")
        
        logger.info(f"Initialized ArxivLoader with base path: {self.base_path}")
    
    def load_papers(
        self,
        num_papers: int = 2000,
        categories: Optional[List[str]] = None,
        time_range: Optional[Tuple[str, str]] = None,
        sampling_strategy: str = "diverse",
        exclude_ids: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Load papers with specified criteria.
        
        Args:
            num_papers: Number of papers to load
            categories: List of arXiv categories to include (None = all)
            time_range: Tuple of (start_date, end_date) in YYYY-MM-DD format
            sampling_strategy: "diverse", "random", or "sequential"
            exclude_ids: Paper IDs to exclude
            
        Returns:
            List of paper dictionaries with metadata
        """
        logger.info(f"Loading {num_papers} papers with strategy: {sampling_strategy}")
        
        # Get all available PDFs
        all_pdfs = list(self.pdf_dir.glob("*.pdf"))
        logger.info(f"Found {len(all_pdfs)} total PDFs")
        
        # Filter by criteria
        candidate_papers = []
        
        for pdf_path in all_pdfs:
            paper_id = pdf_path.stem
            
            # Skip excluded IDs
            if exclude_ids and paper_id in exclude_ids:
                continue
            
            # Load metadata if available
            metadata = self._load_metadata(paper_id)
            
            # Apply filters
            if not self._passes_filters(metadata, categories, time_range):
                continue
            
            # Create paper record
            paper = {
                "id": paper_id,
                "pdf_path": str(pdf_path),
                "has_metadata": metadata is not None
            }
            
            if metadata:
                paper.update({
                    "title": metadata.get("title", "Unknown"),
                    "abstract": metadata.get("abstract", ""),
                    "authors": metadata.get("authors", []),
                    "categories": metadata.get("categories", []),
                    "date": metadata.get("date", ""),
                    "arxiv_id": metadata.get("arxiv_id", paper_id)
                })
            else:
                # Infer basic info from filename
                paper.update({
                    "title": f"Paper {paper_id}",
                    "abstract": "",
                    "authors": [],
                    "categories": self._infer_category(paper_id),
                    "date": self._infer_date(paper_id),
                    "arxiv_id": paper_id
                })
            
            candidate_papers.append(paper)
        
        logger.info(f"Found {len(candidate_papers)} papers matching criteria")
        
        # Apply sampling strategy
        selected_papers = self._apply_sampling(
            candidate_papers, 
            num_papers, 
            sampling_strategy
        )
        
        logger.info(f"Selected {len(selected_papers)} papers")
        
        return selected_papers
    
    def _load_metadata(self, paper_id: str) -> Optional[Dict]:
        """Load metadata for a paper if available."""
        metadata_path = self.metadata_dir / f"{paper_id}.json"
        
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load metadata for {paper_id}: {e}")
        
        return None
    
    def _passes_filters(
        self,
        metadata: Optional[Dict],
        categories: Optional[List[str]],
        time_range: Optional[Tuple[str, str]]
    ) -> bool:
        """Check if paper passes filter criteria."""
        # No metadata means we can't filter accurately
        if not metadata:
            # If filters are specified, exclude papers without metadata
            if categories or time_range:
                return False
            return True  # Include by default only if no filters
        
        # Category filter
        if categories:
            paper_categories = metadata.get("categories", [])
            if not any(cat in categories for cat in paper_categories):
                return False
        
        # Time filter
        if time_range:
            paper_date = metadata.get("date", "")
            if paper_date:
                if paper_date < time_range[0] or paper_date > time_range[1]:
                    return False
        
        return True
    
    def _infer_category(self, paper_id: str) -> List[str]:
        """Infer category from paper ID format."""
        # Old format: hep-th/9901001 -> ["hep-th"]
        if "/" in paper_id:
            return [paper_id.split("/")[0]]
        
        # New format: check common patterns
        # This is a simplified heuristic
        if paper_id.startswith(("astro", "cond", "hep", "math", "physics")):
            return [paper_id.split("-")[0]]
        
        return ["uncategorized"]  # Default fallback
    
    def _infer_date(self, paper_id: str) -> str:
        """Infer submission date from paper ID."""
        # Try new format: YYMM.NNNNN
        if "." in paper_id and paper_id.replace(".", "").isdigit():
            year_month = paper_id.split(".")[0]
            if len(year_month) == 4:
                year = int("20" + year_month[:2])
                month = int(year_month[2:4])
                return f"{year:04d}-{month:02d}-01"
        
        # Old format: category/YYMMNNN
        if "/" in paper_id:
            id_part = paper_id.split("/")[1]
            if len(id_part) >= 4:
                year = int("19" + id_part[:2]) if int(id_part[:2]) > 90 else int("20" + id_part[:2])
                month = int(id_part[2:4])
                return f"{year:04d}-{month:02d}-01"
        
        return "2020-01-01"  # Default
    
    def _apply_sampling(
        self,
        papers: List[Dict],
        num_papers: int,
        strategy: str
    ) -> List[Dict]:
        """Apply sampling strategy to select papers."""
        if len(papers) <= num_papers:
            return papers
        
        if strategy == "random":
            return random.sample(papers, num_papers)
        
        elif strategy == "sequential":
            # Sort by date and take first N
            papers.sort(key=lambda p: p.get("date", ""))
            return papers[:num_papers]
        
        elif strategy == "diverse":
            # Balance across categories and time
            return self._diverse_sampling(papers, num_papers)
        
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")
    
    def _diverse_sampling(self, papers: List[Dict], num_papers: int) -> List[Dict]:
        """
        Diverse sampling to ensure broad coverage.
        
        Uses inverse frequency weighting to avoid clustering.
        """
        selected = []
        
        # Group by category and year
        groups = defaultdict(list)
        for paper in papers:
            # Get primary category
            categories = paper.get("categories", ["unknown"])
            primary_cat = categories[0] if categories else "unknown"
            
            # Get year
            date = paper.get("date", "2020-01-01")
            year = date[:4]
            
            groups[(primary_cat, year)].append(paper)
        
        # Sample from each group proportionally
        group_sizes = {k: len(v) for k, v in groups.items()}
        total_size = sum(group_sizes.values())
        
        for (cat, year), group_papers in groups.items():
            # Calculate how many to sample from this group
            proportion = len(group_papers) / total_size
            group_sample_size = max(1, int(num_papers * proportion))
            
            # Don't sample more than available
            group_sample_size = min(group_sample_size, len(group_papers))
            
            # Sample from group
            sampled = random.sample(group_papers, group_sample_size)
            selected.extend(sampled)
        
        # Trim to exact size if over
        if len(selected) > num_papers:
            selected = random.sample(selected, num_papers)
        
        # Fill remaining slots if under
        while len(selected) < num_papers:
            remaining = [p for p in papers if p not in selected]
            if not remaining:
                break
            selected.append(random.choice(remaining))
        
        return selected
    
    def get_statistics(self, papers: List[Dict]) -> Dict[str, Any]:
        """Get statistics about loaded papers."""
        stats = {
            "total": len(papers),
            "by_category": defaultdict(int),
            "by_year": defaultdict(int),
            "missing_metadata": 0
        }
        
        for paper in papers:
            # Category statistics
            categories = paper.get("categories", ["unknown"])
            for cat in categories:
                stats["by_category"][cat] += 1
            
            # Year statistics
            date = paper.get("date", "")
            if date:
                year = date[:4]
                stats["by_year"][year] += 1
            
            # Missing metadata
            if not paper.get("has_metadata", True):
                stats["missing_metadata"] += 1
        
        # Convert defaultdicts to regular dicts for JSON serialization
        stats["by_category"] = dict(stats["by_category"])
        stats["by_year"] = dict(stats["by_year"])
        
        return stats