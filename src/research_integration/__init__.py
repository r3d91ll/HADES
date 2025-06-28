"""
Research Integration Module

This module implements the self-improving research-driven development system
that continuously:
1. Discovers new research papers
2. Integrates them into the knowledge graph
3. Identifies connections to existing code
4. Generates improvements based on research insights
5. Validates and implements improvements

The system creates a feedback loop where better code leads to better research
understanding, which leads to even better code.
"""

from .discovery import ResearchDiscovery
from .analyzer import ResearchAnalyzer
from .mapper import CodeResearchMapper
from .generator import ImprovementGenerator
from .validator import ImprovementValidator

__all__ = [
    "ResearchDiscovery",
    "ResearchAnalyzer",
    "CodeResearchMapper",
    "ImprovementGenerator",
    "ImprovementValidator"
]