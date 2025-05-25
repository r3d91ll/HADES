"""Common test fixtures and utilities for unit tests.

This module contains shared fixtures and helper functions that can be
used across different unit test modules to provide consistent testing
patterns and reduce code duplication.
"""

import os
import pytest
from typing import Dict, Any, List, Optional
from pathlib import Path


# Sample document contents for testing
SAMPLE_TEXT_CONTENT = """
# Introduction to Machine Learning

Machine learning is a field of inquiry devoted to understanding and building methods that 'learn', that is, methods that leverage data to improve performance on some set of tasks.

## Supervised Learning

Supervised learning algorithms build a model based on sample data in order to make predictions or decisions without being explicitly programmed to do so. Examples include:

- Classification
- Regression
- Forecasting

## Unsupervised Learning

Unsupervised learning is a type of algorithm that learns patterns from untagged data. The hope is that through mimicry, which is an important mode of learning in people, the machine is forced to build a compact internal representation of its world.

### Clustering

Clustering is the task of grouping a set of objects in such a way that objects in the same group (called a cluster) are more similar to each other than to those in other groups.

## Reinforcement Learning

Reinforcement learning is an area of machine learning concerned with how software agents ought to take actions in an environment in order to maximize the notion of cumulative reward.
"""

SAMPLE_CODE_CONTENT = '''
def calculate_mean(numbers):
    """Calculate the mean of a list of numbers.
    
    Args:
        numbers: A list of numbers
        
    Returns:
        The mean value
    """
    total = sum(numbers)
    count = len(numbers)
    return total / count if count > 0 else 0

class DataProcessor:
    """A class for processing data."""
    
    def __init__(self, data=None):
        """Initialize with optional data.
        
        Args:
            data: Initial data to process
        """
        self.data = data or []
        
    def add_item(self, item):
        """Add an item to the data.
        
        Args:
            item: The item to add
        """
        self.data.append(item)
        
    def process(self):
        """Process the data.
        
        Returns:
            Processed data
        """
        if not self.data:
            return []
        
        return [item * 2 for item in self.data]
'''


def create_sample_document(
    content: str = SAMPLE_TEXT_CONTENT,
    doc_id: str = "test-doc-001",
    path: str = "/path/to/document.md",
    doc_type: str = "text"
) -> Dict[str, Any]:
    """Create a sample document dictionary for testing.
    
    Args:
        content: The document content
        doc_id: Document ID
        path: Document path
        doc_type: Document type
        
    Returns:
        A document dictionary
    """
    return {
        "id": doc_id,
        "content": content,
        "path": path,
        "type": doc_type,
        "metadata": {
            "source": "test",
            "created_at": "2025-05-15T10:00:00Z"
        }
    }


def create_expected_chunks(document: Dict[str, Any], count: int = 5) -> List[Dict[str, Any]]:
    """Create expected chunk structures for a document.
    
    Args:
        document: Source document
        count: Number of chunks to generate
        
    Returns:
        List of chunk dictionaries
    """
    doc_id = document["id"]
    chunks = []
    
    content_length = len(document["content"])
    chunk_size = content_length // count
    
    for i in range(count):
        start = i * chunk_size
        end = start + chunk_size if i < count - 1 else content_length
        chunk_content = document["content"][start:end]
        
        chunks.append({
            "id": f"{doc_id}-chunk-{i}",
            "content": chunk_content,
            "metadata": {
                "source_id": doc_id,
                "chunk_index": i,
                "chunk_type": "text" if document["type"] == "text" else "code"
            }
        })
    
    return chunks


@pytest.fixture
def sample_text_document() -> Dict[str, Any]:
    """Fixture providing a sample text document for testing."""
    return create_sample_document(
        content=SAMPLE_TEXT_CONTENT,
        doc_id="text-doc-001",
        path="/path/to/document.md",
        doc_type="text"
    )


@pytest.fixture
def sample_code_document() -> Dict[str, Any]:
    """Fixture providing a sample code document for testing."""
    return create_sample_document(
        content=SAMPLE_CODE_CONTENT,
        doc_id="code-doc-001",
        path="/path/to/document.py",
        doc_type="python"
    )
