#!/usr/bin/env python3
"""
Add research papers from arXiv to the Sequential-ISNE test dataset.
Downloads PDFs, extracts abstracts, and creates natural README bridges.
"""

import sys
import re
import requests
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse
import argparse

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.tools.pdf_to_markdown import convert_pdf_to_markdown

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_arxiv_url(url: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Validate and convert arXiv URL to PDF download URL.
    
    Returns:
        (is_valid, pdf_url, paper_id)
    """
    # First check if this looks like it will download a PDF
    if not any(indicator in url.lower() for indicator in ['pdf', '/abs/']):
        logger.error(f"URL does not appear to be a PDF download link: {url}")
        logger.error("Please provide either:")
        logger.error("  - Direct PDF URL (e.g., arxiv.org/pdf/2506.20081.pdf)")
        logger.error("  - Abstract page URL (e.g., arxiv.org/abs/2506.20081)")
        return False, None, None
    
    # Add protocol if missing
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url
    
    # Common arXiv URL patterns - now more flexible
    patterns = [
        r'https?://arxiv\.org/abs/(\d+\.\d+)(v\d+)?',  # Abstract page
        r'https?://arxiv\.org/pdf/(\d+\.\d+)(v\d+)?\.pdf',  # Direct PDF
        r'https?://arxiv\.org/pdf/(\d+\.\d+)(v\d+)?',  # PDF without extension
    ]
    
    for pattern in patterns:
        match = re.match(pattern, url)
        if match:
            paper_id = match.group(1)
            version = match.group(2) if match.group(2) else ''
            # Use http like wget does - arxiv redirects to https if needed
            pdf_url = f"http://arxiv.org/pdf/{paper_id}{version}.pdf"
            
            return True, pdf_url, f"{paper_id}{version}"
    
    # If we get here, it's not a recognized arXiv URL
    logger.error(f"Not a valid arXiv URL format: {url}")
    return False, None, None


def download_arxiv_pdf(pdf_url: str, output_path: Path) -> bool:
    """Download PDF from arXiv."""
    try:
        logger.info(f"Downloading: {pdf_url}")
        # Minimal headers - be like wget
        response = requests.get(pdf_url, timeout=30, allow_redirects=True)
        response.raise_for_status()
        
        # Check if we got a PDF
        content_type = response.headers.get('content-type', '')
        if 'pdf' not in content_type.lower() and len(response.content) < 1000:
            logger.error(f"Response is not a PDF. Content-Type: {content_type}")
            return False
        
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        logger.info(f"✅ Downloaded to: {output_path} ({len(response.content)/1024/1024:.1f} MB)")
        return True
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            logger.error(f"❌ Paper not found (404): {pdf_url}")
        else:
            logger.error(f"❌ HTTP error {e.response.status_code}: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        return False


def extract_paper_metadata_from_arxiv(paper_id: str) -> Dict[str, str]:
    """Extract metadata including abstract from arXiv API."""
    try:
        # Use arXiv API to get metadata
        api_url = f"http://export.arxiv.org/api/query?id_list={paper_id}"
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()
        
        # Parse the response (simplified - real implementation would use XML parser)
        content = response.text
        
        # Extract title
        title_match = re.search(r'<title>(.*?)</title>', content, re.DOTALL)
        title = title_match.group(1).strip() if title_match else paper_id
        
        # Extract abstract
        abstract_match = re.search(r'<summary>(.*?)</summary>', content, re.DOTALL)
        abstract = abstract_match.group(1).strip() if abstract_match else "Abstract not found"
        
        # Extract authors
        authors = re.findall(r'<name>(.*?)</name>', content)
        
        return {
            'title': title,
            'abstract': abstract,
            'authors': authors,
            'paper_id': paper_id
        }
        
    except Exception as e:
        logger.error(f"Failed to extract metadata: {e}")
        return {
            'title': paper_id,
            'abstract': "Failed to extract abstract",
            'authors': [],
            'paper_id': paper_id
        }


def analyze_abstract_for_concepts(abstract: str) -> Dict[str, any]:
    """
    Analyze abstract to extract concepts and suggest HADES connections.
    This is where we'd use NLP, but for now we'll use keyword matching.
    """
    # Keywords that suggest HADES connections
    keyword_mappings = {
        'retrieval': ('src/pathrag/', 'PathRAG retrieval implementation'),
        'graph': ('src/pathrag/storage.py', 'Graph storage and operations'),
        'embedding': ('src/components/embedding/', 'Embedding generation system'),
        'augment': ('src/pathrag/base.py', 'Augmentation strategies'),
        'knowledge': ('src/types/knowledge/', 'Knowledge representation'),
        'semantic': ('src/isne/', 'Semantic-aware embeddings'),
        'multi-hop': ('src/pathrag/operate.py', 'Multi-hop traversal'),
        'context': ('src/core/process_first.py', 'Context as process'),
        'dynamic': ('src/research_integration/', 'Dynamic system updates'),
        'agent': ('src/pipelines/process_ethnography.py', 'Agent-based design'),
        'memory': ('src/types/storage/', 'Memory and storage systems'),
        'attention': ('src/isne/models/', 'Attention mechanisms'),
        'neural': ('src/isne/models/isne_model.py', 'Neural architectures'),
        'code': ('src/components/docproc/adapters/', 'Code processing'),
        'document': ('src/components/docproc/', 'Document processing'),
        'rag': ('src/pathrag/', 'RAG implementation'),
        'llm': ('src/pathrag/llm.py', 'LLM integration'),
        'vector': ('src/pathrag/storage.py', 'Vector storage'),
        'query': ('src/pathrag/base.py', 'Query processing'),
        'chunk': ('src/components/chunking/', 'Chunking strategies')
    }
    
    connections = []
    abstract_lower = abstract.lower()
    
    for keyword, (path, description) in keyword_mappings.items():
        if keyword in abstract_lower:
            connections.append({
                'concept': keyword.title(),
                'path': path,
                'description': description,
                'relevance': abstract_lower.count(keyword)
            })
    
    # Sort by relevance
    connections.sort(key=lambda x: x['relevance'], reverse=True)
    
    # Build connections text
    connections_text = ""
    for conn in connections[:5]:  # Top 5 connections
        connections_text += f"""### {conn['concept']}
- **HADES Location**: `{conn['path']}`
- **Connection**: {conn['description']}
- **Relevance**: Mentioned {conn['relevance']} times in abstract

"""
    
    return {
        'concepts': [c['concept'] for c in connections],
        'connections': connections,
        'connections_text': connections_text
    }


def create_paper_readme_with_abstract(metadata: Dict[str, str], abstract_analysis: Dict[str, any]) -> str:
    """
    Create a README that treats the abstract as the paper's natural documentation.
    The abstract IS the README - we just enhance it with HADES connections.
    """
    readme_content = f"""# {metadata['title']}

**Authors**: {', '.join(metadata['authors']) if metadata['authors'] else 'Unknown'}  
**Paper ID**: {metadata['paper_id']}  
**Added**: {datetime.now().strftime('%Y-%m-%d')}

## Abstract (The Paper's Natural README)

{metadata['abstract']}

## Why This Abstract Matters

The abstract above is not just a summary - it's the paper's README.md. Just as a README explains what code does and how to use it, this abstract explains:
- **What**: The problem being solved
- **How**: The approach taken
- **Why**: The significance of the work
- **Results**: What was achieved

## Detected Concepts & HADES Connections

Based on the abstract, these concepts connect to HADES:

{abstract_analysis.get('connections_text', 'Analysis pending...')}

## Abstract as Process Documentation

Following our process-first philosophy, this abstract documents the *process* of research:
1. **Problem Formation**: How the authors identified the gap
2. **Method Development**: The process of creating their solution  
3. **Validation Process**: How they proved it works
4. **Knowledge Creation**: The new understanding generated

## Integration Notes

- The abstract naturally bridges theory (paper) and practice (our implementation)
- Each concept in the abstract is a potential enhancement point for HADES
- The paper's methodology can inform our process design

---
*Abstract recognized as natural README: {datetime.now().isoformat()}*
"""
    return readme_content


def add_papers_from_urls(arxiv_urls: List[str], dataset_dir: Optional[Path] = None):
    """
    Add papers from arXiv URLs to the test dataset.
    Downloads PDFs, extracts abstracts, and creates natural README bridges.
    """
    # Target directory for new papers
    if dataset_dir is None:
        base_dir = Path("/home/todd/ML-Lab/Olympus/sequential-ISNE-testdata/isne-testdata")
    else:
        base_dir = dataset_dir
        
    target_dir = base_dir / "theory-papers" / "complex_systems" / "RAG_Systems" / "2025_01_papers"
    target_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=== Adding Research Papers from arXiv ===")
    logger.info(f"Target directory: {target_dir}")
    
    successful_papers = []
    
    for url in arxiv_urls:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {url}")
        
        # Validate and get PDF URL
        is_valid, pdf_url, paper_id = validate_arxiv_url(url)
        
        if not is_valid:
            logger.error(f"❌ Invalid arXiv URL: {url}")
            continue
        
        logger.info(f"✅ Valid arXiv paper: {paper_id}")
        
        # Extract metadata including abstract
        metadata = extract_paper_metadata_from_arxiv(paper_id)
        logger.info(f"📄 Title: {metadata['title']}")
        
        # Create paper directory
        paper_name = re.sub(r'[^\w\s-]', '', metadata['title'])[:100]  # Clean title for directory
        paper_dir = target_dir / paper_name
        paper_dir.mkdir(exist_ok=True)
        
        # Download PDF
        pdf_path = paper_dir / f"{paper_id}.pdf"
        if download_arxiv_pdf(pdf_url, pdf_path):
            # Convert to markdown using docling
            try:
                markdown_path = paper_dir / f"{paper_name}.md"
                logger.info("Converting PDF to markdown...")
                convert_pdf_to_markdown(str(pdf_path), str(markdown_path))
                logger.info(f"✅ Converted to: {markdown_path.name}")
            except Exception as e:
                logger.error(f"❌ Conversion failed: {e}")
                # Continue anyway - we have the abstract
        
        # Analyze abstract for concepts
        abstract_analysis = analyze_abstract_for_concepts(metadata['abstract'])
        logger.info(f"🔍 Found {len(abstract_analysis['concepts'])} concept connections")
        
        # Create README with abstract as natural documentation
        readme_path = paper_dir / "README.md"
        readme_content = create_paper_readme_with_abstract(metadata, abstract_analysis)
        
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        logger.info(f"📝 Created README with abstract as natural bridge")
        
        # Also save just the abstract
        abstract_path = paper_dir / "ABSTRACT.md"
        with open(abstract_path, 'w') as f:
            f.write(f"# Abstract\n\n{metadata['abstract']}")
        
        successful_papers.append({
            'title': metadata['title'],
            'paper_id': paper_id,
            'concepts': abstract_analysis['concepts'],
            'path': paper_dir
        })
    
    # Update main theory-practice bridges document
    if successful_papers:
        update_main_bridges_document(base_dir, successful_papers)
    
    # Create summary
    create_processing_summary(target_dir, successful_papers, arxiv_urls)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"✅ Successfully processed {len(successful_papers)} of {len(arxiv_urls)} papers")
    logger.info(f"📁 Location: {target_dir}")
    
    return successful_papers


def update_main_bridges_document(base_dir: Path, papers: List[Dict]):
    """Update the main THEORY_PRACTICE_BRIDGES.md document."""
    main_bridges_path = base_dir / "theory-papers" / "THEORY_PRACTICE_BRIDGES.md"
    
    with open(main_bridges_path, 'a') as f:
        f.write(f"\n\n## Papers Added via Abstract-as-README ({datetime.now().strftime('%Y-%m-%d')})\n\n")
        f.write("### Recognition: Abstracts ARE Natural READMEs\n\n")
        f.write("These papers demonstrate that academic abstracts function as natural README files:\n\n")
        
        for paper in papers:
            f.write(f"- **[{paper['title']}](./complex_systems/RAG_Systems/2025_01_papers/{paper['path'].name}/)**\n")
            f.write(f"  - Paper ID: {paper['paper_id']}\n")
            f.write(f"  - Key concepts: {', '.join(paper['concepts'][:3])}\n")
            f.write(f"  - Abstract serves as natural documentation\n\n")


def create_processing_summary(target_dir: Path, successful_papers: List[Dict], all_urls: List[str]):
    """Create a summary of the processing results."""
    summary_path = target_dir / "PROCESSING_SUMMARY.md"
    
    summary_content = f"""# Research Papers Processing Summary

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}  
**Total URLs**: {len(all_urls)}  
**Successfully Processed**: {len(successful_papers)}

## Philosophy: Abstract as README

This processing recognizes that **abstracts ARE the natural README files** of academic papers:
- They explain WHAT the work does
- They document HOW it works  
- They justify WHY it matters
- They summarize the RESULTS

## Papers Processed

"""
    
    for i, paper in enumerate(successful_papers, 1):
        summary_content += f"""### {i}. {paper['title']}
- **Paper ID**: {paper['paper_id']}
- **Location**: {paper['path'].name}/
- **Detected Concepts**: {', '.join(paper['concepts'])}
- **README Status**: Abstract successfully recognized as natural documentation

"""
    
    summary_content += """
## Integration with HADES

Each paper's abstract was analyzed for connections to HADES implementation:
- Keyword detection identified relevant concepts
- Automatic mapping to HADES code locations
- Abstract-driven theory-practice bridging

## Next Steps

1. Process these papers through the Sequential-ISNE pipeline
2. Let abstracts guide implementation enhancements
3. Use paper methodologies to improve HADES processes
4. Validate that our implementation aligns with paper abstracts

---
*Generated by Abstract-as-README processor*
"""
    
    with open(summary_path, 'w') as f:
        f.write(summary_content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add arXiv papers to Sequential-ISNE dataset with abstract-as-README philosophy"
    )
    parser.add_argument(
        "urls",
        nargs="+",
        help="arXiv URLs to process (e.g., https://arxiv.org/abs/2501.01234)"
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        help="Dataset directory (defaults to standard location)"
    )
    
    args = parser.parse_args()
    
    # Process the papers
    add_papers_from_urls(args.urls, args.dataset_dir)