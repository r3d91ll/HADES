#!/usr/bin/env python3
"""
Process PDFs with citations - Version 6.7
==========================================

Source-fidelity design with ArangoDB best practices:
- Single base_arxiv collection (no edge collections)
- Citations stored as arrays within paper documents
- Sparse compound indexes for mixed document types
- Global UID convention for cross-source identity
- Batched writes (3 transactions) for paper + chunks
- Content-hash guard to skip unchanged content
- Abstract embedding fallback to first 2000 chars

Collections:
- base_arxiv: Document collection storing papers (kind='paper') and chunks (kind='chunk')
- NO edge collections - citations are arrays in paper documents

Best Practices:
- Array indexes for reverse lookups
- Sparse compound indexes by 'kind'
- Global UIDs for cross-source identity
- Provenance tracking with ingest_run_id
"""

import os
import re
import sys
import time
import json
import gzip
import uuid
import hashlib
import logging
import argparse
import requests
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import torch
import gc
from arango import ArangoClient

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.enhanced_docling_processor_v2 import EnhancedDoclingProcessorV2
from core.jina_v4_embedder import JinaV4Embedder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class CitationOccurrence:
    """Single occurrence of a citation in text."""
    position: int
    context: str
    section: Optional[str] = None


@dataclass
class Citation:
    """Structured citation with all metadata."""
    id: str  # Target paper ID (e.g., "2301.12345" or "[1]")
    type: str  # 'numbered', 'author_year', 'arxiv', etc.
    raw_text: str  # Original citation text
    confidence: float  # Extraction confidence
    occurrences: List[CitationOccurrence]
    bib_number: Optional[int] = None  # For numbered citations
    bib_entry: Optional[str] = None  # Bibliography entry if found
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return {
            'target_id': self.id,
            'type': self.type,
            'raw_text': self.raw_text[:200],  # Limit size
            'confidence': self.confidence,
            'occurrence_count': len(self.occurrences),
            'positions': [occ.position for occ in self.occurrences[:20]],  # First 20
            'bib_number': self.bib_number,
            'bib_entry': self.bib_entry[:500] if self.bib_entry else None
        }


class EnhancedCitationExtractor:
    """Extract and resolve citations from academic text."""
    
    def __init__(self):
        # Regex patterns for different citation styles
        self.patterns = {
            'numbered': re.compile(r'\[(\d+(?:,\s*\d+)*)\]'),
            'numbered_range': re.compile(r'\[(\d+)\s*-\s*(\d+)\]'),
            'author_year': re.compile(
                r'(?:[A-Z][a-z]+(?:\s+(?:et\s+al\.|and\s+[A-Z][a-z]+))?)' +
                r'\s*[\(\[]?\s*(\d{4}[a-z]?)\s*[\)\]]?'
            ),
            'arxiv': re.compile(r'(?:arXiv:)?(\d{4}\.\d{4,5})(?:v\d+)?'),
            'arxiv_old': re.compile(r'(?:arXiv:)?([a-z-]+/\d{7})(?:v\d+)?'),
            'doi': re.compile(r'10\.\d{4,}/[-._;()/:\w]+'),
        }
        
        # Section headers that typically contain references
        self.ref_sections = [
            'references', 'bibliography', 'works cited', 'literature cited',
            'citations', 'sources', 'endnotes'
        ]
    
    def extract_citations(self, text: str, sections: Optional[List[Dict]] = None) -> List[Citation]:
        """Extract all citations from text."""
        citations = {}  # id -> Citation
        
        # Find bibliography section
        bib_text, bib_entries = self._extract_bibliography(text, sections)
        
        # Extract numbered citations
        for match in self.patterns['numbered'].finditer(text):
            numbers = match.group(1).split(',')
            for num_str in numbers:
                num = int(num_str.strip())
                cit_id = f"ref_{num}"
                
                if cit_id not in citations:
                    citations[cit_id] = Citation(
                        id=cit_id,
                        type='numbered',
                        raw_text=f"[{num}]",
                        confidence=0.95,
                        occurrences=[],
                        bib_number=num,
                        bib_entry=bib_entries.get(num)
                    )
                
                citations[cit_id].occurrences.append(
                    CitationOccurrence(
                        position=match.start(),
                        context=text[max(0, match.start()-50):min(len(text), match.end()+50)],
                        section=self._find_section(match.start(), sections)
                    )
                )
        
        # Extract numbered ranges
        for match in self.patterns['numbered_range'].finditer(text):
            start_num = int(match.group(1))
            end_num = int(match.group(2))
            for num in range(start_num, end_num + 1):
                cit_id = f"ref_{num}"
                
                if cit_id not in citations:
                    citations[cit_id] = Citation(
                        id=cit_id,
                        type='numbered',
                        raw_text=f"[{num}]",
                        confidence=0.9,
                        occurrences=[],
                        bib_number=num,
                        bib_entry=bib_entries.get(num)
                    )
                
                citations[cit_id].occurrences.append(
                    CitationOccurrence(
                        position=match.start(),
                        context=text[max(0, match.start()-50):min(len(text), match.end()+50)],
                        section=self._find_section(match.start(), sections)
                    )
                )
        
        # Extract arXiv citations
        for pattern_name in ['arxiv', 'arxiv_old']:
            for match in self.patterns[pattern_name].finditer(text):
                arxiv_id = match.group(1)
                
                if arxiv_id not in citations:
                    citations[arxiv_id] = Citation(
                        id=arxiv_id,
                        type='arxiv',
                        raw_text=match.group(0),
                        confidence=0.98,
                        occurrences=[]
                    )
                
                citations[arxiv_id].occurrences.append(
                    CitationOccurrence(
                        position=match.start(),
                        context=text[max(0, match.start()-50):min(len(text), match.end()+50)],
                        section=self._find_section(match.start(), sections)
                    )
                )
        
        # Extract DOI citations
        for match in self.patterns['doi'].finditer(text):
            doi = match.group(0)
            doi_id = f"doi:{doi}"
            
            if doi_id not in citations:
                citations[doi_id] = Citation(
                    id=doi_id,
                    type='doi',
                    raw_text=doi,
                    confidence=0.95,
                    occurrences=[]
                )
            
            citations[doi_id].occurrences.append(
                CitationOccurrence(
                    position=match.start(),
                    context=text[max(0, match.start()-50):min(len(text), match.end()+50)],
                    section=self._find_section(match.start(), sections)
                )
            )
        
        # Extract author-year citations
        for match in self.patterns['author_year'].finditer(text):
            # Extract author name and year
            full_match = match.group(0)
            year_match = match.group(1) if match.lastindex >= 1 else None
            
            if year_match:
                cit_id = f"author_year_{full_match.replace(' ', '_')}"
                
                if cit_id not in citations:
                    citations[cit_id] = Citation(
                        id=cit_id,
                        type='author_year',
                        raw_text=full_match,
                        confidence=0.7,  # Lower confidence for author-year
                        occurrences=[]
                    )
                
                citations[cit_id].occurrences.append(
                    CitationOccurrence(
                        position=match.start(),
                        context=text[max(0, match.start()-50):min(len(text), match.end()+50)],
                        section=self._find_section(match.start(), sections)
                    )
                )
        
        return list(citations.values())
    
    def _extract_bibliography(self, text: str, sections: Optional[List[Dict]] = None) -> Tuple[Optional[str], Dict[int, str]]:
        """Extract bibliography section and parse entries."""
        bib_text = None
        bib_entries = {}
        
        # Try to find bibliography section
        if sections:
            for section in sections:
                if any(ref in section.get('title', '').lower() for ref in self.ref_sections):
                    start = section.get('start', 0)
                    end = section.get('end', len(text))
                    bib_text = text[start:end]
                    break
        
        if not bib_text:
            # Fallback: look for "References" header
            for ref_header in ['References', 'REFERENCES', 'Bibliography', 'BIBLIOGRAPHY']:
                idx = text.rfind(ref_header)
                if idx != -1:
                    bib_text = text[idx:]
                    break
        
        if bib_text:
            # Parse numbered bibliography entries
            pattern = re.compile(r'^\s*\[(\d+)\]\s+(.+?)(?=^\s*\[\d+\]|\Z)', re.MULTILINE | re.DOTALL)
            for match in pattern.finditer(bib_text):
                num = int(match.group(1))
                entry = match.group(2).strip()
                bib_entries[num] = entry[:500]  # Limit size
        
        return bib_text, bib_entries
    
    def _find_section(self, position: int, sections: Optional[List[Dict]] = None) -> Optional[str]:
        """Find which section a position falls into."""
        if not sections:
            return None
        
        for section in sections:
            if section.get('start', 0) <= position < section.get('end', float('inf')):
                return section.get('title')
        
        return None


class PDFCitationProcessorV67:
    """Process PDFs with source-fidelity design and ArangoDB best practices."""
    
    def __init__(self, 
                 output_dir: str = '/tmp/arxiv_pdfs',
                 db_config: Optional[Dict] = None,
                 collections: Optional[Dict] = None):
        """Initialize processor with configurations."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize processors
        self.docling_processor = EnhancedDoclingProcessorV2()
        self.embedder = JinaV4Embedder()
        self.citation_extractor = EnhancedCitationExtractor()
        
        # Setup database
        self.db_config = db_config or {
            'host': 'localhost',
            'port': 8529,
            'username': 'root',
            'password': os.getenv('ARANGO_PASSWORD'),
            'database': '_system'
        }
        
        self.collections = collections or {
            'base_repo': 'base_arxiv'  # Single collection only
        }
        
        self._setup_database()
    
    def _setup_database(self):
        """Setup ArangoDB connection and ensure collections exist."""
        client = ArangoClient(hosts=f"http://{self.db_config['host']}:{self.db_config['port']}")
        sys_db = client.db('_system', 
                           username=self.db_config['username'],
                           password=self.db_config['password'])
        
        # Use specified database or create it
        db_name = self.db_config['database']
        if not sys_db.has_database(db_name):
            sys_db.create_database(db_name)
        
        self.db = client.db(db_name,
                            username=self.db_config['username'],
                            password=self.db_config['password'])
        
        # Ensure base collection exists
        if not self.db.has_collection(self.collections['base_repo']):
            self.base_repo = self.db.create_collection(self.collections['base_repo'])
            logger.info(f"Created collection {self.collections['base_repo']}")
        else:
            self.base_repo = self.db.collection(self.collections['base_repo'])
        
        # Create sparse compound indexes for mixed document types
        self._setup_indexes()
    
    def _setup_indexes(self):
        """Setup optimal indexes for mixed document types."""
        coll = self.base_repo
        
        # Sparse compound indexes by 'kind' to avoid bloat
        try:
            # For papers
            coll.add_persistent_index(
                fields=['kind', 'arxiv_id'],
                unique=False,
                sparse=True,
                name='idx_paper_arxiv'
            )
            
            # For chunks  
            coll.add_persistent_index(
                fields=['kind', 'doc_id', 'chunk_index'],
                unique=True,
                sparse=True,
                name='idx_chunk_lookup'
            )
            
            # Sparse index for fetching all chunks for a doc (non-unique)
            coll.add_persistent_index(
                fields=['kind', 'doc_id'],
                unique=False,
                sparse=True,
                name='idx_doc_chunks'
            )
            
            # Array index for citation lookups (who cites X?)
            coll.add_persistent_index(
                fields=['citations[*].target_id'],
                unique=False,
                sparse=True,
                name='idx_citation_targets'
            )
            
            # Global UID for cross-source identity
            coll.add_persistent_index(
                fields=['uid'],
                unique=True,
                sparse=True,
                name='idx_global_uid'
            )
            
            logger.info("Created sparse compound indexes")
            
        except Exception as e:
            logger.info(f"Some indexes may already exist: {e}")
    
    def _download_pdf(self, arxiv_id: str) -> Optional[Path]:
        """Download PDF from arXiv with retry logic."""
        pdf_path = self.output_dir / f"{arxiv_id.replace('/', '_')}.pdf"
        
        if pdf_path.exists():
            logger.info(f"Using existing PDF: {pdf_path}")
            return pdf_path
        
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        
        # Retry logic with backoff
        for attempt in range(3):
            try:
                logger.info(f"Downloading PDF from {pdf_url} (attempt {attempt + 1})")
                response = requests.get(pdf_url, timeout=30, headers={
                    'User-Agent': 'PDFCitationProcessor/6.7 (source-fidelity design)'
                })
                response.raise_for_status()
                
                pdf_path.write_bytes(response.content)
                logger.info(f"Downloaded PDF to {pdf_path}")
                return pdf_path
                
            except requests.HTTPError as e:
                if e.response and e.response.status_code in (429, 503) and attempt < 2:
                    # Respect Retry-After header if present
                    retry_after = int(e.response.headers.get('Retry-After', '2'))
                    logger.warning(f"Rate limited, waiting {retry_after} seconds before retry")
                    time.sleep(retry_after)
                else:
                    logger.error(f"Failed to download PDF: {e}")
                    if attempt < 2:
                        time.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        return None
            except Exception as e:
                logger.error(f"Failed to download PDF: {e}")
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    return None
        
        return None
    
    def _fetch_arxiv_metadata(self, arxiv_id: str) -> Optional[Dict]:
        """Fetch metadata from arXiv API with retry logic."""
        base_url = "http://export.arxiv.org/api/query"
        params = {
            'id_list': arxiv_id,
            'max_results': 1
        }
        
        # Retry logic for API calls
        for attempt in range(3):
            try:
                response = requests.get(base_url, params=params, timeout=10)
                response.raise_for_status()
                break  # Success, exit retry loop
            except requests.HTTPError as e:
                if e.response and e.response.status_code in (429, 503) and attempt < 2:
                    retry_after = int(e.response.headers.get('Retry-After', '2'))
                    logger.warning(f"API rate limited, waiting {retry_after} seconds")
                    time.sleep(retry_after)
                    continue
                elif attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                else:
                    logger.error(f"Failed to fetch metadata after {attempt + 1} attempts: {e}")
                    return None
            except Exception as e:
                if attempt < 2:
                    logger.warning(f"Metadata fetch attempt {attempt + 1} failed: {e}")
                    time.sleep(2 ** attempt)
                    continue
                else:
                    logger.error(f"Failed to fetch metadata after {attempt + 1} attempts: {e}")
                    return None
        
        try:
            
            root = ET.fromstring(response.text)
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            
            entry = root.find('.//atom:entry', ns)
            if not entry:
                return None
            
            # Extract metadata
            metadata = {
                'arxiv_id': arxiv_id,
                'title': entry.find('atom:title', ns).text.strip().replace('\n', ' '),
                'abstract': entry.find('atom:summary', ns).text.strip(),
                'authors': [author.find('atom:name', ns).text 
                           for author in entry.findall('atom:author', ns)],
                'published': entry.find('atom:published', ns).text,
                'updated': entry.find('atom:updated', ns).text,
                'categories': [cat.get('term') 
                              for cat in entry.findall('atom:category', ns)],
            }
            
            # Optional fields
            for field in ['doi', 'journal_ref', 'comment']:
                elem = entry.find(f'.//{{http://arxiv.org/schemas/atom}}{field}')
                if elem is not None and elem.text:
                    metadata[field] = elem.text.strip()
            
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to fetch metadata: {e}")
            return None
    
    def _embed_with_retry(self, texts: List[str], max_retries: int = 3) -> Optional[np.ndarray]:
        """Embed texts with retry logic."""
        for attempt in range(max_retries):
            try:
                embeddings = self.embedder.embed_texts(texts)
                return embeddings
            except Exception as e:
                logger.warning(f"Embedding attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
        return None
    
    def _to_list_f32(self, arr: Any) -> Tuple[List[float], str]:
        """Convert array to list of float32."""
        if isinstance(arr, (list, tuple)):
            return list(map(float, arr)), 'list'
        elif hasattr(arr, 'tolist'):
            return arr.astype(np.float32).tolist(), 'numpy'
        else:
            return list(arr), 'unknown'
    
    def _create_structure_aware_chunks(self, text: str, sections: Optional[List[Dict]] = None,
                                      chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
        """Create chunks with structure awareness and overlap."""
        chunks = []
        
        if sections:
            # Structure-aware chunking
            for section in sections:
                section_text = text[section.get('start', 0):section.get('end', len(text))]
                section_chunks = self._chunk_text(
                    section_text,
                    chunk_size=chunk_size,
                    overlap=overlap,
                    base_offset=section.get('start', 0)
                )
                
                for chunk in section_chunks:
                    chunk['section'] = section.get('title')
                    chunk['strategy'] = 'structure_aware'
                    chunks.append(chunk)
        else:
            # Simple overlapping chunks
            chunks = self._chunk_text(text, chunk_size=chunk_size, overlap=overlap)
            for chunk in chunks:
                chunk['strategy'] = 'simple_overlap'
        
        return chunks
    
    def _chunk_text(self, text: str, chunk_size: int = 1000, 
                   overlap: int = 200, base_offset: int = 0) -> List[Dict]:
        """Create overlapping text chunks."""
        # Safety guard: overlap must be less than chunk_size
        if overlap >= chunk_size:
            raise ValueError(f"overlap ({overlap}) must be < chunk_size ({chunk_size})")
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            
            # Try to break at sentence boundary
            if end < len(text):
                last_period = text.rfind('.', start, end)
                if last_period != -1 and last_period > start + chunk_size // 2:
                    end = last_period + 1
            
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    'text': chunk_text,
                    'start': base_offset + start,
                    'end': base_offset + end
                })
            
            # Move with overlap (proper calculation)
            if end >= len(text):
                break
            start = end - overlap  # Simple overlap, no min/max needed
        
        return chunks
    
    def _write_atomic(self, paper_doc: Dict, chunk_docs: List[Dict], clean: bool = True) -> Dict:
        """Batched writes to store paper and chunks (3 separate transactions for ArangoDB compatibility)."""
        try:
            cleaned_count = 0
            
            # Step 1: Clean existing chunks if requested (separate transaction)
            if clean:
                # Count existing chunks
                count_result = self.db.aql.execute("""
                    FOR c IN @@collection
                    FILTER c.kind == 'chunk' AND c.doc_id == @doc_id
                    COLLECT WITH COUNT INTO total
                    RETURN total
                """, bind_vars={
                    '@collection': self.collections['base_repo'],
                    'doc_id': paper_doc['_key']
                })
                result_list = list(count_result)
                cleaned_count = result_list[0] if result_list else 0
                
                # Delete existing chunks
                if cleaned_count > 0:
                    self.db.aql.execute("""
                        FOR c IN @@collection
                        FILTER c.kind == 'chunk' AND c.doc_id == @doc_id
                        REMOVE c IN @@collection
                    """, bind_vars={
                        '@collection': self.collections['base_repo'],
                        'doc_id': paper_doc['_key']
                    })
            
            # Step 2: Upsert paper document (separate transaction)
            self.db.aql.execute("""
                UPSERT { _key: @paper._key }
                INSERT @paper
                REPLACE @paper
                IN @@collection
            """, bind_vars={
                '@collection': self.collections['base_repo'],
                'paper': paper_doc
            })
            
            # Step 3: Upsert chunks (separate transaction) - handles re-runs without clean
            if chunk_docs:
                self.db.aql.execute("""
                    FOR chunk IN @chunks
                        UPSERT { _key: chunk._key }
                        INSERT chunk
                        REPLACE chunk
                        IN @@collection
                """, bind_vars={
                    '@collection': self.collections['base_repo'],
                    'chunks': chunk_docs
                })
            
            # Return stats
            stats = {
                'paper': 1,
                'chunks': len(chunk_docs),
                'cleaned': cleaned_count
            }
            
            logger.info(f"Write complete: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Write failed: {e}")
            raise
    
    def process_pdf(self, arxiv_id: str, pdf_path: Optional[str] = None,
                   clean: bool = False, dry_run: bool = False, 
                   skip_embeddings: bool = False, force: bool = False) -> Dict:
        """Process a PDF with atomic transactions and content-hash guard."""
        start_time = time.time()
        doc_key = arxiv_id.replace('/', '_')
        ingest_run_id = str(uuid.uuid4())
        
        # Global UID for cross-source identity
        global_uid = f"arxiv:{arxiv_id}"
        
        try:
            # Fetch fresh metadata from arXiv API
            logger.info(f"Fetching fresh metadata for {arxiv_id}...")
            metadata = self._fetch_arxiv_metadata(arxiv_id)
            if not metadata:
                logger.warning(f"Could not fetch metadata for {arxiv_id}, using minimal info")
                metadata = {
                    'arxiv_id': arxiv_id,
                    'title': f"Paper {arxiv_id}",
                    'abstract': "",
                    'authors': [],
                    'categories': [],
                    'published': "",
                    'updated': ""
                }
            
            # Check existing document BEFORE any cleanup
            existing = None
            if not clean and self.base_repo.has(doc_key):
                try:
                    existing = self.base_repo.get(doc_key)
                    logger.info(f"Found existing document for {arxiv_id}")
                except Exception:
                    existing = None
            
            # Download or use provided PDF
            if pdf_path:
                pdf_path = Path(pdf_path)
            else:
                pdf_path = self._download_pdf(arxiv_id)
                if not pdf_path:
                    return {'error': f'Failed to download PDF for {arxiv_id}'}
            
            # Process with Docling
            logger.info(f"Processing {arxiv_id} with Docling...")
            docling_result = self.docling_processor.process_pdf(arxiv_id=arxiv_id, pdf_path=str(pdf_path))
            
            # Extract text from Docling result
            full_text, sections = "", None
            if docling_result.get('markdown'):
                full_text = docling_result['markdown']
                sections = docling_result.get('sections')
            else:
                if not docling_result.get('success'):
                    return {'error': docling_result.get('error', 'Docling processing failed')}
                output_path = docling_result.get('output_path')
                if not output_path or not Path(output_path).exists():
                    return {'error': 'Markdown output file not found'}
                with open(output_path, 'r', encoding='utf-8') as f:
                    full_text = f.read()
                sections = docling_result.get('sections')
            
            if not full_text:
                return {'error': 'No text extracted from PDF'}
            
            # Check content hash BEFORE any database changes
            new_hash = hashlib.sha256(full_text.encode()).hexdigest()
            if existing and not clean and not force:
                if (existing.get('full_text_hash') == new_hash and 
                    existing.get('processor_version') == 'v6.7'):
                    logger.info("Content unchanged and same version; skipping rebuild")
                    return {
                        'status': 'noop',
                        'reason': 'unchanged',
                        'arxiv_id': arxiv_id,
                        'processing_time': time.time() - start_time
                    }
            
            # Save markdown to compressed file
            md_gz_path = pdf_path.with_suffix('.md.gz')
            with gzip.open(md_gz_path, 'wt', encoding='utf-8') as f:
                f.write(full_text)
            
            # Extract citations
            logger.info(f"Extracting citations from {arxiv_id}...")
            citations = self.citation_extractor.extract_citations(full_text, sections)
            logger.info(f"Found {len(citations)} total citations")
            
            # Filter high confidence citations
            high_confidence = [c for c in citations if c.confidence >= 0.7]
            logger.info(f"Keeping {len(high_confidence)} high-confidence citations")
            
            # Generate document embedding with fallback
            doc_embedding = None
            embedding_dim = 0
            embedding_dtype = 'unknown'
            
            if not skip_embeddings:
                # Use abstract if available, otherwise first 2000 chars
                embed_text = metadata.get('abstract') or full_text[:2000]
                if embed_text and embed_text.strip():
                    logger.info(f"Generating document embedding...")
                    doc_embeddings = self._embed_with_retry([embed_text])
                    if doc_embeddings is not None:
                        doc_embedding, embedding_dtype = self._to_list_f32(doc_embeddings[0])
                        embedding_dim = len(doc_embedding)
                        logger.info(f"Generated embedding with {embedding_dim} dimensions")
            
            # Create chunks with overlap
            chunks = self._create_structure_aware_chunks(full_text, sections)
            logger.info(f"Created {len(chunks)} chunks")
            
            # Add citation references to chunks
            for i, chunk in enumerate(chunks):
                chunk_start = chunk['start']
                chunk_end = chunk['end']
                chunk_citations = []
                
                for citation in high_confidence:
                    for occ in citation.occurrences:
                        if occ.position >= chunk_start and occ.position < chunk_end:
                            if citation.id not in chunk_citations:
                                chunk_citations.append(citation.id)
                
                chunk['citation_ids'] = chunk_citations
                chunk['chunk_index'] = i
            
            # Generate chunk embeddings using LATE CHUNKING
            if not skip_embeddings:
                logger.info(f"Generating embeddings for {len(chunks)} chunks using late chunking...")
                
                # Get late-chunked embeddings with position information
                # Using smaller windows for better granularity
                # Approx 1k tokens = 4k chars, with 25% overlap for context
                late_windows = self.embedder.embed_with_late_chunking(
                    full_text,
                    chunk_size=4000,  # ~1k tokens in characters
                    chunk_overlap=1000  # 25% overlap for context preservation
                )
                
                # Guard against empty/None windows and fallback to simple embedding
                if not late_windows:
                    logger.warning("Late chunking returned no windows, falling back to simple embeddings")
                    chunk_texts = [c['text'] for c in chunks]
                    chunk_embeddings = self._embed_with_retry(chunk_texts)
                    if chunk_embeddings is not None:
                        for i, emb in enumerate(chunk_embeddings[:len(chunks)]):
                            chunks[i]['embedding'], _ = self._to_list_f32(emb)
                            chunks[i]['embedding_method'] = 'fallback_simple'
                else:
                    # Verify late_windows contract
                    assert all({'start', 'end', 'embedding'} <= set(w.keys()) for w in late_windows), \
                        "late_windows schema changed - expected keys: start, end, embedding"
                    
                    logger.info(f"Created {len(late_windows)} late-chunked windows for {len(full_text)} chars")
                    
                    # Ensure embeddings are numpy arrays for consistent math
                    for w in late_windows:
                        w['embedding'] = np.asarray(w['embedding'], dtype=np.float32)
                    
                    # Map embeddings to our structure-aware chunks using weighted pooling
                    for chunk in chunks:
                        chunk_start = chunk['start']
                        chunk_end = chunk['end']
                        chunk_mid = (chunk_start + chunk_end) // 2
                        
                        # Find all overlapping windows
                        overlapping = []
                        for window in late_windows:
                            # Calculate overlap
                            overlap_start = max(chunk_start, window['start'])
                            overlap_end = min(chunk_end, window['end'])
                            overlap_len = max(0, overlap_end - overlap_start)
                            
                            if overlap_len > 0:
                                overlapping.append({
                                    'embedding': window['embedding'],
                                    'weight': overlap_len
                                })
                        
                        if overlapping:
                            # Weighted average of overlapping embeddings
                            total_weight = sum(w['weight'] for w in overlapping)
                            # Use float32 numpy array for consistent math
                            weighted_embedding = np.zeros_like(overlapping[0]['embedding'], dtype=np.float32)
                            
                            for item in overlapping:
                                weighted_embedding += item['embedding'] * (item['weight'] / total_weight)
                            
                            emb, _ = self._to_list_f32(weighted_embedding)
                            chunk['embedding'] = emb
                            chunk['embedding_method'] = 'late_chunking_weighted'
                        else:
                            # Fallback: find closest window by center distance
                            closest = min(late_windows, 
                                        key=lambda w: abs((w['start'] + w['end'])/2 - chunk_mid))
                            emb, _ = self._to_list_f32(closest['embedding'])
                            chunk['embedding'] = emb
                            chunk['embedding_method'] = 'late_chunking_closest'
            
            if dry_run:
                return {
                    'status': 'dry_run',
                    'arxiv_id': arxiv_id,
                    'text_length': len(full_text),
                    'chunk_count': len(chunks),
                    'citation_count': len(high_confidence),
                    'embedding_dim': embedding_dim
                }
            
            # Build paper document with citations as array
            paper_doc = {
                '_key': doc_key,
                'uid': global_uid,  # Global identifier
                'kind': 'paper',
                'repository': 'arxiv',
                
                # Metadata from arXiv API
                'arxiv_id': arxiv_id,
                'title': metadata['title'],
                'abstract': metadata['abstract'],
                'authors': metadata['authors'],
                'categories': metadata['categories'],
                'published': metadata['published'],
                'updated': metadata['updated'],
                'doi': metadata.get('doi'),
                'comment': metadata.get('comment'),
                'journal_ref': metadata.get('journal_ref'),
                
                # PDF processing results
                'pdf_status': 'processed',
                'pdf_local_path': str(pdf_path),
                'pdf_url': f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                'source_url': f"https://arxiv.org/abs/{arxiv_id}",
                'full_text_sample': full_text[:500],
                'full_text_length': len(full_text),
                'full_text_hash': new_hash,
                'full_text_path': str(md_gz_path),
                
                # Citations as array (NO edges!)
                'citations': [c.to_dict() for c in high_confidence],
                'citation_count': len(high_confidence),
                'numbered_refs': len([c for c in high_confidence if c.type == 'numbered']),
                'arxiv_refs': len([c for c in high_confidence if c.type == 'arxiv']),
                
                # Sections
                'sections': sections,
                
                # Provenance tracking
                'processing_date': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'processor_version': 'v6.7',
                'docling_version': '2.12.0',  # Add Docling version for reproducibility
                'schema_version': '1.0',
                'chunk_count': len(chunks),
                'embedding_dim': embedding_dim,
                'embedding_dtype': embedding_dtype,
                'embedding_model': 'jinaai/jina-embeddings-v4',
                'embedding_task': 'retrieval',
                'ingest_run_id': ingest_run_id
            }
            
            # Add document embedding if generated
            if doc_embedding is not None:
                paper_doc['embedding'] = doc_embedding
                paper_doc['storage_dtype'] = 'float32'
            
            # Build chunk documents
            chunk_docs = []
            for i, chunk in enumerate(chunks):
                chunk_doc = {
                    '_key': f"{doc_key}_chunk_{i}",
                    'uid': f"{global_uid}#chunk_{i}",  # Global identifier
                    'kind': 'chunk',
                    'repository': 'arxiv',
                    'doc_id': doc_key,
                    'chunk_index': i,
                    'chunk_type': 'full_text',
                    
                    'text': chunk['text'],
                    'start_char': chunk['start'],
                    'end_char': chunk['end'],
                    'strategy': chunk.get('strategy'),
                    'section': chunk.get('section'),
                    
                    'citation_ids': chunk.get('citation_ids', []),
                    'processing_date': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                    'ingest_run_id': ingest_run_id
                }
                
                if 'embedding' in chunk:
                    chunk_doc['embedding'] = chunk['embedding']
                    chunk_doc['embedding_dim'] = len(chunk['embedding'])
                    chunk_doc['storage_dtype'] = 'float32'
                
                chunk_docs.append(chunk_doc)
            
            # Use batched writes to store everything
            stats = self._write_atomic(paper_doc, chunk_docs, clean=clean)
            
            return {
                'status': 'success',
                'arxiv_id': arxiv_id,
                'uid': global_uid,
                'chunk_count': stats['chunks'],
                'citation_count': len(high_confidence),
                'embedding_dim': embedding_dim,
                'full_text_stored': str(md_gz_path),
                'processing_time': time.time() - start_time,
                'ingest_run_id': ingest_run_id,
                'message': 'Successfully processed with v6.7 source-fidelity design'
            }
            
        except Exception as e:
            logger.error(f"Processing failed for {arxiv_id}: {e}")
            return {
                'error': str(e),
                'arxiv_id': arxiv_id,
                'processing_time': time.time() - start_time
            }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Process PDFs with citations (v6.7 - Source Fidelity)')
    parser.add_argument('arxiv_id', help='ArXiv ID to process')
    parser.add_argument('--pdf-path', help='Path to existing PDF file')
    parser.add_argument('--output-dir', default='/tmp/arxiv_pdfs', 
                       help='Directory for downloaded PDFs')
    parser.add_argument('--clean', action='store_true',
                       help='Clean all existing data for this paper before processing')
    parser.add_argument('--dry-run', action='store_true',
                       help='Estimate processing without database writes')
    parser.add_argument('--skip-embeddings', action='store_true',
                       help='Skip embedding generation')
    parser.add_argument('--force', action='store_true',
                       help='Force reprocessing even if content unchanged')
    parser.add_argument('--db-host', default='localhost', help='ArangoDB host')
    parser.add_argument('--db-port', type=int, default=8529, help='ArangoDB port')
    parser.add_argument('--db-name', default='_system', help='Database name')
    parser.add_argument('--db-user', default='root', help='Database user')
    parser.add_argument('--base-collection', default='base_arxiv',
                       help='Base repository collection name')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Configure database
    db_config = {
        'host': args.db_host,
        'port': args.db_port,
        'username': args.db_user,
        'password': os.getenv('ARANGO_PASSWORD'),
        'database': args.db_name
    }
    
    # Configure collections (single collection only!)
    collections = {
        'base_repo': args.base_collection
    }
    
    # Create processor
    processor = PDFCitationProcessorV67(
        output_dir=args.output_dir,
        db_config=db_config,
        collections=collections
    )
    
    # Process PDF
    result = processor.process_pdf(
        arxiv_id=args.arxiv_id,
        pdf_path=args.pdf_path,
        clean=args.clean,
        dry_run=args.dry_run,
        skip_embeddings=args.skip_embeddings,
        force=args.force
    )
    
    # Print results
    print(json.dumps(result, indent=2))
    
    if result.get('status') == 'success':
        return 0
    else:
        return 1


if __name__ == '__main__':
    sys.exit(main())