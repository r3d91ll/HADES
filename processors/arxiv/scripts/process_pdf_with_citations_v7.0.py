#!/usr/bin/env python3
"""
Process PDFs with citations - Version 7.0
==========================================

Phase 2 Architecture Update: Per-relation edge collections with delta updates.
Incorporates reviewer feedback on graph structure.

Key Changes from v6.7:
- Separate edge collections by relation type (edges_cites, edges_mentions, etc.)
- Delta edge updates (only modify what changed)
- Soft-delete support for removed edges
- PRUNE-friendly edge structure for efficient traversals
- 32-character edge keys to avoid collisions

Collections:
- base_arxiv: Papers and metadata (vertices)
- base_arxiv_chunks: Paper chunks with embeddings
- edges_cites: Paper → Paper citations
- edges_mentions: Paper → Paper mentions
- edges_contains: Paper → Chunk containment
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
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from datetime import datetime, timezone
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


class PDFCitationProcessorV7:
    """Process PDFs with Phase 2 architecture: per-relation edges with delta updates."""
    
    def __init__(self, 
                 output_dir: str = '/tmp/arxiv_pdfs',
                 db_config: Optional[Dict] = None):
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
            'database': 'academy_store'
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
        
        # Ensure vertex collections exist
        vertex_collections = ['base_arxiv', 'base_arxiv_chunks']
        for coll_name in vertex_collections:
            if not self.db.has_collection(coll_name):
                self.db.create_collection(coll_name)
                logger.info(f"Created vertex collection {coll_name}")
        
        # Ensure edge collections exist (per-relation)
        edge_collections = [
            'edges_cites',      # Paper → Paper citations
            'edges_mentions',   # Paper → Paper mentions
            'edges_contains',   # Paper → Chunk containment
        ]
        
        for coll_name in edge_collections:
            if not self.db.has_collection(coll_name):
                self.db.create_collection(coll_name, edge=True)
                logger.info(f"Created edge collection {coll_name}")
        
        # Setup indexes
        self._setup_indexes()
    
    def _setup_indexes(self):
        """Setup optimal indexes for the new architecture."""
        # Vertex indexes for base_arxiv
        arxiv_coll = self.db.collection('base_arxiv')
        try:
            # ArXiv ID lookup
            arxiv_coll.add_persistent_index(
                fields=['arxiv_id'],
                unique=True,
                sparse=False,
                name='idx_arxiv_id'
            )
            
            # Global UID
            arxiv_coll.add_persistent_index(
                fields=['uid'],
                unique=True,
                sparse=True,
                name='idx_uid'
            )
            
            # Vector index for embeddings (if ArangoDB 3.11+)
            # arxiv_coll.add_vector_index(
            #     fields=['embedding'],
            #     dimension=2048,
            #     metric='cosine',
            #     name='idx_embedding'
            # )
            
            logger.info("Created indexes for base_arxiv")
        except Exception as e:
            logger.info(f"Some indexes may already exist: {e}")
        
        # Chunk indexes
        chunk_coll = self.db.collection('base_arxiv_chunks')
        try:
            # Lookup by parent document
            chunk_coll.add_persistent_index(
                fields=['doc_id', 'chunk_index'],
                unique=True,
                sparse=False,
                name='idx_chunk_lookup'
            )
            
            # Vector index for chunk embeddings
            # chunk_coll.add_vector_index(
            #     fields=['embedding'],
            #     dimension=2048,
            #     metric='cosine',
            #     name='idx_chunk_embedding'
            # )
            
            logger.info("Created indexes for base_arxiv_chunks")
        except Exception as e:
            logger.info(f"Some indexes may already exist: {e}")
        
        # Edge indexes (only what we filter on, NOT _from/_to)
        for edge_name in ['edges_cites', 'edges_mentions', 'edges_contains']:
            edge_coll = self.db.collection(edge_name)
            try:
                # Index for filtering by relation and source
                edge_coll.add_persistent_index(
                    fields=['relation', 'source_system'],
                    unique=False,
                    sparse=False,
                    name='idx_relation_source'
                )
                
                # Index for filtering by confidence (for PRUNE operations)
                edge_coll.add_persistent_index(
                    fields=['confidence'],
                    unique=False,
                    sparse=False,
                    name='idx_confidence'
                )
                
                # Index for active/inactive (soft deletes)
                edge_coll.add_persistent_index(
                    fields=['active'],
                    unique=False,
                    sparse=False,
                    name='idx_active'
                )
                
                logger.info(f"Created indexes for {edge_name}")
            except Exception as e:
                logger.info(f"Some indexes may already exist: {e}")
    
    def _generate_edge_key(self, from_id: str, to_id: str, 
                          relation: str, source: str) -> str:
        """Generate deterministic 32-char edge key."""
        components = f"{from_id}|{to_id}|{relation}|{source}"
        return hashlib.sha256(components.encode()).hexdigest()[:32]
    
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
                    'User-Agent': 'PDFCitationProcessor/7.0 (Phase 2 architecture)'
                })
                response.raise_for_status()
                
                pdf_path.write_bytes(response.content)
                logger.info(f"Downloaded PDF to {pdf_path}")
                return pdf_path
                
            except requests.HTTPError as e:
                if e.response and e.response.status_code in (429, 503) and attempt < 2:
                    retry_after = int(e.response.headers.get('Retry-After', '2'))
                    logger.warning(f"Rate limited, waiting {retry_after} seconds before retry")
                    time.sleep(retry_after)
                else:
                    logger.error(f"Failed to download PDF: {e}")
                    if attempt < 2:
                        time.sleep(2 ** attempt)
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
        """Fetch metadata from arXiv API."""
        base_url = "http://export.arxiv.org/api/query"
        params = {
            'id_list': arxiv_id,
            'max_results': 1
        }
        
        try:
            response = requests.get(base_url, params=params, timeout=10)
            response.raise_for_status()
            
            root = ET.fromstring(response.text)
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            
            entry = root.find('.//atom:entry', ns)
            if not entry:
                return None
            
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
            
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to fetch metadata: {e}")
            return None
    
    def _upsert_vertex_and_delta_edges(self,
                                       vertex_col: str,
                                       edge_col: str,
                                       from_id: str,
                                       new_doc: Dict,
                                       relation: str,
                                       to_ids_new: List[str],
                                       source: str = "arxiv",
                                       confidence: float = 1.0) -> Dict:
        """
        Upsert vertex and perform delta edge updates.
        Only modifies edges that have changed.
        """
        now = datetime.now(timezone.utc).isoformat()
        
        # AQL for atomic delta updates
        aql_query = """
        LET from_id = @from_id
        LET new_doc = @new_doc
        LET relation = @relation
        LET source = @source
        LET to_ids_new = @to_ids_new
        LET now = @now
        
        // 1) Upsert vertex
        UPSERT { _id: from_id }
        INSERT MERGE({ _id: from_id, first_seen_at: now }, new_doc)
        UPDATE MERGE(NEW, { last_seen_at: now })
        IN @@vertex_col
        
        // 2) Find existing outbound edges for this relation/source
        LET existing = (
            FOR e IN @@edge_col
            FILTER e._from == from_id 
                AND e.relation == relation 
                AND e.source_system == source
                AND e.active == true
            RETURN e._to
        )
        
        // 3) Calculate additions and removals
        LET to_add = MINUS(to_ids_new, existing)
        LET to_remove = MINUS(existing, to_ids_new)
        
        // 4) Add new edges
        FOR t IN to_add
            LET k = @generate_key(from_id, t, relation, source)
            UPSERT { _key: k }
            INSERT {
                _key: k,
                _from: from_id,
                _to: t,
                relation: relation,
                source_system: source,
                confidence: @confidence,
                active: true,
                extraction_run_id: now,
                added_at: now
            }
            UPDATE { 
                active: true, 
                extraction_run_id: now,
                reactivated_at: now
            }
            IN @@edge_col
        
        // 5) Soft-delete removed edges
        FOR t IN to_remove
            LET k = @generate_key(from_id, t, relation, source)
            UPDATE { 
                _key: k,
                active: false,
                deactivated_at: now
            }
            IN @@edge_col
            OPTIONS { ignoreErrors: true }
        
        RETURN {
            added: LENGTH(to_add),
            removed: LENGTH(to_remove),
            total_active: LENGTH(to_ids_new)
        }
        """
        
        # Helper function for key generation
        def generate_key(from_id, to_id, relation, source):
            return self._generate_edge_key(from_id, to_id, relation, source)
        
        # Execute query
        cursor = self.db.aql.execute(
            aql_query,
            bind_vars={
                '@vertex_col': vertex_col,
                '@edge_col': edge_col,
                'from_id': from_id,
                'new_doc': new_doc,
                'relation': relation,
                'source': source,
                'to_ids_new': to_ids_new,
                'confidence': confidence,
                'now': now,
                'generate_key': generate_key
            }
        )
        
        result = list(cursor)
        return result[0] if result else {'added': 0, 'removed': 0, 'total_active': 0}
    
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
            
            if end >= len(text):
                break
            start = end - overlap
        
        return chunks
    
    def process_pdf(self, arxiv_id: str, pdf_path: Optional[str] = None,
                   force: bool = False) -> Dict:
        """Process a PDF with Phase 2 architecture."""
        start_time = time.time()
        doc_key = arxiv_id.replace('/', '_')
        ingest_run_id = str(uuid.uuid4())
        
        # Global UID for cross-source identity
        global_uid = f"arxiv:{arxiv_id}"
        
        try:
            # Fetch metadata
            logger.info(f"Fetching metadata for {arxiv_id}...")
            metadata = self._fetch_arxiv_metadata(arxiv_id)
            if not metadata:
                metadata = {
                    'arxiv_id': arxiv_id,
                    'title': f"Paper {arxiv_id}",
                    'abstract': "",
                    'authors': [],
                    'categories': [],
                    'published': "",
                    'updated': ""
                }
            
            # Check existing document
            existing = None
            try:
                existing = self.db.collection('base_arxiv').get(doc_key)
                logger.info(f"Found existing document for {arxiv_id}")
            except:
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
            
            # Extract text
            full_text = docling_result.get('markdown', '')
            sections = docling_result.get('sections')
            
            if not full_text:
                return {'error': 'No text extracted from PDF'}
            
            # Check content hash
            new_hash = hashlib.sha256(full_text.encode()).hexdigest()
            if existing and not force:
                if existing.get('full_text_hash') == new_hash:
                    logger.info("Content unchanged; updating edges only")
                    # Still need to update edges in case citations changed
            
            # Extract citations
            logger.info(f"Extracting citations from {arxiv_id}...")
            citations = self.citation_extractor.extract_citations(full_text, sections)
            high_confidence = [c for c in citations if c.confidence >= 0.7]
            logger.info(f"Found {len(high_confidence)} high-confidence citations")
            
            # Prepare citation target IDs for edges
            citation_targets = []
            for citation in high_confidence:
                if citation.type == 'arxiv':
                    # Direct ArXiv reference
                    target_id = f"base_arxiv/{citation.id.replace('/', '_')}"
                    citation_targets.append(target_id)
                elif citation.type == 'numbered' and citation.bib_entry:
                    # Try to extract ArXiv ID from bibliography entry
                    arxiv_match = re.search(r'(\d{4}\.\d{4,5})', citation.bib_entry)
                    if arxiv_match:
                        target_id = f"base_arxiv/{arxiv_match.group(1).replace('/', '_')}"
                        citation_targets.append(target_id)
            
            # Build paper document
            paper_doc = {
                '_key': doc_key,
                '_id': f"base_arxiv/{doc_key}",
                'uid': global_uid,
                'arxiv_id': arxiv_id,
                'title': metadata['title'],
                'abstract': metadata['abstract'],
                'authors': metadata['authors'],
                'categories': metadata['categories'],
                'published': metadata['published'],
                'updated': metadata['updated'],
                'full_text_hash': new_hash,
                'full_text_length': len(full_text),
                'citation_count': len(high_confidence),
                'citations_metadata': [c.to_dict() for c in high_confidence],  # Keep metadata
                'processing_date': datetime.now(timezone.utc).isoformat(),
                'processor_version': 'v7.0',
                'ingest_run_id': ingest_run_id
            }
            
            # Update paper and citation edges (delta update)
            logger.info(f"Updating paper and {len(citation_targets)} citation edges...")
            edge_stats = self._upsert_vertex_and_delta_edges(
                vertex_col='base_arxiv',
                edge_col='edges_cites',
                from_id=f"base_arxiv/{doc_key}",
                new_doc=paper_doc,
                relation='cites',
                to_ids_new=citation_targets,
                source='arxiv',
                confidence=0.95
            )
            
            logger.info(f"Edge updates: {edge_stats}")
            
            # Create chunks
            chunks = self._create_structure_aware_chunks(full_text, sections)
            logger.info(f"Created {len(chunks)} chunks")
            
            # Generate chunk embeddings using late chunking
            logger.info(f"Generating embeddings for {len(chunks)} chunks...")
            late_windows = self.embedder.embed_with_late_chunking(
                full_text,
                chunk_size=4000,
                chunk_overlap=1000
            )
            
            # Process and store chunks
            chunk_ids = []
            for i, chunk in enumerate(chunks):
                chunk_key = f"{doc_key}_chunk_{i}"
                chunk_id = f"base_arxiv_chunks/{chunk_key}"
                chunk_ids.append(chunk_id)
                
                # Find embedding for this chunk (weighted average of overlapping windows)
                chunk_start = chunk['start']
                chunk_end = chunk['end']
                
                overlapping = []
                for window in late_windows:
                    overlap_start = max(chunk_start, window['start'])
                    overlap_end = min(chunk_end, window['end'])
                    overlap_len = max(0, overlap_end - overlap_start)
                    
                    if overlap_len > 0:
                        overlapping.append({
                            'embedding': np.asarray(window['embedding'], dtype=np.float32),
                            'weight': overlap_len
                        })
                
                if overlapping:
                    total_weight = sum(w['weight'] for w in overlapping)
                    weighted_embedding = np.zeros_like(overlapping[0]['embedding'], dtype=np.float32)
                    for item in overlapping:
                        weighted_embedding += item['embedding'] * (item['weight'] / total_weight)
                    embedding = weighted_embedding.tolist()
                else:
                    # Fallback to closest window
                    chunk_mid = (chunk_start + chunk_end) // 2
                    closest = min(late_windows, 
                                key=lambda w: abs((w['start'] + w['end'])/2 - chunk_mid))
                    embedding = np.asarray(closest['embedding'], dtype=np.float32).tolist()
                
                # Create chunk document
                chunk_doc = {
                    '_key': chunk_key,
                    '_id': chunk_id,
                    'doc_id': doc_key,
                    'chunk_index': i,
                    'text': chunk['text'][:5000],  # Limit size
                    'start_char': chunk['start'],
                    'end_char': chunk['end'],
                    'section': chunk.get('section'),
                    'embedding': embedding,
                    'embedding_dim': len(embedding),
                    'processing_date': datetime.now(timezone.utc).isoformat()
                }
                
                # Upsert chunk
                self.db.collection('base_arxiv_chunks').insert(chunk_doc, overwrite=True)
            
            # Create containment edges (Paper → Chunks)
            logger.info(f"Creating containment edges for {len(chunk_ids)} chunks...")
            containment_stats = self._upsert_vertex_and_delta_edges(
                vertex_col='base_arxiv',  # Parent collection
                edge_col='edges_contains',
                from_id=f"base_arxiv/{doc_key}",
                new_doc={},  # No update to parent needed
                relation='contains',
                to_ids_new=chunk_ids,
                source='system',
                confidence=1.0
            )
            
            logger.info(f"Containment edges: {containment_stats}")
            
            return {
                'status': 'success',
                'arxiv_id': arxiv_id,
                'uid': global_uid,
                'chunk_count': len(chunks),
                'citation_count': len(high_confidence),
                'citation_edges': edge_stats,
                'containment_edges': containment_stats,
                'processing_time': time.time() - start_time,
                'ingest_run_id': ingest_run_id,
                'message': 'Successfully processed with v7.0 Phase 2 architecture'
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
    parser = argparse.ArgumentParser(description='Process PDFs with Phase 2 Architecture (v7.0)')
    parser.add_argument('arxiv_id', help='ArXiv ID to process')
    parser.add_argument('--pdf-path', help='Path to existing PDF file')
    parser.add_argument('--output-dir', default='/tmp/arxiv_pdfs', 
                       help='Directory for downloaded PDFs')
    parser.add_argument('--force', action='store_true',
                       help='Force reprocessing even if content unchanged')
    parser.add_argument('--db-host', default='localhost', help='ArangoDB host')
    parser.add_argument('--db-port', type=int, default=8529, help='ArangoDB port')
    parser.add_argument('--db-name', default='academy_store', help='Database name')
    parser.add_argument('--db-user', default='root', help='Database user')
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
    
    # Create processor
    processor = PDFCitationProcessorV7(
        output_dir=args.output_dir,
        db_config=db_config
    )
    
    # Process PDF
    result = processor.process_pdf(
        arxiv_id=args.arxiv_id,
        pdf_path=args.pdf_path,
        force=args.force
    )
    
    # Print results
    print(json.dumps(result, indent=2))
    
    return 0 if result.get('status') == 'success' else 1


if __name__ == '__main__':
    sys.exit(main())