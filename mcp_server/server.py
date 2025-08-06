"""
Main MCP Server implementation for HADES arXiv processor.

Provides async MCP interface to the production processing pipeline.
"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import logging

import torch
from mcp import Server
from mcp.types import TextContent, ToolResult

# Import production processor components
import sys
sys.path.append(str(Path(__file__).parent.parent))

from processors.arxiv.process_arxiv_production import (
    ProductionProcessor,
    ProcessingConfig,
    ProcessingStats
)

logger = logging.getLogger(__name__)


@dataclass
class ProcessingJob:
    """Track async processing jobs"""
    job_id: str
    status: str  # 'running', 'completed', 'failed'
    started_at: datetime
    completed_at: Optional[datetime] = None
    config: Optional[ProcessingConfig] = None
    stats: Optional[ProcessingStats] = None
    error: Optional[str] = None


class ArxivMCPServer:
    """
    MCP Server wrapping the HADES production processor.
    
    Provides tools for:
    - Batch processing of arXiv papers
    - Single paper processing
    - Semantic search
    - Processing monitoring
    - GPU status tracking
    """
    
    def __init__(self, name: str = "hades-arxiv"):
        """
        Initialize the MCP server.
        
        Args:
            name: Server name for MCP registration
        """
        self.server = Server(name)
        self.processor: Optional[ProductionProcessor] = None
        self.jobs: Dict[str, ProcessingJob] = {}
        self.current_job_id: Optional[str] = None
        
        # Register all tools
        self.register_tools()
        
        logger.info(f"Initialized {name} MCP server")
    
    def register_tools(self):
        """Register all MCP tools with the server."""
        
        @self.server.tool()
        async def process_arxiv_batch(
            input_file: str,
            db_name: str = "academy_store",
            collection_name: str = "base_arxiv",
            limit: Optional[int] = None,
            categories: Optional[List[str]] = None,
            resume: bool = True,
            gpu_batch_size: int = 1024
        ) -> ToolResult:
            """
            Process a batch of ArXiv papers from metadata file.
            
            Args:
                input_file: Path to JSON metadata file
                db_name: ArangoDB database name
                collection_name: Collection to store papers
                limit: Maximum papers to process
                categories: Filter by categories
                resume: Resume from checkpoint if available
                gpu_batch_size: Batch size for GPU processing
            
            Returns:
                Job ID for tracking progress
            """
            try:
                # Create processing config
                config = ProcessingConfig(
                    input_file=input_file,
                    db_name=db_name,
                    collection_name=collection_name,
                    gpu_batch_size=gpu_batch_size
                )
                
                # Generate job ID
                job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                # Create job tracking
                job = ProcessingJob(
                    job_id=job_id,
                    status='running',
                    started_at=datetime.now(),
                    config=config
                )
                self.jobs[job_id] = job
                self.current_job_id = job_id
                
                # Start processing in background
                asyncio.create_task(
                    self._run_batch_processing(job_id, config, limit, categories, resume)
                )
                
                return ToolResult(content=[TextContent(
                    text=f"Started batch processing job: {job_id}\n"
                         f"Input: {input_file}\n"
                         f"Use 'check_job_status' with job_id='{job_id}' to monitor progress."
                )])
                
            except Exception as e:
                logger.error(f"Failed to start batch processing: {e}")
                return ToolResult(content=[TextContent(
                    text=f"Error starting batch processing: {str(e)}"
                )])
        
        @self.server.tool()
        async def process_single_paper(
            arxiv_id: str,
            title: str,
            abstract: str,
            categories: List[str],
            authors: Optional[List[str]] = None,
            db_name: str = "academy_store",
            collection_name: str = "base_arxiv"
        ) -> ToolResult:
            """
            Process a single paper immediately.
            
            Args:
                arxiv_id: ArXiv ID of the paper
                title: Paper title
                abstract: Paper abstract
                categories: List of categories
                authors: List of authors
                db_name: Database to store in
                collection_name: Collection to store in
            
            Returns:
                Processing result with embedding info
            """
            try:
                # Initialize processor if needed
                if not self.processor:
                    config = ProcessingConfig(
                        db_name=db_name,
                        collection_name=collection_name
                    )
                    self.processor = ProductionProcessor(config)
                    await asyncio.get_event_loop().run_in_executor(
                        None, self.processor._init_model
                    )
                
                # Generate embedding
                embedding = await self._generate_embedding(abstract)
                
                # Prepare document
                doc = {
                    'arxiv_id': arxiv_id,
                    'title': title,
                    'abstract': abstract,
                    'categories': categories,
                    'authors': authors or [],
                    'embedding': embedding,
                    'processed_at': datetime.now().isoformat()
                }
                
                # Store in database
                success = await self._store_document(doc)
                
                if success:
                    return ToolResult(content=[TextContent(
                        text=f"Successfully processed {arxiv_id}\n"
                             f"Title: {title}\n"
                             f"Categories: {', '.join(categories)}\n"
                             f"Embedding dimension: {len(embedding)}"
                    )])
                else:
                    return ToolResult(content=[TextContent(
                        text=f"Failed to store {arxiv_id} in database"
                    )])
                    
            except Exception as e:
                logger.error(f"Failed to process single paper: {e}")
                return ToolResult(content=[TextContent(
                    text=f"Error processing paper: {str(e)}"
                )])
        
        @self.server.tool()
        async def semantic_search(
            query: str,
            limit: int = 10,
            categories: Optional[List[str]] = None,
            min_similarity: float = 0.5,
            db_name: str = "academy_store",
            collection_name: str = "base_arxiv"
        ) -> ToolResult:
            """
            Search papers by semantic similarity.
            
            Args:
                query: Search query text
                limit: Maximum results to return
                categories: Filter by categories
                min_similarity: Minimum similarity threshold
                db_name: Database to search
                collection_name: Collection to search
            
            Returns:
                Search results with similarity scores
            """
            try:
                # Initialize processor if needed
                if not self.processor:
                    config = ProcessingConfig(
                        db_name=db_name,
                        collection_name=collection_name
                    )
                    self.processor = ProductionProcessor(config)
                    await asyncio.get_event_loop().run_in_executor(
                        None, self.processor._init_model
                    )
                
                # Generate query embedding
                query_embedding = await self._generate_embedding(query)
                
                # Search in database
                results = await self._search_similar(
                    query_embedding,
                    limit,
                    categories,
                    min_similarity
                )
                
                # Format results
                if results:
                    output = f"Found {len(results)} papers matching '{query}':\n\n"
                    for i, result in enumerate(results, 1):
                        output += f"{i}. [{result['arxiv_id']}] {result['title']}\n"
                        output += f"   Similarity: {result['similarity']:.3f}\n"
                        output += f"   Categories: {', '.join(result['categories'])}\n"
                        if result.get('abstract'):
                            output += f"   Abstract: {result['abstract'][:200]}...\n"
                        output += "\n"
                else:
                    output = f"No papers found matching '{query}'"
                
                return ToolResult(content=[TextContent(text=output)])
                
            except Exception as e:
                logger.error(f"Search failed: {e}")
                return ToolResult(content=[TextContent(
                    text=f"Search error: {str(e)}"
                )])
        
        @self.server.tool()
        async def check_job_status(
            job_id: Optional[str] = None
        ) -> ToolResult:
            """
            Check status of a processing job.
            
            Args:
                job_id: Specific job ID to check (or current job if None)
            
            Returns:
                Job status and statistics
            """
            try:
                # Use current job if no ID specified
                if job_id is None:
                    job_id = self.current_job_id
                
                if not job_id:
                    return ToolResult(content=[TextContent(
                        text="No active processing jobs"
                    )])
                
                job = self.jobs.get(job_id)
                if not job:
                    return ToolResult(content=[TextContent(
                        text=f"Job {job_id} not found"
                    )])
                
                # Build status report
                output = f"Job {job_id} Status: {job.status}\n"
                output += f"Started: {job.started_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
                
                if job.completed_at:
                    duration = (job.completed_at - job.started_at).total_seconds()
                    output += f"Completed: {job.completed_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
                    output += f"Duration: {duration:.1f} seconds\n"
                
                if job.stats:
                    output += "\nProcessing Statistics:\n"
                    output += f"  Total documents: {job.stats.total_documents}\n"
                    output += f"  Processed: {job.stats.documents_processed}\n"
                    output += f"  Skipped: {job.stats.documents_skipped}\n"
                    output += f"  Failed: {job.stats.documents_failed}\n"
                    
                    if job.stats.documents_processed > 0:
                        avg_time = job.stats.total_processing_time / job.stats.documents_processed
                        output += f"  Average time per doc: {avg_time:.2f}s\n"
                
                if job.error:
                    output += f"\nError: {job.error}\n"
                
                return ToolResult(content=[TextContent(text=output)])
                
            except Exception as e:
                logger.error(f"Failed to check job status: {e}")
                return ToolResult(content=[TextContent(
                    text=f"Error checking status: {str(e)}"
                )])
        
        @self.server.tool()
        async def get_gpu_status() -> ToolResult:
            """
            Check GPU availability and status.
            
            Returns:
                GPU information including memory usage
            """
            try:
                if not torch.cuda.is_available():
                    return ToolResult(content=[TextContent(
                        text="No GPU available - running in CPU mode"
                    )])
                
                output = "GPU Status:\n"
                for i in range(torch.cuda.device_count()):
                    output += f"\nGPU {i}: {torch.cuda.get_device_name(i)}\n"
                    
                    # Memory info
                    mem_allocated = torch.cuda.memory_allocated(i) / 1e9
                    mem_reserved = torch.cuda.memory_reserved(i) / 1e9
                    mem_total = torch.cuda.get_device_properties(i).total_memory / 1e9
                    
                    output += f"  Memory: {mem_allocated:.1f}GB allocated / "
                    output += f"{mem_reserved:.1f}GB reserved / "
                    output += f"{mem_total:.1f}GB total\n"
                    output += f"  Utilization: {(mem_allocated/mem_total)*100:.1f}%\n"
                
                return ToolResult(content=[TextContent(text=output)])
                
            except Exception as e:
                logger.error(f"Failed to get GPU status: {e}")
                return ToolResult(content=[TextContent(
                    text=f"Error getting GPU status: {str(e)}"
                )])
        
        @self.server.tool()
        async def list_jobs(
            status: Optional[str] = None,
            limit: int = 10
        ) -> ToolResult:
            """
            List processing jobs.
            
            Args:
                status: Filter by status ('running', 'completed', 'failed')
                limit: Maximum jobs to return
            
            Returns:
                List of jobs with basic info
            """
            try:
                # Filter jobs by status if specified
                jobs = self.jobs.values()
                if status:
                    jobs = [j for j in jobs if j.status == status]
                
                # Sort by start time (most recent first)
                jobs = sorted(jobs, key=lambda j: j.started_at, reverse=True)[:limit]
                
                if not jobs:
                    return ToolResult(content=[TextContent(
                        text="No jobs found"
                    )])
                
                output = f"Processing Jobs (showing {len(jobs)} of {len(self.jobs)} total):\n\n"
                for job in jobs:
                    output += f"• {job.job_id}: {job.status}\n"
                    output += f"  Started: {job.started_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
                    if job.stats:
                        output += f"  Processed: {job.stats.documents_processed} documents\n"
                    output += "\n"
                
                return ToolResult(content=[TextContent(text=output)])
                
            except Exception as e:
                logger.error(f"Failed to list jobs: {e}")
                return ToolResult(content=[TextContent(
                    text=f"Error listing jobs: {str(e)}"
                )])
    
    async def _run_batch_processing(
        self,
        job_id: str,
        config: ProcessingConfig,
        limit: Optional[int],
        categories: Optional[List[str]],
        resume: bool
    ):
        """Run batch processing in background."""
        job = self.jobs[job_id]
        
        try:
            # Create processor
            self.processor = ProductionProcessor(config)
            
            # Run processing (blocking operation in thread pool)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self.processor.run,
                limit,
                categories,
                resume
            )
            
            # Update job status
            job.status = 'completed'
            job.completed_at = datetime.now()
            job.stats = self.processor.stats
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            job.status = 'failed'
            job.completed_at = datetime.now()
            job.error = str(e)
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using GPU model."""
        loop = asyncio.get_event_loop()
        
        # Run in thread pool to avoid blocking
        embedding = await loop.run_in_executor(
            None,
            lambda: self.processor.model.encode([text])[0]
        )
        
        return embedding.tolist()
    
    async def _store_document(self, doc: Dict[str, Any]) -> bool:
        """Store document in ArangoDB."""
        try:
            loop = asyncio.get_event_loop()
            
            # Store in database (run in thread pool)
            await loop.run_in_executor(
                None,
                self.processor.collection.insert,
                doc
            )
            
            return True
        except Exception as e:
            logger.error(f"Failed to store document: {e}")
            return False
    
    async def _search_similar(
        self,
        query_embedding: List[float],
        limit: int,
        categories: Optional[List[str]],
        min_similarity: float
    ) -> List[Dict[str, Any]]:
        """Search for similar documents using vector similarity."""
        
        # Build AQL query
        query = """
        FOR doc IN @@collection
            LET similarity = COSINE_SIMILARITY(doc.embedding, @query_embedding)
        """
        
        # Add filters
        filters = [f"similarity >= {min_similarity}"]
        if categories:
            filters.append("doc.categories ANY IN @categories")
        
        if filters:
            query += " FILTER " + " AND ".join(filters)
        
        query += """
            SORT similarity DESC
            LIMIT @limit
            RETURN {
                arxiv_id: doc.arxiv_id,
                title: doc.title,
                abstract: doc.abstract,
                categories: doc.categories,
                similarity: similarity
            }
        """
        
        bind_vars = {
            '@collection': self.processor.config.collection_name,
            'query_embedding': query_embedding,
            'limit': limit
        }
        
        if categories:
            bind_vars['categories'] = categories
        
        # Execute query
        loop = asyncio.get_event_loop()
        cursor = await loop.run_in_executor(
            None,
            self.processor.db.aql.execute,
            query,
            bind_vars
        )
        
        return list(cursor)
    
    async def run(self):
        """Run the MCP server."""
        logger.info("Starting HADES MCP server...")
        async with self.server:
            await self.server.serve()


# Entry point for running the server
if __name__ == "__main__":
    import logging
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    server = ArxivMCPServer()
    asyncio.run(server.run())