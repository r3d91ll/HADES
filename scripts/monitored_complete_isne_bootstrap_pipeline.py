#!/usr/bin/env python3
"""
Monitored Complete ISNE Bootstrap Pipeline

Enhanced version of the complete end-to-end ISNE bootstrap pipeline with comprehensive
monitoring integration following HADES component monitoring patterns:
docproc → chunking → embedding → graph construction → ISNE training

Features:
- AlertManager integration for pipeline alerts
- Prometheus metrics export at each stage
- Performance monitoring and bottleneck detection
- Real-time progress tracking with timestamps
- Component-level metrics collection
- Infrastructure resource monitoring
- Configurable alert thresholds

Usage:
    python scripts/monitored_complete_isne_bootstrap_pipeline.py --input-dir ./test-data --output-dir ./models/isne
"""

import sys
import json
import logging
import argparse
import psutil
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
import traceback
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import monitoring and alerting components
from src.alerts.alert_manager import AlertManager
from src.alerts import AlertLevel

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('monitored_isne_bootstrap.log')
    ]
)
logger = logging.getLogger(__name__)


class MonitoredISNEBootstrapPipeline:
    """Complete ISNE Bootstrap Pipeline with comprehensive monitoring integration."""
    
    def __init__(self, input_dir: Path, output_dir: Path, enable_alerts: bool = True):
        """Initialize the monitored bootstrap pipeline."""
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.enable_alerts = enable_alerts
        
        # Initialize alert manager
        if self.enable_alerts:
            alert_dir = self.output_dir / "alerts"
            self.alert_manager = AlertManager(
                alert_dir=str(alert_dir),
                min_level=AlertLevel.LOW
            )
        else:
            self.alert_manager = None
        
        # Pipeline monitoring state
        self.pipeline_start_time = time.time()
        self.stage_timings = {}
        self.stage_metrics = {}
        self.performance_baseline = {
            "max_memory_usage_mb": 0,
            "max_cpu_usage_percent": 0,
            "total_pipeline_time": 0,
            "bottlenecks": []
        }
        
        # Results tracking
        self.results = {
            "bootstrap_timestamp": datetime.now().isoformat(),
            "bootstrap_type": "monitored_complete_end_to_end_isne_training",
            "monitoring_enabled": enable_alerts,
            "pipeline_stages": ["docproc", "chunking", "embedding", "graph_construction", "isne_training"],
            "stages": {},
            "model_info": {},
            "monitoring": {
                "performance_metrics": {},
                "infrastructure_metrics": {},
                "alerts_generated": [],
                "stage_timings": {}
            },
            "summary": {}
        }
        
        # Find input files
        self.input_files = []
        for pattern in ["*.pdf", "*.md", "*.py", "*.yaml", "*.txt"]:
            self.input_files.extend(list(input_dir.glob(pattern)))
        
        logger.info(f"Initialized monitored ISNE bootstrap pipeline")
        logger.info(f"Found {len(self.input_files)} input files: {[f.name for f in self.input_files]}")
        logger.info(f"Monitoring enabled: {enable_alerts}")
        
        # Initial alert
        if self.alert_manager:
            self.alert_manager.alert(
                "Monitored ISNE bootstrap pipeline started",
                AlertLevel.LOW,
                "pipeline_initialization",
                {
                    "input_files": len(self.input_files),
                    "output_dir": str(output_dir),
                    "timestamp": self.timestamp
                }
            )
    
    def _start_stage_monitoring(self, stage_name: str) -> Dict[str, Any]:
        """Start monitoring for a pipeline stage."""
        stage_start = time.time()
        process = psutil.Process()
        
        initial_metrics = {
            "stage_name": stage_name,
            "start_time": stage_start,
            "start_timestamp": datetime.now().isoformat(),
            "initial_memory_mb": process.memory_info().rss / 1024 / 1024,
            "initial_cpu_percent": process.cpu_percent(),
            "system_load": psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0,
            "available_memory_mb": psutil.virtual_memory().available / 1024 / 1024
        }
        
        logger.info(f"Started monitoring stage: {stage_name}")
        logger.info(f"  Initial memory: {initial_metrics['initial_memory_mb']:.1f} MB")
        logger.info(f"  System load: {initial_metrics['system_load']:.2f}")
        
        return initial_metrics
    
    def _end_stage_monitoring(self, stage_name: str, initial_metrics: Dict[str, Any], 
                             stage_success: bool) -> Dict[str, Any]:
        """End monitoring for a pipeline stage and collect final metrics."""
        stage_end = time.time()
        process = psutil.Process()
        
        final_memory_mb = process.memory_info().rss / 1024 / 1024
        final_cpu_percent = process.cpu_percent()
        stage_duration = stage_end - initial_metrics["start_time"]
        
        # Calculate resource usage
        memory_delta = final_memory_mb - initial_metrics["initial_memory_mb"]
        max_memory_usage = max(final_memory_mb, initial_metrics["initial_memory_mb"])
        
        # Update performance baseline
        self.performance_baseline["max_memory_usage_mb"] = max(
            self.performance_baseline["max_memory_usage_mb"],
            max_memory_usage
        )
        self.performance_baseline["max_cpu_usage_percent"] = max(
            self.performance_baseline["max_cpu_usage_percent"],
            final_cpu_percent
        )
        
        monitoring_result = {
            "stage_name": stage_name,
            "duration_seconds": stage_duration,
            "success": stage_success,
            "resource_usage": {
                "initial_memory_mb": initial_metrics["initial_memory_mb"],
                "final_memory_mb": final_memory_mb,
                "memory_delta_mb": memory_delta,
                "max_memory_mb": max_memory_usage,
                "initial_cpu_percent": initial_metrics["initial_cpu_percent"],
                "final_cpu_percent": final_cpu_percent,
                "system_load": initial_metrics["system_load"],
                "available_memory_mb": initial_metrics["available_memory_mb"]
            },
            "timestamps": {
                "start": initial_metrics["start_timestamp"],
                "end": datetime.now().isoformat()
            }
        }
        
        # Store stage timing
        self.stage_timings[stage_name] = stage_duration
        self.stage_metrics[stage_name] = monitoring_result
        
        # Generate alerts based on thresholds
        if self.alert_manager:
            self._check_performance_thresholds(stage_name, monitoring_result)
        
        logger.info(f"Completed monitoring stage: {stage_name}")
        logger.info(f"  Duration: {stage_duration:.2f} seconds")
        logger.info(f"  Memory usage: {memory_delta:+.1f} MB (final: {final_memory_mb:.1f} MB)")
        logger.info(f"  Success: {stage_success}")
        
        return monitoring_result
    
    def _check_performance_thresholds(self, stage_name: str, metrics: Dict[str, Any]) -> None:
        """Check performance thresholds and generate alerts if needed."""
        if not self.alert_manager:
            return
        
        duration = metrics["duration_seconds"]
        memory_mb = metrics["resource_usage"]["max_memory_mb"]
        memory_delta = metrics["resource_usage"]["memory_delta_mb"]
        
        # Performance thresholds
        stage_time_thresholds = {
            "document_processing": 30,  # 30 seconds
            "chunking": 60,             # 1 minute
            "embedding": 300,           # 5 minutes
            "graph_construction": 120,  # 2 minutes
            "isne_training": 1200       # 20 minutes
        }
        
        memory_growth_threshold = 500  # 500 MB growth
        absolute_memory_threshold = 4000  # 4 GB absolute
        
        # Check duration threshold
        threshold = stage_time_thresholds.get(stage_name, 60)
        if duration > threshold:
            self.alert_manager.alert(
                f"Stage {stage_name} exceeded time threshold ({duration:.1f}s > {threshold}s)",
                AlertLevel.MEDIUM,
                f"stage_performance_{stage_name}",
                {
                    "stage": stage_name,
                    "duration_seconds": duration,
                    "threshold_seconds": threshold,
                    "performance_impact": "high_latency"
                }
            )
        
        # Check memory growth threshold
        if memory_delta > memory_growth_threshold:
            self.alert_manager.alert(
                f"Stage {stage_name} high memory growth ({memory_delta:.1f} MB)",
                AlertLevel.MEDIUM,
                f"stage_memory_{stage_name}",
                {
                    "stage": stage_name,
                    "memory_growth_mb": memory_delta,
                    "threshold_mb": memory_growth_threshold,
                    "current_memory_mb": memory_mb
                }
            )
        
        # Check absolute memory threshold
        if memory_mb > absolute_memory_threshold:
            self.alert_manager.alert(
                f"Stage {stage_name} high absolute memory usage ({memory_mb:.1f} MB)",
                AlertLevel.HIGH,
                f"stage_memory_absolute_{stage_name}",
                {
                    "stage": stage_name,
                    "memory_usage_mb": memory_mb,
                    "threshold_mb": absolute_memory_threshold,
                    "risk": "memory_exhaustion"
                }
            )
    
    def _export_stage_prometheus_metrics(self, stage_name: str, stage_data: Dict[str, Any]) -> str:
        """Export stage metrics in Prometheus format."""
        lines = [
            f"# HELP hades_isne_bootstrap_stage_duration_seconds Duration of pipeline stage",
            f"# TYPE hades_isne_bootstrap_stage_duration_seconds gauge",
            f'hades_isne_bootstrap_stage_duration_seconds{{stage="{stage_name}",pipeline="isne_bootstrap"}} {stage_data.get("duration_seconds", 0):.6f}',
            "",
            f"# HELP hades_isne_bootstrap_stage_memory_usage_bytes Memory usage during stage",
            f"# TYPE hades_isne_bootstrap_stage_memory_usage_bytes gauge",
            f'hades_isne_bootstrap_stage_memory_usage_bytes{{stage="{stage_name}",pipeline="isne_bootstrap",type="max"}} {stage_data.get("resource_usage", {}).get("max_memory_mb", 0) * 1024 * 1024:.0f}',
            "",
            f"# HELP hades_isne_bootstrap_stage_success Stage completion status",
            f"# TYPE hades_isne_bootstrap_stage_success gauge",
            f'hades_isne_bootstrap_stage_success{{stage="{stage_name}",pipeline="isne_bootstrap"}} {1 if stage_data.get("success", False) else 0}',
            ""
        ]
        
        # Add stage-specific metrics
        stats = stage_data.get("stats", {})
        if stage_name == "document_processing":
            lines.extend([
                f"# HELP hades_isne_bootstrap_documents_processed_total Documents processed in docproc stage",
                f"# TYPE hades_isne_bootstrap_documents_processed_total counter",
                f'hades_isne_bootstrap_documents_processed_total{{pipeline="isne_bootstrap"}} {stats.get("documents_generated", 0)}',
                ""
            ])
        elif stage_name == "chunking":
            lines.extend([
                f"# HELP hades_isne_bootstrap_chunks_created_total Chunks created in chunking stage",
                f"# TYPE hades_isne_bootstrap_chunks_created_total counter",
                f'hades_isne_bootstrap_chunks_created_total{{pipeline="isne_bootstrap"}} {stats.get("output_chunks", 0)}',
                ""
            ])
        elif stage_name == "embedding":
            lines.extend([
                f"# HELP hades_isne_bootstrap_embeddings_generated_total Embeddings generated",
                f"# TYPE hades_isne_bootstrap_embeddings_generated_total counter",
                f'hades_isne_bootstrap_embeddings_generated_total{{pipeline="isne_bootstrap"}} {stats.get("valid_embeddings", 0)}',
                "",
                f"# HELP hades_isne_bootstrap_embedding_dimension Embedding dimension",
                f"# TYPE hades_isne_bootstrap_embedding_dimension gauge",
                f'hades_isne_bootstrap_embedding_dimension{{pipeline="isne_bootstrap"}} {stats.get("embedding_dimension", 0)}',
                ""
            ])
        elif stage_name == "graph_construction":
            lines.extend([
                f"# HELP hades_isne_bootstrap_graph_nodes_total Graph nodes created",
                f"# TYPE hades_isne_bootstrap_graph_nodes_total counter",
                f'hades_isne_bootstrap_graph_nodes_total{{pipeline="isne_bootstrap"}} {stats.get("num_nodes", 0)}',
                "",
                f"# HELP hades_isne_bootstrap_graph_edges_total Graph edges created",
                f"# TYPE hades_isne_bootstrap_graph_edges_total counter",
                f'hades_isne_bootstrap_graph_edges_total{{pipeline="isne_bootstrap"}} {stats.get("num_edges", 0)}',
                ""
            ])
        elif stage_name == "isne_training":
            lines.extend([
                f"# HELP hades_isne_bootstrap_training_epochs_total Training epochs completed",
                f"# TYPE hades_isne_bootstrap_training_epochs_total counter",
                f'hades_isne_bootstrap_training_epochs_total{{pipeline="isne_bootstrap"}} {stats.get("training_epochs", 0)}',
                "",
                f"# HELP hades_isne_bootstrap_model_parameters_total Model parameters",
                f"# TYPE hades_isne_bootstrap_model_parameters_total gauge",
                f'hades_isne_bootstrap_model_parameters_total{{pipeline="isne_bootstrap"}} {stats.get("model_architecture", {}).get("total_parameters", 0)}',
                ""
            ])
        
        return "\\n".join(lines)
    
    def save_stage_results(self, stage_name: str, stage_data: Dict[str, Any]) -> None:
        """Save results for a bootstrap stage with monitoring data."""
        self.results["stages"][stage_name] = stage_data
        
        # Add monitoring data if available
        if stage_name in self.stage_metrics:
            stage_data["monitoring"] = self.stage_metrics[stage_name]
        
        # Export Prometheus metrics for this stage
        prometheus_metrics = self._export_stage_prometheus_metrics(stage_name, stage_data)
        metrics_file = self.output_dir / f"{self.timestamp}_{stage_name}_metrics.txt"
        with open(metrics_file, 'w') as f:
            f.write(prometheus_metrics)
        
        # Save intermediate results
        results_file = self.output_dir / f"{self.timestamp}_monitored_bootstrap_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info(f"Saved stage results and metrics for {stage_name}")
    
    def stage_1_document_processing(self) -> bool:
        """Stage 1: Process documents using the document processing component."""
        logger.info("\\n=== Monitored Bootstrap Stage 1: Document Processing ===")
        
        # Start stage monitoring
        monitor_data = self._start_stage_monitoring("document_processing")
        stage_success = False
        
        try:
            # Import the document processing component system
            from src.components.docproc.factory import create_docproc_component
            from src.types.components.contracts import DocumentProcessingInput
            
            # Use core processor for reliable processing
            processor = create_docproc_component("core")
            
            documents = []
            processing_stats = {
                "files_processed": 0,
                "documents_generated": 0,
                "total_content_chars": 0,
                "file_details": []
            }
            
            for input_file in self.input_files:
                try:
                    logger.info(f"Processing {input_file.name}...")
                    
                    doc_input = DocumentProcessingInput(
                        file_path=str(input_file),
                        processing_options={
                            "extract_metadata": True,
                            "extract_sections": True,
                            "extract_entities": True
                        },
                        metadata={"bootstrap_file": input_file.name, "stage": "docproc"}
                    )
                    
                    output = processor.process(doc_input)
                    
                    file_docs = len(output.documents)
                    file_chars = sum(len(doc.content) for doc in output.documents)
                    
                    documents.extend(output.documents)
                    processing_stats["files_processed"] += 1
                    processing_stats["documents_generated"] += file_docs
                    processing_stats["total_content_chars"] += file_chars
                    processing_stats["file_details"].append({
                        "filename": input_file.name,
                        "documents": file_docs,
                        "characters": file_chars
                    })
                    
                    logger.info(f"  ✓ {input_file.name}: {file_docs} documents, {file_chars} characters")
                    
                except Exception as e:
                    logger.error(f"  ✗ Failed to process {input_file.name}: {e}")
                    if self.alert_manager:
                        self.alert_manager.alert(
                            f"Document processing failed for {input_file.name}: {e}",
                            AlertLevel.MEDIUM,
                            "docproc_file_error",
                            {"filename": input_file.name, "error": str(e)}
                        )
                    continue
            
            if not documents:
                error_msg = "No documents were successfully processed"
                logger.error(error_msg)
                if self.alert_manager:
                    self.alert_manager.alert(
                        error_msg,
                        AlertLevel.HIGH,
                        "docproc_complete_failure"
                    )
                return False
            
            stage_results = {
                "stage_name": "document_processing",
                "pipeline_position": 1,
                "stats": processing_stats,
                "documents": documents  # Store full documents for next stage
            }
            
            stage_success = True
            
            # End stage monitoring
            monitoring_result = self._end_stage_monitoring("document_processing", monitor_data, stage_success)
            stage_results.update(monitoring_result)
            
            self.save_stage_results("document_processing", stage_results)
            logger.info(f"Stage 1 complete: {len(documents)} documents from {processing_stats['files_processed']} files")
            return True
            
        except Exception as e:
            logger.error(f"Document processing stage failed: {e}")
            traceback.print_exc()
            
            if self.alert_manager:
                self.alert_manager.alert(
                    f"Document processing stage critical failure: {e}",
                    AlertLevel.CRITICAL,
                    "docproc_stage_failure",
                    {"error": str(e), "traceback": traceback.format_exc()}
                )
            
            # End stage monitoring with failure
            self._end_stage_monitoring("document_processing", monitor_data, stage_success)
            return False
    
    def stage_2_chunking(self) -> bool:
        """Stage 2: Chunk documents using the chunking component."""
        logger.info("\\n=== Monitored Bootstrap Stage 2: Document Chunking ===")
        
        # Start stage monitoring
        monitor_data = self._start_stage_monitoring("chunking")
        stage_success = False
        
        try:
            # Get documents from previous stage
            doc_stage = self.results["stages"].get("document_processing")
            if not doc_stage:
                logger.error("Document processing stage must be completed first")
                return False
            
            documents = doc_stage["documents"]
            if not documents:
                logger.error("No documents available for chunking")
                return False
            
            # Import chunking components
            from src.components.chunking.factory import create_chunking_component
            from src.types.components.contracts import ChunkingInput, DocumentChunk
            
            # Use core chunker for reliable chunking
            chunker = create_chunking_component("core")
            
            all_chunks = []
            chunking_stats = {
                "input_documents": len(documents),
                "output_chunks": 0,
                "total_characters": 0,
                "document_details": []
            }
            
            for doc in documents:
                try:
                    logger.info(f"Chunking document {doc.id}...")
                    
                    # Create chunking input
                    chunking_input = ChunkingInput(
                        text=doc.content,
                        document_id=doc.id,
                        chunking_strategy="semantic",  # Best for ISNE training
                        chunk_size=512,  # Optimal for embedding models
                        chunk_overlap=50,
                        processing_options={
                            "preserve_structure": True,
                            "extract_metadata": True
                        },
                        metadata={
                            "bootstrap_stage": "chunking", 
                            "document_id": doc.id,
                            "source_file": doc.metadata.get("source_file", "unknown")
                        }
                    )
                    
                    # Process chunks for this document
                    doc_output = chunker.chunk(chunking_input)
                    
                    # Convert TextChunks to DocumentChunks for consistency
                    doc_chunks = []
                    for i, text_chunk in enumerate(doc_output.chunks):
                        doc_chunk = DocumentChunk(
                            id=f"{doc.id}_chunk_{i}",
                            content=text_chunk.text,
                            document_id=doc.id,
                            chunk_index=i,
                            chunk_size=len(text_chunk.text),
                            metadata={
                                **text_chunk.metadata,
                                "source_document": doc.id,
                                "chunk_strategy": "semantic"
                            }
                        )
                        doc_chunks.append(doc_chunk)
                        all_chunks.append(doc_chunk)
                    
                    doc_chars = sum(len(chunk.content) for chunk in doc_chunks)
                    chunking_stats["output_chunks"] += len(doc_chunks)
                    chunking_stats["total_characters"] += doc_chars
                    chunking_stats["document_details"].append({
                        "document_id": doc.id,
                        "chunks_created": len(doc_chunks),
                        "characters": doc_chars
                    })
                    
                    logger.info(f"  ✓ {doc.id}: {len(doc_chunks)} chunks, {doc_chars} characters")
                    
                except Exception as e:
                    logger.error(f"  ✗ Failed to chunk {doc.id}: {e}")
                    if self.alert_manager:
                        self.alert_manager.alert(
                            f"Chunking failed for document {doc.id}: {e}",
                            AlertLevel.MEDIUM,
                            "chunking_document_error",
                            {"document_id": doc.id, "error": str(e)}
                        )
                    continue
            
            if not all_chunks:
                error_msg = "No chunks were successfully created"
                logger.error(error_msg)
                if self.alert_manager:
                    self.alert_manager.alert(
                        error_msg,
                        AlertLevel.HIGH,
                        "chunking_complete_failure"
                    )
                return False
            
            stage_results = {
                "stage_name": "chunking",
                "pipeline_position": 2,
                "stats": chunking_stats,
                "chunks": all_chunks  # Store for next stage
            }
            
            stage_success = True
            
            # End stage monitoring
            monitoring_result = self._end_stage_monitoring("chunking", monitor_data, stage_success)
            stage_results.update(monitoring_result)
            
            self.save_stage_results("chunking", stage_results)
            logger.info(f"Stage 2 complete: {len(all_chunks)} chunks from {len(documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Chunking stage failed: {e}")
            traceback.print_exc()
            
            if self.alert_manager:
                self.alert_manager.alert(
                    f"Chunking stage critical failure: {e}",
                    AlertLevel.CRITICAL,
                    "chunking_stage_failure",
                    {"error": str(e), "traceback": traceback.format_exc()}
                )
            
            # End stage monitoring with failure
            self._end_stage_monitoring("chunking", monitor_data, stage_success)
            return False
    
    def stage_3_embedding(self) -> bool:
        """Stage 3: Generate embeddings using the embedding component."""
        logger.info("\\n=== Monitored Bootstrap Stage 3: Embedding Generation ===")
        
        # Start stage monitoring
        monitor_data = self._start_stage_monitoring("embedding")
        stage_success = False
        
        try:
            # Get chunks from previous stage
            chunk_stage = self.results["stages"].get("chunking")
            if not chunk_stage:
                logger.error("Chunking stage must be completed first")
                return False
            
            chunks = chunk_stage["chunks"]
            if not chunks:
                logger.error("No chunks available for embedding")
                return False
            
            # Import embedding components
            from src.components.embedding.factory import create_embedding_component
            from src.types.components.contracts import EmbeddingInput
            
            # Use CPU embedder for reliable processing
            embedder = create_embedding_component("cpu")
            
            logger.info(f"Generating embeddings for {len(chunks)} chunks...")
            
            # Create embedding input
            embedding_input = EmbeddingInput(
                chunks=chunks,
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                embedding_options={
                    "normalize": True,
                    "batch_size": 32
                },
                batch_size=32,
                metadata={"bootstrap_stage": "embedding", "pipeline": "monitored_isne"}
            )
            
            # Generate embeddings
            embedding_start = time.time()
            output = embedder.embed(embedding_input)
            embedding_duration = time.time() - embedding_start
            
            # Validate embeddings quality
            valid_embeddings = [emb for emb in output.embeddings if len(emb.embedding) > 0]
            
            # Check for embedding quality issues
            if len(valid_embeddings) < len(chunks) * 0.9:  # Less than 90% success rate
                if self.alert_manager:
                    self.alert_manager.alert(
                        f"Low embedding success rate: {len(valid_embeddings)}/{len(chunks)} ({len(valid_embeddings)/len(chunks)*100:.1f}%)",
                        AlertLevel.MEDIUM,
                        "embedding_quality_warning",
                        {
                            "valid_embeddings": len(valid_embeddings),
                            "total_chunks": len(chunks),
                            "success_rate": len(valid_embeddings)/len(chunks)*100
                        }
                    )
            
            embedding_stats = {
                "input_chunks": len(chunks),
                "output_embeddings": len(output.embeddings),
                "valid_embeddings": len(valid_embeddings),
                "embedding_dimension": valid_embeddings[0].embedding_dimension if valid_embeddings else 0,
                "processing_time": output.metadata.processing_time,
                "embedding_duration_seconds": embedding_duration,
                "embeddings_per_second": len(valid_embeddings) / max(embedding_duration, 0.001),
                "model_info": {
                    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                    "embedding_dimension": valid_embeddings[0].embedding_dimension if valid_embeddings else 0
                }
            }
            
            stage_results = {
                "stage_name": "embedding",
                "pipeline_position": 3,
                "stats": embedding_stats,
                "embeddings": output.embeddings  # Store for graph construction
            }
            
            stage_success = True
            
            # End stage monitoring
            monitoring_result = self._end_stage_monitoring("embedding", monitor_data, stage_success)
            stage_results.update(monitoring_result)
            
            self.save_stage_results("embedding", stage_results)
            logger.info(f"Stage 3 complete: {len(valid_embeddings)} embeddings generated")
            logger.info(f"  Throughput: {embedding_stats['embeddings_per_second']:.1f} embeddings/second")
            return True
            
        except Exception as e:
            logger.error(f"Embedding stage failed: {e}")
            traceback.print_exc()
            
            if self.alert_manager:
                self.alert_manager.alert(
                    f"Embedding stage critical failure: {e}",
                    AlertLevel.CRITICAL,
                    "embedding_stage_failure",
                    {"error": str(e), "traceback": traceback.format_exc()}
                )
            
            # End stage monitoring with failure
            self._end_stage_monitoring("embedding", monitor_data, stage_success)
            return False
    
    def stage_4_graph_construction(self) -> bool:
        """Stage 4: Construct graph from embeddings."""
        logger.info("\\n=== Monitored Bootstrap Stage 4: Graph Construction ===")
        
        # Start stage monitoring
        monitor_data = self._start_stage_monitoring("graph_construction")
        stage_success = False
        
        try:
            # Get embeddings from previous stage
            embedding_stage = self.results["stages"].get("embedding")
            if not embedding_stage:
                logger.error("Embedding stage must be completed first")
                return False
            
            embeddings = embedding_stage["embeddings"]
            if not embeddings:
                logger.error("No embeddings available for graph construction")
                return False
            
            logger.info(f"Constructing graph from {len(embeddings)} embeddings...")
            
            # Create node features and metadata
            graph_start = time.time()
            
            node_features = []
            node_metadata = []
            chunk_stage = self.results["stages"]["chunking"]
            chunks = chunk_stage["chunks"]
            
            # Build mapping from chunk_id to chunk data
            chunk_map = {chunk.id: chunk for chunk in chunks}
            
            for i, emb_data in enumerate(embeddings):
                node_features.append(emb_data.embedding)
                
                # Get corresponding chunk data
                chunk_id = emb_data.chunk_id
                chunk_data = chunk_map.get(chunk_id, {})
                
                node_metadata.append({
                    "node_id": i,
                    "chunk_id": chunk_id,
                    "document_id": getattr(chunk_data, 'document_id', f'unknown_doc_{i}'),
                    "text": getattr(chunk_data, 'content', '')[:200],  # Truncate for storage
                    "embedding_model": "all-MiniLM-L6-v2",
                    "metadata": getattr(chunk_data, 'metadata', {})
                })
            
            # Convert to PyTorch tensors
            node_features_tensor = torch.tensor(node_features, dtype=torch.float)
            num_nodes = len(node_features)
            
            logger.info(f"Created {num_nodes} nodes with {node_features_tensor.size(1)}-dimensional features")
            
            # Create graph edges
            edge_construction_start = time.time()
            edge_index_src = []
            edge_index_dst = []
            
            # 1. Sequential edges (document flow)
            doc_chunks = {}
            for i, meta in enumerate(node_metadata):
                doc_id = meta["document_id"]
                if doc_id not in doc_chunks:
                    doc_chunks[doc_id] = []
                doc_chunks[doc_id].append(i)
            
            # Add sequential edges within each document
            sequential_edges = 0
            for doc_id, chunk_indices in doc_chunks.items():
                chunk_indices.sort()  # Ensure proper order
                for i in range(len(chunk_indices) - 1):
                    src_idx = chunk_indices[i]
                    dst_idx = chunk_indices[i + 1]
                    # Add bidirectional edges
                    edge_index_src.extend([src_idx, dst_idx])
                    edge_index_dst.extend([dst_idx, src_idx])
                    sequential_edges += 2
            
            # 2. Similarity-based edges (using embedding similarity)
            import random
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np
            
            # Sample a subset for similarity calculation to avoid O(n²) complexity
            max_similarity_nodes = min(500, num_nodes)
            if num_nodes > max_similarity_nodes:
                similarity_indices = random.sample(range(num_nodes), max_similarity_nodes)
            else:
                similarity_indices = list(range(num_nodes))
            
            # Calculate similarities for sampled nodes
            similarity_start = time.time()
            sampled_features = node_features_tensor[similarity_indices].numpy()
            similarities = cosine_similarity(sampled_features)
            similarity_duration = time.time() - similarity_start
            
            # Add high-similarity edges
            similarity_threshold = 0.8
            similarity_edges = 0
            for i in range(len(similarity_indices)):
                for j in range(i + 1, len(similarity_indices)):
                    if similarities[i, j] > similarity_threshold:
                        actual_i = similarity_indices[i]
                        actual_j = similarity_indices[j]
                        edge_index_src.extend([actual_i, actual_j])
                        edge_index_dst.extend([actual_j, actual_i])
                        similarity_edges += 2
            
            # 3. Random edges for connectivity
            num_random_edges = min(200, num_nodes // 5)
            random_edges = 0
            for _ in range(num_random_edges):
                src = random.randint(0, num_nodes - 1)
                dst = random.randint(0, num_nodes - 1)
                if src != dst:
                    edge_index_src.append(src)
                    edge_index_dst.append(dst)
                    random_edges += 1
            
            # Create final edge index tensor
            edge_index = torch.tensor([edge_index_src, edge_index_dst], dtype=torch.long)
            edge_construction_duration = time.time() - edge_construction_start
            graph_duration = time.time() - graph_start
            
            graph_stats = {
                "num_nodes": num_nodes,
                "num_edges": len(edge_index_src),
                "embedding_dimension": node_features_tensor.size(1),
                "documents_represented": len(doc_chunks),
                "edge_types": {
                    "sequential": sequential_edges,
                    "similarity": similarity_edges,
                    "random": random_edges
                },
                "timing": {
                    "total_graph_construction_seconds": graph_duration,
                    "edge_construction_seconds": edge_construction_duration,
                    "similarity_calculation_seconds": similarity_duration
                },
                "performance": {
                    "nodes_per_second": num_nodes / max(graph_duration, 0.001),
                    "edges_per_second": len(edge_index_src) / max(edge_construction_duration, 0.001)
                }
            }
            
            # Check for graph quality issues
            if len(edge_index_src) < num_nodes:  # Very sparse graph
                if self.alert_manager:
                    self.alert_manager.alert(
                        f"Sparse graph detected: {len(edge_index_src)} edges for {num_nodes} nodes",
                        AlertLevel.MEDIUM,
                        "graph_quality_warning",
                        {
                            "num_nodes": num_nodes,
                            "num_edges": len(edge_index_src),
                            "edge_to_node_ratio": len(edge_index_src) / num_nodes
                        }
                    )
            
            stage_results = {
                "stage_name": "graph_construction",
                "pipeline_position": 4,
                "stats": graph_stats,
                "node_features": node_features_tensor,
                "edge_index": edge_index,
                "node_metadata": node_metadata
            }
            
            stage_success = True
            
            # End stage monitoring
            monitoring_result = self._end_stage_monitoring("graph_construction", monitor_data, stage_success)
            stage_results.update(monitoring_result)
            
            self.save_stage_results("graph_construction", stage_results)
            logger.info(f"Stage 4 complete: Graph with {num_nodes} nodes, {len(edge_index_src)} edges")
            logger.info(f"  Construction rate: {graph_stats['performance']['nodes_per_second']:.1f} nodes/second")
            return True
            
        except Exception as e:
            logger.error(f"Graph construction stage failed: {e}")
            traceback.print_exc()
            
            if self.alert_manager:
                self.alert_manager.alert(
                    f"Graph construction stage critical failure: {e}",
                    AlertLevel.CRITICAL,
                    "graph_construction_stage_failure",
                    {"error": str(e), "traceback": traceback.format_exc()}
                )
            
            # End stage monitoring with failure
            self._end_stage_monitoring("graph_construction", monitor_data, stage_success)
            return False
    
    def stage_5_isne_training(self) -> bool:
        """Stage 5: Train ISNE model on the constructed graph."""
        logger.info("\\n=== Monitored Bootstrap Stage 5: ISNE Model Training ===")
        
        # Start stage monitoring
        monitor_data = self._start_stage_monitoring("isne_training")
        stage_success = False
        
        try:
            # Get graph data from previous stage
            graph_stage = self.results["stages"].get("graph_construction")
            if not graph_stage:
                logger.error("Graph construction stage must be completed first")
                return False
            
            node_features = graph_stage["node_features"]
            edge_index = graph_stage["edge_index"]
            
            if node_features is None or edge_index is None:
                logger.error("No graph data available for ISNE training")
                return False
            
            # Import ISNE trainer
            from src.isne.training.trainer import ISNETrainer
            
            # Get model configuration
            embedding_dim = node_features.size(1)
            hidden_dim = 256
            output_dim = 128
            
            logger.info(f"Training ISNE model: {embedding_dim} → {hidden_dim} → {output_dim}")
            
            # Create trainer
            trainer_start = time.time()
            trainer = ISNETrainer(
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                num_layers=3,  # Slightly deeper for better representation
                num_heads=8,
                dropout=0.1,
                learning_rate=0.001,
                weight_decay=1e-4,
                lambda_feat=1.0,      # Feature preservation
                lambda_struct=1.0,    # Structural preservation  
                lambda_contrast=0.5,  # Contrastive learning
                device="cpu"
            )
            
            # Prepare the model
            trainer.prepare_model()
            trainer_init_duration = time.time() - trainer_start
            
            # Train the model
            logger.info("Starting ISNE model training...")
            training_start = time.time()
            training_results = trainer.train(
                features=node_features,
                edge_index=edge_index,
                epochs=50,  # More epochs for better convergence
                batch_size=64,
                num_hops=2,
                neighbor_size=15,
                eval_interval=10,
                early_stopping_patience=15,
                verbose=True
            )
            training_duration = time.time() - training_start
            
            # Save the trained model
            model_path = self.output_dir / f"monitored_isne_model_{self.timestamp}.pth"
            trainer.save_model(str(model_path))
            
            # Calculate model statistics
            total_params = sum(p.numel() for p in trainer.model.parameters()) if trainer.model else 0
            trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad) if trainer.model else 0
            
            # Check training quality
            final_total_loss = training_results.get("total_loss", [])[-1] if training_results.get("total_loss") else float('inf')
            
            if final_total_loss > 10.0:  # High loss threshold
                if self.alert_manager:
                    self.alert_manager.alert(
                        f"High final training loss: {final_total_loss:.4f}",
                        AlertLevel.MEDIUM,
                        "training_quality_warning",
                        {
                            "final_loss": final_total_loss,
                            "training_epochs": training_results.get("epochs", 0),
                            "model_path": str(model_path)
                        }
                    )
            
            training_stats = {
                "model_path": str(model_path),
                "training_epochs": training_results.get("epochs", 0),
                "final_losses": {
                    "total": training_results.get("total_loss", [])[-1] if training_results.get("total_loss") else 0.0,
                    "feature": training_results.get("feature_loss", [])[-1] if training_results.get("feature_loss") else 0.0,
                    "structural": training_results.get("structural_loss", [])[-1] if training_results.get("structural_loss") else 0.0,
                    "contrastive": training_results.get("contrastive_loss", [])[-1] if training_results.get("contrastive_loss") else 0.0
                },
                "model_architecture": {
                    "input_dim": embedding_dim,
                    "hidden_dim": hidden_dim,
                    "output_dim": output_dim,
                    "num_layers": 3,
                    "num_heads": 8,
                    "total_parameters": total_params,
                    "trainable_parameters": trainable_params
                },
                "training_config": {
                    "epochs": 50,
                    "batch_size": 64,
                    "learning_rate": 0.001,
                    "weight_decay": 1e-4,
                    "loss_weights": {
                        "feature": 1.0,
                        "structural": 1.0,
                        "contrastive": 0.5
                    }
                },
                "timing": {
                    "trainer_initialization_seconds": trainer_init_duration,
                    "training_duration_seconds": training_duration,
                    "total_training_time_seconds": trainer_init_duration + training_duration
                },
                "performance": {
                    "epochs_per_second": training_results.get("epochs", 0) / max(training_duration, 0.001),
                    "parameters_per_second": total_params / max(training_duration, 0.001)
                }
            }
            
            stage_results = {
                "stage_name": "isne_training",
                "pipeline_position": 5,
                "stats": training_stats
            }
            
            stage_success = True
            
            # End stage monitoring
            monitoring_result = self._end_stage_monitoring("isne_training", monitor_data, stage_success)
            stage_results.update(monitoring_result)
            
            self.save_stage_results("isne_training", stage_results)
            self.results["model_info"] = training_stats
            
            logger.info(f"Stage 5 complete: ISNE model trained and saved to {model_path}")
            logger.info(f"Model: {total_params:,} parameters, Final loss: {training_stats['final_losses']['total']:.4f}")
            logger.info(f"Training rate: {training_stats['performance']['epochs_per_second']:.2f} epochs/second")
            return True
            
        except Exception as e:
            logger.error(f"ISNE training stage failed: {e}")
            traceback.print_exc()
            
            if self.alert_manager:
                self.alert_manager.alert(
                    f"ISNE training stage critical failure: {e}",
                    AlertLevel.CRITICAL,
                    "isne_training_stage_failure",
                    {"error": str(e), "traceback": traceback.format_exc()}
                )
            
            # End stage monitoring with failure
            self._end_stage_monitoring("isne_training", monitor_data, stage_success)
            return False
    
    def generate_comprehensive_summary(self) -> Dict[str, Any]:
        """Generate comprehensive pipeline summary with monitoring data."""
        stages_completed = len(self.results["stages"])
        stages_successful = len([s for s in self.results["stages"].values() 
                               if s.get("stage_name") and "error" not in s and s.get("success", True)])
        
        # Calculate pipeline metrics
        pipeline_metrics = {}
        if "document_processing" in self.results["stages"]:
            doc_stage = self.results["stages"]["document_processing"]
            pipeline_metrics["documents"] = doc_stage["stats"]["documents_generated"]
            pipeline_metrics["input_files"] = doc_stage["stats"]["files_processed"]
        
        if "chunking" in self.results["stages"]:
            chunk_stage = self.results["stages"]["chunking"]
            pipeline_metrics["chunks"] = chunk_stage["stats"]["output_chunks"]
        
        if "embedding" in self.results["stages"]:
            emb_stage = self.results["stages"]["embedding"]
            pipeline_metrics["embeddings"] = emb_stage["stats"]["valid_embeddings"]
            pipeline_metrics["embedding_dimension"] = emb_stage["stats"]["embedding_dimension"]
            pipeline_metrics["embedding_throughput"] = emb_stage["stats"].get("embeddings_per_second", 0)
        
        if "graph_construction" in self.results["stages"]:
            graph_stage = self.results["stages"]["graph_construction"]
            pipeline_metrics["graph_nodes"] = graph_stage["stats"]["num_nodes"]
            pipeline_metrics["graph_edges"] = graph_stage["stats"]["num_edges"]
            pipeline_metrics["graph_throughput"] = graph_stage["stats"]["performance"]["nodes_per_second"]
        
        if "isne_training" in self.results["stages"]:
            train_stage = self.results["stages"]["isne_training"]
            pipeline_metrics["model_parameters"] = train_stage["stats"]["model_architecture"]["total_parameters"]
            pipeline_metrics["training_epochs"] = train_stage["stats"]["training_epochs"]
            pipeline_metrics["training_throughput"] = train_stage["stats"]["performance"]["epochs_per_second"]
        
        # Monitoring summary
        total_pipeline_time = time.time() - self.pipeline_start_time
        self.performance_baseline["total_pipeline_time"] = total_pipeline_time
        
        # Identify bottlenecks
        bottlenecks = []
        if self.stage_timings:
            max_time = max(self.stage_timings.values())
            for stage, duration in self.stage_timings.items():
                if duration > max_time * 0.7:  # Stages taking > 70% of max time
                    bottlenecks.append({
                        "stage": stage,
                        "duration_seconds": duration,
                        "percentage_of_total": (duration / total_pipeline_time) * 100
                    })
        
        self.performance_baseline["bottlenecks"] = bottlenecks
        
        # Alert summary
        alert_summary = {"total_alerts": 0, "by_level": {}, "by_source": {}}
        if self.alert_manager:
            alerts = self.alert_manager.get_alerts(limit=1000)  # Get all alerts
            alert_summary["total_alerts"] = len(alerts)
            
            for alert in alerts:
                level_name = alert.level.name
                alert_summary["by_level"][level_name] = alert_summary["by_level"].get(level_name, 0) + 1
                alert_summary["by_source"][alert.source] = alert_summary["by_source"].get(alert.source, 0) + 1
        
        summary = {
            "total_stages_planned": 5,
            "stages_completed": stages_completed,
            "stages_successful": stages_successful,
            "overall_success": stages_successful == 5,
            "completion_rate": stages_completed / 5,
            "success_rate": stages_successful / stages_completed if stages_completed > 0 else 0,
            "pipeline_metrics": pipeline_metrics,
            "model_ready": self.results.get("model_info", {}).get("model_path") is not None,
            "monitoring_summary": {
                "total_pipeline_duration_seconds": total_pipeline_time,
                "performance_baseline": self.performance_baseline,
                "stage_timings": self.stage_timings,
                "bottlenecks_identified": len(bottlenecks),
                "alert_summary": alert_summary,
                "monitoring_enabled": self.enable_alerts
            },
            "bootstrap_duration": datetime.now().isoformat()
        }
        
        self.results["summary"] = summary
        self.results["monitoring"]["performance_metrics"] = self.performance_baseline
        self.results["monitoring"]["stage_timings"] = self.stage_timings
        self.results["monitoring"]["alerts_generated"] = alert_summary
        
        return summary
    
    def export_comprehensive_prometheus_metrics(self) -> str:
        """Export comprehensive pipeline metrics in Prometheus format."""
        summary = self.results.get("summary", {})
        monitoring = summary.get("monitoring_summary", {})
        
        lines = [
            "# HELP hades_isne_bootstrap_pipeline_duration_seconds Total pipeline duration",
            "# TYPE hades_isne_bootstrap_pipeline_duration_seconds gauge",
            f'hades_isne_bootstrap_pipeline_duration_seconds{{pipeline="monitored_isne_bootstrap"}} {monitoring.get("total_pipeline_duration_seconds", 0):.6f}',
            "",
            "# HELP hades_isne_bootstrap_pipeline_success Pipeline completion success",
            "# TYPE hades_isne_bootstrap_pipeline_success gauge",
            f'hades_isne_bootstrap_pipeline_success{{pipeline="monitored_isne_bootstrap"}} {1 if summary.get("overall_success", False) else 0}',
            "",
            "# HELP hades_isne_bootstrap_pipeline_completion_rate Pipeline completion rate",
            "# TYPE hades_isne_bootstrap_pipeline_completion_rate gauge",
            f'hades_isne_bootstrap_pipeline_completion_rate{{pipeline="monitored_isne_bootstrap"}} {summary.get("completion_rate", 0):.4f}',
            "",
            "# HELP hades_isne_bootstrap_pipeline_memory_peak_bytes Peak memory usage",
            "# TYPE hades_isne_bootstrap_pipeline_memory_peak_bytes gauge",
            f'hades_isne_bootstrap_pipeline_memory_peak_bytes{{pipeline="monitored_isne_bootstrap"}} {monitoring.get("performance_baseline", {}).get("max_memory_usage_mb", 0) * 1024 * 1024:.0f}',
            "",
            "# HELP hades_isne_bootstrap_alerts_total Total alerts generated",
            "# TYPE hades_isne_bootstrap_alerts_total counter",
            f'hades_isne_bootstrap_alerts_total{{pipeline="monitored_isne_bootstrap"}} {monitoring.get("alert_summary", {}).get("total_alerts", 0)}',
            ""
        ]
        
        # Add per-stage timing metrics
        stage_timings = monitoring.get("stage_timings", {})
        for stage, duration in stage_timings.items():
            lines.extend([
                f"# HELP hades_isne_bootstrap_stage_timing_seconds Stage timing",
                f"# TYPE hades_isne_bootstrap_stage_timing_seconds gauge",
                f'hades_isne_bootstrap_stage_timing_seconds{{pipeline="monitored_isne_bootstrap",stage="{stage}"}} {duration:.6f}',
                ""
            ])
        
        return "\\n".join(lines)
    
    def run_monitored_complete_bootstrap(self) -> bool:
        """Run the complete end-to-end bootstrap pipeline with monitoring."""
        logger.info("=== MONITORED COMPLETE ISNE BOOTSTRAP PIPELINE ===")
        logger.info("Pipeline: docproc → chunking → embedding → graph construction → ISNE training")
        logger.info(f"Bootstrap timestamp: {self.timestamp}")
        logger.info(f"Input files: {[f.name for f in self.input_files]}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Monitoring enabled: {self.enable_alerts}")
        
        try:
            # Stage 1: Document Processing
            if not self.stage_1_document_processing():
                logger.error("Monitored bootstrap failed at document processing stage")
                return False
            
            # Stage 2: Chunking
            if not self.stage_2_chunking():
                logger.error("Monitored bootstrap failed at chunking stage")
                return False
            
            # Stage 3: Embedding
            if not self.stage_3_embedding():
                logger.error("Monitored bootstrap failed at embedding stage")
                return False
            
            # Stage 4: Graph Construction
            if not self.stage_4_graph_construction():
                logger.error("Monitored bootstrap failed at graph construction stage")
                return False
            
            # Stage 5: ISNE Training
            if not self.stage_5_isne_training():
                logger.error("Monitored bootstrap failed at ISNE training stage")
                return False
            
            # Generate comprehensive summary
            summary = self.generate_comprehensive_summary()
            
            # Export comprehensive Prometheus metrics
            prometheus_metrics = self.export_comprehensive_prometheus_metrics()
            metrics_file = self.output_dir / f"{self.timestamp}_pipeline_metrics.txt"
            with open(metrics_file, 'w') as f:
                f.write(prometheus_metrics)
            
            # Save final results with monitoring data
            results_file = self.output_dir / f"{self.timestamp}_monitored_bootstrap_results.json"
            with open(results_file, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            
            logger.info(f"\\n=== MONITORED BOOTSTRAP SUMMARY ===")
            logger.info(f"Pipeline Success: {summary['overall_success']}")
            logger.info(f"Stages Completed: {summary['stages_completed']}/5")
            logger.info(f"Total Duration: {summary['monitoring_summary']['total_pipeline_duration_seconds']:.2f} seconds")
            logger.info(f"Peak Memory: {summary['monitoring_summary']['performance_baseline']['max_memory_usage_mb']:.1f} MB")
            
            if summary.get("monitoring_summary", {}).get("alert_summary", {}).get("total_alerts", 0) > 0:
                logger.info(f"Alerts Generated: {summary['monitoring_summary']['alert_summary']['total_alerts']}")
            
            if "pipeline_metrics" in summary:
                metrics = summary["pipeline_metrics"]
                logger.info(f"\\nPipeline Flow:")
                logger.info(f"  {metrics.get('input_files', 0)} input files")
                logger.info(f"  → {metrics.get('documents', 0)} documents")
                logger.info(f"  → {metrics.get('chunks', 0)} chunks") 
                logger.info(f"  → {metrics.get('embeddings', 0)} embeddings ({metrics.get('embedding_dimension', 0)}D)")
                logger.info(f"  → Graph: {metrics.get('graph_nodes', 0)} nodes, {metrics.get('graph_edges', 0)} edges")
                logger.info(f"  → ISNE Model: {metrics.get('model_parameters', 0):,} parameters, {metrics.get('training_epochs', 0)} epochs")
            
            # Log performance insights
            bottlenecks = summary.get("monitoring_summary", {}).get("performance_baseline", {}).get("bottlenecks", [])
            if bottlenecks:
                logger.info(f"\\nPerformance Bottlenecks:")
                for bottleneck in bottlenecks:
                    logger.info(f"  {bottleneck['stage']}: {bottleneck['duration_seconds']:.1f}s ({bottleneck['percentage_of_total']:.1f}% of total)")
            
            logger.info(f"\\nResults saved to: {results_file}")
            logger.info(f"Metrics exported to: {metrics_file}")
            
            if summary["model_ready"]:
                model_path = self.results["model_info"]["model_path"]
                logger.info(f"✓ Monitored ISNE model successfully trained: {model_path}")
                logger.info("✓ End-to-end monitored bootstrap pipeline completed successfully!")
                logger.info("✓ ISNE model ready for integration with HADES RAG system")
            
            # Final success alert
            if self.alert_manager:
                self.alert_manager.alert(
                    "Monitored ISNE bootstrap pipeline completed successfully",
                    AlertLevel.LOW,
                    "pipeline_completion",
                    {
                        "total_duration": summary['monitoring_summary']['total_pipeline_duration_seconds'],
                        "peak_memory_mb": summary['monitoring_summary']['performance_baseline']['max_memory_usage_mb'],
                        "model_path": self.results.get("model_info", {}).get("model_path"),
                        "success": True
                    }
                )
            
            return summary["overall_success"]
            
        except Exception as e:
            logger.error(f"Monitored bootstrap pipeline failed: {e}")
            traceback.print_exc()
            
            if self.alert_manager:
                self.alert_manager.alert(
                    f"Monitored bootstrap pipeline critical failure: {e}",
                    AlertLevel.CRITICAL,
                    "pipeline_critical_failure",
                    {"error": str(e), "traceback": traceback.format_exc()}
                )
            
            return False


def main():
    """Main function for monitored bootstrap pipeline."""
    parser = argparse.ArgumentParser(
        description="Monitored Complete ISNE Bootstrap Pipeline - End-to-end training with monitoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument("--input-dir", default="./test-data",
                       help="Input directory containing documents to process")
    parser.add_argument("--output-dir", default="./models/isne",
                       help="Output directory for trained models and results")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       default="INFO", help="Logging level")
    parser.add_argument("--disable-alerts", action="store_true",
                       help="Disable alert system and monitoring")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Validate paths
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return False
    
    output_dir = Path(args.output_dir)
    
    # Run monitored bootstrap
    bootstrap = MonitoredISNEBootstrapPipeline(
        input_dir, 
        output_dir, 
        enable_alerts=not args.disable_alerts
    )
    success = bootstrap.run_monitored_complete_bootstrap()
    
    if success:
        logger.info("✓ Monitored complete ISNE bootstrap pipeline completed successfully!")
        return True
    else:
        logger.error("✗ Monitored complete ISNE bootstrap pipeline failed!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)