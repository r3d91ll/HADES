"""
ISNE Bootstrap Pipeline Monitoring

Specialized monitoring for the ISNE bootstrap pipeline.
Provides performance tracking, alerting, and metrics export.
"""

import time
import psutil
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

from src.alerts.alert_manager import AlertManager
from src.alerts import AlertLevel

logger = logging.getLogger(__name__)


@dataclass
class StageMetrics:
    """Metrics for a single pipeline stage."""
    stage_name: str
    start_time: float
    end_time: Optional[float] = None
    duration_seconds: Optional[float] = None
    success: bool = False
    
    # Resource usage
    initial_memory_mb: float = 0.0
    peak_memory_mb: float = 0.0
    memory_delta_mb: float = 0.0
    initial_cpu_percent: float = 0.0
    peak_cpu_percent: float = 0.0
    
    # Stage-specific stats
    stats: Dict[str, Any] = field(default_factory=dict)
    
    # Error information
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None
    
    def complete(self, success: bool, stats: Optional[Dict[str, Any]] = None, 
                error_message: Optional[str] = None, error_traceback: Optional[str] = None):
        """Mark stage as complete and calculate final metrics."""
        self.end_time = time.time()
        self.duration_seconds = self.end_time - self.start_time
        self.success = success
        if stats:
            self.stats.update(stats)
        self.error_message = error_message
        self.error_traceback = error_traceback
        
        # Calculate memory delta
        self.memory_delta_mb = self.peak_memory_mb - self.initial_memory_mb


@dataclass
class PipelineMetrics:
    """Overall pipeline metrics."""
    pipeline_name: str
    start_time: float
    end_time: Optional[float] = None
    total_duration_seconds: Optional[float] = None
    overall_success: bool = False
    
    # Stage metrics
    stages: Dict[str, StageMetrics] = field(default_factory=dict)
    
    # Performance baselines
    peak_memory_mb: float = 0.0
    peak_cpu_percent: float = 0.0
    
    # Alerts generated
    alerts_generated: List[Dict[str, Any]] = field(default_factory=list)
    
    def complete(self, success: bool):
        """Mark pipeline as complete."""
        self.end_time = time.time()
        self.total_duration_seconds = self.end_time - self.start_time
        self.overall_success = success


class BootstrapMonitor:
    """Monitoring and alerting for ISNE bootstrap pipeline."""
    
    def __init__(self, pipeline_name: str, output_dir: Path, 
                 enable_alerts: bool = True, alert_thresholds: Optional[Dict[str, Any]] = None):
        """Initialize bootstrap monitor."""
        self.pipeline_name = pipeline_name
        self.output_dir = output_dir
        self.enable_alerts = enable_alerts
        self.alert_thresholds = alert_thresholds or {
            "max_memory_mb": 4000,
            "max_duration_seconds": 3600,
            "stage_timeout_multiplier": 2.0
        }
        
        # Initialize alert manager
        if enable_alerts:
            alert_dir = output_dir / "alerts"
            alert_dir.mkdir(parents=True, exist_ok=True)
            self.alert_manager = AlertManager(
                alert_dir=str(alert_dir),
                min_level=AlertLevel.LOW
            )
        else:
            self.alert_manager = None
        
        # Pipeline metrics
        self.pipeline_metrics = PipelineMetrics(
            pipeline_name=pipeline_name,
            start_time=time.time()
        )
        
        # Stage timing thresholds (seconds)
        self.stage_time_thresholds = {
            "document_processing": 60,
            "chunking": 120,
            "embedding": 600,
            "graph_construction": 300,
            "model_training": 1800
        }
        
        logger.info(f"Initialized bootstrap monitor for {pipeline_name}")
        
        # Initial pipeline start alert
        if self.alert_manager:
            self.alert_manager.alert(
                f"ISNE bootstrap pipeline started: {pipeline_name}",
                AlertLevel.LOW,
                "pipeline_start",
                {
                    "pipeline_name": pipeline_name,
                    "start_time": datetime.now().isoformat(),
                    "output_dir": str(output_dir)
                }
            )
    
    def start_stage(self, stage_name: str) -> StageMetrics:
        """Start monitoring a pipeline stage."""
        logger.info(f"Starting stage monitoring: {stage_name}")
        
        process = psutil.Process()
        initial_memory_mb = process.memory_info().rss / 1024 / 1024
        initial_cpu_percent = process.cpu_percent()
        
        stage_metrics = StageMetrics(
            stage_name=stage_name,
            start_time=time.time(),
            initial_memory_mb=initial_memory_mb,
            peak_memory_mb=initial_memory_mb,
            initial_cpu_percent=initial_cpu_percent,
            peak_cpu_percent=initial_cpu_percent
        )
        
        self.pipeline_metrics.stages[stage_name] = stage_metrics
        
        logger.info(f"Stage {stage_name} monitoring started - Initial memory: {initial_memory_mb:.1f} MB")
        return stage_metrics
    
    def update_stage_resources(self, stage_name: str) -> None:
        """Update resource usage for a stage (call periodically during execution)."""
        if stage_name not in self.pipeline_metrics.stages:
            return
        
        stage_metrics = self.pipeline_metrics.stages[stage_name]
        process = psutil.Process()
        
        current_memory_mb = process.memory_info().rss / 1024 / 1024
        current_cpu_percent = process.cpu_percent()
        
        # Update peak values
        stage_metrics.peak_memory_mb = max(stage_metrics.peak_memory_mb, current_memory_mb)
        stage_metrics.peak_cpu_percent = max(stage_metrics.peak_cpu_percent, current_cpu_percent)
        
        # Update pipeline peak values
        self.pipeline_metrics.peak_memory_mb = max(self.pipeline_metrics.peak_memory_mb, current_memory_mb)
        self.pipeline_metrics.peak_cpu_percent = max(self.pipeline_metrics.peak_cpu_percent, current_cpu_percent)
    
    def complete_stage(self, stage_name: str, success: bool, 
                      stats: Optional[Dict[str, Any]] = None,
                      error_message: Optional[str] = None,
                      error_traceback: Optional[str] = None) -> None:
        """Complete stage monitoring and check thresholds."""
        if stage_name not in self.pipeline_metrics.stages:
            logger.warning(f"Stage {stage_name} not found in metrics")
            return
        
        stage_metrics = self.pipeline_metrics.stages[stage_name]
        
        # Final resource update
        self.update_stage_resources(stage_name)
        
        # Complete stage metrics
        stage_metrics.complete(success, stats, error_message, error_traceback)
        
        logger.info(f"Stage {stage_name} completed - Duration: {stage_metrics.duration_seconds:.2f}s, "
                   f"Memory delta: {stage_metrics.memory_delta_mb:+.1f} MB, Success: {success}")
        
        # Check performance thresholds and generate alerts
        self._check_stage_thresholds(stage_metrics)
        
        # Export stage metrics
        self._export_stage_metrics(stage_metrics)
    
    def _check_stage_thresholds(self, stage_metrics: StageMetrics) -> None:
        """Check stage performance against thresholds and generate alerts."""
        if not self.alert_manager:
            return
        
        stage_name = stage_metrics.stage_name
        
        # Check duration threshold
        threshold = self.stage_time_thresholds.get(stage_name, 300)
        if stage_metrics.duration_seconds and stage_metrics.duration_seconds > threshold:
            alert_data = {
                "stage": stage_name,
                "duration_seconds": stage_metrics.duration_seconds,
                "threshold_seconds": threshold,
                "performance_impact": "high_latency"
            }
            self.alert_manager.alert(
                f"Stage {stage_name} exceeded time threshold ({stage_metrics.duration_seconds:.1f}s > {threshold}s)",
                AlertLevel.MEDIUM,
                f"stage_performance_{stage_name}",
                alert_data
            )
            self.pipeline_metrics.alerts_generated.append({
                "type": "performance_threshold",
                "stage": stage_name,
                "data": alert_data
            })
        
        # Check memory growth threshold
        memory_growth_threshold = 500  # 500 MB
        if stage_metrics.memory_delta_mb > memory_growth_threshold:
            alert_data = {
                "stage": stage_name,
                "memory_growth_mb": stage_metrics.memory_delta_mb,
                "threshold_mb": memory_growth_threshold,
                "peak_memory_mb": stage_metrics.peak_memory_mb
            }
            self.alert_manager.alert(
                f"Stage {stage_name} high memory growth ({stage_metrics.memory_delta_mb:.1f} MB)",
                AlertLevel.MEDIUM,
                f"stage_memory_{stage_name}",
                alert_data
            )
            self.pipeline_metrics.alerts_generated.append({
                "type": "memory_growth",
                "stage": stage_name,
                "data": alert_data
            })
        
        # Check absolute memory threshold
        absolute_memory_threshold = self.alert_thresholds.get("max_memory_mb", 4000)
        if stage_metrics.peak_memory_mb > absolute_memory_threshold:
            alert_data = {
                "stage": stage_name,
                "memory_usage_mb": stage_metrics.peak_memory_mb,
                "threshold_mb": absolute_memory_threshold,
                "risk": "memory_exhaustion"
            }
            self.alert_manager.alert(
                f"Stage {stage_name} high absolute memory usage ({stage_metrics.peak_memory_mb:.1f} MB)",
                AlertLevel.HIGH,
                f"stage_memory_absolute_{stage_name}",
                alert_data
            )
            self.pipeline_metrics.alerts_generated.append({
                "type": "memory_absolute",
                "stage": stage_name,
                "data": alert_data
            })
        
        # Check for stage failure
        if not stage_metrics.success:
            alert_data = {
                "stage": stage_name,
                "error_message": stage_metrics.error_message,
                "duration_seconds": stage_metrics.duration_seconds
            }
            self.alert_manager.alert(
                f"Stage {stage_name} failed: {stage_metrics.error_message}",
                AlertLevel.CRITICAL,
                f"stage_failure_{stage_name}",
                alert_data
            )
            self.pipeline_metrics.alerts_generated.append({
                "type": "stage_failure",
                "stage": stage_name,
                "data": alert_data
            })
    
    def _export_stage_metrics(self, stage_metrics: StageMetrics) -> None:
        """Export stage metrics in Prometheus format."""
        metrics_lines = [
            f"# HELP hades_isne_bootstrap_stage_duration_seconds Duration of bootstrap stage",
            f"# TYPE hades_isne_bootstrap_stage_duration_seconds gauge",
            f'hades_isne_bootstrap_stage_duration_seconds{{stage="{stage_metrics.stage_name}",pipeline="{self.pipeline_name}"}} {stage_metrics.duration_seconds or 0:.6f}',
            "",
            f"# HELP hades_isne_bootstrap_stage_memory_usage_bytes Memory usage during stage",
            f"# TYPE hades_isne_bootstrap_stage_memory_usage_bytes gauge",
            f'hades_isne_bootstrap_stage_memory_usage_bytes{{stage="{stage_metrics.stage_name}",pipeline="{self.pipeline_name}",type="peak"}} {stage_metrics.peak_memory_mb * 1024 * 1024:.0f}',
            "",
            f"# HELP hades_isne_bootstrap_stage_success Stage completion status",
            f"# TYPE hades_isne_bootstrap_stage_success gauge",
            f'hades_isne_bootstrap_stage_success{{stage="{stage_metrics.stage_name}",pipeline="{self.pipeline_name}"}} {1 if stage_metrics.success else 0}',
            ""
        ]
        
        # Add stage-specific metrics
        stats = stage_metrics.stats
        stage_name = stage_metrics.stage_name
        
        if stage_name == "document_processing" and "documents_generated" in stats:
            metrics_lines.extend([
                f"# HELP hades_isne_bootstrap_documents_processed_total Documents processed",
                f"# TYPE hades_isne_bootstrap_documents_processed_total counter",
                f'hades_isne_bootstrap_documents_processed_total{{pipeline="{self.pipeline_name}"}} {stats["documents_generated"]}',
                ""
            ])
        
        if stage_name == "chunking" and "output_chunks" in stats:
            metrics_lines.extend([
                f"# HELP hades_isne_bootstrap_chunks_created_total Chunks created",
                f"# TYPE hades_isne_bootstrap_chunks_created_total counter",
                f'hades_isne_bootstrap_chunks_created_total{{pipeline="{self.pipeline_name}"}} {stats["output_chunks"]}',
                ""
            ])
        
        if stage_name == "embedding" and "valid_embeddings" in stats:
            metrics_lines.extend([
                f"# HELP hades_isne_bootstrap_embeddings_generated_total Embeddings generated",
                f"# TYPE hades_isne_bootstrap_embeddings_generated_total counter",
                f'hades_isne_bootstrap_embeddings_generated_total{{pipeline="{self.pipeline_name}"}} {stats["valid_embeddings"]}',
                ""
            ])
        
        if stage_name == "graph_construction" and "num_nodes" in stats:
            metrics_lines.extend([
                f"# HELP hades_isne_bootstrap_graph_nodes_total Graph nodes created",
                f"# TYPE hades_isne_bootstrap_graph_nodes_total counter",
                f'hades_isne_bootstrap_graph_nodes_total{{pipeline="{self.pipeline_name}"}} {stats["num_nodes"]}',
                "",
                f"# HELP hades_isne_bootstrap_graph_edges_total Graph edges created",
                f"# TYPE hades_isne_bootstrap_graph_edges_total counter",
                f'hades_isne_bootstrap_graph_edges_total{{pipeline="{self.pipeline_name}"}} {stats.get("num_edges", 0)}',
                ""
            ])
        
        if stage_name == "model_training" and "training_epochs" in stats:
            metrics_lines.extend([
                f"# HELP hades_isne_bootstrap_training_epochs_total Training epochs completed",
                f"# TYPE hades_isne_bootstrap_training_epochs_total counter",
                f'hades_isne_bootstrap_training_epochs_total{{pipeline="{self.pipeline_name}"}} {stats["training_epochs"]}',
                ""
            ])
        
        # Write metrics to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_file = self.output_dir / f"{timestamp}_{stage_name}_metrics.txt"
        with open(metrics_file, 'w') as f:
            f.write("\\n".join(metrics_lines))
    
    def complete_pipeline(self, success: bool) -> Dict[str, Any]:
        """Complete pipeline monitoring and generate summary."""
        self.pipeline_metrics.complete(success)
        
        # Calculate summary statistics
        completed_stages = len([s for s in self.pipeline_metrics.stages.values() if s.end_time is not None])
        successful_stages = len([s for s in self.pipeline_metrics.stages.values() if s.success])
        
        summary = {
            "pipeline_name": self.pipeline_name,
            "overall_success": success,
            "total_duration_seconds": self.pipeline_metrics.total_duration_seconds,
            "stages_completed": completed_stages,
            "stages_successful": successful_stages,
            "completion_rate": completed_stages / 5.0,  # 5 total stages
            "success_rate": successful_stages / max(completed_stages, 1),
            "peak_memory_mb": self.pipeline_metrics.peak_memory_mb,
            "peak_cpu_percent": self.pipeline_metrics.peak_cpu_percent,
            "alerts_generated": len(self.pipeline_metrics.alerts_generated),
            "stage_timings": {
                name: metrics.duration_seconds 
                for name, metrics in self.pipeline_metrics.stages.items()
                if metrics.duration_seconds is not None
            }
        }
        
        # Export pipeline-level metrics
        self._export_pipeline_metrics(summary)
        
        # Final pipeline alert
        if self.alert_manager:
            alert_level = AlertLevel.LOW if success else AlertLevel.CRITICAL
            message = f"ISNE bootstrap pipeline {'completed successfully' if success else 'failed'}: {self.pipeline_name}"
            self.alert_manager.alert(
                message,
                alert_level,
                "pipeline_completion",
                summary
            )
        
        logger.info(f"Pipeline monitoring completed - Success: {success}, "
                   f"Duration: {self.pipeline_metrics.total_duration_seconds:.2f}s, "
                   f"Peak memory: {self.pipeline_metrics.peak_memory_mb:.1f} MB")
        
        return summary
    
    def _export_pipeline_metrics(self, summary: Dict[str, Any]) -> None:
        """Export pipeline-level metrics in Prometheus format."""
        metrics_lines = [
            f"# HELP hades_isne_bootstrap_pipeline_duration_seconds Total pipeline duration",
            f"# TYPE hades_isne_bootstrap_pipeline_duration_seconds gauge",
            f'hades_isne_bootstrap_pipeline_duration_seconds{{pipeline="{self.pipeline_name}"}} {summary["total_duration_seconds"]:.6f}',
            "",
            f"# HELP hades_isne_bootstrap_pipeline_success Pipeline completion success",
            f"# TYPE hades_isne_bootstrap_pipeline_success gauge",
            f'hades_isne_bootstrap_pipeline_success{{pipeline="{self.pipeline_name}"}} {1 if summary["overall_success"] else 0}',
            "",
            f"# HELP hades_isne_bootstrap_pipeline_completion_rate Pipeline completion rate",
            f"# TYPE hades_isne_bootstrap_pipeline_completion_rate gauge",
            f'hades_isne_bootstrap_pipeline_completion_rate{{pipeline="{self.pipeline_name}"}} {summary["completion_rate"]:.4f}',
            "",
            f"# HELP hades_isne_bootstrap_pipeline_memory_peak_bytes Peak memory usage",
            f"# TYPE hades_isne_bootstrap_pipeline_memory_peak_bytes gauge",
            f'hades_isne_bootstrap_pipeline_memory_peak_bytes{{pipeline="{self.pipeline_name}"}} {summary["peak_memory_mb"] * 1024 * 1024:.0f}',
            "",
            f"# HELP hades_isne_bootstrap_alerts_total Total alerts generated",
            f"# TYPE hades_isne_bootstrap_alerts_total counter",
            f'hades_isne_bootstrap_alerts_total{{pipeline="{self.pipeline_name}"}} {summary["alerts_generated"]}',
            ""
        ]
        
        # Write pipeline metrics to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_file = self.output_dir / f"{timestamp}_pipeline_metrics.txt"
        with open(metrics_file, 'w') as f:
            f.write("\\n".join(metrics_lines))
    
    def get_metrics(self) -> PipelineMetrics:
        """Get current pipeline metrics."""
        return self.pipeline_metrics