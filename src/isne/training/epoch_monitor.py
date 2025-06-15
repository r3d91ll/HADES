"""
ISNE Training Epoch Monitor

Real-time monitoring and Prometheus export for per-epoch training metrics.
Tracks loss progression, learning dynamics, and performance indicators.
"""

import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class EpochMetrics:
    """Metrics for a single training epoch."""
    epoch: int
    loss: float
    duration_seconds: float
    timestamp: float = field(default_factory=time.time)
    
    # Optional additional metrics
    learning_rate: Optional[float] = None
    gradient_norm: Optional[float] = None
    gpu_memory_mb: Optional[float] = None
    
    def to_prometheus(self, job_name: str, model_name: str) -> List[str]:
        """Convert epoch metrics to Prometheus format."""
        lines = [
            f'# HELP isne_training_epoch_loss Loss value for ISNE training epoch',
            f'# TYPE isne_training_epoch_loss gauge',
            f'isne_training_epoch_loss{{job="{job_name}",model="{model_name}",epoch="{self.epoch}"}} {self.loss:.6f}',
            '',
            f'# HELP isne_training_epoch_duration_seconds Duration of training epoch',
            f'# TYPE isne_training_epoch_duration_seconds gauge',
            f'isne_training_epoch_duration_seconds{{job="{job_name}",model="{model_name}",epoch="{self.epoch}"}} {self.duration_seconds:.2f}',
            '',
        ]
        
        if self.learning_rate is not None:
            lines.extend([
                f'# HELP isne_training_learning_rate Current learning rate',
                f'# TYPE isne_training_learning_rate gauge',
                f'isne_training_learning_rate{{job="{job_name}",model="{model_name}",epoch="{self.epoch}"}} {self.learning_rate:.6f}',
                '',
            ])
        
        if self.gradient_norm is not None:
            lines.extend([
                f'# HELP isne_training_gradient_norm L2 norm of gradients',
                f'# TYPE isne_training_gradient_norm gauge',
                f'isne_training_gradient_norm{{job="{job_name}",model="{model_name}",epoch="{self.epoch}"}} {self.gradient_norm:.6f}',
                '',
            ])
        
        if self.gpu_memory_mb is not None:
            lines.extend([
                f'# HELP isne_training_gpu_memory_bytes GPU memory usage',
                f'# TYPE isne_training_gpu_memory_bytes gauge',
                f'isne_training_gpu_memory_bytes{{job="{job_name}",model="{model_name}",epoch="{self.epoch}"}} {self.gpu_memory_mb * 1024 * 1024:.0f}',
                '',
            ])
        
        return lines


@dataclass
class TrainingRunMetrics:
    """Metrics for entire training run."""
    job_name: str
    model_name: str
    start_time: float = field(default_factory=time.time)
    total_epochs: int = 0
    
    # Epoch history
    epochs: List[EpochMetrics] = field(default_factory=list)
    
    # Training dynamics
    best_loss: Optional[float] = None
    best_epoch: Optional[int] = None
    loss_improvement_rate: Optional[float] = None
    convergence_epoch: Optional[int] = None
    
    def add_epoch(self, epoch_metrics: EpochMetrics):
        """Add epoch metrics and update training dynamics."""
        self.epochs.append(epoch_metrics)
        self.total_epochs = len(self.epochs)
        
        # Update best loss
        if self.best_loss is None or epoch_metrics.loss < self.best_loss:
            self.best_loss = epoch_metrics.loss
            self.best_epoch = epoch_metrics.epoch
        
        # Calculate improvement rate
        if len(self.epochs) >= 2:
            recent_losses = [e.loss for e in self.epochs[-5:]]
            if len(recent_losses) >= 2:
                self.loss_improvement_rate = (recent_losses[0] - recent_losses[-1]) / len(recent_losses)
        
        # Detect convergence (when improvement rate drops below threshold)
        if self.loss_improvement_rate is not None and abs(self.loss_improvement_rate) < 0.001:
            if self.convergence_epoch is None:
                self.convergence_epoch = epoch_metrics.epoch
    
    def to_prometheus(self) -> List[str]:
        """Export all training metrics to Prometheus format."""
        lines = [
            f'# Training run metrics for {self.model_name}',
            f'# Generated at {datetime.now().isoformat()}',
            '',
            f'# HELP isne_training_total_epochs Total epochs completed',
            f'# TYPE isne_training_total_epochs counter',
            f'isne_training_total_epochs{{job="{self.job_name}",model="{self.model_name}"}} {self.total_epochs}',
            '',
        ]
        
        if self.best_loss is not None:
            lines.extend([
                f'# HELP isne_training_best_loss Best loss achieved during training',
                f'# TYPE isne_training_best_loss gauge',
                f'isne_training_best_loss{{job="{self.job_name}",model="{self.model_name}"}} {self.best_loss:.6f}',
                '',
                f'# HELP isne_training_best_epoch Epoch with best loss',
                f'# TYPE isne_training_best_epoch gauge',
                f'isne_training_best_epoch{{job="{self.job_name}",model="{self.model_name}"}} {self.best_epoch}',
                '',
            ])
        
        if self.loss_improvement_rate is not None:
            lines.extend([
                f'# HELP isne_training_improvement_rate Recent loss improvement rate',
                f'# TYPE isne_training_improvement_rate gauge',
                f'isne_training_improvement_rate{{job="{self.job_name}",model="{self.model_name}"}} {self.loss_improvement_rate:.6f}',
                '',
            ])
        
        if self.convergence_epoch is not None:
            lines.extend([
                f'# HELP isne_training_convergence_epoch Epoch where convergence detected',
                f'# TYPE isne_training_convergence_epoch gauge',
                f'isne_training_convergence_epoch{{job="{self.job_name}",model="{self.model_name}"}} {self.convergence_epoch}',
                '',
            ])
        
        # Add current epoch metrics
        if self.epochs:
            latest_epoch = self.epochs[-1]
            lines.extend([
                f'# HELP isne_training_current_loss Current training loss',
                f'# TYPE isne_training_current_loss gauge',
                f'isne_training_current_loss{{job="{self.job_name}",model="{self.model_name}"}} {latest_epoch.loss:.6f}',
                '',
                f'# HELP isne_training_current_epoch Current training epoch',
                f'# TYPE isne_training_current_epoch gauge',
                f'isne_training_current_epoch{{job="{self.job_name}",model="{self.model_name}"}} {latest_epoch.epoch}',
                '',
            ])
        
        # Add per-epoch metrics
        for epoch_metrics in self.epochs:
            lines.extend(epoch_metrics.to_prometheus(self.job_name, self.model_name))
        
        return lines


class ISNEEpochMonitor:
    """Monitor and export per-epoch training metrics for ISNE."""
    
    def __init__(self, 
                 output_dir: Path,
                 job_name: str = "isne_training",
                 model_name: str = "isne_model",
                 export_interval: int = 1,
                 prometheus_dir: Optional[Path] = None):
        """
        Initialize epoch monitor.
        
        Args:
            output_dir: Directory for output files
            job_name: Prometheus job name
            model_name: Model identifier
            export_interval: Export metrics every N epochs
            prometheus_dir: Directory for Prometheus metric files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.prometheus_dir = prometheus_dir or self.output_dir / "prometheus_metrics"
        self.prometheus_dir.mkdir(parents=True, exist_ok=True)
        
        self.export_interval = export_interval
        
        # Initialize training metrics
        self.training_metrics = TrainingRunMetrics(
            job_name=job_name,
            model_name=model_name
        )
        
        # Metrics file paths
        self.current_metrics_file = self.prometheus_dir / "isne_training_current.prom"
        self.history_file = self.output_dir / "training_history.json"
        
        logger.info(f"Initialized ISNE epoch monitor - Metrics will be exported to {self.prometheus_dir}")
    
    def record_epoch(self, 
                    epoch: int, 
                    loss: float, 
                    duration_seconds: float,
                    learning_rate: Optional[float] = None,
                    gradient_norm: Optional[float] = None,
                    gpu_memory_mb: Optional[float] = None) -> None:
        """
        Record metrics for a completed epoch.
        
        Args:
            epoch: Epoch number
            loss: Training loss value
            duration_seconds: Time taken for epoch
            learning_rate: Current learning rate
            gradient_norm: L2 norm of gradients
            gpu_memory_mb: GPU memory usage in MB
        """
        # Create epoch metrics
        epoch_metrics = EpochMetrics(
            epoch=epoch,
            loss=loss,
            duration_seconds=duration_seconds,
            learning_rate=learning_rate,
            gradient_norm=gradient_norm,
            gpu_memory_mb=gpu_memory_mb
        )
        
        # Add to training metrics
        self.training_metrics.add_epoch(epoch_metrics)
        
        # Log progress
        logger.info(f"Epoch {epoch} - Loss: {loss:.6f}, Duration: {duration_seconds:.2f}s")
        
        if self.training_metrics.loss_improvement_rate is not None:
            logger.info(f"  Loss improvement rate: {self.training_metrics.loss_improvement_rate:.6f}")
        
        # Export metrics if interval reached
        if epoch % self.export_interval == 0:
            self.export_metrics()
    
    def export_metrics(self) -> None:
        """Export current metrics to Prometheus format."""
        # Generate Prometheus metrics
        prometheus_lines = self.training_metrics.to_prometheus()
        
        # Write to current metrics file (for Prometheus scraping)
        with open(self.current_metrics_file, 'w') as f:
            f.write('\n'.join(prometheus_lines))
        
        # Also save timestamped snapshot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_file = self.prometheus_dir / f"isne_training_{timestamp}.prom"
        with open(snapshot_file, 'w') as f:
            f.write('\n'.join(prometheus_lines))
        
        logger.info(f"Exported metrics to {self.current_metrics_file}")
    
    def save_history(self) -> None:
        """Save complete training history as JSON."""
        import json
        
        history = {
            "job_name": self.training_metrics.job_name,
            "model_name": self.training_metrics.model_name,
            "total_epochs": self.training_metrics.total_epochs,
            "best_loss": self.training_metrics.best_loss,
            "best_epoch": self.training_metrics.best_epoch,
            "convergence_epoch": self.training_metrics.convergence_epoch,
            "epochs": [
                {
                    "epoch": e.epoch,
                    "loss": e.loss,
                    "duration_seconds": e.duration_seconds,
                    "learning_rate": e.learning_rate,
                    "gradient_norm": e.gradient_norm,
                    "gpu_memory_mb": e.gpu_memory_mb,
                    "timestamp": e.timestamp
                }
                for e in self.training_metrics.epochs
            ]
        }
        
        with open(self.history_file, 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info(f"Saved training history to {self.history_file}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get training summary statistics."""
        if not self.training_metrics.epochs:
            return {"status": "no_data"}
        
        losses = [e.loss for e in self.training_metrics.epochs]
        durations = [e.duration_seconds for e in self.training_metrics.epochs]
        
        return {
            "total_epochs": self.training_metrics.total_epochs,
            "best_loss": self.training_metrics.best_loss,
            "best_epoch": self.training_metrics.best_epoch,
            "current_loss": losses[-1],
            "current_epoch": self.training_metrics.epochs[-1].epoch,
            "avg_epoch_duration": sum(durations) / len(durations),
            "total_duration": sum(durations),
            "convergence_detected": self.training_metrics.convergence_epoch is not None,
            "convergence_epoch": self.training_metrics.convergence_epoch,
            "improvement_rate": self.training_metrics.loss_improvement_rate
        }


# Integration example for trainer
def integrate_with_trainer(trainer, epoch_monitor: ISNEEpochMonitor):
    """
    Example of how to integrate epoch monitor with ISNE trainer.
    
    This would be added to the training loop:
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # ... training code ...
        
        epoch_time = time.time() - epoch_start
        
        # Record epoch metrics
        epoch_monitor.record_epoch(
            epoch=epoch + 1,
            loss=avg_loss,
            duration_seconds=epoch_time,
            learning_rate=trainer.optimizer.param_groups[0]['lr'],
            gpu_memory_mb=torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else None
        )
    
    # Save final history
    epoch_monitor.save_history()
    """
    pass