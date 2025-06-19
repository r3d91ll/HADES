#!/usr/bin/env python3
"""
Add Epoch Monitoring to ISNE Training

Quick patch to add per-epoch Prometheus metrics to existing ISNE training.
This can be integrated into the trainer for future runs.

Usage:
    # In the trainer's train_isne_simple method, add:
    
    from src.isne.training.epoch_monitor import ISNEEpochMonitor
    
    # Initialize monitor
    epoch_monitor = ISNEEpochMonitor(
        output_dir=Path("output/olympus_production/metrics"),
        model_name="olympus_production_rag_model",
        export_interval=1  # Export every epoch
    )
    
    # In the training loop, after calculating epoch loss:
    epoch_monitor.record_epoch(
        epoch=epoch + 1,
        loss=avg_loss,
        duration_seconds=epoch_time
    )
"""

# Example configuration for Prometheus scraping
PROMETHEUS_CONFIG = """
# Add to your prometheus.yml:

scrape_configs:
  - job_name: 'isne_training'
    scrape_interval: 30s
    static_configs:
      - targets: ['localhost:9090']  # Or use file_sd_configs
    file_sd_configs:
      - files:
        - '/home/todd/ML-Lab/Olympus/HADES/output/olympus_production/metrics/prometheus_metrics/isne_training_current.prom'
    
# Or use node_exporter textfile collector:
# Place metrics in: /var/lib/prometheus/node-exporter/

# Then you can create Grafana dashboards with queries like:
# - rate(isne_training_epoch_loss[5m]) - Loss change rate
# - isne_training_improvement_rate - Training dynamics
# - isne_training_epoch_duration_seconds - Performance metrics
# - isne_training_best_loss - Best achieved loss
"""

# Example Grafana dashboard panels
GRAFANA_DASHBOARD = {
    "training_progress": {
        "title": "ISNE Training Progress",
        "panels": [
            {
                "title": "Training Loss",
                "query": "isne_training_epoch_loss{job='isne_training'}",
                "legend": "Loss per epoch"
            },
            {
                "title": "Epoch Duration",
                "query": "isne_training_epoch_duration_seconds{job='isne_training'}",
                "unit": "seconds"
            },
            {
                "title": "Loss Improvement Rate",
                "query": "isne_training_improvement_rate{job='isne_training'}",
                "description": "Rate of loss improvement (negative = improving)"
            },
            {
                "title": "Best Loss",
                "query": "isne_training_best_loss{job='isne_training'}",
                "stat": True
            },
            {
                "title": "Training Efficiency",
                "query": "rate(isne_training_total_epochs[5m]) * 60",
                "unit": "epochs/min"
            }
        ]
    }
}


def print_integration_guide():
    """Print integration guide for adding epoch monitoring."""
    print("ISNE Epoch Monitoring Integration Guide")
    print("=" * 50)
    print()
    print("1. Add to trainer imports:")
    print("   from src.isne.training.epoch_monitor import ISNEEpochMonitor")
    print()
    print("2. Initialize before training loop:")
    print("   epoch_monitor = ISNEEpochMonitor(")
    print("       output_dir=Path('output/metrics'),")
    print("       model_name=self.model_name,")
    print("       prometheus_dir=Path('/var/lib/prometheus/node-exporter')")
    print("   )")
    print()
    print("3. In training loop after loss calculation:")
    print("   epoch_monitor.record_epoch(")
    print("       epoch=epoch + 1,")
    print("       loss=avg_loss,")
    print("       duration_seconds=epoch_time")
    print("   )")
    print()
    print("4. After training completes:")
    print("   epoch_monitor.save_history()")
    print("   summary = epoch_monitor.get_summary()")
    print()
    print("Prometheus Configuration:")
    print("-" * 50)
    print(PROMETHEUS_CONFIG)
    print()
    print("Benefits:")
    print("- Real-time training monitoring in Grafana")
    print("- Historical analysis of training runs")
    print("- Early detection of training issues")
    print("- Comparison across different model configurations")
    print("- Automated alerting on training anomalies")


if __name__ == "__main__":
    print_integration_guide()