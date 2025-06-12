# Alerts Module

The alerts module provides comprehensive monitoring and alerting infrastructure for the HADES system. It enables real-time detection of issues, performance degradation, and system health monitoring across all pipeline stages.

## 📋 Overview

This module implements a centralized alerting system that:

- Monitors pipeline execution and detects failures
- Tracks data quality and embedding performance
- Provides configurable alert levels and notification channels
- Maintains alert history for analysis and debugging
- Integrates with all system components for comprehensive coverage

## 🏗️ Architecture

### Core Components

- **`AlertManager`** - Central alert coordination and dispatch
- **`AlertLevel`** - Severity classification system
- **`AlertSource`** - Component identification for alert origin
- **Alert Storage** - Persistent alert history and metrics

### Alert Flow

```text
System Component → Alert Generation → AlertManager → Classification → Storage + Notification
```

## 📁 Module Contents

### Core Files

- **`alert_manager.py`** - Main AlertManager class and alert handling logic
- **`__init__.py`** - Module exports and convenience imports
- **`alerts_readme.md`** - Legacy documentation (superseded by this README)

### Data Storage

- **`alerts/alerts.json`** - Persistent alert storage (created automatically)
- Log files generated in `./training_logs/` for training-related alerts

## 🚀 Key Features

### Alert Levels

```python
from src.alerts import AlertLevel

AlertLevel.LOW     # Informational - system events, successful operations
AlertLevel.MEDIUM  # Warning - performance issues, minor errors
AlertLevel.HIGH    # Critical - failures, data corruption, system unavailable
```

### Alert Sources

Alerts can originate from any system component:

- **Pipeline stages** (document processing, chunking, embedding, ISNE)
- **Training processes** (bootstrap, incremental, full retraining)
- **Storage operations** (database connections, data integrity)
- **API endpoints** (request failures, authentication issues)
- **System resources** (memory, disk space, GPU availability)

### Alert Management

```python
from src.alerts import AlertManager

# Initialize alert manager
alert_manager = AlertManager(alert_dir="./alerts")

# Generate alerts
alert_manager.alert(
    message="Processing error in chunking stage",
    level=AlertLevel.HIGH,
    source="chunking_pipeline",
    metadata={"error_type": "validation_failed", "chunk_count": 0}
)

# Retrieve recent alerts
recent_alerts = alert_manager.get_recent_alerts(hours=24)
high_priority = alert_manager.get_alerts_by_level(AlertLevel.HIGH)
```

## 🔧 Configuration

### Alert Settings

Configure alerting behavior in your pipeline configurations:

```yaml
# In pipeline configs
alerts:
  enabled: true
  storage_dir: "./alerts"
  max_alert_history: 1000
  alert_levels:
    - HIGH    # Always stored and logged
    - MEDIUM  # Stored, logged with sampling
    - LOW     # Stored, minimal logging

# In training configs  
logging:
  alert_on_failure: true
  alert_on_quality_drop: true
  quality_threshold: 0.8
```

### Integration Points

The alert system integrates with:

- **Pipeline stages** - Automatic error detection and reporting
- **Training processes** - Quality monitoring and failure alerts
- **Scheduled jobs** - Cron job monitoring and status reporting
- **API services** - Request monitoring and error tracking

## 📊 Alert Types and Use Cases

### Pipeline Alerts

**Document Processing Failures**:

```python
alert_manager.alert(
    message="Failed to process PDF document: corruption detected",
    level=AlertLevel.HIGH,
    source="docproc_stage",
    metadata={"file_path": "document.pdf", "error": "InvalidPDF"}
)
```

**Chunking Quality Issues**:

```python
alert_manager.alert(
    message="Chunking produced unusually small chunks",
    level=AlertLevel.MEDIUM,
    source="chunking_stage", 
    metadata={"avg_chunk_size": 45, "threshold": 100}
)
```

**Embedding Generation Problems**:

```python
alert_manager.alert(
    message="Embedding quality below threshold",
    level=AlertLevel.HIGH,
    source="isne_pipeline",
    metadata={"quality_score": 0.65, "threshold": 0.8}
)
```

### Training Alerts

**Model Training Failures**:

```python
alert_manager.alert(
    message="ISNE training failed to converge", 
    level=AlertLevel.HIGH,
    source="isne_trainer",
    metadata={"final_loss": 2.34, "epochs": 100}
)
```

**Scheduled Training Issues**:

```python
alert_manager.alert(
    message="Daily incremental training found no new data",
    level=AlertLevel.LOW,
    source="scheduled_trainer",
    metadata={"date": "2024-01-15", "expected_chunks": ">0"}
)
```

### System Resource Alerts

**Database Connection Issues**:

```python
alert_manager.alert(
    message="ArangoDB connection timeout",
    level=AlertLevel.HIGH, 
    source="arango_client",
    metadata={"timeout_duration": 30, "retry_count": 3}
)
```

**Storage Space Warnings**:

```python
alert_manager.alert(
    message="Training log directory approaching capacity",
    level=AlertLevel.MEDIUM,
    source="storage_monitor", 
    metadata={"free_space_gb": 2.1, "threshold_gb": 5.0}
)
```

## 📈 Monitoring and Analysis

### Alert History Analysis

```python
# Get alert trends
alert_stats = alert_manager.get_alert_statistics(days=7)
print(f"High priority alerts this week: {alert_stats['HIGH']}")

# Check for recurring issues
recurring = alert_manager.find_recurring_alerts(
    pattern="chunking.*failed",
    time_window_hours=24
)
```

### Integration with Status Monitoring

The alert system integrates with system status monitoring:

```bash
# Check for recent alerts
python scripts/check_training_status.py --alerts

# Get alert summary in JSON
python scripts/check_training_status.py --json | jq '.alerts'
```

## 🛠️ Development and Extension

### Adding Custom Alert Sources

```python
from src.alerts import AlertManager, AlertLevel

class CustomPipelineStage:
    def __init__(self):
        self.alert_manager = AlertManager()
    
    def process_data(self, data):
        try:
            result = self.complex_processing(data)
            
            # Success alert (optional for debugging)
            self.alert_manager.alert(
                message=f"Successfully processed {len(data)} items",
                level=AlertLevel.LOW,
                source="custom_stage"
            )
            return result
            
        except Exception as e:
            # Error alert
            self.alert_manager.alert(
                message=f"Processing failed: {str(e)}",
                level=AlertLevel.HIGH,
                source="custom_stage",
                metadata={"error_type": type(e).__name__, "data_size": len(data)}
            )
            raise
```

### Custom Alert Processors

Extend the alert system with custom notification channels:

```python
class SlackAlertNotifier:
    def __init__(self, webhook_url):
        self.webhook_url = webhook_url
    
    def send_alert(self, alert):
        if alert.level == AlertLevel.HIGH:
            # Send to Slack for critical alerts
            self.send_slack_message(alert.message)
```

## 🔍 Troubleshooting

### Common Issues

**Alert Storage Full**:

- Check `alert_dir` disk space
- Configure alert retention policies
- Archive old alerts periodically

**Missing Alerts**:

- Verify `AlertManager` initialization in components
- Check alert level configuration
- Ensure proper error handling in pipeline stages

**Performance Impact**:

- Use appropriate alert levels (avoid excessive LOW level alerts)
- Configure alert sampling for high-frequency events
- Monitor alert storage growth

### Debugging

Enable detailed alert logging:

```python
import logging
logging.getLogger('src.alerts').setLevel(logging.DEBUG)
```

Check alert file integrity:

```bash
# Validate alert JSON structure
python -c "import json; json.load(open('./alerts/alerts.json'))"
```

## 📚 Related Documentation

- **Training Monitoring**: See `isne/bootstrap/isne_bootstrap.md` for training-specific alerts
- **Pipeline Integration**: See `orchestration/README.md` for pipeline alert integration
- **Status Monitoring**: See `scripts/check_training_status.py` for automated alert checking
- **Configuration**: See `config/README.md` for alert configuration options

## 🎯 Best Practices

1. **Use appropriate alert levels** - Reserve HIGH for actionable issues
2. **Include diagnostic metadata** - Help debugging with context information
3. **Monitor alert trends** - Regular analysis prevents issue accumulation
4. **Test alert integration** - Verify alerts work in all pipeline stages
5. **Document alert meanings** - Clear alert messages aid troubleshooting
