# Alert System for HADES-PathRAG

The HADES-PathRAG alert system is designed to monitor and report on critical data quality issues during document processing, embedding generation, and ISNE application. This document describes the alert system architecture, integration with the ISNE pipeline, and usage examples.

## Overview

The alert system provides real-time monitoring of document processing and embedding generation, with configurable alert levels (LOW, MEDIUM, HIGH, CRITICAL) and notification channels (file logging and email). The system is designed to:

1. Detect and report validation issues in real-time
2. Track discrepancies in embedding generation
3. Provide contextual information for troubleshooting
4. Support multiple notification channels

## Components

### AlertManager

The `AlertManager` class is the core component responsible for:

- Managing alert thresholds and notification rules
- Logging alerts to files with timestamps and context
- Sending email notifications for critical alerts
- Tracking alert history

### Alert Levels

The system defines four alert levels:

- **LOW**: Informational messages, minor issues
- **MEDIUM**: Issues that should be reviewed but don't affect critical functionality
- **HIGH**: Serious issues that may affect system functionality
- **CRITICAL**: Critical issues that require immediate attention

### Integration with ISNE Pipeline

The alert system integrates with the ISNE pipeline at key validation points:

1. **Pre-ISNE Validation**: Validates documents before ISNE application, checking for:
   - Missing base embeddings
   - Inconsistent embedding dimensions
   - Document structure issues

2. **Post-ISNE Validation**: Validates documents after ISNE application, checking for:
   - Missing ISNE embeddings
   - Embedding dimension mismatches
   - Unexpected changes in document structure

## Usage Examples

### Basic Usage

```python
from src.alerts import AlertManager, AlertLevel

# Initialize alert manager
alert_manager = AlertManager(
    log_dir="./alert_logs",
    alert_threshold=AlertLevel.MEDIUM,
    email_config={
        "smtp_server": "smtp.example.com",
        "smtp_port": 587,
        "sender_email": "alerts@example.com",
        "recipient_emails": ["admin@example.com"],
        "username": "alerts@example.com",
        "password": "password"
    }
)

# Create an alert
alert_manager.alert(
    message="Missing base embeddings detected",
    level=AlertLevel.HIGH,
    source="isne_pipeline",
    context={
        "missing_count": 5,
        "affected_documents": ["doc1", "doc2"]
    }
)
```

### Integration with ISNE Pipeline

See the script `scripts/run_isne_with_alerts.py` for a complete example of integrating the alert system with the ISNE pipeline.

### Running Tests

To test the alert system integration:

```bash
python tests/integration/alert_isne_test.py --output-dir ./alert_test_output --alert-threshold MEDIUM
```

## Configuration

### Alert Thresholds

Configure the minimum alert level to be recorded:

```python
alert_manager = AlertManager(
    log_dir="./alert_logs",
    alert_threshold=AlertLevel.HIGH  # Only HIGH and CRITICAL alerts will be recorded
)
```

### Email Notifications

To enable email notifications for critical alerts:

```python
email_config = {
    "smtp_server": "smtp.example.com",
    "smtp_port": 587,
    "sender_email": "alerts@example.com",
    "recipient_emails": ["admin@example.com"],
    "username": "alerts@example.com",
    "password": "password"
}

alert_manager = AlertManager(
    log_dir="./alert_logs",
    alert_threshold=AlertLevel.MEDIUM,
    email_config=email_config
)
```

## Best Practices

1. **Set Appropriate Thresholds**: In production, set appropriate alert thresholds to avoid alert fatigue
2. **Provide Detailed Context**: Always include detailed context with alerts to aid in troubleshooting
3. **Review Alert Logs Regularly**: Establish a process for regularly reviewing alert logs
4. **Adjust Validation Criteria**: Fine-tune validation criteria based on alert patterns
5. **Secure Email Credentials**: Use environment variables or secure storage for email credentials

## Future Enhancements

1. **Additional Notification Channels**: Slack, Microsoft Teams, or SMS notifications
2. **Alert Aggregation**: Group similar alerts to reduce noise
3. **Alert Dashboard**: Web-based dashboard for monitoring alerts
4. **Alert Analytics**: Track and analyze alert patterns over time
