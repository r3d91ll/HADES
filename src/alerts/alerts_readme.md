# Alert System Module

## Overview

The Alert System provides a centralized framework for generating, tracking, and managing alerts throughout the HADES-PathRAG system. It enables systematic reporting of validation discrepancies, data quality issues, and other critical events that might affect the integrity of the embedding pipeline.

## Components

### Alert Class

The `Alert` class encapsulates a single alert with the following properties:

- **Message**: Description of the alert
- **Level**: Severity level (LOW, MEDIUM, HIGH, CRITICAL)
- **Source**: Component or module that generated the alert
- **Context**: Additional contextual data
- **Timestamp**: When the alert was generated

### AlertManager Class

The `AlertManager` serves as the central handler for alerts with capabilities for:

- Generating alerts with appropriate severity levels
- Logging alerts to various outputs (console, files)
- Sending critical alerts via email
- Filtering and retrieving alert statistics
- Custom alert handler registration

## Integration Points

### With ISNE Pipeline

The alert system is integrated with the ISNE pipeline to provide:

1. Real-time validation alerts during embedding generation
2. Critical alerts for embedding consistency issues
3. Performance monitoring alerts

### With Validation Module

Tight integration with the validation module to:

1. Convert validation discrepancies into appropriately leveled alerts
2. Track validation statistics across pipeline runs
3. Provide audit trails for data quality issues

## Usage Examples

### Basic Alert Generation

```python
from src.alerts import AlertManager, AlertLevel

# Initialize the alert manager
alert_mgr = AlertManager(alert_dir="./alerts")

# Generate an alert
alert_mgr.alert(
    message="Missing base embeddings for 10 chunks",
    level=AlertLevel.MEDIUM,
    source="embedding_validator",
    context={"document_id": "doc123", "affected_chunks": [1, 2, 3]}
)
```

### Email Configuration

```python
# Configure email notifications
alert_mgr.configure_email({
    "smtp_server": "smtp.example.com",
    "smtp_port": 587,
    "username": "alerts@example.com",
    "password": "****",  # Use environment variables in production
    "from_addr": "alerts@example.com",
    "to_addrs": ["admin@example.com", "team@example.com"]
})
```

### Custom Alert Handler

```python
# Define a custom alert handler
def slack_alert_handler(alert):
    # Logic to send alert to Slack
    pass

# Register the custom handler
alert_mgr.register_handler("slack", slack_alert_handler)
```

## Alert Levels

The system defines four alert levels:

1. **LOW**: Informational alerts, no action required
2. **MEDIUM**: Warnings that may require attention
3. **HIGH**: Serious issues that need prompt attention
4. **CRITICAL**: Severe problems requiring immediate intervention

## Performance Considerations

- The alert system is designed to be lightweight with minimal impact on pipeline performance
- Email notifications are only sent for HIGH and CRITICAL alerts by default
- Alert processing is asynchronous where possible

## Testing

The alert system includes:

- Unit tests covering all alert functionality
- Integration tests demonstrating system behavior in the ISNE pipeline
- Performance benchmarks to ensure minimal overhead

## Future Enhancements

- Webhook integration for third-party notification systems
- Alert aggregation to prevent alert storms
- Interactive dashboard for alert monitoring
- Automated response actions for common alert patterns
