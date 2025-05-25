"""
Alert management system for HADES-PathRAG.

This module provides a central alert management system for detecting and 
handling various types of alerts throughout the HADES-PathRAG system.
"""

import json
import logging
import os
import time
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Set, Callable
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import traceback

# Configure logging
logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    """Alert severity levels."""
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()

class Alert:
    """
    Alert class representing a single alert.
    
    This class encapsulates all information about an alert, including
    its message, level, source, and timestamp.
    """
    
    def __init__(
        self,
        message: str,
        level: AlertLevel,
        source: str,
        context: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None
    ):
        """
        Initialize an alert.
        
        Args:
            message: The alert message
            level: Alert severity level
            source: Component or module that generated the alert
            context: Additional context data related to the alert
            timestamp: Alert timestamp (defaults to current time)
        """
        self.message = message
        self.level = level
        self.source = source
        self.context = context or {}
        self.timestamp = timestamp or time.time()
        self.id = f"{int(self.timestamp)}_{hash(message) % 10000:04d}"
        
    def __str__(self) -> str:
        """Return string representation of the alert."""
        level_name = self.level.name
        time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.timestamp))
        return f"[{time_str}] {level_name}: {self.message} (Source: {self.source})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary for serialization."""
        return {
            "id": self.id,
            "message": self.message,
            "level": self.level.name,
            "source": self.source,
            "timestamp": self.timestamp,
            "timestamp_formatted": time.strftime("%Y-%m-%d %H:%M:%S", 
                                               time.localtime(self.timestamp)),
            "context": self.context
        }

class AlertManager:
    """
    Central manager for handling alerts throughout the system.
    
    This class provides a unified interface for generating, logging,
    and dispatching alerts to various handlers (log files, email, etc.).
    """
    
    def __init__(
        self,
        alert_dir: str = "./alerts",
        min_level: AlertLevel = AlertLevel.LOW,
        handlers: Optional[Dict[str, Callable]] = None,
        email_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the alert manager.
        
        Args:
            alert_dir: Directory to store alert logs
            min_level: Minimum level of alerts to process
            handlers: Custom alert handlers {name: handler_function}
            email_config: Email configuration for sending alerts
        """
        self.alert_dir = Path(alert_dir)
        self.min_level = min_level
        self.handlers = handlers or {}
        self.email_config = email_config or {}
        self.alerts: List[Alert] = []
        self.alert_counts: Dict[AlertLevel, int] = {
            level: 0 for level in AlertLevel
        }
        
        # Set up alert directory
        self.alert_dir.mkdir(parents=True, exist_ok=True)
        
        # Register default handlers
        if "log" not in self.handlers:
            self.handlers["log"] = self._log_alert
        if "file" not in self.handlers:
            self.handlers["file"] = self._file_alert
            
        # Register email handler if configured
        if self.email_config and "email" not in self.handlers:
            self.handlers["email"] = self._email_alert
    
    def alert(
        self,
        message: str,
        level: Union[AlertLevel, str],
        source: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Alert:
        """
        Create and process an alert.
        
        Args:
            message: Alert message
            level: Alert level (AlertLevel enum or string name)
            source: Component or module that generated the alert
            context: Additional context data
            
        Returns:
            The created Alert object
        """
        # Convert string level to enum if needed
        if isinstance(level, str):
            try:
                level = AlertLevel[level.upper()]
            except KeyError:
                logger.warning(f"Invalid alert level: {level}, defaulting to MEDIUM")
                level = AlertLevel.MEDIUM
        
        # Create alert
        alert = Alert(message, level, source, context)
        
        # Only process if alert level meets minimum threshold
        if alert.level.value >= self.min_level.value:
            # Add to internal list
            self.alerts.append(alert)
            self.alert_counts[alert.level] += 1
            
            # Process through handlers
            for handler_name, handler_func in self.handlers.items():
                try:
                    handler_func(alert)
                except Exception as e:
                    logger.error(f"Error in alert handler '{handler_name}': {e}")
                    logger.debug(traceback.format_exc())
        
        return alert
    
    def _log_alert(self, alert: Alert) -> None:
        """
        Log alert to the standard logging system.
        
        Args:
            alert: Alert to log
        """
        log_message = f"{alert.source}: {alert.message}"
        
        if alert.level in (AlertLevel.CRITICAL, AlertLevel.HIGH):
            logger.error(f"🚨 {log_message}")
        elif alert.level == AlertLevel.MEDIUM:
            logger.warning(f"⚠️ {log_message}")
        else:
            logger.info(f"ℹ️ {log_message}")
    
    def _file_alert(self, alert: Alert) -> None:
        """
        Log alert to a file.
        
        Args:
            alert: Alert to log
        """
        # Determine log file based on alert level
        if alert.level in (AlertLevel.CRITICAL, AlertLevel.HIGH):
            log_file = self.alert_dir / "critical_alerts.log"
        else:
            log_file = self.alert_dir / "alerts.log"
        
        # Write to log file
        try:
            with open(log_file, "a") as f:
                f.write(f"{str(alert)}\n")
        except Exception as e:
            logger.error(f"Error writing to alert log file: {e}")
            
        # Also write to JSON for structured access
        try:
            json_file = self.alert_dir / "alerts.json"
            alerts_data = []
            
            # Read existing data if file exists
            if json_file.exists():
                try:
                    with open(json_file, "r") as f:
                        alerts_data = json.load(f)
                except json.JSONDecodeError:
                    # If file is corrupted, start fresh
                    alerts_data = []
            
            # Add new alert
            alerts_data.append(alert.to_dict())
            
            # Write back (limit to most recent 1000 alerts)
            with open(json_file, "w") as f:
                json.dump(alerts_data[-1000:], f, indent=2)
        except Exception as e:
            logger.error(f"Error writing to JSON alert log: {e}")
    
    def _email_alert(self, alert: Alert) -> None:
        """
        Send alert via email.
        
        Args:
            alert: Alert to send
        """
        # Only send emails for high/critical alerts by default
        if alert.level not in (AlertLevel.HIGH, AlertLevel.CRITICAL):
            return
            
        # Check if email config is available
        if not self.email_config:
            logger.warning("Email alert handler called but no email configuration provided")
            return
            
        # Extract email config
        smtp_server = self.email_config.get("smtp_server")
        smtp_port = self.email_config.get("smtp_port", 587)
        username = self.email_config.get("username")
        password = self.email_config.get("password")
        from_addr = self.email_config.get("from_addr")
        to_addrs = self.email_config.get("to_addrs", [])
        
        if not all([smtp_server, username, password, from_addr, to_addrs]):
            logger.warning("Incomplete email configuration, cannot send alert email")
            return
            
        try:
            # Create message
            msg = MIMEMultipart()
            msg["From"] = from_addr
            msg["To"] = ", ".join(to_addrs)
            msg["Subject"] = f"HADES-PathRAG Alert: {alert.level.name} - {alert.source}"
            
            # Create email body
            body = f"""
            <html>
            <body>
                <h2>HADES-PathRAG Alert</h2>
                <p><strong>Level:</strong> {alert.level.name}</p>
                <p><strong>Source:</strong> {alert.source}</p>
                <p><strong>Time:</strong> {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(alert.timestamp))}</p>
                <p><strong>Message:</strong> {alert.message}</p>
                
                {self._format_context_for_email(alert.context) if alert.context else ""}
            </body>
            </html>
            """
            
            msg.attach(MIMEText(body, "html"))
            
            # Send email
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(username, password)
                server.send_message(msg)
                
            logger.info(f"Alert email sent for {alert.level.name} alert")
        except Exception as e:
            logger.error(f"Error sending alert email: {e}")
    
    def _format_context_for_email(self, context: Dict[str, Any]) -> str:
        """Format context data for email display."""
        if not context:
            return ""
            
        html = "<h3>Additional Context:</h3><ul>"
        for key, value in context.items():
            if isinstance(value, dict):
                html += f"<li><strong>{key}:</strong><ul>"
                for sub_key, sub_value in value.items():
                    html += f"<li><strong>{sub_key}:</strong> {sub_value}</li>"
                html += "</ul></li>"
            else:
                html += f"<li><strong>{key}:</strong> {value}</li>"
        html += "</ul>"
        
        return html
    
    def get_alerts(
        self,
        min_level: Optional[AlertLevel] = None,
        source: Optional[str] = None,
        limit: int = 100
    ) -> List[Alert]:
        """
        Get filtered alerts.
        
        Args:
            min_level: Minimum alert level to include
            source: Filter by source
            limit: Maximum number of alerts to return
            
        Returns:
            Filtered list of alerts
        """
        filtered = self.alerts
        
        if min_level:
            filtered = [a for a in filtered if a.level.value >= min_level.value]
            
        if source:
            filtered = [a for a in filtered if a.source == source]
            
        # Return most recent alerts first
        return sorted(filtered, key=lambda a: a.timestamp, reverse=True)[:limit]
    
    def get_alert_stats(self) -> Dict[str, int]:
        """
        Get alert statistics.
        
        Returns:
            Dictionary with alert counts by level
        """
        return {level.name: count for level, count in self.alert_counts.items()}
    
    def register_handler(self, name: str, handler: Callable[[Alert], None]) -> None:
        """
        Register a custom alert handler.
        
        Args:
            name: Handler name
            handler: Handler function taking an Alert argument
        """
        self.handlers[name] = handler
        logger.info(f"Registered alert handler: {name}")
    
    def configure_email(self, config: Dict[str, Any]) -> None:
        """
        Configure email alerts.
        
        Args:
            config: Email configuration dictionary with keys:
                   smtp_server, smtp_port, username, password,
                   from_addr, to_addrs
        """
        required_keys = ["smtp_server", "username", "password", "from_addr", "to_addrs"]
        missing = [k for k in required_keys if k not in config]
        
        if missing:
            logger.error(f"Missing required email config keys: {', '.join(missing)}")
            return
            
        self.email_config = config
        
        # Register email handler if not already registered
        if "email" not in self.handlers:
            self.handlers["email"] = self._email_alert
            
        logger.info("Email alert configuration updated")
