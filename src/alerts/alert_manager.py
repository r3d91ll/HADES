"""
Alert management system for HADES-PathRAG.

This module provides a central alert management system for detecting and 
handling various types of alerts throughout the HADES-PathRAG system.
"""

import time
import json
import logging
import smtplib
from pathlib import Path
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional, Any, Callable, Union, cast

from src.types.alerts import AlertLevel, Alert

# Set up logging
logger = logging.getLogger(__name__)


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
        alert_level: AlertLevel
        if isinstance(level, str):
            try:
                alert_level = getattr(AlertLevel, level.upper())
            except AttributeError:
                logger.warning(f"Invalid alert level: {level}, using MEDIUM")
                alert_level = AlertLevel.MEDIUM
        else:
            alert_level = level
        
        # Create alert object
        alert = Alert(message, alert_level, source, context)
        
        # Skip processing if below minimum level
        if alert.level.value < self.min_level.value:
            return alert
        
        # Store alert
        self.alerts.append(alert)
        self.alert_counts[alert.level] += 1
        
        # Process through handlers
        for handler_name, handler_fn in self.handlers.items():
            try:
                handler_fn(alert)
            except Exception as e:
                logger.exception(f"Error in alert handler '{handler_name}': {e}")
        
        return alert
    
    def _log_alert(self, alert: Alert) -> None:
        """
        Log alert to the standard logging system.
        
        Args:
            alert: Alert to log
        """
        log_levels = {
            AlertLevel.LOW: logging.INFO,
            AlertLevel.MEDIUM: logging.WARNING,
            AlertLevel.HIGH: logging.ERROR,
            AlertLevel.CRITICAL: logging.CRITICAL
        }
        
        log_level = log_levels.get(alert.level, logging.WARNING)
        logger.log(log_level, str(alert))
    
    def _file_alert(self, alert: Alert) -> None:
        """
        Log alert to a file.
        
        Args:
            alert: Alert to log
        """
        # Create level-specific directory
        level_dir = self.alert_dir / alert.level.name.lower()
        level_dir.mkdir(exist_ok=True)
        
        # Write alert to JSON file
        alert_file = level_dir / f"{alert.id}.json"
        with open(alert_file, "w") as f:
            json.dump(alert.to_dict(), f, indent=2)
    
    def _email_alert(self, alert: Alert) -> None:
        """
        Send alert via email.
        
        Args:
            alert: Alert to send
        """
        # Only send email for higher severity alerts
        if alert.level.value < AlertLevel.HIGH.value:
            return
        
        # Extract email configuration
        smtp_server = self.email_config.get("smtp_server")
        smtp_port = self.email_config.get("smtp_port", 587)
        username = self.email_config.get("username")
        password = self.email_config.get("password")
        from_addr = self.email_config.get("from_addr")
        to_addrs = self.email_config.get("to_addrs", [])
        
        # Ensure all required configuration is present and properly typed
        if not all([smtp_server, username, password, from_addr, to_addrs]):
            logger.warning("Incomplete email configuration, cannot send alert email")
            return
            
        # Ensure we have proper string types for SMTP connection
        smtp_server_str = cast(str, smtp_server)
        username_str = cast(str, username)
        password_str = cast(str, password)
        from_addr_str = cast(str, from_addr)
        to_addrs_list = cast(List[str], to_addrs)
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg["From"] = from_addr_str
            msg["To"] = ", ".join(to_addrs_list)
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
            with smtplib.SMTP(smtp_server_str, smtp_port) as server:
                server.starttls()
                server.login(username_str, password_str)
                server.send_message(msg)
                
            logger.info(f"Alert email sent for {alert.level.name} alert")
            
        except Exception as e:
            logger.exception(f"Failed to send alert email: {e}")
    
    def _format_context_for_email(self, context: Dict[str, Any]) -> str:
        """Format context data for email display."""
        if not context:
            return ""
            
        html = "<h3>Additional Context</h3><table border='1' cellpadding='5'>"
        for key, value in context.items():
            # Format value based on type
            if isinstance(value, dict):
                formatted_value = "<pre>" + json.dumps(value, indent=2) + "</pre>"
            elif isinstance(value, (list, tuple)):
                formatted_value = "<ul>" + "".join(f"<li>{item}</li>" for item in value) + "</ul>"
            else:
                formatted_value = str(value)
                
            html += f"<tr><td><strong>{key}</strong></td><td>{formatted_value}</td></tr>"
            
        html += "</table>"
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
        # Apply filters
        filtered = self.alerts
        
        if min_level:
            filtered = [a for a in filtered if a.level.value >= min_level.value]
            
        if source:
            filtered = [a for a in filtered if a.source == source]
            
        # Return latest alerts first, up to limit
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
        self.email_config.update(config)
        
        # Register email handler if not already registered
        if "email" not in self.handlers:
            self.handlers["email"] = self._email_alert
            
        logger.info("Email alert configuration updated")
