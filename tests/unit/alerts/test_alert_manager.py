"""
Unit tests for the alert manager module.

This test module verifies the functionality of the alert system, including
alert creation, management, and dispatching to various handlers.
"""

import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.alerts.alert_manager import Alert, AlertLevel, AlertManager


class TestAlert(unittest.TestCase):
    """Test cases for the Alert class."""

    def test_alert_initialization(self):
        """Test that an alert is correctly initialized with all properties."""
        # Arrange
        message = "Test alert message"
        level = AlertLevel.MEDIUM
        source = "test_module"
        context = {"key": "value"}
        
        # Act
        alert = Alert(message, level, source, context)
        
        # Assert
        self.assertEqual(alert.message, message)
        self.assertEqual(alert.level, level)
        self.assertEqual(alert.source, source)
        self.assertEqual(alert.context, context)
        self.assertIsNotNone(alert.timestamp)
        self.assertIsNotNone(alert.id)
    
    def test_string_representation(self):
        """Test that the string representation of an alert is correct."""
        # Arrange
        message = "Test alert message"
        level = AlertLevel.HIGH
        source = "test_module"
        
        # Act
        alert = Alert(message, level, source)
        alert_str = str(alert)
        
        # Assert
        self.assertIn(message, alert_str)
        self.assertIn(level.name, alert_str)
        self.assertIn(source, alert_str)
    
    def test_to_dict_conversion(self):
        """Test that an alert correctly converts to a dictionary."""
        # Arrange
        message = "Test alert message"
        level = AlertLevel.LOW
        source = "test_module"
        context = {"key": "value"}
        
        # Act
        alert = Alert(message, level, source, context)
        alert_dict = alert.to_dict()
        
        # Assert
        self.assertEqual(alert_dict["message"], message)
        self.assertEqual(alert_dict["level"], level.name)
        self.assertEqual(alert_dict["source"], source)
        self.assertEqual(alert_dict["context"], context)
        self.assertIn("timestamp", alert_dict)
        self.assertIn("timestamp_formatted", alert_dict)
        self.assertIn("id", alert_dict)


class TestAlertManager(unittest.TestCase):
    """Test cases for the AlertManager class."""
    
    def setUp(self):
        """Set up a temporary directory for alert logs."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.alert_dir = Path(self.temp_dir.name)
    
    def tearDown(self):
        """Clean up the temporary directory."""
        self.temp_dir.cleanup()
    
    def test_alert_manager_initialization(self):
        """Test that the AlertManager is correctly initialized."""
        # Act
        manager = AlertManager(alert_dir=self.alert_dir)
        
        # Assert
        self.assertEqual(manager.alert_dir, Path(self.alert_dir))
        self.assertEqual(manager.min_level, AlertLevel.LOW)
        self.assertIn("log", manager.handlers)
        self.assertIn("file", manager.handlers)
        self.assertEqual(len(manager.alerts), 0)
        self.assertTrue(Path(self.alert_dir).exists())
    
    def test_alert_creation(self):
        """Test that an alert is correctly created and processed."""
        # Arrange
        manager = AlertManager(alert_dir=self.alert_dir)
        manager._log_alert = MagicMock()
        manager._file_alert = MagicMock()
        
        # Act
        alert = manager.alert(
            message="Test alert",
            level=AlertLevel.MEDIUM,
            source="test_source"
        )
        
        # Assert
        self.assertEqual(len(manager.alerts), 1)
        self.assertEqual(manager.alerts[0], alert)
        self.assertEqual(manager.alert_counts[AlertLevel.MEDIUM], 1)
        manager._log_alert.assert_called_once()
        manager._file_alert.assert_called_once()
    
    def test_alert_level_filtering(self):
        """Test that alerts are filtered by minimum level."""
        # Arrange
        manager = AlertManager(
            alert_dir=self.alert_dir,
            min_level=AlertLevel.MEDIUM
        )
        manager._log_alert = MagicMock()
        manager._file_alert = MagicMock()
        
        # Act
        manager.alert("Low alert", AlertLevel.LOW, "test_source")
        manager.alert("Medium alert", AlertLevel.MEDIUM, "test_source")
        
        # Assert
        self.assertEqual(len(manager.alerts), 1)
        self.assertEqual(manager.alerts[0].message, "Medium alert")
        self.assertEqual(manager.alert_counts[AlertLevel.LOW], 0)
        self.assertEqual(manager.alert_counts[AlertLevel.MEDIUM], 1)
        manager._log_alert.assert_called_once()
        manager._file_alert.assert_called_once()
    
    def test_string_level_conversion(self):
        """Test that string alert levels are correctly converted."""
        # Arrange
        manager = AlertManager(alert_dir=self.alert_dir)
        manager._log_alert = MagicMock()
        manager._file_alert = MagicMock()
        
        # Act
        alert = manager.alert(
            message="Test alert",
            level="HIGH",  # String level
            source="test_source"
        )
        
        # Assert
        self.assertEqual(alert.level, AlertLevel.HIGH)
        
        # Test invalid level defaults to MEDIUM
        with self.assertLogs(level='WARNING'):
            alert = manager.alert(
                message="Test alert",
                level="INVALID",  # Invalid level
                source="test_source"
            )
            self.assertEqual(alert.level, AlertLevel.MEDIUM)
    
    def test_file_alert_handler(self):
        """Test that alerts are correctly written to log files."""
        # Arrange
        manager = AlertManager(alert_dir=self.alert_dir)
        critical_log = self.alert_dir / "critical_alerts.log"
        normal_log = self.alert_dir / "alerts.log"
        json_log = self.alert_dir / "alerts.json"
        
        # Act
        # Create alerts of different levels
        manager.alert("Low alert", AlertLevel.LOW, "test_source")
        manager.alert("Critical alert", AlertLevel.CRITICAL, "test_source")
        
        # Assert
        # Check log files exist
        self.assertTrue(critical_log.exists())
        self.assertTrue(normal_log.exists())
        self.assertTrue(json_log.exists())
        
        # Check critical log content
        with open(critical_log, "r") as f:
            content = f.read()
            self.assertIn("CRITICAL", content)
            self.assertIn("Critical alert", content)
        
        # Check normal log content
        with open(normal_log, "r") as f:
            content = f.read()
            self.assertIn("LOW", content)
            self.assertIn("Low alert", content)
        
        # Check JSON log content
        with open(json_log, "r") as f:
            alerts_data = json.load(f)
            self.assertEqual(len(alerts_data), 2)
            self.assertEqual(alerts_data[0]["level"], "LOW")
            self.assertEqual(alerts_data[1]["level"], "CRITICAL")
    
    def test_email_alert_handler(self):
        """Test that email alerts are correctly configured and sent."""
        # Arrange
        with patch("smtplib.SMTP") as mock_smtp:
            # Mock SMTP instance
            mock_server = MagicMock()
            mock_smtp.return_value.__enter__.return_value = mock_server
            
            # Setup manager with email config
            manager = AlertManager(
                alert_dir=self.alert_dir,
                email_config={
                    "smtp_server": "smtp.example.com",
                    "smtp_port": 587,
                    "username": "test_user",
                    "password": "test_pass",
                    "from_addr": "from@example.com",
                    "to_addrs": ["to@example.com"]
                }
            )
            
            # Act
            # Only HIGH/CRITICAL alerts should trigger emails
            manager.alert("Low alert", AlertLevel.LOW, "test_source")
            manager.alert("Critical alert", AlertLevel.CRITICAL, "test_source")
            
            # Assert
            # Check email was sent once (only for CRITICAL)
            mock_server.send_message.assert_called_once()
            
            # Check SMTP connection was properly configured
            mock_smtp.assert_called_once_with("smtp.example.com", 587)
            mock_server.starttls.assert_called_once()
            mock_server.login.assert_called_once_with("test_user", "test_pass")
    
    def test_custom_handler_registration(self):
        """Test that custom alert handlers can be registered and called."""
        # Arrange
        manager = AlertManager(alert_dir=self.alert_dir)
        custom_handler = MagicMock()
        
        # Act
        manager.register_handler("custom", custom_handler)
        manager.alert("Test alert", AlertLevel.MEDIUM, "test_source")
        
        # Assert
        custom_handler.assert_called_once()
        self.assertIn("custom", manager.handlers)
    
    def test_get_filtered_alerts(self):
        """Test that alerts can be filtered by level and source."""
        # Arrange
        manager = AlertManager(alert_dir=self.alert_dir)
        
        # Create various alerts
        manager.alert("Low alert 1", AlertLevel.LOW, "source_a")
        manager.alert("Medium alert", AlertLevel.MEDIUM, "source_b")
        manager.alert("High alert", AlertLevel.HIGH, "source_a")
        manager.alert("Low alert 2", AlertLevel.LOW, "source_a")
        
        # Act & Assert
        # Filter by level
        high_alerts = manager.get_alerts(min_level=AlertLevel.HIGH)
        self.assertEqual(len(high_alerts), 1)
        self.assertEqual(high_alerts[0].level, AlertLevel.HIGH)
        
        # Filter by source
        source_a_alerts = manager.get_alerts(source="source_a")
        self.assertEqual(len(source_a_alerts), 3)
        
        # Combined filter
        filtered_alerts = manager.get_alerts(min_level=AlertLevel.MEDIUM, source="source_b")
        self.assertEqual(len(filtered_alerts), 1)
        self.assertEqual(filtered_alerts[0].message, "Medium alert")
    
    def test_get_alert_stats(self):
        """Test that alert statistics are correctly calculated."""
        # Arrange
        manager = AlertManager(alert_dir=self.alert_dir)
        
        # Create various alerts
        manager.alert("Low alert 1", AlertLevel.LOW, "test_source")
        manager.alert("Low alert 2", AlertLevel.LOW, "test_source")
        manager.alert("Medium alert", AlertLevel.MEDIUM, "test_source")
        manager.alert("High alert", AlertLevel.HIGH, "test_source")
        manager.alert("Critical alert", AlertLevel.CRITICAL, "test_source")
        
        # Act
        stats = manager.get_alert_stats()
        
        # Assert
        self.assertEqual(stats["LOW"], 2)
        self.assertEqual(stats["MEDIUM"], 1)
        self.assertEqual(stats["HIGH"], 1)
        self.assertEqual(stats["CRITICAL"], 1)
    
    def test_handler_exception_handling(self):
        """Test that exceptions in handlers are caught and don't affect other handlers."""
        # Arrange
        manager = AlertManager(alert_dir=self.alert_dir)
        
        # Create handlers - one that works, one that raises exception
        working_handler = MagicMock()
        failing_handler = MagicMock(side_effect=Exception("Test exception"))
        
        manager.register_handler("working", working_handler)
        manager.register_handler("failing", failing_handler)
        
        # Act - this should log the exception but not raise it
        with self.assertLogs(level='ERROR'):
            manager.alert("Test alert", AlertLevel.MEDIUM, "test_source")
        
        # Assert
        working_handler.assert_called_once()  # This should still have been called
        failing_handler.assert_called_once()  # This was called but raised exception
        self.assertEqual(len(manager.alerts), 1)  # Alert should still be recorded
    
    def test_email_configuration_validation(self):
        """Test that email configuration is properly validated."""
        # Arrange
        manager = AlertManager(alert_dir=self.alert_dir)
        
        # Act & Assert
        # Invalid config missing required keys
        with self.assertLogs(level='ERROR'):
            manager.configure_email({
                "smtp_server": "smtp.example.com",
                # Missing other required keys
            })
            # Email config should not be updated
            self.assertEqual(manager.email_config, {})
        
        # Valid config
        manager.configure_email({
            "smtp_server": "smtp.example.com",
            "username": "test_user",
            "password": "test_pass",
            "from_addr": "from@example.com",
            "to_addrs": ["to@example.com"]
        })
        
        # Assert config was updated
        self.assertEqual(manager.email_config["smtp_server"], "smtp.example.com")
        self.assertIn("email", manager.handlers)


if __name__ == "__main__":
    unittest.main()
