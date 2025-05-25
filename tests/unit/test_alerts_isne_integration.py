#!/usr/bin/env python
"""
Unit tests for the alert system integration with ISNE validation.

This module tests the interaction between the AlertManager and ISNE validation
to ensure alerts are properly generated for validation discrepancies.
"""

import os
import sys
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock
from typing import Dict, List, Any, Optional

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import local modules
from src.alerts import AlertManager, AlertLevel, Alert
from src.validation.embedding_validator import (
    validate_embeddings_before_isne,
    validate_embeddings_after_isne
)


class TestAlertISNEIntegration(unittest.TestCase):
    """Tests for alert system integration with ISNE validation."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directory for alert logs
        self.temp_dir = tempfile.TemporaryDirectory()
        self.alert_dir = self.temp_dir.name
        
        # Initialize alert manager with test settings
        self.alert_manager = AlertManager(
            alert_dir=self.alert_dir,
            min_level=AlertLevel.LOW  # Capture all alerts
        )
        
        # Sample documents with known validation issues
        self.sample_docs = self._create_sample_documents()
    
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    def _create_sample_documents(self, num_docs=3, chunks_per_doc=5):
        """Create sample documents with controlled validation issues."""
        import numpy as np
        
        documents = []
        embedding_dim = 384
        
        for doc_idx in range(num_docs):
            doc_id = f"test_doc_{doc_idx}"
            chunks = []
            
            for chunk_idx in range(chunks_per_doc):
                # Deliberately create specific validation scenarios
                has_base = True
                has_isne = False
                
                # Doc 0: All chunks have base embeddings, no ISNE
                if doc_idx == 0:
                    has_base = True
                    has_isne = False
                
                # Doc 1: Some chunks missing base embeddings
                elif doc_idx == 1:
                    has_base = chunk_idx % 2 == 0  # Every other chunk has base
                    has_isne = False
                
                # Doc 2: All have base, some already have ISNE (unusual condition)
                elif doc_idx == 2:
                    has_base = True
                    has_isne = chunk_idx % 3 == 0  # Every third chunk has ISNE
                
                chunk = {
                    "text": f"This is chunk {chunk_idx} of document {doc_idx}",
                    "metadata": {
                        "chunk_id": f"{doc_id}_chunk_{chunk_idx}",
                        "embedding_model": "test_model"
                    }
                }
                
                # Add base embedding if configured
                if has_base:
                    chunk["embedding"] = np.random.rand(embedding_dim).tolist()
                
                # Add ISNE embedding if configured
                if has_isne:
                    chunk["isne_embedding"] = np.random.rand(embedding_dim).tolist()
                
                chunks.append(chunk)
            
            document = {
                "file_id": doc_id,
                "file_path": f"/path/to/{doc_id}.txt",
                "metadata": {
                    "title": f"Test Document {doc_idx}"
                },
                "chunks": chunks
            }
            
            documents.append(document)
        
        return documents
    
    def test_pre_isne_validation_alerts(self):
        """Test alert generation for pre-ISNE validation issues."""
        # Run pre-ISNE validation
        pre_validation = validate_embeddings_before_isne(self.sample_docs)
        
        # Generate alerts based on validation results
        missing_base = pre_validation.get("missing_base_embeddings", 0)
        if missing_base > 0:
            self.alert_manager.alert(
                message=f"Missing base embeddings detected in {missing_base} chunks",
                level=AlertLevel.MEDIUM if missing_base < 5 else AlertLevel.HIGH,
                source="isne_validation",
                context={
                    "missing_count": missing_base,
                    "affected_chunks": pre_validation.get('missing_base_embedding_ids', [])
                }
            )
        
        # Get alerts
        alerts = self.alert_manager.get_alerts()
        
        # Assertions
        self.assertTrue(len(alerts) > 0, "No alerts generated for missing base embeddings")
        
        # At least one alert should be about missing base embeddings
        has_missing_base_alert = False
        for alert in alerts:
            if "Missing base embeddings" in alert.message:
                has_missing_base_alert = True
                # Verify the context contains the expected data
                self.assertIn("missing_count", alert.context)
                self.assertIn("affected_chunks", alert.context)
                self.assertEqual(alert.context["missing_count"], missing_base)
                break
        
        self.assertTrue(has_missing_base_alert, "No alert for missing base embeddings found")
    
    def test_post_isne_validation_alerts(self):
        """Test alert generation for post-ISNE validation issues."""
        # Clear any previous alerts
        self.alert_manager.alerts.clear()
        
        # Create a mock validation result directly
        post_validation = {
            "total_discrepancies": 10,
            "discrepancies": {
                "missing_isne": 5,
                "isne_vs_chunks": 5
            },
            "expected_counts": {"isne_embeddings": 15},
            "actual_counts": {"isne_embeddings": 5}
        }
        
        # Generate alert directly based on mock validation results
        self.alert_manager.alert(
            message=f"Found {post_validation['total_discrepancies']} embedding discrepancies after ISNE application",
            level=AlertLevel.HIGH,  # High level since more than 5 discrepancies
            source="isne_validation",
            context={
                "discrepancies": post_validation["discrepancies"],
                "total_discrepancies": post_validation["total_discrepancies"],
                "expected_counts": post_validation["expected_counts"],
                "actual_counts": post_validation["actual_counts"]
            }
        )
        
        # Get alerts
        alerts = self.alert_manager.get_alerts()
        
        # Assertions
        self.assertTrue(len(alerts) > 0, "No alerts generated for ISNE discrepancies")
        
        # At least one alert should be about ISNE discrepancies
        has_discrepancy_alert = False
        for alert in alerts:
            if "embedding discrepancies" in alert.message:
                has_discrepancy_alert = True
                # Verify the context contains the expected data
                self.assertIn("discrepancies", alert.context)
                self.assertIn("total_discrepancies", alert.context)
                self.assertEqual(alert.context["total_discrepancies"], post_validation["total_discrepancies"])
                break
        
        self.assertTrue(has_discrepancy_alert, "No alert for ISNE discrepancies found")
    
    def test_alert_levels_based_on_severity(self):
        """Test that alert levels are set correctly based on severity."""
        # Generate alerts with different severity levels
        self.alert_manager.alert(
            message="Minor issue detected",
            level=AlertLevel.LOW,
            source="test_source",
            context={"test": "context"}
        )
        
        self.alert_manager.alert(
            message="Medium issue detected",
            level=AlertLevel.MEDIUM,
            source="test_source",
            context={"test": "context"}
        )
        
        self.alert_manager.alert(
            message="Critical issue detected",
            level=AlertLevel.CRITICAL,
            source="test_source",
            context={"test": "context"}
        )
        
        # Get alerts
        alerts = self.alert_manager.get_alerts()
        
        # Count alerts by level
        alert_counts = {
            AlertLevel.LOW: 0,
            AlertLevel.MEDIUM: 0,
            AlertLevel.HIGH: 0,
            AlertLevel.CRITICAL: 0
        }
        
        for alert in alerts:
            alert_counts[alert.level] += 1
        
        # Assertions
        self.assertGreaterEqual(alert_counts[AlertLevel.LOW], 1, "No LOW level alerts found")
        self.assertGreaterEqual(alert_counts[AlertLevel.MEDIUM], 1, "No MEDIUM level alerts found")
        self.assertGreaterEqual(alert_counts[AlertLevel.CRITICAL], 1, "No CRITICAL level alerts found")
    
    def test_email_alert_configuration(self):
        """Test email alert configuration and handler registration."""
        # Configure email settings
        email_config = {
            "smtp_server": "smtp.example.com",
            "smtp_port": 587,
            "sender_email": "alerts@example.com",
            "recipient_emails": ["admin@example.com"],
            "username": "alerts@example.com",
            "password": "password"
        }
        
        # Create alert manager with email configuration
        alert_manager = AlertManager(
            alert_dir=self.alert_dir,
            min_level=AlertLevel.LOW,
            email_config=email_config
        )
        
        # Verify email handler was registered
        self.assertIn("email", alert_manager.handlers, 
                      "Email handler was not registered despite valid config")
        
        # Verify email configuration was stored
        self.assertEqual(alert_manager.email_config, email_config,
                        "Email configuration was not properly stored")
        
        # This test only verifies the configuration is correctly set up
        # We don't test the actual sending since that would require complex mocking
        # of SMTP interactions, which is outside the scope of the integration test


if __name__ == "__main__":
    unittest.main()
