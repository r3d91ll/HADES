"""
Integration tests for the alert system with ISNE pipeline.

These tests demonstrate how the alert system integrates with the ISNE validation 
pipeline and handles real-world validation discrepancies.
"""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pytest

from src.alerts import AlertManager, AlertLevel
from src.validation.embedding_validator import (
    validate_embeddings_before_isne,
    validate_embeddings_after_isne,
    create_validation_summary
)


def create_test_document(doc_id: str, num_chunks: int = 5, 
                         missing_embeddings: List[int] = None) -> Dict[str, Any]:
    """Create a test document with chunks and embeddings."""
    if missing_embeddings is None:
        missing_embeddings = []
        
    chunks = []
    for i in range(num_chunks):
        chunk = {
            "id": f"{doc_id}_chunk_{i}",
            "text": f"This is test chunk {i} of document {doc_id}",
            "metadata": {"position": i}
        }
        
        # Add base embedding unless in missing list
        if i not in missing_embeddings:
            chunk["embedding"] = np.random.rand(384).tolist()
            chunk["embedding_model"] = "test-base-embedding"
            chunk["embedding_type"] = "base"
            
        chunks.append(chunk)
    
    return {
        "id": doc_id,
        "file_id": f"file_{doc_id}",
        "chunks": chunks
    }


def apply_isne_embeddings(docs: List[Dict[str, Any]], 
                          skip_chunks: List[Dict[str, int]] = None) -> List[Dict[str, Any]]:
    """
    Simulate applying ISNE embeddings to documents.
    
    Args:
        docs: List of documents with base embeddings
        skip_chunks: List of {doc_index, chunk_index} to skip ISNE application
        
    Returns:
        Documents with ISNE embeddings applied
    """
    if skip_chunks is None:
        skip_chunks = []
        
    skip_map = {(item["doc_index"], item["chunk_index"]) for item in skip_chunks}
    
    for doc_idx, doc in enumerate(docs):
        for chunk_idx, chunk in enumerate(doc["chunks"]):
            # Skip if no base embedding or in skip list
            if "embedding" not in chunk or (doc_idx, chunk_idx) in skip_map:
                continue
                
            # Apply mock ISNE embedding
            chunk["isne_embedding"] = np.random.rand(384).tolist()
            chunk["isne_embedding_model"] = "test-isne-model"
            
    return docs


class TestAlertIntegration(unittest.TestCase):
    """Test the integration of alerts with the validation system."""
    
    def setUp(self):
        """Set up test environment with documents and alert manager."""
        # Create a temporary directory for alerts
        self.temp_dir = tempfile.TemporaryDirectory()
        self.alert_dir = Path(self.temp_dir.name) / "alerts"
        self.alert_dir.mkdir(parents=True, exist_ok=True)
        
        # Create alert manager
        self.alert_manager = AlertManager(alert_dir=self.alert_dir)
        
        # Create test documents
        self.documents = [
            create_test_document("doc1", num_chunks=5),
            create_test_document("doc2", num_chunks=3, missing_embeddings=[1]),
            create_test_document("doc3", num_chunks=4)
        ]
    
    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()
    
    def test_validation_alerts_before_isne(self):
        """Test that validation generates appropriate alerts before ISNE."""
        # Validate documents before ISNE
        pre_validation = validate_embeddings_before_isne(self.documents)
        
        # Create alerts from validation results
        self._create_alerts_from_validation(pre_validation, "pre_validation")
        
        # Check the alert log file exists
        alert_log_file = self.alert_dir / "alerts.log"
        self.assertTrue(alert_log_file.exists())
        
        # Check that alerts were generated
        with open(alert_log_file, "r") as f:
            log_content = f.read()
            self.assertIn("Missing base embeddings", log_content)
            self.assertIn("MEDIUM", log_content)
            
        # Check alert counts
        stats = self.alert_manager.get_alert_stats()
        self.assertEqual(stats["MEDIUM"], 1)  # One alert for missing embeddings
    
    def test_validation_alerts_after_isne(self):
        """Test that validation generates appropriate alerts after ISNE."""
        # First validate before ISNE
        pre_validation = validate_embeddings_before_isne(self.documents)
        
        # Apply ISNE embeddings with some intentional issues
        enhanced_docs = apply_isne_embeddings(
            self.documents, 
            skip_chunks=[
                {"doc_index": 0, "chunk_index": 2},  # Skip a chunk in doc1
                {"doc_index": 2, "chunk_index": 1}   # Skip a chunk in doc3
            ]
        )
        
        # Validate after ISNE
        post_validation = validate_embeddings_after_isne(enhanced_docs, pre_validation)
        
        # Create validation summary
        summary = create_validation_summary(pre_validation, post_validation)
        
        # Create alerts from validation results
        self._create_alerts_from_validation(post_validation, "post_validation")
        self._create_alerts_from_summary(summary)
        
        # Check alert counts
        stats = self.alert_manager.get_alert_stats()
        self.assertGreaterEqual(stats["HIGH"], 1)  # High alert for missing ISNE embeddings
        
        # Check critical alerts log
        critical_log_file = self.alert_dir / "critical_alerts.log"
        self.assertTrue(critical_log_file.exists())
        
        # Get filtered alerts
        high_alerts = self.alert_manager.get_alerts(min_level=AlertLevel.HIGH)
        self.assertGreaterEqual(len(high_alerts), 1)
        
        # Check alert context contains relevant information
        for alert in high_alerts:
            self.assertIn("affected_chunks", alert.context)
    
    def test_end_to_end_validation_alerts(self):
        """Test the complete validation and alert workflow."""
        # 1. Pre-validation
        pre_validation = validate_embeddings_before_isne(self.documents)
        self._create_alerts_from_validation(pre_validation, "pre_validation")
        
        # 2. Apply ISNE with intentional issues
        enhanced_docs = apply_isne_embeddings(
            self.documents,
            skip_chunks=[{"doc_index": 0, "chunk_index": 1}]
        )
        
        # 3. Post-validation
        post_validation = validate_embeddings_after_isne(enhanced_docs, pre_validation)
        self._create_alerts_from_validation(post_validation, "post_validation")
        
        # 4. Create and alert on summary
        summary = create_validation_summary(pre_validation, post_validation)
        self._create_alerts_from_summary(summary)
        
        # Check the JSON alert log for structured data
        json_log_file = self.alert_dir / "alerts.json"
        self.assertTrue(json_log_file.exists())
        
        with open(json_log_file, "r") as f:
            alerts_data = json.load(f)
            
            # Find alerts related to ISNE application
            isne_alerts = [a for a in alerts_data if "ISNE" in a["message"]]
            self.assertGreaterEqual(len(isne_alerts), 1)
            
            # Check for high-level alerts
            high_alerts = [a for a in alerts_data if a["level"] in ("HIGH", "CRITICAL")]
            self.assertGreaterEqual(len(high_alerts), 1)
    
    def _create_alerts_from_validation(self, validation: Dict[str, Any], stage: str):
        """Create alerts from validation results."""
        # Alert on missing base embeddings
        if "missing_base_embeddings" in validation and validation["missing_base_embeddings"]:
            count = len(validation["missing_base_embeddings"])
            self.alert_manager.alert(
                message=f"Missing base embeddings detected in {count} chunks",
                level=AlertLevel.MEDIUM,
                source="embedding_validator",
                context={
                    "stage": stage,
                    "affected_chunks": validation["missing_base_embeddings"]
                }
            )
        
        # Alert on missing ISNE embeddings
        if "missing_isne_embeddings" in validation and validation["missing_isne_embeddings"]:
            count = len(validation["missing_isne_embeddings"])
            self.alert_manager.alert(
                message=f"Missing ISNE embeddings detected in {count} chunks",
                level=AlertLevel.HIGH,
                source="embedding_validator",
                context={
                    "stage": stage,
                    "affected_chunks": validation["missing_isne_embeddings"]
                }
            )
    
    def _create_alerts_from_summary(self, summary: Dict[str, Any]):
        """Create alerts from validation summary."""
        # Alert on validation discrepancies
        if "discrepancies" in summary and summary["discrepancies"]:
            count = len(summary["discrepancies"])
            severity = AlertLevel.CRITICAL if count > 5 else AlertLevel.HIGH
            
            self.alert_manager.alert(
                message=f"ISNE embedding discrepancies detected in {count} chunks",
                level=severity,
                source="embedding_validator",
                context={
                    "discrepancies": summary["discrepancies"],
                    "expected_counts": summary.get("expected_counts"),
                    "actual_counts": summary.get("actual_counts")
                }
            )


if __name__ == "__main__":
    unittest.main()
