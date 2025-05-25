"""
Comprehensive validation test for the integrated ISNE pipeline with alerts.

This test script demonstrates the complete ISNE validation and alert system 
working together with real-world data to provide actionable alerts for
embedding discrepancies and quality issues.
"""

import json
import logging
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add the project root directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest

from src.alerts import AlertManager, AlertLevel
from src.isne.pipeline.isne_pipeline import ISNEPipeline
from src.validation.embedding_validator import (
    validate_embeddings_before_isne,
    validate_embeddings_after_isne,
    create_validation_summary
)


class TestIntegratedISNEValidation(unittest.TestCase):
    """Test the integrated ISNE validation and alert system."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment with sample data."""
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
        
        # Create temp directory for test artifacts
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.test_dir = Path(cls.temp_dir.name)
        cls.output_dir = cls.test_dir / "output"
        cls.alerts_dir = cls.test_dir / "alerts"
        
        # Create directories
        cls.output_dir.mkdir(parents=True, exist_ok=True)
        cls.alerts_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate test documents
        cls.documents = cls._generate_test_documents(
            num_docs=5,
            chunks_per_doc=10,
            embed_dim=384
        )
        
        # Create various test scenarios
        cls.perfect_docs = cls.documents.copy()
        cls.missing_base_docs = cls._create_missing_base_embeddings(cls.documents.copy())
        cls.missing_isne_docs = cls._apply_isne_with_issues(cls.documents.copy())
        
        # Save test documents for inspection
        with open(cls.test_dir / "perfect_docs.json", "w") as f:
            json.dump(cls.perfect_docs, f, indent=2)
            
        with open(cls.test_dir / "missing_base_docs.json", "w") as f:
            json.dump(cls.missing_base_docs, f, indent=2)
            
        with open(cls.test_dir / "missing_isne_docs.json", "w") as f:
            json.dump(cls.missing_isne_docs, f, indent=2)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up temporary files."""
        cls.temp_dir.cleanup()
    
    @staticmethod
    def _generate_test_documents(num_docs: int, chunks_per_doc: int, 
                                embed_dim: int = 384) -> List[Dict[str, Any]]:
        """Generate test documents with embeddings."""
        docs = []
        
        for i in range(num_docs):
            chunks = []
            
            for j in range(chunks_per_doc):
                # Create embedding
                embedding = np.random.rand(embed_dim).tolist()
                
                chunks.append({
                    "id": f"chunk_{i}_{j}",
                    "text": f"This is test chunk {j} of document {i}",
                    "embedding": embedding,
                    "embedding_model": "test-base-embedding",
                    "embedding_type": "base",
                    "metadata": {"position": j}
                })
            
            docs.append({
                "id": f"doc_{i}",
                "file_id": f"file_{i}",
                "file_name": f"test_document_{i}.txt",
                "chunks": chunks
            })
        
        return docs
    
    @staticmethod
    def _create_missing_base_embeddings(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create test documents with some missing base embeddings."""
        # Remove base embeddings from ~20% of chunks
        import random
        
        for doc in docs:
            for chunk in random.sample(doc["chunks"], k=max(1, len(doc["chunks"]) // 5)):
                if "embedding" in chunk:
                    del chunk["embedding"]
                    del chunk["embedding_model"]
                    del chunk["embedding_type"]
        
        return docs
    
    @staticmethod
    def _apply_isne_with_issues(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply mock ISNE embeddings with intentional issues."""
        # Apply ISNE embeddings to all chunks that have base embeddings
        for doc in docs:
            for chunk in doc["chunks"]:
                if "embedding" in chunk:
                    # Add ISNE embedding (same dimension as base)
                    chunk["isne_embedding"] = np.random.rand(len(chunk["embedding"])).tolist()
                    chunk["isne_embedding_model"] = "test-isne-model"
        
        # Now remove some ISNE embeddings to create discrepancies
        import random
        
        for doc in docs:
            valid_chunks = [c for c in doc["chunks"] if "isne_embedding" in c]
            if valid_chunks:
                # Remove ISNE from ~10% of chunks that have it
                for chunk in random.sample(valid_chunks, k=max(1, len(valid_chunks) // 10)):
                    if "isne_embedding" in chunk:
                        del chunk["isne_embedding"]
                        del chunk["isne_embedding_model"]
        
        return docs
    
    def test_perfect_pipeline(self):
        """Test pipeline with perfect documents (no issues)."""
        # Initialize alert manager and pipeline
        alert_manager = AlertManager(alert_dir=str(self.alerts_dir / "perfect"))
        
        pipeline = ISNEPipeline(
            validate=True,
            alert_threshold="low",  # Sensitive to any issues
            alert_manager=alert_manager
        )
        
        # Process documents
        enhanced_docs, stats = pipeline.process_documents(
            documents=self.perfect_docs,
            save_report=True,
            output_dir=str(self.output_dir / "perfect")
        )
        
        # Check alert stats - should have no alerts
        alert_stats = alert_manager.get_alert_stats()
        self.assertEqual(alert_stats["LOW"], 0)
        self.assertEqual(alert_stats["MEDIUM"], 0)
        self.assertEqual(alert_stats["HIGH"], 0)
        self.assertEqual(alert_stats["CRITICAL"], 0)
        
        # Verify validation summary
        validation_summary = stats.get("validation_summary", {})
        self.assertIn("discrepancies", validation_summary)
        
        # No discrepancies should be present
        discrepancies = validation_summary.get("discrepancies", {})
        total_discrepancies = sum(abs(value) for value in discrepancies.values())
        self.assertEqual(total_discrepancies, 0)
    
    def test_missing_base_embeddings(self):
        """Test pipeline with missing base embeddings."""
        # Initialize alert manager and pipeline
        alert_manager = AlertManager(alert_dir=str(self.alerts_dir / "missing_base"))
        
        pipeline = ISNEPipeline(
            validate=True,
            alert_threshold="low",  # Sensitive to any issues
            alert_manager=alert_manager
        )
        
        # Process documents
        enhanced_docs, stats = pipeline.process_documents(
            documents=self.missing_base_docs,
            save_report=True,
            output_dir=str(self.output_dir / "missing_base")
        )
        
        # Check alert stats - should have alerts for missing base embeddings
        alert_stats = alert_manager.get_alert_stats()
        self.assertGreater(sum(alert_stats.values()), 0)
        
        # Verify validation summary
        validation_summary = stats.get("validation_summary", {})
        self.assertIn("discrepancies", validation_summary)
        
        # Get alerts and check content
        alerts = alert_manager.get_alerts(min_level=AlertLevel.LOW)
        self.assertGreater(len(alerts), 0)
        
        # Check that at least one alert mentions missing base embeddings
        missing_alerts = [a for a in alerts if "missing" in a.message.lower() and "base" in a.message.lower()]
        self.assertGreater(len(missing_alerts), 0)
    
    def test_missing_isne_embeddings(self):
        """Test pipeline validation with missing ISNE embeddings."""
        # For this test, we'll use direct validation rather than the pipeline
        # since we're testing documents that already have partial ISNE embeddings
        
        # Initialize alert manager
        alert_manager = AlertManager(alert_dir=str(self.alerts_dir / "missing_isne"))
        
        # Pre-validation
        pre_validation = validate_embeddings_before_isne(self.missing_isne_docs)
        
        # Post-validation
        post_validation = validate_embeddings_after_isne(self.missing_isne_docs, pre_validation)
        
        # Create validation summary
        validation_summary = create_validation_summary(pre_validation, post_validation)
        
        # Check for discrepancies
        discrepancies = validation_summary.get("discrepancies", {})
        total_discrepancies = sum(abs(value) for value in discrepancies.values())
        self.assertGreater(total_discrepancies, 0)
        
        # Create alerts from validation results
        if "missing_isne_embeddings" in post_validation and post_validation["missing_isne_embeddings"]:
            count = len(post_validation["missing_isne_embeddings"])
            alert_manager.alert(
                message=f"Missing ISNE embeddings detected in {count} chunks",
                level=AlertLevel.HIGH,
                source="embedding_validator",
                context={
                    "affected_chunks": post_validation["missing_isne_embeddings"]
                }
            )
        
        # Check that alerts were created
        alerts = alert_manager.get_alerts(min_level=AlertLevel.HIGH)
        self.assertGreater(len(alerts), 0)
        
        # Check alert contents
        for alert in alerts:
            self.assertIn("Missing ISNE embeddings", alert.message)
            self.assertEqual(alert.level, AlertLevel.HIGH)
            self.assertIn("affected_chunks", alert.context)
    
    def test_end_to_end_alert_integration(self):
        """Test the complete end-to-end alert and validation system."""
        # Create a dataset with various issues
        mixed_docs = self.documents.copy()
        
        # Remove base embeddings from some chunks
        for doc in mixed_docs[:2]:
            for chunk in doc["chunks"][:2]:
                if "embedding" in chunk:
                    del chunk["embedding"]
                    del chunk["embedding_model"]
                    del chunk["embedding_type"]
        
        # Initialize alert manager
        alert_manager = AlertManager(
            alert_dir=str(self.alerts_dir / "end_to_end"),
            min_level=AlertLevel.LOW
        )
        
        # Initialize pipeline
        pipeline = ISNEPipeline(
            validate=True,
            alert_threshold="low",
            alert_manager=alert_manager
        )
        
        # Process documents
        enhanced_docs, stats = pipeline.process_documents(
            documents=mixed_docs,
            save_report=True,
            output_dir=str(self.output_dir / "end_to_end")
        )
        
        # Check alert stats
        alert_stats = alert_manager.get_alert_stats()
        self.assertGreater(alert_stats["LOW"] + alert_stats["MEDIUM"] + alert_stats["HIGH"], 0)
        
        # Check validation report was saved
        validation_report_path = self.output_dir / "end_to_end" / "isne_validation_report.json"
        self.assertTrue(validation_report_path.exists())
        
        # Load and check validation report
        with open(validation_report_path, "r") as f:
            validation_report = json.load(f)
        
        self.assertIn("discrepancies", validation_report)
        
        # Check alert log was created
        alert_log_path = self.alerts_dir / "end_to_end" / "alerts.log"
        self.assertTrue(alert_log_path.exists())
        
        # Check alert JSON was created
        alert_json_path = self.alerts_dir / "end_to_end" / "alerts.json"
        self.assertTrue(alert_json_path.exists())
        
        # Load and check alert JSON
        with open(alert_json_path, "r") as f:
            alert_data = json.load(f)
        
        self.assertGreater(len(alert_data), 0)
        
        # Print summary for debugging
        print("\n===== End-to-End Alert Test Summary =====")
        print(f"Alert counts: {alert_stats}")
        print(f"Validation report: {validation_report_path}")
        print(f"Alert log: {alert_log_path}")
        print(f"Number of JSON alerts: {len(alert_data)}")


if __name__ == "__main__":
    unittest.main()
