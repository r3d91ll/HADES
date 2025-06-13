#!/usr/bin/env python3
"""
Service-Integrated Bootstrap Example

This is an EXAMPLE showing how the bootstrap pipeline COULD integrate with
the HADES service APIs for ISNE training, even though we decided to keep
the bootstrap as a standalone script.

This demonstrates the pattern for other pipelines that SHOULD use service APIs.

Usage:
    python scripts/service_integrated_bootstrap_example.py --use-service-training
"""

import sys
import asyncio
import httpx
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)

class ServiceIntegratedBootstrapExample:
    """Example of bootstrap pipeline with service integration for training."""
    
    def __init__(self, hades_service_url: str = "http://localhost:8000"):
        """Initialize with HADES service URL."""
        self.service_url = hades_service_url
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def check_service_health(self) -> bool:
        """Check if HADES service is available."""
        try:
            response = await self.client.get(f"{self.service_url}/health")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"HADES service not available: {e}")
            return False
    
    async def run_local_bootstrap_stages(self) -> Dict[str, Any]:
        """Run bootstrap stages 1-4 locally (docproc, chunking, embedding, graph)."""
        logger.info("Running local bootstrap stages...")
        
        # This would be the same as the current bootstrap pipeline
        # Stages 1-4: docproc → chunking → embedding → graph construction
        
        # Simulate the stages for this example
        bootstrap_results = {
            "documents": 4,
            "chunks": 1354,
            "embeddings": 1354,
            "graph_nodes": 1354,
            "graph_edges": 231098,
            "embedding_dimension": 384,
            "prepared_for_training": True
        }
        
        logger.info(f"Local bootstrap complete: {bootstrap_results}")
        return bootstrap_results
    
    async def trigger_service_training(self, bootstrap_results: Dict[str, Any]) -> Optional[str]:
        """Trigger ISNE training via HADES service API."""
        logger.info("Triggering ISNE training via HADES service...")
        
        # Prepare training request
        training_request = {
            "training_data_path": "./models/isne/bootstrap_data.json",
            "model_config": {
                "input_dim": bootstrap_results["embedding_dimension"],
                "hidden_dim": 256,
                "output_dim": 128,
                "num_layers": 3,
                "num_heads": 8
            },
            "training_config": {
                "epochs": 50,
                "batch_size": 64,
                "learning_rate": 0.001,
                "device": "cpu"
            }
        }
        
        try:
            response = await self.client.post(
                f"{self.service_url}/api/v1/pipelines/isne_training",
                json=training_request
            )
            
            if response.status_code == 200:
                result = response.json()
                job_id = result["job_id"]
                logger.info(f"Training job started: {job_id}")
                logger.info(f"Progress endpoint: {self.service_url}{result['progress_endpoint']}")
                return job_id
            else:
                logger.error(f"Training request failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to trigger training: {e}")
            return None
    
    async def monitor_training_progress(self, job_id: str) -> Dict[str, Any]:
        """Monitor training job progress."""
        logger.info(f"Monitoring training job {job_id}...")
        
        while True:
            try:
                response = await self.client.get(
                    f"{self.service_url}/api/v1/pipelines/jobs/{job_id}/status"
                )
                
                if response.status_code == 200:
                    status = response.json()
                    logger.info(f"Training progress: {status['progress_percent']:.1f}% - {status['stage']}")
                    
                    if status["status"] == "completed":
                        logger.info("Training completed successfully!")
                        return status["results"]
                    elif status["status"] == "failed":
                        logger.error(f"Training failed: {status.get('error_message', 'Unknown error')}")
                        return {}
                    
                    # Wait before next check
                    await asyncio.sleep(5)
                else:
                    logger.error(f"Failed to get status: {response.status_code}")
                    break
                    
            except Exception as e:
                logger.error(f"Error monitoring progress: {e}")
                break
        
        return {}
    
    async def run_hybrid_bootstrap(self) -> bool:
        """Run hybrid bootstrap: local stages + service training."""
        logger.info("=== SERVICE-INTEGRATED BOOTSTRAP EXAMPLE ===")
        
        try:
            # Check service availability
            if not await self.check_service_health():
                logger.error("HADES service not available, falling back to local training")
                return False
            
            # Stage 1-4: Run locally (same as current bootstrap)
            bootstrap_results = await self.run_local_bootstrap_stages()
            
            # Stage 5: Use service for training
            job_id = await self.trigger_service_training(bootstrap_results)
            if not job_id:
                logger.error("Failed to start service training")
                return False
            
            # Monitor training
            training_results = await self.monitor_training_progress(job_id)
            
            if training_results:
                logger.info("=== HYBRID BOOTSTRAP COMPLETE ===")
                logger.info(f"Local stages: {bootstrap_results}")
                logger.info(f"Service training: {training_results}")
                return True
            else:
                logger.error("Training did not complete successfully")
                return False
                
        except Exception as e:
            logger.error(f"Hybrid bootstrap failed: {e}")
            return False
        finally:
            await self.client.aclose()

async def main():
    """Main function demonstrating service integration pattern."""
    logging.basicConfig(level=logging.INFO)
    
    # This example shows the pattern, but we decided bootstrap should remain standalone
    logger.info("🔍 This is an EXAMPLE of service integration")
    logger.info("📝 The actual bootstrap pipeline remains a standalone script")
    logger.info("🎯 This pattern SHOULD be used for operational pipelines like:")
    logger.info("   - Query pipeline (frequent)")
    logger.info("   - Ingestion pipeline (regular)")
    logger.info("   - ISNE retraining (periodic)")
    
    example = ServiceIntegratedBootstrapExample()
    
    # Check if service is available
    service_available = await example.check_service_health()
    logger.info(f"HADES service available: {service_available}")
    
    if service_available:
        logger.info("✅ Service integration pattern demonstrated")
        logger.info("📍 Available endpoints:")
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get("http://localhost:8000/openapi.json")
                if response.status_code == 200:
                    openapi = response.json()
                    endpoints = list(openapi.get("paths", {}).keys())
                    for endpoint in endpoints:
                        logger.info(f"   - {endpoint}")
            except Exception:
                logger.info("   - Could not fetch endpoint list")
    else:
        logger.info("❌ HADES service not available")
        logger.info("💡 Start the service with: poetry run python -m src.api.main")
    
    await example.client.aclose()

if __name__ == "__main__":
    asyncio.run(main())