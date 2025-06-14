"""
Bootstrap Pipeline for HADES

This module implements the bootstrap pipeline for initializing ISNE models
and other graph enhancement components from scratch.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# Import from data ingestion stages (temporarily)
from ..data_ingestion.stages.document_processor import DocumentProcessorStage
from ..data_ingestion.stages.chunking import ChunkingStage
from ..data_ingestion.stages.embedding import EmbeddingStage

# Import ISNE components
from src.isne.bootstrap.adaptive_training import AdaptiveISNETrainer
from src.isne.pipeline.isne_pipeline import ISNEPipeline
from src.config.config_loader import load_config
from src.alerts.alert_manager import AlertManager, AlertLevel
from src.validation.embedding_validator import validate_embeddings_after_isne

logger = logging.getLogger(__name__)


class BootstrapPipeline:
    """
    Bootstrap pipeline for initializing HADES components.
    
    This pipeline handles the initial setup and training of graph enhancement
    models, particularly ISNE, from scratch using initial document sets.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize bootstrap pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config("bootstrap_config") if not config_path else load_config(config_path)
        self.alert_manager = AlertManager()
        
        # Initialize stages
        self.doc_processor = DocumentProcessorStage(self.config.get('document_processing', {}))
        self.chunker = ChunkingStage(self.config.get('chunking', {}))
        self.embedder = EmbeddingStage(self.config.get('embedding', {}))
        
        # ISNE components
        self.isne_pipeline: Optional[ISNEPipeline] = None
        self.adaptive_trainer: Optional[AdaptiveISNETrainer] = None
        
        logger.info("Bootstrap pipeline initialized")
    
    def bootstrap_from_documents(
        self, 
        document_paths: List[str], 
        output_dir: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Bootstrap ISNE model from initial document set.
        
        Args:
            document_paths: List of document file paths
            output_dir: Directory to save results
            **kwargs: Additional parameters
            
        Returns:
            Bootstrap results dictionary
        """
        start_time = time.time()
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting bootstrap process with {len(document_paths)} documents")
        
        try:
            # Stage 1: Document Processing
            logger.info("Stage 1: Processing documents")
            documents = self.doc_processor.execute(document_paths)
            
            if not documents or documents.status.name != 'SUCCESS':
                raise ValueError("Document processing failed")
            
            # Stage 2: Chunking
            logger.info("Stage 2: Chunking documents")
            chunks = self.chunker.execute(documents.data)
            
            if not chunks or chunks.status.name != 'SUCCESS':
                raise ValueError("Document chunking failed")
            
            # Stage 3: Generate base embeddings
            logger.info("Stage 3: Generating base embeddings")
            embeddings = self.embedder.execute(chunks.data)
            
            if not embeddings or embeddings.status.name != 'SUCCESS':
                raise ValueError("Embedding generation failed")
            
            # Stage 4: Bootstrap ISNE model
            logger.info("Stage 4: Bootstrapping ISNE model")
            isne_results = self._bootstrap_isne_model(
                embeddings.data, 
                output_dir,
                **kwargs
            )
            
            # Stage 5: Validation
            logger.info("Stage 5: Validating bootstrap results")
            validation_results = self._validate_bootstrap(isne_results, output_dir)
            
            # Compile results
            bootstrap_results = {
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'execution_time': time.time() - start_time,
                'statistics': {
                    'documents_processed': len(documents.data) if documents.data else 0,
                    'chunks_created': len(chunks.data) if chunks.data else 0,
                    'embeddings_generated': len(embeddings.data) if embeddings.data else 0,
                    'isne_model_trained': isne_results.get('model_trained', False)
                },
                'model_info': isne_results.get('model_info', {}),
                'validation': validation_results,
                'output_directory': str(output_path)
            }
            
            logger.info(f"Bootstrap completed successfully in {bootstrap_results['execution_time']:.2f}s")
            return bootstrap_results
            
        except Exception as e:
            logger.error(f"Bootstrap pipeline failed: {e}")
            self.alert_manager.alert(
                f"Bootstrap pipeline failed: {e}",
                AlertLevel.HIGH,
                "bootstrap_pipeline"
            )
            
            return {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'execution_time': time.time() - start_time
            }
    
    def _bootstrap_isne_model(
        self, 
        embeddings_data: List[Dict[str, Any]], 
        output_dir: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Bootstrap ISNE model from embeddings.
        
        Args:
            embeddings_data: List of embedding data
            output_dir: Output directory
            **kwargs: Additional parameters
            
        Returns:
            ISNE bootstrap results
        """
        try:
            # Initialize ISNE pipeline if not already done
            if self.isne_pipeline is None:
                # Extract ISNE config parameters
                isne_config = self.config.get('isne', {})
                
                self.isne_pipeline = ISNEPipeline(
                    model_path=isne_config.get('model_path'),
                    validate=isne_config.get('validate', True),
                    alert_threshold=isne_config.get('alert_threshold', 'high'),
                    device=isne_config.get('device'),
                    alert_dir=isne_config.get('alert_dir', output_dir),
                    alert_manager=None
                )
            
            # Initialize adaptive trainer
            if self.adaptive_trainer is None:
                self.adaptive_trainer = AdaptiveISNETrainer(
                    isne_pipeline=self.isne_pipeline,
                    retrain_threshold=self.config.get('retrain_threshold', 0.15),
                    min_retrain_interval=1,
                    max_incremental_updates=kwargs.get('max_incremental_updates', 500)
                )
            
            # Perform initial training
            logger.info("Training ISNE model from scratch")
            if self.adaptive_trainer is not None:
                training_results = self.adaptive_trainer.train_from_embeddings(
                    embeddings_data,
                    force_retrain=True  # Force full training for bootstrap
                )
            else:
                logger.warning("Adaptive trainer is None, skipping training")
                training_results = {}
            
            # Save model
            model_path = Path(output_dir) / "models" / "isne_bootstrap_model.pt"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            
            if self.isne_pipeline is not None and hasattr(self.isne_pipeline, 'save_model'):
                self.isne_pipeline.save_model(str(model_path))
                logger.info(f"ISNE model saved to {model_path}")
            else:
                logger.warning("ISNE pipeline is None or doesn't have save_model method")
            
            return {
                'model_trained': True,
                'model_path': str(model_path),
                'training_results': training_results,
                'model_info': self._get_model_info()
            }
            
        except Exception as e:
            logger.error(f"ISNE bootstrap failed: {e}")
            return {
                'model_trained': False,
                'error': str(e)
            }
    
    def _validate_bootstrap(
        self, 
        isne_results: Dict[str, Any], 
        output_dir: str
    ) -> Dict[str, Any]:
        """
        Validate bootstrap results.
        
        Args:
            isne_results: ISNE training results
            output_dir: Output directory
            
        Returns:
            Validation results
        """
        try:
            if not isne_results.get('model_trained', False):
                return {
                    'status': 'failed',
                    'reason': 'ISNE model not trained'
                }
            
            # Perform embedding validation if available
            if hasattr(self, 'isne_pipeline') and self.isne_pipeline:
                validation_result = validate_embeddings_after_isne(
                    isne_pipeline=self.isne_pipeline,
                    output_dir=output_dir
                )
                
                return {
                    'status': 'success',
                    'embedding_validation': validation_result
                }
            
            return {
                'status': 'success',
                'note': 'Basic validation passed'
            }
            
        except Exception as e:
            logger.warning(f"Validation failed: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def _get_model_info(self) -> Dict[str, Any]:
        """Get information about the trained model."""
        if self.isne_pipeline and hasattr(self.isne_pipeline, 'model'):
            try:
                return {
                    'model_type': 'ISNE',
                    'parameters': getattr(self.isne_pipeline.model, 'config', {}),
                    'timestamp': datetime.now().isoformat()
                }
            except Exception:
                pass
        
        return {
            'model_type': 'ISNE',
            'status': 'basic_info_only',
            'timestamp': datetime.now().isoformat()
        }
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        return {
            'initialized': True,
            'isne_pipeline_ready': self.isne_pipeline is not None,
            'adaptive_trainer_ready': self.adaptive_trainer is not None,
            'config_loaded': self.config is not None
        }


def run_bootstrap_pipeline(
    document_paths: List[str],
    output_dir: str,
    config_path: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to run bootstrap pipeline.
    
    Args:
        document_paths: List of document paths
        output_dir: Output directory
        config_path: Optional config file path
        **kwargs: Additional parameters
        
    Returns:
        Bootstrap results
    """
    pipeline = BootstrapPipeline(config_path)
    return pipeline.bootstrap_from_documents(document_paths, output_dir, **kwargs)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run HADES Bootstrap Pipeline")
    parser.add_argument("--input-dir", required=True, help="Input directory with documents")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--max-files", type=int, help="Maximum number of files to process")
    
    args = parser.parse_args()
    
    # Collect document paths
    input_path = Path(args.input_dir)
    if input_path.is_file():
        document_paths = [str(input_path)]
    else:
        document_paths = []
        for ext in ['*.pdf', '*.txt', '*.md', '*.py']:
            document_paths.extend([str(p) for p in input_path.glob(f"**/{ext}")])
        
        if args.max_files:
            document_paths = document_paths[:args.max_files]
    
    print(f"Found {len(document_paths)} documents to process")
    
    # Run bootstrap
    results = run_bootstrap_pipeline(
        document_paths=document_paths,
        output_dir=args.output_dir,
        config_path=args.config
    )
    
    print(f"Bootstrap completed with status: {results.get('status', 'unknown')}")
    if results.get('status') == 'success':
        print(f"Results saved to: {results.get('output_directory')}")
        print(f"Execution time: {results.get('execution_time', 0):.2f}s")
    else:
        print(f"Error: {results.get('error', 'Unknown error')}")