"""
Adaptive ISNE Training Strategy for Dynamic Knowledge Graphs

This module implements an intelligent training strategy that determines when
to retrain the ISNE model versus when to use existing model weights for
new embeddings.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path

from src.isne.pipeline.isne_pipeline import ISNEPipeline
from src.validation.embedding_validator import validate_embeddings_after_isne
from src.types.validation.results import PreValidationResult, PostValidationResult

logger = logging.getLogger(__name__)


class ScheduledISNETrainer:
    """
    Manages hierarchical ISNE training schedules with different frequencies.
    
    Supports:
    - Daily incremental training
    - Weekly/Monthly full retraining
    - Cron job integration
    """
    
    def __init__(self,
                 isne_pipeline: ISNEPipeline,
                 training_schedule_config: Dict[str, Any]):
        """
        Initialize scheduled trainer.
        
        Args:
            isne_pipeline: Configured ISNE pipeline
            training_schedule_config: Configuration for training schedules
        """
        self.isne_pipeline = isne_pipeline
        self.config = training_schedule_config
        
        # Initialize adaptive trainer for daily incremental updates
        self.adaptive_trainer = AdaptiveISNETrainer(
            isne_pipeline=isne_pipeline,
            retrain_threshold=self.config.get('incremental', {}).get('retrain_threshold', 0.15),
            min_retrain_interval=1,  # Allow daily updates
            max_incremental_updates=self.config.get('incremental', {}).get('max_daily_chunks', 500)
        )
        
        # State tracking
        self.last_full_retrain = datetime.now(timezone.utc)
        self.daily_updates_since_full_retrain = 0
        
    def run_daily_incremental(self, date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Run daily incremental training on new chunks from the specified date.
        
        Args:
            date: Date to process (defaults to yesterday)
            
        Returns:
            Training results
        """
        if date is None:
            date = datetime.now(timezone.utc) - timedelta(days=1)
            
        logger.info(f"Running daily incremental training for {date.strftime('%Y-%m-%d')}")
        
        # Get new chunks from the specified date
        new_chunks = self._get_chunks_by_date(date)
        
        if not new_chunks:
            logger.info(f"No new chunks found for {date.strftime('%Y-%m-%d')}")
            return {'status': 'no_new_data', 'date': date.isoformat()}
        
        # Force incremental processing (don't trigger full retrain)
        logger.info(f"Processing {len(new_chunks)} chunks incrementally")
        results = self.adaptive_trainer._incremental_process(new_chunks)
        
        # Update tracking
        self.daily_updates_since_full_retrain += 1
        
        # Enhanced results
        results.update({
            'training_type': 'daily_incremental',
            'date': date.isoformat(),
            'daily_updates_since_full_retrain': self.daily_updates_since_full_retrain
        })
        
        # Save daily training log
        self._save_training_log(results, 'daily')
        
        return results
    
    def run_full_retraining(self, scope: str = 'weekly') -> Dict[str, Any]:
        """
        Run full retraining on all data within the specified scope.
        
        Args:
            scope: 'weekly', 'monthly', or 'full'
            
        Returns:
            Training results
        """
        logger.info(f"Running {scope} full retraining")
        
        # Get date range for retraining
        if scope == 'weekly':
            start_date = datetime.now(timezone.utc) - timedelta(weeks=1)
        elif scope == 'monthly':
            start_date = datetime.now(timezone.utc) - timedelta(days=30)
        else:  # full
            start_date = None  # All data
            
        # Get all chunks in scope
        all_chunks = self._get_chunks_by_date_range(start_date, datetime.now(timezone.utc))
        current_graph_stats = self._get_current_graph_stats()
        
        logger.info(f"Retraining with {len(all_chunks)} total chunks")
        
        # Force full retraining
        results = self.adaptive_trainer._retrain_and_process(all_chunks, current_graph_stats)
        
        # Update tracking
        self.last_full_retrain = datetime.now(timezone.utc)
        self.daily_updates_since_full_retrain = 0
        
        # Enhanced results
        results.update({
            'training_type': f'{scope}_full_retrain',
            'scope': scope,
            'start_date': start_date.isoformat() if start_date else None,
            'end_date': datetime.now(timezone.utc).isoformat()
        })
        
        # Save full training log
        self._save_training_log(results, scope)
        
        return results
    
    def should_run_full_retrain(self) -> Tuple[bool, str]:
        """
        Check if a full retraining should be triggered based on schedule or conditions.
        
        Returns:
            Tuple of (should_retrain, reason)
        """
        now = datetime.now(timezone.utc)
        
        # Check weekly schedule
        weekly_config = self.config.get('weekly', {})
        if weekly_config.get('enabled', True):
            days_since_retrain = (now - self.last_full_retrain).days
            weekly_interval = weekly_config.get('interval_days', 7)
            
            if days_since_retrain >= weekly_interval:
                return True, f"Weekly schedule: {days_since_retrain} days since last full retrain"
        
        # Check monthly schedule
        monthly_config = self.config.get('monthly', {})
        if monthly_config.get('enabled', True):
            if now.day == 1 and (now - self.last_full_retrain).days >= 28:
                return True, "Monthly schedule: First of month trigger"
        
        # Check accumulation threshold
        max_daily_updates = self.config.get('incremental', {}).get('max_daily_updates_before_full', 30)
        if self.daily_updates_since_full_retrain >= max_daily_updates:
            return True, f"Accumulation threshold: {self.daily_updates_since_full_retrain} daily updates"
        
        return False, "No full retraining trigger met"
    
    def _get_chunks_by_date(self, date: datetime) -> List[Dict]:
        """Get chunks that were added on a specific date."""
        # This would interface with your ArangoDB to get chunks by date
        # Implementation depends on your chunk metadata schema
        # Placeholder implementation:
        return []
    
    def _get_chunks_by_date_range(self, start_date: Optional[datetime], end_date: datetime) -> List[Dict]:
        """Get all chunks within a date range."""
        # This would interface with your ArangoDB to get chunks by date range
        # Implementation depends on your chunk metadata schema
        # Placeholder implementation:
        return []
    
    def _get_current_graph_stats(self) -> Dict:
        """Get current graph statistics from ArangoDB."""
        # This would query your ArangoDB for current graph stats
        # Placeholder implementation:
        return {
            'total_chunks': 0,
            'total_edges': 0,
            'edge_density': 0.0
        }
    
    def _save_training_log(self, results: Dict, training_type: str) -> None:
        """Save training results to log file."""
        log_dir = Path("./training_logs")
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{training_type}_training_{timestamp}.json"
        
        import json
        with open(log_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Training log saved to {log_file}")


class AdaptiveISNETrainer:
    """
    Manages ISNE model training decisions based on graph evolution.
    
    Key Principles:
    1. Use inductive capabilities when possible (no retraining)
    2. Retrain when graph structure significantly changes
    3. Maintain embedding quality through validation
    """
    
    def __init__(self, 
                 isne_pipeline: ISNEPipeline,
                 retrain_threshold: float = 0.15,  # 15% structure change
                 min_retrain_interval: int = 24,   # hours
                 max_incremental_updates: int = 1000):
        """
        Initialize adaptive trainer.
        
        Args:
            isne_pipeline: Configured ISNE pipeline
            retrain_threshold: Fraction of graph change that triggers retraining
            min_retrain_interval: Minimum hours between retraining
            max_incremental_updates: Max chunks before forced retraining
        """
        self.isne_pipeline = isne_pipeline
        self.retrain_threshold = retrain_threshold
        self.min_retrain_interval = timedelta(hours=min_retrain_interval)
        self.max_incremental_updates = max_incremental_updates
        
        # State tracking
        self.last_retrain_time = datetime.now(timezone.utc)
        self.incremental_updates_count = 0
        self.graph_stats_at_last_train: Dict[str, Any] = {}
        self.embedding_quality_history: List[float] = []
        
    def should_retrain(self, 
                      new_chunks: List[Dict],
                      current_graph_stats: Dict) -> Tuple[bool, str]:
        """
        Determine if ISNE model should be retrained.
        
        Args:
            new_chunks: List of new chunks to be added
            current_graph_stats: Current graph statistics
            
        Returns:
            Tuple of (should_retrain, reason)
        """
        
        # 1. Check minimum time interval
        time_since_retrain = datetime.now(timezone.utc) - self.last_retrain_time
        if time_since_retrain < self.min_retrain_interval:
            return False, f"Too soon since last retrain ({time_since_retrain})"
        
        # 2. Check incremental update limit
        new_count = len(new_chunks)
        if self.incremental_updates_count + new_count > self.max_incremental_updates:
            return True, f"Incremental update limit reached ({self.incremental_updates_count + new_count})"
        
        # 3. Check graph structure change
        if self.graph_stats_at_last_train:
            structure_change = self._calculate_structure_change(
                current_graph_stats, new_count
            )
            
            if structure_change > self.retrain_threshold:
                return True, f"Significant structure change ({structure_change:.2%})"
        
        # 4. Check embedding quality degradation
        if len(self.embedding_quality_history) >= 3:
            quality_trend = self._check_quality_trend()
            if quality_trend < -0.05:  # 5% quality drop
                return True, f"Embedding quality degrading (trend: {quality_trend:.2%})"
        
        return False, "Incremental update recommended"
    
    def process_new_chunks(self, 
                          new_chunks: List[Dict],
                          current_graph_stats: Dict) -> Dict:
        """
        Process new chunks with appropriate training strategy.
        
        Args:
            new_chunks: New chunks to process
            current_graph_stats: Current graph statistics
            
        Returns:
            Dictionary with processing results and metrics
        """
        should_retrain, reason = self.should_retrain(new_chunks, current_graph_stats)
        
        logger.info(f"Processing {len(new_chunks)} new chunks")
        logger.info(f"Training decision: {'RETRAIN' if should_retrain else 'INCREMENTAL'} - {reason}")
        
        if should_retrain:
            return self._retrain_and_process(new_chunks, current_graph_stats)
        else:
            return self._incremental_process(new_chunks)
    
    def _retrain_and_process(self, 
                           new_chunks: List[Dict],
                           current_graph_stats: Dict) -> Dict:
        """Retrain ISNE model and process chunks."""
        
        start_time = datetime.now(timezone.utc)
        
        # 1. Create training documents from new chunks
        logger.info("Processing documents for retraining")
        documents_to_process = [{'chunks': new_chunks}]  # Wrap chunks in document format
        
        # 2. Process documents through ISNE pipeline (includes training)
        logger.info("Processing documents with ISNE pipeline")
        pipeline_result = self.isne_pipeline.process_documents(
            documents=documents_to_process,
            output_dir="./temp_isne_output"
        )
        
        # 3. Extract results from pipeline processing
        logger.info("Processing completed")
        if isinstance(pipeline_result, dict):
            all_embeddings: List[Any] = pipeline_result.get('enhanced_documents', [])
        else:
            all_embeddings = []
        
        # 4. Update state tracking
        self.last_retrain_time = datetime.now(timezone.utc)
        self.incremental_updates_count = 0
        self.graph_stats_at_last_train = current_graph_stats.copy()
        
        # 5. Validate embedding quality
        quality_metrics = validate_embeddings_after_isne([], {})  # Simplified for now
        if isinstance(quality_metrics, dict) and 'chunks_with_isne' in quality_metrics:
            # Use chunks_with_isne as a proxy for overall quality
            overall_score = float(quality_metrics['chunks_with_isne']) / max(1, float(quality_metrics.get('total_isne_count', 1)))
            self.embedding_quality_history.append(overall_score)
        
        processing_time = datetime.now(timezone.utc) - start_time
        
        return {
            'strategy': 'retrain',
            'reason': 'Model retrained for optimal performance',
            'chunks_processed': len(new_chunks),
            'total_chunks': len(all_embeddings),
            'processing_time': processing_time.total_seconds(),
            'training_metrics': training_metrics,
            'quality_metrics': quality_metrics,
            'model_updated': True
        }
    
    def _incremental_process(self, new_chunks: List[Dict]) -> Dict:
        """Process chunks incrementally without retraining."""
        
        start_time = datetime.now(timezone.utc)
        
        # 1. Generate embeddings using existing model
        logger.info(f"Generating ISNE embeddings for {len(new_chunks)} chunks")
        documents_to_process = [{'chunks': new_chunks}]  # Wrap chunks in document format
        
        # 2. Process documents through ISNE pipeline (incremental)
        logger.info("Processing new chunks through ISNE pipeline")
        pipeline_result = self.isne_pipeline.process_documents(
            documents=documents_to_process,
            output_dir="./temp_isne_output"
        )
        
        # Extract relationship statistics from pipeline result
        total_processed = 0
        if isinstance(pipeline_result, dict):
            total_processed = pipeline_result.get('total_chunks_processed', 0)
        
        relationship_stats = {
            'new_relationships': len(new_chunks),  # Simplified
            'total_relationships': total_processed
        }
        
        # 3. Update tracking
        self.incremental_updates_count += len(new_chunks)
        
        # 4. Validate embedding quality
        quality_metrics = validate_embeddings_after_isne([], {})  # Simplified for now
        if isinstance(quality_metrics, dict) and 'chunks_with_isne' in quality_metrics:
            # Use chunks_with_isne as a proxy for overall quality
            overall_score = float(quality_metrics['chunks_with_isne']) / max(1, float(quality_metrics.get('total_isne_count', 1)))
            self.embedding_quality_history.append(overall_score)
        
        processing_time = datetime.now(timezone.utc) - start_time
        
        return {
            'strategy': 'incremental',
            'reason': 'Used existing model for new embeddings',
            'chunks_processed': len(new_chunks),
            'relationships_created': relationship_stats['new_relationships'],
            'processing_time': processing_time.total_seconds(),
            'quality_metrics': quality_metrics,
            'model_updated': False
        }
    
    def _calculate_structure_change(self, 
                                  current_stats: Dict,
                                  new_chunk_count: int) -> float:
        """Calculate the degree of structural change in the graph."""
        
        if not self.graph_stats_at_last_train:
            return 0.0
        
        baseline = self.graph_stats_at_last_train
        
        # Calculate relative changes
        node_change = new_chunk_count / max(baseline.get('total_chunks', 1), 1)
        
        density_change = abs(
            current_stats.get('edge_density', 0) - 
            baseline.get('edge_density', 0)
        )
        
        # Weight the changes
        structure_change = (node_change * 0.7) + (density_change * 0.3)
        
        return float(structure_change)
    
    def _check_quality_trend(self) -> float:
        """Check trend in embedding quality over recent updates."""
        
        if len(self.embedding_quality_history) < 3:
            return 0.0
        
        recent_scores = self.embedding_quality_history[-3:]
        
        # Simple linear trend
        trend = (recent_scores[-1] - recent_scores[0]) / len(recent_scores)
        
        return float(trend)
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get current training statistics."""
        
        return {
            'last_retrain_time': self.last_retrain_time.isoformat(),
            'incremental_updates_since_retrain': self.incremental_updates_count,
            'time_since_retrain_hours': (datetime.now(timezone.utc) - self.last_retrain_time).total_seconds() / 3600,
            'embedding_quality_history': self.embedding_quality_history[-10:],  # Last 10 scores
            'current_quality': self.embedding_quality_history[-1] if self.embedding_quality_history else None
        }