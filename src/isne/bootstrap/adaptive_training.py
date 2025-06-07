"""
Adaptive ISNE Training Strategy for Dynamic Knowledge Graphs

This module implements an intelligent training strategy that determines when
to retrain the ISNE model versus when to use existing model weights for
new embeddings.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from pathlib import Path

from src.isne.pipeline.isne_pipeline import ISNEPipeline
from src.validation.embedding_validator import validate_embeddings_after_isne

logger = logging.getLogger(__name__)


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
        self.last_retrain_time = datetime.now()
        self.incremental_updates_count = 0
        self.graph_stats_at_last_train = {}
        self.embedding_quality_history = []
        
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
        time_since_retrain = datetime.now() - self.last_retrain_time
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
        
        start_time = datetime.now()
        
        # 1. Create combined training data
        logger.info("Creating combined training graph for retraining")
        combined_graph = self.isne_pipeline.create_training_graph(
            include_existing=True,
            new_chunks=new_chunks
        )
        
        # 2. Retrain ISNE model
        logger.info("Retraining ISNE model with updated graph")
        training_metrics = self.isne_pipeline.train(
            graph_data=combined_graph,
            validation_split=0.2
        )
        
        # 3. Re-embed all chunks with updated model
        logger.info("Re-embedding all chunks with updated model")
        all_embeddings = self.isne_pipeline.encode_all_chunks()
        
        # 4. Update state tracking
        self.last_retrain_time = datetime.now()
        self.incremental_updates_count = 0
        self.graph_stats_at_last_train = current_graph_stats.copy()
        
        # 5. Validate embedding quality
        quality_metrics = validate_embeddings_after_isne([], {})  # Simplified for now
        self.embedding_quality_history.append(quality_metrics['overall_score'])
        
        processing_time = datetime.now() - start_time
        
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
        
        start_time = datetime.now()
        
        # 1. Generate embeddings using existing model
        logger.info(f"Generating ISNE embeddings for {len(new_chunks)} chunks")
        new_embeddings = self.isne_pipeline.encode_chunks(new_chunks)
        
        # 2. Add to graph with relationship building
        logger.info("Building relationships with existing chunks")
        relationship_stats = self.isne_pipeline.add_chunks_to_graph(
            new_chunks, 
            new_embeddings,
            build_relationships=True
        )
        
        # 3. Update tracking
        self.incremental_updates_count += len(new_chunks)
        
        # 4. Validate embedding quality
        quality_metrics = validate_embeddings_after_isne([], {})  # Simplified for now
        self.embedding_quality_history.append(quality_metrics['overall_score'])
        
        processing_time = datetime.now() - start_time
        
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
        
        return structure_change
    
    def _check_quality_trend(self) -> float:
        """Check trend in embedding quality over recent updates."""
        
        if len(self.embedding_quality_history) < 3:
            return 0.0
        
        recent_scores = self.embedding_quality_history[-3:]
        
        # Simple linear trend
        trend = (recent_scores[-1] - recent_scores[0]) / len(recent_scores)
        
        return trend
    
    def get_training_stats(self) -> Dict:
        """Get current training statistics."""
        
        return {
            'last_retrain_time': self.last_retrain_time.isoformat(),
            'incremental_updates_since_retrain': self.incremental_updates_count,
            'time_since_retrain_hours': (datetime.now() - self.last_retrain_time).total_seconds() / 3600,
            'embedding_quality_history': self.embedding_quality_history[-10:],  # Last 10 scores
            'current_quality': self.embedding_quality_history[-1] if self.embedding_quality_history else None
        }