"""
A/B testing and comparison framework for multiple RAG engines.

This module provides functionality to:
- Run side-by-side comparisons between engines
- Collect detailed metrics and performance data
- Manage controlled experiments
- Generate comparative analysis reports
"""

import asyncio
import json
import time
import uuid
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum

from .base import (
    EngineType, RetrievalMode, EngineRetrievalRequest, EngineRetrievalResponse,
    ExperimentRequest, ExperimentResult, ExperimentResponse, ExperimentStatus,
    get_engine_registry
)


class ComparisonMetric(str, Enum):
    """Metrics available for engine comparison."""
    RELEVANCE = "relevance"
    SPEED = "speed"
    DIVERSITY = "diversity"
    COVERAGE = "coverage"
    CONSISTENCY = "consistency"
    RESOURCE_USAGE = "resource_usage"


@dataclass
class ComparisonResult:
    """Result of comparing two or more engines."""
    query: str
    engines: List[EngineType]
    responses: Dict[EngineType, EngineRetrievalResponse]
    metrics: Dict[str, Dict[EngineType, float]]
    winner: Optional[EngineType]
    analysis: Dict[str, Any]


class ExperimentManager:
    """Manages A/B testing experiments and comparisons."""
    
    def __init__(self):
        self._experiments: Dict[str, ExperimentResponse] = {}
        self._running_experiments: Dict[str, asyncio.Task] = {}
    
    async def run_comparison(
        self,
        query: str,
        engines: List[EngineType],
        mode: RetrievalMode = RetrievalMode.FULL_ENGINE,
        top_k: int = 10,
        engine_configs: Optional[Dict[EngineType, Dict[str, Any]]] = None,
        metrics: Optional[List[ComparisonMetric]] = None
    ) -> ComparisonResult:
        """
        Run a single query comparison across multiple engines.
        
        Args:
            query: Query to test
            engines: List of engines to compare
            mode: Retrieval mode to use
            top_k: Number of results to retrieve
            engine_configs: Optional engine-specific configurations
            metrics: Metrics to calculate
            
        Returns:
            Comparison result with metrics and analysis
        """
        registry = get_engine_registry()
        responses: Dict[EngineType, EngineRetrievalResponse] = {}
        
        # Prepare request
        base_request = EngineRetrievalRequest(
            query=query,
            mode=mode,
            top_k=top_k,
            max_token_for_context=4000
        )
        
        # Run queries in parallel
        tasks = []
        for engine_type in engines:
            engine = registry.get_engine(engine_type)
            if not engine:
                continue
            
            # Apply engine-specific config if provided
            if engine_configs and engine_type in engine_configs:
                await engine.configure(engine_configs[engine_type])
            
            # Create request with engine-specific params
            request = EngineRetrievalRequest(
                **base_request.dict(),
                engine_params=engine_configs.get(engine_type) if engine_configs else None
            )
            
            tasks.append(self._run_single_engine(engine, request))
        
        # Execute all queries
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect valid responses
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Create error response
                responses[engines[i]] = EngineRetrievalResponse(
                    engine_type=engines[i],
                    mode=mode,
                    query=query,
                    results=[],
                    execution_time_ms=0.0,
                    metadata={"error": str(result)}
                )
            else:
                responses[engines[i]] = result
        
        # Calculate metrics
        calculated_metrics = await self._calculate_metrics(
            query, responses, metrics or [ComparisonMetric.RELEVANCE, ComparisonMetric.SPEED]
        )
        
        # Determine winner
        winner = self._determine_winner(calculated_metrics)
        
        # Generate analysis
        analysis = await self._generate_analysis(query, responses, calculated_metrics)
        
        return ComparisonResult(
            query=query,
            engines=engines,
            responses=responses,
            metrics=calculated_metrics,
            winner=winner,
            analysis=analysis
        )
    
    async def start_experiment(self, request: ExperimentRequest) -> str:
        """Start a new A/B testing experiment."""
        experiment_id = str(uuid.uuid4())
        
        # Create experiment response
        experiment = ExperimentResponse(
            experiment_id=experiment_id,
            name=request.name,
            status=ExperimentStatus.RUNNING,
            engines_tested=request.engines,
            total_queries=len(request.test_queries),
            results=[],
            created_at=datetime.utcnow()
        )
        
        self._experiments[experiment_id] = experiment
        
        # Start experiment task
        task = asyncio.create_task(
            self._run_experiment_task(experiment_id, request)
        )
        self._running_experiments[experiment_id] = task
        
        return experiment_id
    
    async def get_experiment(self, experiment_id: str) -> Optional[ExperimentResponse]:
        """Get experiment by ID."""
        return self._experiments.get(experiment_id)
    
    async def list_experiments(self) -> List[ExperimentResponse]:
        """List all experiments."""
        return list(self._experiments.values())
    
    async def cancel_experiment(self, experiment_id: str) -> bool:
        """Cancel a running experiment."""
        if experiment_id in self._running_experiments:
            task = self._running_experiments[experiment_id]
            task.cancel()
            
            if experiment_id in self._experiments:
                self._experiments[experiment_id].status = ExperimentStatus.CANCELLED
            
            return True
        return False
    
    async def _run_single_engine(
        self, 
        engine, 
        request: EngineRetrievalRequest
    ) -> EngineRetrievalResponse:
        """Run a single engine query."""
        return await engine.retrieve(request)
    
    async def _calculate_metrics(
        self,
        query: str,
        responses: Dict[EngineType, EngineRetrievalResponse],
        metrics: List[ComparisonMetric]
    ) -> Dict[str, Dict[EngineType, float]]:
        """Calculate comparison metrics for engine responses."""
        calculated = {}
        
        for metric in metrics:
            calculated[metric.value] = {}
            
            if metric == ComparisonMetric.SPEED:
                # Speed metric (lower is better, so we invert)
                max_time = max(r.execution_time_ms for r in responses.values())
                for engine_type, response in responses.items():
                    if max_time > 0:
                        calculated[metric.value][engine_type] = 1.0 - (response.execution_time_ms / max_time)
                    else:
                        calculated[metric.value][engine_type] = 1.0
            
            elif metric == ComparisonMetric.RELEVANCE:
                # Relevance based on average scores
                for engine_type, response in responses.items():
                    if response.results:
                        avg_score = sum(r.score for r in response.results) / len(response.results)
                        calculated[metric.value][engine_type] = avg_score
                    else:
                        calculated[metric.value][engine_type] = 0.0
            
            elif metric == ComparisonMetric.DIVERSITY:
                # Diversity based on unique sources
                for engine_type, response in responses.items():
                    if response.results:
                        unique_sources = len(set(r.source for r in response.results))
                        total_results = len(response.results)
                        calculated[metric.value][engine_type] = unique_sources / total_results if total_results > 0 else 0.0
                    else:
                        calculated[metric.value][engine_type] = 0.0
            
            elif metric == ComparisonMetric.COVERAGE:
                # Coverage based on number of results
                max_results = max(len(r.results) for r in responses.values())
                for engine_type, response in responses.items():
                    if max_results > 0:
                        calculated[metric.value][engine_type] = len(response.results) / max_results
                    else:
                        calculated[metric.value][engine_type] = 0.0
            
            elif metric == ComparisonMetric.CONSISTENCY:
                # Consistency based on score variance
                for engine_type, response in responses.items():
                    if response.results and len(response.results) > 1:
                        scores = [r.score for r in response.results]
                        variance = sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores)
                        calculated[metric.value][engine_type] = 1.0 - min(variance, 1.0)  # Lower variance = higher consistency
                    else:
                        calculated[metric.value][engine_type] = 1.0
        
        return calculated
    
    def _determine_winner(self, metrics: Dict[str, Dict[EngineType, float]]) -> Optional[EngineType]:
        """Determine the winning engine based on metrics."""
        if not metrics:
            return None
        
        # Calculate overall scores (simple average for now)
        overall_scores = defaultdict(float)
        metric_count = len(metrics)
        
        for metric_name, engine_scores in metrics.items():
            for engine_type, score in engine_scores.items():
                overall_scores[engine_type] += score
        
        # Average the scores
        for engine_type in overall_scores:
            overall_scores[engine_type] /= metric_count
        
        # Find winner
        if overall_scores:
            winner = max(overall_scores.items(), key=lambda x: x[1])
            return winner[0]
        
        return None
    
    async def _generate_analysis(
        self,
        query: str,
        responses: Dict[EngineType, EngineRetrievalResponse],
        metrics: Dict[str, Dict[EngineType, float]]
    ) -> Dict[str, Any]:
        """Generate detailed analysis of the comparison."""
        analysis = {
            "query_complexity": len(query.split()),
            "total_engines": len(responses),
            "successful_responses": sum(1 for r in responses.values() if r.results),
            "avg_execution_time": sum(r.execution_time_ms for r in responses.values()) / len(responses),
            "total_results": sum(len(r.results) for r in responses.values()),
            "metric_summary": {},
            "recommendations": [],
            "detailed_comparison": {}
        }
        
        # Metric summary
        for metric_name, engine_scores in metrics.items():
            analysis["metric_summary"][metric_name] = {
                "best_engine": max(engine_scores.items(), key=lambda x: x[1])[0].value if engine_scores else None,
                "worst_engine": min(engine_scores.items(), key=lambda x: x[1])[0].value if engine_scores else None,
                "score_range": max(engine_scores.values()) - min(engine_scores.values()) if engine_scores else 0,
                "average_score": sum(engine_scores.values()) / len(engine_scores) if engine_scores else 0
            }
        
        # Generate recommendations
        if "speed" in metrics and "relevance" in metrics:
            speed_leader = max(metrics["speed"].items(), key=lambda x: x[1])[0]
            relevance_leader = max(metrics["relevance"].items(), key=lambda x: x[1])[0]
            
            if speed_leader == relevance_leader:
                analysis["recommendations"].append(f"{speed_leader.value} provides the best balance of speed and relevance")
            else:
                analysis["recommendations"].append(f"Use {speed_leader.value} for speed-critical queries")
                analysis["recommendations"].append(f"Use {relevance_leader.value} for relevance-critical queries")
        
        # Detailed comparison
        for engine_type, response in responses.items():
            analysis["detailed_comparison"][engine_type.value] = {
                "execution_time_ms": response.execution_time_ms,
                "result_count": len(response.results),
                "avg_relevance": sum(r.score for r in response.results) / len(response.results) if response.results else 0,
                "unique_sources": len(set(r.source for r in response.results)) if response.results else 0,
                "has_errors": "error" in response.metadata
            }
        
        return analysis
    
    async def _run_experiment_task(self, experiment_id: str, request: ExperimentRequest):
        """Run the experiment in the background."""
        try:
            experiment = self._experiments[experiment_id]
            results = []
            
            total_operations = len(request.test_queries) * len(request.engines) * request.iterations
            completed_operations = 0
            
            for iteration in range(request.iterations):
                for query in request.test_queries:
                    # Run comparison for this query
                    comparison = await self.run_comparison(
                        query=query,
                        engines=request.engines,
                        engine_configs=request.engine_configs,
                        metrics=[ComparisonMetric(m) for m in request.metrics]
                    )
                    
                    # Create experiment results for each engine
                    for engine_type, response in comparison.responses.items():
                        result = ExperimentResult(
                            experiment_id=experiment_id,
                            engine_type=engine_type,
                            query=query,
                            iteration=iteration,
                            execution_time_ms=response.execution_time_ms,
                            num_results=len(response.results),
                            avg_score=sum(r.score for r in response.results) / len(response.results) if response.results else 0,
                            metadata={
                                "metrics": {
                                    metric_name: engine_scores.get(engine_type, 0.0)
                                    for metric_name, engine_scores in comparison.metrics.items()
                                },
                                "total_candidates": response.total_candidates,
                                "response_metadata": response.metadata
                            }
                        )
                        results.append(result)
                        completed_operations += 1
            
            # Calculate summary metrics
            summary_metrics = self._calculate_experiment_summary(results, request.engines)
            
            # Generate recommendations
            recommendations = self._generate_experiment_recommendations(summary_metrics, request.engines)
            
            # Update experiment
            experiment.results = results
            experiment.summary_metrics = summary_metrics
            experiment.recommendations = recommendations
            experiment.status = ExperimentStatus.COMPLETED
            experiment.completed_at = datetime.utcnow()
            
        except asyncio.CancelledError:
            experiment.status = ExperimentStatus.CANCELLED
        except Exception as e:
            experiment.status = ExperimentStatus.FAILED
            experiment.summary_metrics = {"error": str(e)}
        finally:
            # Clean up running task
            if experiment_id in self._running_experiments:
                del self._running_experiments[experiment_id]
    
    def _calculate_experiment_summary(
        self, 
        results: List[ExperimentResult], 
        engines: List[EngineType]
    ) -> Dict[str, Any]:
        """Calculate summary metrics for the experiment."""
        summary = {}
        
        # Group results by engine
        engine_results = defaultdict(list)
        for result in results:
            engine_results[result.engine_type].append(result)
        
        # Calculate per-engine summaries
        for engine_type in engines:
            engine_data = engine_results[engine_type]
            if not engine_data:
                continue
            
            summary[engine_type.value] = {
                "total_queries": len(engine_data),
                "avg_execution_time_ms": sum(r.execution_time_ms for r in engine_data) / len(engine_data),
                "avg_results_per_query": sum(r.num_results for r in engine_data) / len(engine_data),
                "avg_relevance_score": sum(r.avg_score for r in engine_data) / len(engine_data),
                "success_rate": sum(1 for r in engine_data if r.num_results > 0) / len(engine_data),
                "min_execution_time": min(r.execution_time_ms for r in engine_data),
                "max_execution_time": max(r.execution_time_ms for r in engine_data)
            }
        
        # Overall comparison
        if len(engines) > 1:
            summary["comparison"] = {
                "fastest_engine": min(engines, key=lambda e: summary.get(e.value, {}).get("avg_execution_time_ms", float('inf'))).value,
                "most_relevant_engine": max(engines, key=lambda e: summary.get(e.value, {}).get("avg_relevance_score", 0)).value,
                "most_reliable_engine": max(engines, key=lambda e: summary.get(e.value, {}).get("success_rate", 0)).value
            }
        
        return summary
    
    def _generate_experiment_recommendations(
        self, 
        summary_metrics: Dict[str, Any], 
        engines: List[EngineType]
    ) -> List[str]:
        """Generate recommendations based on experiment results."""
        recommendations = []
        
        if "comparison" in summary_metrics:
            comp = summary_metrics["comparison"]
            
            recommendations.append(f"For speed-critical applications, use {comp['fastest_engine']}")
            recommendations.append(f"For relevance-critical applications, use {comp['most_relevant_engine']}")
            recommendations.append(f"For reliability, use {comp['most_reliable_engine']}")
            
            # Check if one engine dominates
            if (comp["fastest_engine"] == comp["most_relevant_engine"] == comp["most_reliable_engine"]):
                recommendations.append(f"{comp['fastest_engine']} dominates across all metrics")
        
        # Performance recommendations
        for engine_type in engines:
            engine_stats = summary_metrics.get(engine_type.value, {})
            avg_time = engine_stats.get("avg_execution_time_ms", 0)
            
            if avg_time > 5000:  # 5 seconds
                recommendations.append(f"{engine_type.value} has high latency - consider optimization")
            
            success_rate = engine_stats.get("success_rate", 0)
            if success_rate < 0.9:
                recommendations.append(f"{engine_type.value} has low success rate - investigate failures")
        
        return recommendations


# Global experiment manager
_experiment_manager = ExperimentManager()


def get_experiment_manager() -> ExperimentManager:
    """Get the global experiment manager."""
    return _experiment_manager