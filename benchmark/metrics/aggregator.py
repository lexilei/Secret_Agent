"""
Metrics aggregation for benchmark results.

Computes summary statistics across experiments, calculators, and categories.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from collections import defaultdict
import statistics

from ..experiments.base import ExperimentResult


@dataclass
class CalculatorMetrics:
    """Metrics for a single calculator type."""
    calculator_name: str
    total_instances: int
    correct_exact: int
    correct_tolerance: int
    accuracy_exact: float
    accuracy_tolerance: float
    avg_latency_ms: float
    total_cost_usd: float
    avg_cost_usd: float
    error_count: int


@dataclass
class CategoryMetrics:
    """Metrics for a category (equation-based or rule-based)."""
    category: str
    total_instances: int
    correct_exact: int
    correct_tolerance: int
    accuracy_exact: float
    accuracy_tolerance: float
    avg_latency_ms: float
    total_cost_usd: float
    calculators: List[str]


@dataclass
class AggregatedMetrics:
    """Complete aggregated metrics for an experiment."""

    # Overall metrics
    experiment_name: str
    experiment_level: str
    total_instances: int
    correct_exact: int
    correct_tolerance: int
    accuracy_exact: float
    accuracy_tolerance: float
    error_count: int
    error_rate: float

    # Cost metrics
    total_cost_usd: float
    avg_cost_per_instance: float
    total_tokens: int
    avg_tokens_per_instance: float

    # Latency metrics
    total_latency_ms: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float

    # Framework-specific metrics
    total_steps: int
    avg_steps_per_instance: float
    repair_attempts_total: int
    repair_success_count: int
    repair_success_rate: float
    patterns_used_total: int
    patterns_learned_total: int

    # Breakdowns
    by_calculator: Dict[str, CalculatorMetrics] = field(default_factory=dict)
    by_category: Dict[str, CategoryMetrics] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "experiment_name": self.experiment_name,
            "experiment_level": self.experiment_level,
            "total_instances": self.total_instances,
            "correct_exact": self.correct_exact,
            "correct_tolerance": self.correct_tolerance,
            "accuracy_exact": self.accuracy_exact,
            "accuracy_tolerance": self.accuracy_tolerance,
            "error_count": self.error_count,
            "error_rate": self.error_rate,
            "total_cost_usd": self.total_cost_usd,
            "avg_cost_per_instance": self.avg_cost_per_instance,
            "total_tokens": self.total_tokens,
            "avg_tokens_per_instance": self.avg_tokens_per_instance,
            "total_latency_ms": self.total_latency_ms,
            "avg_latency_ms": self.avg_latency_ms,
            "p50_latency_ms": self.p50_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "min_latency_ms": self.min_latency_ms,
            "max_latency_ms": self.max_latency_ms,
            "total_steps": self.total_steps,
            "avg_steps_per_instance": self.avg_steps_per_instance,
            "repair_attempts_total": self.repair_attempts_total,
            "repair_success_count": self.repair_success_count,
            "repair_success_rate": self.repair_success_rate,
            "patterns_used_total": self.patterns_used_total,
            "patterns_learned_total": self.patterns_learned_total,
            "by_calculator": {k: vars(v) for k, v in self.by_calculator.items()},
            "by_category": {k: vars(v) for k, v in self.by_category.items()},
        }


class MetricsAggregator:
    """
    Aggregates experiment results into summary statistics.

    Usage:
        aggregator = MetricsAggregator()
        metrics = aggregator.aggregate(results, "baseline", "L0")
    """

    def aggregate(
        self,
        results: List[ExperimentResult],
        experiment_name: str,
        experiment_level: str,
    ) -> AggregatedMetrics:
        """
        Aggregate a list of experiment results.

        Args:
            results: List of ExperimentResult objects
            experiment_name: Name of the experiment
            experiment_level: Level (L0, L1, etc.)

        Returns:
            AggregatedMetrics with all statistics
        """
        if not results:
            return self._empty_metrics(experiment_name, experiment_level)

        total = len(results)

        # Basic counts
        correct_exact = sum(1 for r in results if r.is_correct_exact)
        correct_tolerance = sum(1 for r in results if r.is_correct_tolerance)
        errors = sum(1 for r in results if r.error is not None)

        # Latencies (exclude errors)
        latencies = [r.latency_ms for r in results if r.error is None and r.latency_ms > 0]
        if not latencies:
            latencies = [0.0]

        # Cost and tokens
        total_cost = sum(r.cost_usd for r in results)
        total_tokens = sum(r.total_tokens for r in results)

        # Framework metrics
        total_steps = sum(r.num_steps for r in results)
        repair_attempts = sum(r.repair_attempts for r in results)
        repair_successes = sum(1 for r in results if r.repair_success)
        patterns_used = sum(r.patterns_used for r in results)
        patterns_learned = sum(r.patterns_learned for r in results)

        # Compute latency percentiles
        sorted_latencies = sorted(latencies)
        p50_idx = len(sorted_latencies) // 2
        p95_idx = int(len(sorted_latencies) * 0.95)

        # Build aggregated metrics
        metrics = AggregatedMetrics(
            experiment_name=experiment_name,
            experiment_level=experiment_level,
            total_instances=total,
            correct_exact=correct_exact,
            correct_tolerance=correct_tolerance,
            accuracy_exact=correct_exact / total,
            accuracy_tolerance=correct_tolerance / total,
            error_count=errors,
            error_rate=errors / total,

            total_cost_usd=total_cost,
            avg_cost_per_instance=total_cost / total,
            total_tokens=total_tokens,
            avg_tokens_per_instance=total_tokens / total,

            total_latency_ms=sum(latencies),
            avg_latency_ms=statistics.mean(latencies),
            p50_latency_ms=sorted_latencies[p50_idx] if sorted_latencies else 0,
            p95_latency_ms=sorted_latencies[p95_idx] if sorted_latencies else 0,
            min_latency_ms=min(latencies) if latencies else 0,
            max_latency_ms=max(latencies) if latencies else 0,

            total_steps=total_steps,
            avg_steps_per_instance=total_steps / total,
            repair_attempts_total=repair_attempts,
            repair_success_count=repair_successes,
            repair_success_rate=repair_successes / repair_attempts if repair_attempts > 0 else 0,
            patterns_used_total=patterns_used,
            patterns_learned_total=patterns_learned,
        )

        # Compute per-calculator breakdown
        metrics.by_calculator = self._aggregate_by_calculator(results)
        metrics.by_category = self._aggregate_by_category(results)

        return metrics

    def _aggregate_by_calculator(
        self,
        results: List[ExperimentResult],
    ) -> Dict[str, CalculatorMetrics]:
        """Aggregate results by calculator name."""
        by_calc: Dict[str, List[ExperimentResult]] = defaultdict(list)
        for r in results:
            by_calc[r.calculator_name].append(r)

        calc_metrics = {}
        for calc_name, calc_results in by_calc.items():
            total = len(calc_results)
            correct_exact = sum(1 for r in calc_results if r.is_correct_exact)
            correct_tol = sum(1 for r in calc_results if r.is_correct_tolerance)
            errors = sum(1 for r in calc_results if r.error is not None)
            latencies = [r.latency_ms for r in calc_results if r.latency_ms > 0]
            costs = [r.cost_usd for r in calc_results]

            calc_metrics[calc_name] = CalculatorMetrics(
                calculator_name=calc_name,
                total_instances=total,
                correct_exact=correct_exact,
                correct_tolerance=correct_tol,
                accuracy_exact=correct_exact / total if total > 0 else 0,
                accuracy_tolerance=correct_tol / total if total > 0 else 0,
                avg_latency_ms=statistics.mean(latencies) if latencies else 0,
                total_cost_usd=sum(costs),
                avg_cost_usd=sum(costs) / total if total > 0 else 0,
                error_count=errors,
            )

        return calc_metrics

    def _aggregate_by_category(
        self,
        results: List[ExperimentResult],
    ) -> Dict[str, CategoryMetrics]:
        """Aggregate results by category."""
        by_cat: Dict[str, List[ExperimentResult]] = defaultdict(list)
        for r in results:
            by_cat[r.category].append(r)

        cat_metrics = {}
        for cat_name, cat_results in by_cat.items():
            total = len(cat_results)
            correct_exact = sum(1 for r in cat_results if r.is_correct_exact)
            correct_tol = sum(1 for r in cat_results if r.is_correct_tolerance)
            latencies = [r.latency_ms for r in cat_results if r.latency_ms > 0]
            costs = [r.cost_usd for r in cat_results]
            calculators = list(set(r.calculator_name for r in cat_results))

            cat_metrics[cat_name] = CategoryMetrics(
                category=cat_name,
                total_instances=total,
                correct_exact=correct_exact,
                correct_tolerance=correct_tol,
                accuracy_exact=correct_exact / total if total > 0 else 0,
                accuracy_tolerance=correct_tol / total if total > 0 else 0,
                avg_latency_ms=statistics.mean(latencies) if latencies else 0,
                total_cost_usd=sum(costs),
                calculators=calculators,
            )

        return cat_metrics

    def _empty_metrics(self, name: str, level: str) -> AggregatedMetrics:
        """Return empty metrics when no results."""
        return AggregatedMetrics(
            experiment_name=name,
            experiment_level=level,
            total_instances=0,
            correct_exact=0,
            correct_tolerance=0,
            accuracy_exact=0,
            accuracy_tolerance=0,
            error_count=0,
            error_rate=0,
            total_cost_usd=0,
            avg_cost_per_instance=0,
            total_tokens=0,
            avg_tokens_per_instance=0,
            total_latency_ms=0,
            avg_latency_ms=0,
            p50_latency_ms=0,
            p95_latency_ms=0,
            min_latency_ms=0,
            max_latency_ms=0,
            total_steps=0,
            avg_steps_per_instance=0,
            repair_attempts_total=0,
            repair_success_count=0,
            repair_success_rate=0,
            patterns_used_total=0,
            patterns_learned_total=0,
        )

    def compare_to_baseline(
        self,
        baseline: AggregatedMetrics,
        experiment: AggregatedMetrics,
    ) -> Dict[str, Any]:
        """
        Compare experiment metrics to baseline.

        Args:
            baseline: Baseline experiment metrics
            experiment: Experiment to compare

        Returns:
            Dict with improvements/changes
        """
        def safe_pct_change(new: float, old: float) -> float:
            if old == 0:
                return 0 if new == 0 else float('inf')
            return ((new - old) / old) * 100

        return {
            "experiment": experiment.experiment_name,
            "baseline": baseline.experiment_name,
            "accuracy_improvement_pct": (
                experiment.accuracy_tolerance - baseline.accuracy_tolerance
            ) * 100,
            "accuracy_improvement_abs": (
                experiment.accuracy_tolerance - baseline.accuracy_tolerance
            ),
            "cost_change_pct": safe_pct_change(
                experiment.total_cost_usd, baseline.total_cost_usd
            ),
            "latency_change_pct": safe_pct_change(
                experiment.avg_latency_ms, baseline.avg_latency_ms
            ),
            "tokens_change_pct": safe_pct_change(
                experiment.total_tokens, baseline.total_tokens
            ),
        }
