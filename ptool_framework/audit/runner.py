"""
Audit Runner: Execute audits on traces.

This module provides:
- AuditRunner: Run audits on individual traces or batches
- TypicalityTrainer: Train typicality models from trace stores
- BatchAuditResult: Results from batch auditing

Example:
    >>> from ptool_framework.audit import AuditRunner, TypicalityTrainer
    >>>
    >>> # Create runner with audits
    >>> runner = AuditRunner(audits=[
    ...     NonEmptyTraceAudit(),
    ...     NoFailedStepsAudit(),
    ...     my_domain_audit,
    ... ])
    >>>
    >>> # Audit a single trace
    >>> report = runner.audit_trace(trace)
    >>> print(report.summary())
    >>>
    >>> # Batch audit from trace store
    >>> batch_result = runner.audit_from_store("my_ptool", limit=100)
    >>> print(f"Pass rate: {batch_result.pass_rate:.1%}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    from ..traces import WorkflowTrace
    from ..react import ReActTrajectory
    from ..trace_store import ExecutionTrace, TraceStore
    from ..react import ReActStore

from .base import (
    AuditResult,
    AuditReport,
    AuditViolation,
    BaseAudit,
    TypicalityAudit,
    BaseTypicalityModel,
    get_audit_registry,
)
from .dataframe_converter import TraceDataFrameConverter, convert_trace
from .typicality.patterns import extract_pattern, PatternStats
from .typicality.models import BigramModel, create_typicality_model


@dataclass
class BatchAuditResult:
    """
    Results from auditing multiple traces.

    Attributes:
        reports: List of individual audit reports
        total_count: Total number of traces audited
        pass_count: Number of traces that passed
        fail_count: Number of traces that failed
        abstain_count: Number of abstentions
        pass_rate: Ratio of passes to total
        common_violations: Most common violations across traces

    Example:
        >>> result = runner.run_batch(traces)
        >>> print(f"Pass rate: {result.pass_rate:.1%}")
        >>> for violation, count in result.common_violations[:5]:
        ...     print(f"  {violation}: {count}")
    """

    reports: List[AuditReport] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    audit_names: List[str] = field(default_factory=list)

    @property
    def total_count(self) -> int:
        """Total traces audited."""
        return len(self.reports)

    @property
    def pass_count(self) -> int:
        """Number of traces that passed."""
        return sum(1 for r in self.reports if r.result == AuditResult.PASS)

    @property
    def fail_count(self) -> int:
        """Number of traces that failed."""
        return sum(1 for r in self.reports if r.result == AuditResult.FAIL)

    @property
    def abstain_count(self) -> int:
        """Number of abstentions."""
        return sum(1 for r in self.reports if r.result == AuditResult.ABSTAIN)

    @property
    def pass_rate(self) -> float:
        """Ratio of passes to total (excluding abstentions)."""
        decided = self.pass_count + self.fail_count
        if decided == 0:
            return 1.0
        return self.pass_count / decided

    @property
    def common_violations(self) -> List[Tuple[str, int]]:
        """Most common violation types across all reports."""
        from collections import Counter
        violation_counts = Counter()

        for report in self.reports:
            for violation in report.violations:
                key = f"{violation.audit_name}.{violation.rule_name}"
                violation_counts[key] += 1

        return violation_counts.most_common()

    def get_failed_traces(self) -> List[str]:
        """Get trace IDs of failed traces."""
        return [r.trace_id for r in self.reports if r.result == AuditResult.FAIL]

    def get_violation_stats(self) -> Dict[str, int]:
        """Get statistics about violations by audit."""
        stats = {}
        for report in self.reports:
            for violation in report.violations:
                stats[violation.audit_name] = stats.get(violation.audit_name, 0) + 1
        return stats

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Batch Audit Result",
            f"  Total traces: {self.total_count}",
            f"  Passed: {self.pass_count} ({self.pass_rate:.1%})",
            f"  Failed: {self.fail_count}",
            f"  Abstain: {self.abstain_count}",
        ]
        if self.common_violations:
            lines.append("  Top violations:")
            for violation, count in self.common_violations[:3]:
                lines.append(f"    - {violation}: {count}")
        return "\n".join(lines)


class AuditRunner:
    """
    Run audits on traces.

    The runner coordinates multiple audits and aggregates results.

    Attributes:
        audits: List of audits to run
        converter: DataFrame converter for traces

    Example:
        >>> runner = AuditRunner()
        >>> runner.add_audit(NonEmptyTraceAudit())
        >>> runner.add_audit(NoFailedStepsAudit())
        >>>
        >>> # Audit single trace
        >>> report = runner.audit_trace(my_trace)
        >>>
        >>> # Audit batch
        >>> results = runner.run_batch([trace1, trace2, trace3])
    """

    def __init__(
        self,
        audits: Optional[List[BaseAudit]] = None,
        use_registry: bool = True,
        converter: Optional[TraceDataFrameConverter] = None,
    ):
        """
        Initialize the runner.

        Args:
            audits: List of audits to run (if None, uses registry)
            use_registry: Whether to include audits from global registry
            converter: Custom DataFrame converter
        """
        self.audits: List[BaseAudit] = audits or []
        self.converter = converter or TraceDataFrameConverter()

        if use_registry and not audits:
            registry = get_audit_registry()
            self.audits.extend(registry.list_all())

    def add_audit(self, audit: BaseAudit) -> "AuditRunner":
        """
        Add an audit to the runner.

        Args:
            audit: Audit to add

        Returns:
            self for method chaining
        """
        self.audits.append(audit)
        return self

    def remove_audit(self, name: str) -> "AuditRunner":
        """
        Remove an audit by name.

        Args:
            name: Name of audit to remove

        Returns:
            self for method chaining
        """
        self.audits = [a for a in self.audits if a.name != name]
        return self

    def audit_trace(
        self,
        trace: Union["WorkflowTrace", "ReActTrajectory", "ExecutionTrace", List[Dict]],
        domain: Optional[str] = None,
    ) -> AuditReport:
        """
        Run all audits on a single trace.

        Args:
            trace: The trace to audit
            domain: Optional domain to filter audits by

        Returns:
            Combined AuditReport with all results

        Example:
            >>> report = runner.audit_trace(my_trace)
            >>> if report.is_pass:
            ...     print("All audits passed!")
            >>> else:
            ...     for v in report.violations:
            ...         print(f"Violation: {v.message}")
        """
        # Convert trace to DataFrame
        df, metadata = self.converter.convert(trace)

        # Determine which audits to run
        audits_to_run = self.audits
        if domain:
            registry = get_audit_registry()
            domain_audits = registry.list_by_domain(domain)
            audits_to_run = self.audits + [a for a in domain_audits if a not in self.audits]

        # Run all audits and collect results
        all_violations = []
        all_passed = []
        typicality_scores = []

        for audit in audits_to_run:
            report = audit.run(df, metadata)

            all_violations.extend(report.violations)
            all_passed.extend(report.passed_checks)

            if report.typicality_score is not None:
                typicality_scores.append(report.typicality_score)

        # Determine overall result
        has_errors = any(v.severity == "error" for v in all_violations)
        if has_errors:
            overall_result = AuditResult.FAIL
        elif all_violations:
            # Only warnings - could pass or abstain based on policy
            overall_result = AuditResult.PASS
        else:
            overall_result = AuditResult.PASS

        # Calculate average typicality score if any
        avg_typicality = None
        if typicality_scores:
            avg_typicality = sum(typicality_scores) / len(typicality_scores)

        return AuditReport(
            trace_id=metadata.get("trace_id", "unknown"),
            result=overall_result,
            violations=all_violations,
            passed_checks=all_passed,
            metadata={
                "audits_run": [a.name for a in audits_to_run],
                "domain": domain,
            },
            typicality_score=avg_typicality,
        )

    def run_batch(
        self,
        traces: List[Union["WorkflowTrace", "ReActTrajectory", "ExecutionTrace", List[Dict]]],
        domain: Optional[str] = None,
    ) -> BatchAuditResult:
        """
        Run audits on multiple traces.

        Args:
            traces: List of traces to audit
            domain: Optional domain filter

        Returns:
            BatchAuditResult with aggregated statistics

        Example:
            >>> result = runner.run_batch(all_traces)
            >>> print(f"Pass rate: {result.pass_rate:.1%}")
        """
        reports = []
        for trace in traces:
            try:
                report = self.audit_trace(trace, domain=domain)
                reports.append(report)
            except Exception as e:
                # Create a failure report for traces that couldn't be audited
                reports.append(AuditReport(
                    trace_id="unknown",
                    result=AuditResult.FAIL,
                    violations=[AuditViolation(
                        audit_name="runner",
                        rule_name="conversion",
                        message=f"Failed to audit trace: {e}",
                        severity="error",
                    )],
                ))

        return BatchAuditResult(
            reports=reports,
            audit_names=[a.name for a in self.audits],
        )

    def audit_from_store(
        self,
        ptool_name: str,
        trace_store: Optional["TraceStore"] = None,
        limit: int = 100,
        success_only: bool = True,
        domain: Optional[str] = None,
    ) -> BatchAuditResult:
        """
        Audit traces from a TraceStore.

        Args:
            ptool_name: Name of ptool to audit
            trace_store: TraceStore instance (if None, uses global)
            limit: Maximum number of traces to audit
            success_only: Only audit successful traces
            domain: Optional domain filter

        Returns:
            BatchAuditResult with aggregated statistics

        Example:
            >>> result = runner.audit_from_store("extract_data", limit=50)
            >>> print(f"Audited {result.total_count} traces")
        """
        from ..trace_store import get_trace_store

        store = trace_store or get_trace_store()
        traces = store.get_traces(ptool_name=ptool_name, limit=limit, success_only=success_only)

        return self.run_batch(traces, domain=domain)

    def audit_from_react_store(
        self,
        react_store: Optional["ReActStore"] = None,
        success_only: bool = True,
        limit: int = 100,
        domain: Optional[str] = None,
    ) -> BatchAuditResult:
        """
        Audit trajectories from a ReActStore.

        Args:
            react_store: ReActStore instance (if None, uses global)
            success_only: Only audit successful trajectories
            limit: Maximum number of trajectories to audit
            domain: Optional domain filter

        Returns:
            BatchAuditResult with aggregated statistics

        Example:
            >>> result = runner.audit_from_react_store(success_only=True)
        """
        from ..react import get_react_store

        store = react_store or get_react_store()

        if success_only:
            trajectories = store.get_successful_trajectories(limit=limit)
        else:
            summaries = store.list_trajectories(limit=limit)
            trajectories = []
            for summary in summaries:
                traj = store.get_trajectory(summary["trajectory_id"])
                if traj:
                    trajectories.append(traj)

        return self.run_batch(trajectories, domain=domain)


class TypicalityTrainer:
    """
    Train typicality models from trace stores.

    Example:
        >>> trainer = TypicalityTrainer(model_type="bigram")
        >>> trainer.train_from_store(ptool_name="extract_data", success_only=True)
        >>>
        >>> # Create typicality audit
        >>> audit = trainer.create_audit(name="extract_typicality")
        >>> runner.add_audit(audit)
    """

    def __init__(
        self,
        model_type: str = "bigram",
        model_kwargs: Optional[Dict[str, Any]] = None,
        converter: Optional[TraceDataFrameConverter] = None,
    ):
        """
        Initialize the trainer.

        Args:
            model_type: Type of model ("unigram", "bigram", "trigram", "hmm")
            model_kwargs: Additional arguments for model creation
            converter: Custom DataFrame converter
        """
        self.model_type = model_type
        self.model_kwargs = model_kwargs or {}
        self.converter = converter or TraceDataFrameConverter()

        self.model: Optional[BaseTypicalityModel] = None
        self.patterns: List[List[str]] = []
        self.stats: Optional[PatternStats] = None

    def train(
        self,
        traces: List[Union["WorkflowTrace", "ReActTrajectory", "ExecutionTrace", List[Dict]]],
    ) -> "TypicalityTrainer":
        """
        Train the model on traces.

        Args:
            traces: List of traces to train on

        Returns:
            self for method chaining

        Example:
            >>> trainer.train(good_traces)
            >>> print(f"Trained on {len(trainer.patterns)} patterns")
        """
        self.patterns = []

        for trace in traces:
            df, _ = self.converter.convert(trace)
            pattern = extract_pattern(df)
            self.patterns.append(pattern)

        # Compute statistics
        self.stats = PatternStats.from_patterns(self.patterns)

        # Create and train model
        self.model = create_typicality_model(self.model_type, **self.model_kwargs)
        self.model.fit(self.patterns)

        return self

    def train_from_store(
        self,
        ptool_name: Optional[str] = None,
        trace_store: Optional["TraceStore"] = None,
        success_only: bool = True,
        limit: int = 1000,
    ) -> "TypicalityTrainer":
        """
        Train from a TraceStore.

        Args:
            ptool_name: Name of ptool (if None, uses all)
            trace_store: TraceStore instance (if None, uses global)
            success_only: Only use successful traces
            limit: Maximum traces to use

        Returns:
            self for method chaining

        Example:
            >>> trainer.train_from_store("extract_data", success_only=True)
        """
        from ..trace_store import get_trace_store

        store = trace_store or get_trace_store()
        traces = store.get_traces(
            ptool_name=ptool_name,
            limit=limit,
            success_only=success_only,
        )

        return self.train(traces)

    def train_from_react_store(
        self,
        react_store: Optional["ReActStore"] = None,
        success_only: bool = True,
        limit: int = 1000,
        goal_filter: Optional[str] = None,
    ) -> "TypicalityTrainer":
        """
        Train from a ReActStore.

        Args:
            react_store: ReActStore instance (if None, uses global)
            success_only: Only use successful trajectories
            limit: Maximum trajectories to use
            goal_filter: Optional goal substring filter

        Returns:
            self for method chaining

        Example:
            >>> trainer.train_from_react_store(success_only=True)
        """
        from ..react import get_react_store

        store = react_store or get_react_store()

        if success_only:
            trajectories = store.get_successful_trajectories(
                limit=limit,
                goal_filter=goal_filter,
            )
        else:
            summaries = store.list_trajectories(limit=limit)
            if goal_filter:
                summaries = [
                    s for s in summaries
                    if goal_filter.lower() in s["goal"].lower()
                ]
            trajectories = []
            for summary in summaries[:limit]:
                traj = store.get_trajectory(summary["trajectory_id"])
                if traj:
                    trajectories.append(traj)

        return self.train(trajectories)

    def create_audit(
        self,
        name: str = "typicality",
        quantile_thresholds: Tuple[float, float] = (0.2, 0.8),
        description: str = "",
    ) -> TypicalityAudit:
        """
        Create a TypicalityAudit from the trained model.

        Args:
            name: Audit name
            quantile_thresholds: (fail_threshold, pass_threshold)
            description: Human-readable description

        Returns:
            TypicalityAudit instance

        Raises:
            ValueError: If model hasn't been trained

        Example:
            >>> trainer.train(traces)
            >>> audit = trainer.create_audit("my_typicality")
            >>> runner.add_audit(audit)
        """
        if self.model is None:
            raise ValueError("Model must be trained before creating audit. Call train() first.")

        return TypicalityAudit(
            name=name,
            model=self.model,
            quantile_thresholds=quantile_thresholds,
            description=description or f"Typicality audit using {self.model_type} model",
        )

    def score_trace(
        self,
        trace: Union["WorkflowTrace", "ReActTrajectory", "ExecutionTrace", List[Dict]],
    ) -> float:
        """
        Score a trace's typicality.

        Args:
            trace: Trace to score

        Returns:
            Typicality score (0-1)

        Raises:
            ValueError: If model hasn't been trained
        """
        if self.model is None:
            raise ValueError("Model must be trained before scoring. Call train() first.")

        df, _ = self.converter.convert(trace)
        pattern = extract_pattern(df)
        return self.model.score(pattern)

    def get_statistics(self) -> Optional[PatternStats]:
        """Get training pattern statistics."""
        return self.stats

    def save_model(self, path: str) -> None:
        """
        Save the trained model to disk.

        Args:
            path: File path to save to
        """
        if self.model is None:
            raise ValueError("No model to save. Call train() first.")
        self.model.save(path)

    def load_model(self, path: str) -> "TypicalityTrainer":
        """
        Load a model from disk.

        Args:
            path: File path to load from

        Returns:
            self for method chaining
        """
        # Determine model type from file if possible
        self.model = create_typicality_model(self.model_type)
        self.model = type(self.model).load(path)
        return self


# =============================================================================
# Convenience Functions
# =============================================================================

def audit_trace(
    trace: Union["WorkflowTrace", "ReActTrajectory", "ExecutionTrace", List[Dict]],
    audits: Optional[List[BaseAudit]] = None,
    domain: Optional[str] = None,
) -> AuditReport:
    """
    Convenience function to audit a single trace.

    Args:
        trace: Trace to audit
        audits: Optional list of audits (if None, uses registry)
        domain: Optional domain filter

    Returns:
        AuditReport with results

    Example:
        >>> report = audit_trace(my_trace)
        >>> print(report.summary())
    """
    runner = AuditRunner(audits=audits)
    return runner.audit_trace(trace, domain=domain)


def batch_audit(
    traces: List[Union["WorkflowTrace", "ReActTrajectory", "ExecutionTrace", List[Dict]]],
    audits: Optional[List[BaseAudit]] = None,
    domain: Optional[str] = None,
) -> BatchAuditResult:
    """
    Convenience function to audit multiple traces.

    Args:
        traces: List of traces to audit
        audits: Optional list of audits
        domain: Optional domain filter

    Returns:
        BatchAuditResult with aggregated statistics

    Example:
        >>> result = batch_audit(all_traces)
        >>> print(f"Pass rate: {result.pass_rate:.1%}")
    """
    runner = AuditRunner(audits=audits)
    return runner.run_batch(traces, domain=domain)


def train_typicality_model(
    traces: List[Union["WorkflowTrace", "ReActTrajectory", "ExecutionTrace", List[Dict]]],
    model_type: str = "bigram",
    **model_kwargs,
) -> BaseTypicalityModel:
    """
    Train a typicality model on traces.

    Args:
        traces: Traces to train on
        model_type: Type of model
        **model_kwargs: Additional model arguments

    Returns:
        Trained model

    Example:
        >>> model = train_typicality_model(good_traces, model_type="bigram")
        >>> score = model.score(new_pattern)
    """
    trainer = TypicalityTrainer(model_type=model_type, model_kwargs=model_kwargs)
    trainer.train(traces)
    return trainer.model
