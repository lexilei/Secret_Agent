"""
Trace Critic: Evaluate trace quality and suggest repairs.

This module implements the critic component of William's Approach 3:
using audits to evaluate traces and determine if they need repair.

Key Classes:
    - CriticVerdict: Evaluation outcomes (ACCEPT, REPAIR_NEEDED, REJECT)
    - CriticEvaluation: Detailed evaluation with confidence and suggestions
    - TraceCritic: Main critic class that runs audits and generates evaluations

The critic integrates:
- Structured audits (SSRM-style rule checks)
- Typicality audits (probabilistic pattern models)
- LLM-based reasoning analysis

Example:
    >>> from ptool_framework.critic import TraceCritic, CriticVerdict
    >>>
    >>> # Create critic with audits
    >>> critic = TraceCritic(
    ...     audits=[structure_audit, typicality_audit],
    ...     model="deepseek-v3-0324"
    ... )
    >>>
    >>> # Evaluate a trace
    >>> evaluation = critic.evaluate(trace, trajectory, goal)
    >>>
    >>> if evaluation.verdict == CriticVerdict.REPAIR_NEEDED:
    ...     print("Trace needs repair:")
    ...     for suggestion in evaluation.repair_suggestions:
    ...         print(f"  - {suggestion['action']}: {suggestion['reason']}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    from .traces import WorkflowTrace
    from .react import ReActTrajectory
    from .trace_store import ExecutionTrace, TraceStore
    from .react import ReActStore

from .audit.base import (
    AuditResult,
    AuditReport,
    AuditViolation,
    BaseAudit,
    get_audit_registry,
)
from .audit.dataframe_converter import TraceDataFrameConverter, convert_trace
from .audit.runner import AuditRunner
from .audit.typicality.patterns import extract_pattern
from .audit.typicality.models import BaseTypicalityModel


class CriticVerdict(Enum):
    """
    Verdict from the critic evaluation.

    Attributes:
        ACCEPT: Trace is good, no action needed
        REPAIR_NEEDED: Trace has issues but can be repaired
        REJECT: Trace is fundamentally flawed, needs re-execution
    """
    ACCEPT = "accept"
    REPAIR_NEEDED = "repair_needed"
    REJECT = "reject"


@dataclass
class CriticEvaluation:
    """
    Complete evaluation from the critic.

    Attributes:
        verdict: Overall verdict (ACCEPT, REPAIR_NEEDED, REJECT)
        confidence: Confidence in the verdict (0-1)
        audit_reports: Results from individual audits
        failed_steps: Indices of steps that failed
        reasoning_issues: List of reasoning issues found
        completeness_score: How complete the trace is (0-1)
        correctness_score: How correct the trace appears (0-1)
        efficiency_score: How efficient the trace is (0-1)
        repair_suggestions: List of suggested repairs
        trace_id: ID of the evaluated trace
        goal: The goal being evaluated
        timestamp: When evaluation was performed

    Example:
        >>> eval = critic.evaluate(trace, trajectory, goal)
        >>> print(f"Verdict: {eval.verdict.value}")
        >>> print(f"Confidence: {eval.confidence:.2f}")
        >>> if eval.repair_suggestions:
        ...     print("Suggested repairs:")
        ...     for s in eval.repair_suggestions:
        ...         print(f"  - {s['action']}")
    """

    verdict: CriticVerdict
    confidence: float
    audit_reports: List[AuditReport] = field(default_factory=list)
    failed_steps: List[int] = field(default_factory=list)
    reasoning_issues: List[str] = field(default_factory=list)

    # Scores
    completeness_score: float = 1.0
    correctness_score: float = 1.0
    efficiency_score: float = 1.0

    # Repair suggestions
    repair_suggestions: List[Dict[str, Any]] = field(default_factory=list)

    # Metadata
    trace_id: str = ""
    goal: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def is_acceptable(self) -> bool:
        """Check if trace is acceptable."""
        return self.verdict == CriticVerdict.ACCEPT

    @property
    def needs_repair(self) -> bool:
        """Check if trace needs repair."""
        return self.verdict == CriticVerdict.REPAIR_NEEDED

    @property
    def should_reject(self) -> bool:
        """Check if trace should be rejected."""
        return self.verdict == CriticVerdict.REJECT

    @property
    def total_violations(self) -> int:
        """Total number of violations across all audits."""
        return sum(len(r.violations) for r in self.audit_reports)

    @property
    def error_violations(self) -> List[AuditViolation]:
        """Get all error-severity violations."""
        violations = []
        for report in self.audit_reports:
            violations.extend(v for v in report.violations if v.severity == "error")
        return violations

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "verdict": self.verdict.value,
            "confidence": self.confidence,
            "audit_reports": [r.to_dict() for r in self.audit_reports],
            "failed_steps": self.failed_steps,
            "reasoning_issues": self.reasoning_issues,
            "completeness_score": self.completeness_score,
            "correctness_score": self.correctness_score,
            "efficiency_score": self.efficiency_score,
            "repair_suggestions": self.repair_suggestions,
            "trace_id": self.trace_id,
            "goal": self.goal,
            "timestamp": self.timestamp,
        }

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Critic Evaluation for {self.trace_id}",
            f"  Verdict: {self.verdict.value.upper()}",
            f"  Confidence: {self.confidence:.2f}",
            f"  Scores: completeness={self.completeness_score:.2f}, "
            f"correctness={self.correctness_score:.2f}, efficiency={self.efficiency_score:.2f}",
            f"  Violations: {self.total_violations} ({len(self.error_violations)} errors)",
        ]
        if self.repair_suggestions:
            lines.append(f"  Repair suggestions: {len(self.repair_suggestions)}")
        return "\n".join(lines)


class TraceCritic:
    """
    Critic that evaluates traces and determines if they need repair.

    The critic runs multiple audits, analyzes reasoning patterns,
    and generates repair suggestions when issues are found.

    Attributes:
        audits: List of audits to run
        typicality_model: Optional model for typicality checking
        model: LLM model for reasoning analysis

    Example:
        >>> critic = TraceCritic(
        ...     audits=[structure_audit, domain_audit],
        ...     typicality_model=trained_bigram_model,
        ...     model="deepseek-v3-0324"
        ... )
        >>>
        >>> eval = critic.evaluate(trace, trajectory, goal)
        >>> print(eval.summary())
    """

    def __init__(
        self,
        audits: Optional[List[BaseAudit]] = None,
        typicality_model: Optional[BaseTypicalityModel] = None,
        trace_store: Optional["TraceStore"] = None,
        react_store: Optional["ReActStore"] = None,
        model: str = "deepseek-v3-0324",
        llm_backend: Optional[Callable] = None,
        # Thresholds
        accept_threshold: float = 0.8,
        repair_threshold: float = 0.4,
        typicality_threshold: float = 0.3,
    ):
        """
        Initialize the critic.

        Args:
            audits: List of audits to run
            typicality_model: Trained model for pattern checking
            trace_store: TraceStore for historical comparison
            react_store: ReActStore for trajectory history
            model: LLM model for reasoning analysis
            llm_backend: Custom LLM backend
            accept_threshold: Score threshold for ACCEPT (default 0.8)
            repair_threshold: Score threshold for REPAIR (vs REJECT) (default 0.4)
            typicality_threshold: Minimum typicality score (default 0.3)
        """
        self.audits = audits or []
        self.typicality_model = typicality_model
        self.trace_store = trace_store
        self.react_store = react_store
        self.model = model
        self.llm_backend = llm_backend

        # Thresholds
        self.accept_threshold = accept_threshold
        self.repair_threshold = repair_threshold
        self.typicality_threshold = typicality_threshold

        # Tools
        self.converter = TraceDataFrameConverter()
        self.runner = AuditRunner(audits=audits, use_registry=False)

    def add_audit(self, audit: BaseAudit) -> "TraceCritic":
        """
        Add an audit to the critic.

        Args:
            audit: Audit to add

        Returns:
            self for method chaining
        """
        self.audits.append(audit)
        self.runner.add_audit(audit)
        return self

    def evaluate(
        self,
        trace: Union["WorkflowTrace", "ReActTrajectory", "ExecutionTrace", List[Dict]],
        trajectory: Optional["ReActTrajectory"] = None,
        goal: str = "",
        domain: Optional[str] = None,
    ) -> CriticEvaluation:
        """
        Evaluate a trace and generate verdict.

        Args:
            trace: The trace to evaluate
            trajectory: Optional ReAct trajectory for additional context
            goal: The goal the trace was trying to achieve
            domain: Optional domain for domain-specific audits

        Returns:
            CriticEvaluation with verdict and suggestions

        Example:
            >>> eval = critic.evaluate(trace, goal="Calculate BMI")
            >>> if eval.needs_repair:
            ...     repaired = repair_agent.repair(trace, eval, goal)
        """
        # Convert trace to DataFrame
        df, metadata = self.converter.convert(trace)
        metadata["goal"] = goal or metadata.get("goal", "")

        # 1. Run structured audits
        audit_reports = self._run_audits(df, metadata, domain)

        # 2. Check typicality
        typicality_score = self._check_typicality(df)

        # 3. Detect failed steps
        failed_steps = self._detect_failed_steps(df)

        # 4. Analyze reasoning (if LLM available)
        reasoning_issues = []
        if trajectory and self.model:
            reasoning_issues = self._analyze_reasoning(trajectory, goal)

        # 5. Calculate scores
        completeness_score = self._calculate_completeness(df, metadata)
        correctness_score = self._calculate_correctness(audit_reports, failed_steps)
        efficiency_score = self._calculate_efficiency(df)

        # 6. Determine verdict
        verdict, confidence = self._determine_verdict(
            audit_reports=audit_reports,
            typicality_score=typicality_score,
            failed_steps=failed_steps,
            reasoning_issues=reasoning_issues,
            completeness_score=completeness_score,
            correctness_score=correctness_score,
        )

        # 7. Generate repair suggestions
        repair_suggestions = []
        if verdict == CriticVerdict.REPAIR_NEEDED:
            repair_suggestions = self._generate_repair_suggestions(
                audit_reports=audit_reports,
                failed_steps=failed_steps,
                reasoning_issues=reasoning_issues,
                df=df,
            )

        return CriticEvaluation(
            verdict=verdict,
            confidence=confidence,
            audit_reports=audit_reports,
            failed_steps=failed_steps,
            reasoning_issues=reasoning_issues,
            completeness_score=completeness_score,
            correctness_score=correctness_score,
            efficiency_score=efficiency_score,
            repair_suggestions=repair_suggestions,
            trace_id=metadata.get("trace_id", "unknown"),
            goal=goal,
        )

    def _run_audits(
        self,
        df: "pd.DataFrame",
        metadata: Dict[str, Any],
        domain: Optional[str],
    ) -> List[AuditReport]:
        """Run all configured audits."""
        reports = []

        # Run configured audits
        for audit in self.audits:
            try:
                report = audit.run(df, metadata)
                reports.append(report)
            except Exception as e:
                # Create error report
                reports.append(AuditReport(
                    trace_id=metadata.get("trace_id", "unknown"),
                    result=AuditResult.FAIL,
                    violations=[AuditViolation(
                        audit_name=audit.name,
                        rule_name="execution",
                        message=f"Audit raised exception: {e}",
                        severity="error",
                    )],
                ))

        # Run domain-specific audits if specified
        if domain:
            registry = get_audit_registry()
            for audit in registry.list_by_domain(domain):
                if audit not in self.audits:
                    try:
                        report = audit.run(df, metadata)
                        reports.append(report)
                    except Exception:
                        pass

        return reports

    def _check_typicality(self, df: "pd.DataFrame") -> Optional[float]:
        """Check pattern typicality against model."""
        if self.typicality_model is None:
            return None

        pattern = extract_pattern(df)
        return self.typicality_model.score(pattern)

    def _detect_failed_steps(self, df: "pd.DataFrame") -> List[int]:
        """Find indices of failed steps."""
        failed = []
        for idx, row in df.iterrows():
            if row.get("status") == "failed":
                failed.append(int(row.get("step_idx", idx)))
        return failed

    def _analyze_reasoning(
        self,
        trajectory: "ReActTrajectory",
        goal: str,
    ) -> List[str]:
        """Use LLM to analyze reasoning quality."""
        from .llm_backend import call_llm

        issues = []

        # Build analysis prompt
        prompt = self._build_reasoning_analysis_prompt(trajectory, goal)

        try:
            if self.llm_backend:
                response = self.llm_backend(prompt, self.model)
            else:
                response = call_llm(prompt, self.model)

            # Parse issues from response
            issues = self._parse_reasoning_issues(response)
        except Exception:
            pass  # LLM analysis is optional

        return issues

    def _build_reasoning_analysis_prompt(
        self,
        trajectory: "ReActTrajectory",
        goal: str,
    ) -> str:
        """Build prompt for reasoning analysis."""
        # Format trajectory steps
        steps_str = []
        for i, step in enumerate(trajectory.steps):
            if step.thought:
                steps_str.append(f"Step {i+1} Thought: {step.thought.content[:200]}")
            if step.action:
                steps_str.append(f"Step {i+1} Action: {step.action.ptool_name}({step.action.args})")
            if step.observation:
                result = str(step.observation.result)[:100]
                steps_str.append(f"Step {i+1} Result: {result}")

        prompt = f"""Analyze this reasoning trace for logical errors or omissions.

GOAL: {goal}

REASONING STEPS:
{chr(10).join(steps_str)}

FINAL ANSWER: {trajectory.final_answer}

Identify any issues with the reasoning. Format each issue as:
ISSUE: <description>

If no issues found, respond with:
NO_ISSUES

Be specific about what's wrong and why."""

        return prompt

    def _parse_reasoning_issues(self, response: str) -> List[str]:
        """Parse issues from LLM response."""
        if "NO_ISSUES" in response.upper():
            return []

        issues = []
        import re
        for match in re.finditer(r'ISSUE:\s*(.+?)(?=ISSUE:|$)', response, re.IGNORECASE | re.DOTALL):
            issue = match.group(1).strip()
            if issue:
                issues.append(issue[:200])  # Truncate long issues

        return issues

    def _calculate_completeness(
        self,
        df: "pd.DataFrame",
        metadata: Dict[str, Any],
    ) -> float:
        """Calculate trace completeness score."""
        if df.empty:
            return 0.0

        # Factors for completeness
        factors = []

        # Has steps?
        factors.append(1.0 if len(df) > 0 else 0.0)

        # Has final answer?
        has_answer = metadata.get("final_answer") is not None
        factors.append(1.0 if has_answer else 0.5)

        # Trace marked as success?
        is_success = metadata.get("success", True)
        factors.append(1.0 if is_success else 0.3)

        return sum(factors) / len(factors)

    def _calculate_correctness(
        self,
        audit_reports: List[AuditReport],
        failed_steps: List[int],
    ) -> float:
        """Calculate correctness score based on audits."""
        if not audit_reports:
            # No audits run - assume correct
            return 1.0 if not failed_steps else 0.5

        # Count passed vs failed audits
        passed = sum(1 for r in audit_reports if r.result == AuditResult.PASS)
        total = len(audit_reports)

        audit_score = passed / total if total > 0 else 1.0

        # Penalize for failed steps
        if failed_steps:
            step_penalty = min(0.3, len(failed_steps) * 0.1)
            audit_score -= step_penalty

        return max(0.0, audit_score)

    def _calculate_efficiency(self, df: "pd.DataFrame") -> float:
        """Calculate efficiency score."""
        if df.empty:
            return 1.0

        # Factors for efficiency
        step_count = len(df)

        # Penalize very long traces
        if step_count > 10:
            step_score = max(0.5, 1.0 - (step_count - 10) * 0.05)
        else:
            step_score = 1.0

        # Check for repeated steps (inefficient)
        fn_counts = df["fn_name"].value_counts()
        max_repeats = fn_counts.max() if not fn_counts.empty else 1
        repeat_score = 1.0 if max_repeats <= 2 else max(0.5, 1.0 - (max_repeats - 2) * 0.1)

        return (step_score + repeat_score) / 2

    def _determine_verdict(
        self,
        audit_reports: List[AuditReport],
        typicality_score: Optional[float],
        failed_steps: List[int],
        reasoning_issues: List[str],
        completeness_score: float,
        correctness_score: float,
    ) -> Tuple[CriticVerdict, float]:
        """Determine overall verdict and confidence."""
        # Calculate overall score
        scores = [completeness_score, correctness_score]
        weights = [0.3, 0.5]

        # Add typicality if available
        if typicality_score is not None:
            scores.append(typicality_score)
            weights.append(0.2)

        # Weight-normalized score
        total_weight = sum(weights)
        overall_score = sum(s * w for s, w in zip(scores, weights)) / total_weight

        # Count error-level violations
        error_count = 0
        for report in audit_reports:
            error_count += sum(1 for v in report.violations if v.severity == "error")

        # Determine verdict
        if overall_score >= self.accept_threshold and error_count == 0 and not failed_steps:
            verdict = CriticVerdict.ACCEPT
            confidence = min(1.0, overall_score + 0.1)
        elif overall_score >= self.repair_threshold or error_count <= 2:
            verdict = CriticVerdict.REPAIR_NEEDED
            confidence = 0.5 + (overall_score - self.repair_threshold) / 2
        else:
            verdict = CriticVerdict.REJECT
            confidence = 1.0 - overall_score

        # Adjust confidence based on evidence
        if typicality_score is not None and typicality_score < self.typicality_threshold:
            confidence *= 0.9  # Less confident if atypical

        if len(reasoning_issues) > 3:
            confidence *= 0.85  # Less confident if many reasoning issues

        return verdict, min(1.0, max(0.0, confidence))

    def _generate_repair_suggestions(
        self,
        audit_reports: List[AuditReport],
        failed_steps: List[int],
        reasoning_issues: List[str],
        df: "pd.DataFrame",
    ) -> List[Dict[str, Any]]:
        """Generate actionable repair suggestions."""
        suggestions = []

        # Suggestions from audit violations
        for report in audit_reports:
            for violation in report.violations:
                if violation.severity == "error":
                    suggestion = self._violation_to_suggestion(violation, df)
                    if suggestion:
                        suggestions.append(suggestion)

        # Suggestions from failed steps
        for step_idx in failed_steps:
            suggestions.append({
                "action": "regenerate_step",
                "step_index": step_idx,
                "reason": f"Step {step_idx} failed during execution",
                "priority": "high",
            })

        # Suggestions from reasoning issues
        for issue in reasoning_issues[:3]:  # Limit to top 3
            suggestions.append({
                "action": "review_reasoning",
                "reason": issue,
                "priority": "medium",
            })

        # Deduplicate and prioritize
        seen = set()
        unique_suggestions = []
        for s in suggestions:
            key = (s["action"], s.get("step_index", ""), s["reason"][:50])
            if key not in seen:
                seen.add(key)
                unique_suggestions.append(s)

        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        unique_suggestions.sort(key=lambda x: priority_order.get(x.get("priority", "low"), 2))

        return unique_suggestions[:10]  # Limit suggestions

    def _violation_to_suggestion(
        self,
        violation: AuditViolation,
        df: "pd.DataFrame",
    ) -> Optional[Dict[str, Any]]:
        """Convert an audit violation to a repair suggestion."""
        rule_name = violation.rule_name.lower()
        message = violation.message.lower()
        step_index = violation.location.get("step_index") if violation.location else None

        # Map common violations to actions
        if "missing" in message or "must have" in message or "required" in message:
            return {
                "action": "add_step",
                "reason": violation.message,
                "step_index": step_index,
                "details": {"rule": violation.rule_name},
                "priority": "high",
            }
        elif "order" in rule_name or "before" in message or "after" in message:
            return {
                "action": "reorder_steps",
                "reason": violation.message,
                "step_index": step_index,
                "details": {"rule": violation.rule_name},
                "priority": "medium",
            }
        elif "failed" in message or "error" in message:
            return {
                "action": "regenerate_step",
                "reason": violation.message,
                "step_index": step_index,
                "priority": "high",
            }
        elif "differs" in message or "incorrect" in message or "wrong" in message or "calculation" in rule_name:
            # Calculation errors - regenerate the calculation step
            return {
                "action": "regenerate_step",
                "reason": violation.message,
                "step_index": step_index,
                "details": violation.context if violation.context else {"rule": violation.rule_name},
                "priority": "high",
            }
        elif "regenerate" in message:
            # Explicit regeneration request in the message
            return {
                "action": "regenerate_step",
                "reason": violation.message,
                "step_index": step_index,
                "details": {"rule": violation.rule_name},
                "priority": "high",
            }
        elif step_index is not None:
            # If we have a step index, try to regenerate that step
            return {
                "action": "regenerate_step",
                "reason": violation.message,
                "step_index": step_index,
                "details": {"rule": violation.rule_name},
                "priority": "medium",
            }

        # Last resort: if no step_index, suggest reviewing (can't regenerate without knowing which step)
        return {
            "action": "review",
            "reason": violation.message,
            "step_index": step_index,
            "details": {"audit": violation.audit_name, "rule": violation.rule_name},
            "priority": "low",
        }


# =============================================================================
# Convenience Functions
# =============================================================================

def evaluate_trace(
    trace: Union["WorkflowTrace", "ReActTrajectory", "ExecutionTrace", List[Dict]],
    audits: Optional[List[BaseAudit]] = None,
    goal: str = "",
    domain: Optional[str] = None,
    model: str = "deepseek-v3-0324",
) -> CriticEvaluation:
    """
    Convenience function to evaluate a trace.

    Args:
        trace: Trace to evaluate
        audits: Optional list of audits
        goal: The goal
        domain: Optional domain
        model: LLM model

    Returns:
        CriticEvaluation

    Example:
        >>> eval = evaluate_trace(my_trace, goal="Calculate BMI")
        >>> print(eval.verdict)
    """
    critic = TraceCritic(audits=audits, model=model)
    return critic.evaluate(trace, goal=goal, domain=domain)


def quick_evaluate(
    trace: Union["WorkflowTrace", "ReActTrajectory", "ExecutionTrace", List[Dict]],
) -> CriticVerdict:
    """
    Quick evaluation returning just the verdict.

    Args:
        trace: Trace to evaluate

    Returns:
        CriticVerdict

    Example:
        >>> if quick_evaluate(trace) == CriticVerdict.ACCEPT:
        ...     print("Trace is good!")
    """
    critic = TraceCritic()
    eval_result = critic.evaluate(trace)
    return eval_result.verdict
