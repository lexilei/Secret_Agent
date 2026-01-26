"""
Base classes for SSRM-style auditing system.

This module provides the foundation for Semi-Structured Reasoning Model (SSRM) audits
as described in "Semi-structured LLM Reasoners Can Be Rigorously Audited" (Cohen et al.).

The audit system enables:
- Structured audits: Python unit tests using assertions on trace DataFrames
- Typicality audits: Probabilistic models over reasoning patterns

Key Classes:
    - AuditResult: Enum for audit outcomes (PASS, FAIL, ABSTAIN)
    - AuditViolation: Details about a specific audit failure
    - AuditReport: Complete report from running audits on a trace
    - BaseAudit: Abstract base class for all audit types
    - StructuredAudit: Base for audits using DataFrame assertions
    - TypicalityAudit: Base for audits using probabilistic models
    - BaseTypicalityModel: Abstract base for probabilistic models
    - AuditRegistry: Registry for managing available audits

Example:
    >>> from ptool_framework.audit import AuditRunner, StructuredAudit
    >>>
    >>> # Create a custom audit
    >>> audit = StructuredAudit("my_audit", "Check trace validity")
    >>> audit.add_rule(
    ...     "has_steps",
    ...     lambda df, _: len(df) > 0,
    ...     "Trace must have at least one step"
    ... )
    >>>
    >>> # Run on a trace
    >>> runner = AuditRunner(audits=[audit])
    >>> report = runner.audit_trace(my_trace)
    >>> print(report.result)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


class AuditResult(Enum):
    """
    Result of an audit check.

    Attributes:
        PASS: Audit passed - trace meets requirements
        FAIL: Audit failed - trace has issues
        ABSTAIN: Audit cannot make a determination (e.g., uncertain typicality score)
    """
    PASS = "pass"
    FAIL = "fail"
    ABSTAIN = "abstain"


@dataclass
class AuditViolation:
    """
    Details about a specific audit violation.

    Attributes:
        audit_name: Name of the audit that detected the violation
        rule_name: Name of the specific rule that failed
        message: Human-readable description of the violation
        severity: Severity level ("error", "warning", "info")
        location: Optional location info (e.g., step index)
        context: Additional context about the violation

    Example:
        >>> violation = AuditViolation(
        ...     audit_name="step_order",
        ...     rule_name="extract_before_analyze",
        ...     message="extract_data must come before analyze_data",
        ...     severity="error",
        ...     location={"step_index": 2}
        ... )
    """
    audit_name: str
    rule_name: str
    message: str
    severity: str = "error"  # "error", "warning", "info"
    location: Optional[Dict[str, Any]] = None  # Step index, etc.
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "audit_name": self.audit_name,
            "rule_name": self.rule_name,
            "message": self.message,
            "severity": self.severity,
            "location": self.location,
            "context": self.context,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuditViolation":
        """Create from dictionary."""
        return cls(
            audit_name=data["audit_name"],
            rule_name=data["rule_name"],
            message=data["message"],
            severity=data.get("severity", "error"),
            location=data.get("location"),
            context=data.get("context", {}),
        )


@dataclass
class AuditReport:
    """
    Complete report from running audits on a trace.

    Attributes:
        trace_id: ID of the audited trace
        result: Overall result (PASS, FAIL, or ABSTAIN)
        violations: List of violations found
        passed_checks: List of rules that passed
        metadata: Additional metadata about the audit
        timestamp: When the audit was run
        typicality_score: Score from typicality audit (if applicable)
        quantile: Which quantile the trace falls into (for typicality)

    Example:
        >>> report = AuditReport(
        ...     trace_id="abc123",
        ...     result=AuditResult.PASS,
        ...     passed_checks=["has_steps", "no_failures"],
        ... )
        >>> print(report.is_pass)  # True
    """
    trace_id: str
    result: AuditResult
    violations: List[AuditViolation] = field(default_factory=list)
    passed_checks: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # For typicality audits
    typicality_score: Optional[float] = None
    quantile: Optional[int] = None  # Which quantile this trace falls into

    @property
    def is_pass(self) -> bool:
        """Check if audit passed."""
        return self.result == AuditResult.PASS

    @property
    def is_fail(self) -> bool:
        """Check if audit failed."""
        return self.result == AuditResult.FAIL

    @property
    def error_count(self) -> int:
        """Count of error-severity violations."""
        return sum(1 for v in self.violations if v.severity == "error")

    @property
    def warning_count(self) -> int:
        """Count of warning-severity violations."""
        return sum(1 for v in self.violations if v.severity == "warning")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "trace_id": self.trace_id,
            "result": self.result.value,
            "violations": [v.to_dict() for v in self.violations],
            "passed_checks": self.passed_checks,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "typicality_score": self.typicality_score,
            "quantile": self.quantile,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuditReport":
        """Create from dictionary."""
        return cls(
            trace_id=data["trace_id"],
            result=AuditResult(data["result"]),
            violations=[AuditViolation.from_dict(v) for v in data.get("violations", [])],
            passed_checks=data.get("passed_checks", []),
            metadata=data.get("metadata", {}),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            typicality_score=data.get("typicality_score"),
            quantile=data.get("quantile"),
        )

    def summary(self) -> str:
        """Human-readable summary of the report."""
        lines = [
            f"Audit Report for {self.trace_id}",
            f"  Result: {self.result.value.upper()}",
            f"  Passed: {len(self.passed_checks)} checks",
            f"  Violations: {len(self.violations)} ({self.error_count} errors, {self.warning_count} warnings)",
        ]
        if self.typicality_score is not None:
            lines.append(f"  Typicality: {self.typicality_score:.4f}")
        return "\n".join(lines)


class BaseAudit(ABC):
    """
    Abstract base class for all audit types.

    Audits evaluate traces (represented as DataFrames) and produce AuditReports.
    Subclasses must implement the `run` method.

    Attributes:
        name: Unique name for this audit
        description: Human-readable description

    Example:
        >>> class MyAudit(BaseAudit):
        ...     def run(self, df, metadata):
        ...         # Check something about the trace
        ...         if len(df) == 0:
        ...             return AuditReport(
        ...                 trace_id=metadata.get("trace_id", "unknown"),
        ...                 result=AuditResult.FAIL,
        ...                 violations=[AuditViolation(...)]
        ...             )
        ...         return AuditReport(
        ...             trace_id=metadata.get("trace_id", "unknown"),
        ...             result=AuditResult.PASS
        ...         )
    """

    def __init__(self, name: str, description: str = ""):
        """
        Initialize the audit.

        Args:
            name: Unique name for this audit
            description: Human-readable description
        """
        self.name = name
        self.description = description

    @abstractmethod
    def run(self, df: "pd.DataFrame", trace_metadata: Dict[str, Any]) -> AuditReport:
        """
        Run the audit on a trace DataFrame.

        Args:
            df: DataFrame representation of the trace with columns:
                - step_idx: int - Index of the step
                - fn_name: str - Name of the ptool/function called
                - input: str - JSON serialized input (all args)
                - output: str - JSON serialized output
                - input_0, input_1, ...: Individual input arguments
                - status: str - Step status (completed, failed, etc.)
                - duration_ms: float - Execution time
                - goal: str - Step's goal/description
                - error: Optional[str] - Error message if failed
            trace_metadata: Additional metadata including:
                - trace_id: Unique identifier for the trace
                - goal: The overall goal of the trace
                - success: Whether the trace succeeded
                - final_answer: The final result (if any)

        Returns:
            AuditReport with the results
        """
        pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} '{self.name}'>"


class StructuredAudit(BaseAudit):
    """
    Base class for structured audits using assertions on DataFrames.

    Structured audits define rules that check trace structure patterns.
    Rules are Python functions that take (df, metadata) and return True if the rule passes.

    Attributes:
        name: Unique name for this audit
        description: Human-readable description

    Example:
        >>> audit = StructuredAudit("trace_structure", "Check trace has proper structure")
        >>> audit.add_rule(
        ...     "has_extraction",
        ...     lambda df, _: "extract" in df["fn_name"].values,
        ...     "Trace must have an extraction step"
        ... )
        >>> audit.add_rule(
        ...     "extraction_first",
        ...     lambda df, _: df.iloc[0]["fn_name"].startswith("extract"),
        ...     "Extraction should be the first step"
        ... )
        >>> report = audit.run(trace_df, {"trace_id": "test"})
    """

    def __init__(self, name: str, description: str = ""):
        """
        Initialize the structured audit.

        Args:
            name: Unique name for this audit
            description: Human-readable description
        """
        super().__init__(name, description)
        self._rules: List[Tuple[str, Callable[["pd.DataFrame", Dict], bool], str]] = []

    def add_rule(
        self,
        rule_name: str,
        check_fn: Callable[["pd.DataFrame", Dict[str, Any]], bool],
        error_message: str = "",
    ) -> "StructuredAudit":
        """
        Add a rule to this audit.

        Args:
            rule_name: Name of the rule
            check_fn: Function taking (df, metadata) and returning True if rule passes
            error_message: Message to show if rule fails

        Returns:
            self for method chaining

        Example:
            >>> audit.add_rule(
            ...     "has_steps",
            ...     lambda df, _: len(df) > 0,
            ...     "Trace must have at least one step"
            ... ).add_rule(
            ...     "all_completed",
            ...     lambda df, _: (df["status"] == "completed").all(),
            ...     "All steps must be completed"
            ... )
        """
        self._rules.append((rule_name, check_fn, error_message))
        return self

    def remove_rule(self, rule_name: str) -> "StructuredAudit":
        """
        Remove a rule by name.

        Args:
            rule_name: Name of the rule to remove

        Returns:
            self for method chaining
        """
        self._rules = [(n, fn, msg) for n, fn, msg in self._rules if n != rule_name]
        return self

    @property
    def rules(self) -> List[str]:
        """Get list of rule names."""
        return [name for name, _, _ in self._rules]

    def run(self, df: "pd.DataFrame", trace_metadata: Dict[str, Any]) -> AuditReport:
        """
        Run all rules in this audit.

        Args:
            df: DataFrame representation of the trace
            trace_metadata: Additional metadata

        Returns:
            AuditReport with results for all rules
        """
        violations = []
        passed = []

        for rule_name, check_fn, error_msg in self._rules:
            try:
                if check_fn(df, trace_metadata):
                    passed.append(rule_name)
                else:
                    violations.append(AuditViolation(
                        audit_name=self.name,
                        rule_name=rule_name,
                        message=error_msg or f"Rule '{rule_name}' failed",
                    ))
            except Exception as e:
                violations.append(AuditViolation(
                    audit_name=self.name,
                    rule_name=rule_name,
                    message=f"Rule raised exception: {e}",
                    severity="error",
                    context={"exception": str(e)},
                ))

        result = AuditResult.PASS if not violations else AuditResult.FAIL

        return AuditReport(
            trace_id=trace_metadata.get("trace_id", "unknown"),
            result=result,
            violations=violations,
            passed_checks=passed,
            metadata={"audit_type": "structured", "audit_name": self.name},
        )


class BaseTypicalityModel(ABC):
    """
    Abstract base class for probabilistic models over reasoning patterns.

    Typicality models assign probability scores to reasoning patterns
    (sequences of step names). Atypical patterns may indicate errors.

    Attributes:
        name: Name of the model type (e.g., "unigram", "bigram", "hmm")

    Example:
        >>> model = BigramModel()
        >>> model.fit([["extract", "analyze", "format"], ...])
        >>> score = model.score(["extract", "analyze", "format"])
        >>> print(f"Pattern typicality: {score:.4f}")
    """

    name: str = "base"

    @abstractmethod
    def fit(self, patterns: List[List[str]]) -> "BaseTypicalityModel":
        """
        Train the model on a corpus of reasoning patterns.

        Args:
            patterns: List of patterns, each pattern is a list of step names.
                      Patterns should include <START> and <END> tokens.

        Returns:
            self for method chaining
        """
        pass

    @abstractmethod
    def score(self, pattern: List[str]) -> float:
        """
        Score a pattern's probability/typicality.

        Args:
            pattern: Sequence of step names

        Returns:
            Probability score in [0, 1] range (higher = more typical)
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save model to disk.

        Args:
            path: File path to save to
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "BaseTypicalityModel":
        """
        Load model from disk.

        Args:
            path: File path to load from

        Returns:
            Loaded model instance
        """
        pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} '{self.name}'>"


class TypicalityAudit(BaseAudit):
    """
    Base class for typicality audits using probabilistic models.

    Typicality audits extract reasoning patterns (sequences of step names)
    and score them using probabilistic models. Atypical patterns may indicate errors.

    The audit uses quantile thresholds to determine pass/fail/abstain:
    - Traces below fail_threshold → FAIL
    - Traces above pass_threshold → PASS
    - Traces in between → ABSTAIN

    Attributes:
        name: Unique name for this audit
        model: Trained probabilistic model
        quantile_thresholds: (fail_threshold, pass_threshold) tuple

    Example:
        >>> model = BigramModel()
        >>> model.fit(training_patterns)
        >>> audit = TypicalityAudit("pattern_typicality", model)
        >>> report = audit.run(trace_df, {"trace_id": "test"})
        >>> print(f"Score: {report.typicality_score}")
    """

    def __init__(
        self,
        name: str,
        model: BaseTypicalityModel,
        quantile_thresholds: Tuple[float, float] = (0.2, 0.8),
        description: str = "",
    ):
        """
        Initialize the typicality audit.

        Args:
            name: Audit name
            model: Trained probabilistic model
            quantile_thresholds: (fail_threshold, pass_threshold) tuple
                - Traces with score < fail_threshold → FAIL
                - Traces with score > pass_threshold → PASS
                - Traces in between → ABSTAIN
            description: Human-readable description
        """
        super().__init__(name, description)
        self.model = model
        self.fail_threshold, self.pass_threshold = quantile_thresholds

    def run(self, df: "pd.DataFrame", trace_metadata: Dict[str, Any]) -> AuditReport:
        """
        Score the trace pattern and determine pass/fail/abstain.

        Args:
            df: DataFrame representation of the trace
            trace_metadata: Additional metadata

        Returns:
            AuditReport with typicality score and result
        """
        # Extract pattern (sequence of step names)
        pattern = df["fn_name"].tolist()

        # Add special tokens
        pattern = ["<START>"] + pattern + ["<END>"]

        # Score using probabilistic model
        score = self.model.score(pattern)

        # Determine result based on quantile thresholds
        if score < self.fail_threshold:
            result = AuditResult.FAIL
            violations = [AuditViolation(
                audit_name=self.name,
                rule_name="typicality",
                message=f"Atypical reasoning pattern (score={score:.4f} < {self.fail_threshold})",
                severity="warning",
                context={"pattern": pattern, "score": score},
            )]
        elif score > self.pass_threshold:
            result = AuditResult.PASS
            violations = []
        else:
            result = AuditResult.ABSTAIN
            violations = []

        return AuditReport(
            trace_id=trace_metadata.get("trace_id", "unknown"),
            result=result,
            violations=violations,
            passed_checks=["typicality"] if result == AuditResult.PASS else [],
            metadata={
                "audit_type": "typicality",
                "model": self.model.name,
                "thresholds": (self.fail_threshold, self.pass_threshold),
            },
            typicality_score=score,
        )


class AuditRegistry:
    """
    Registry for managing available audits.

    Provides a central place to register and retrieve audits,
    optionally organized by domain.

    Example:
        >>> registry = AuditRegistry()
        >>> registry.register(my_audit)
        >>> registry.register(medcalc_audit, domain="medcalc")
        >>>
        >>> # Get all audits
        >>> all_audits = registry.list_all()
        >>>
        >>> # Get domain-specific audits
        >>> medcalc_audits = registry.list_by_domain("medcalc")
    """

    def __init__(self):
        """Initialize empty registry."""
        self._audits: Dict[str, BaseAudit] = {}
        self._domain_audits: Dict[str, List[str]] = {}  # domain -> audit names

    def register(
        self,
        audit: BaseAudit,
        domain: Optional[str] = None,
    ) -> None:
        """
        Register an audit, optionally associating it with a domain.

        Args:
            audit: The audit to register
            domain: Optional domain name (e.g., "medcalc", "hotpotqa")
        """
        self._audits[audit.name] = audit
        if domain:
            self._domain_audits.setdefault(domain, []).append(audit.name)

    def unregister(self, name: str) -> Optional[BaseAudit]:
        """
        Unregister an audit by name.

        Args:
            name: Name of the audit to remove

        Returns:
            The removed audit, or None if not found
        """
        audit = self._audits.pop(name, None)
        if audit:
            for domain_audits in self._domain_audits.values():
                if name in domain_audits:
                    domain_audits.remove(name)
        return audit

    def get(self, name: str) -> Optional[BaseAudit]:
        """
        Get an audit by name.

        Args:
            name: Name of the audit

        Returns:
            The audit, or None if not found
        """
        return self._audits.get(name)

    def list_all(self) -> List[BaseAudit]:
        """
        List all registered audits.

        Returns:
            List of all audits
        """
        return list(self._audits.values())

    def list_names(self) -> List[str]:
        """
        List names of all registered audits.

        Returns:
            List of audit names
        """
        return list(self._audits.keys())

    def list_by_domain(self, domain: str) -> List[BaseAudit]:
        """
        List audits for a specific domain.

        Args:
            domain: Domain name

        Returns:
            List of audits in that domain
        """
        names = self._domain_audits.get(domain, [])
        return [self._audits[n] for n in names if n in self._audits]

    def list_domains(self) -> List[str]:
        """
        List all registered domains.

        Returns:
            List of domain names
        """
        return list(self._domain_audits.keys())

    def __len__(self) -> int:
        return len(self._audits)

    def __contains__(self, name: str) -> bool:
        return name in self._audits


# Global registry
_AUDIT_REGISTRY: Optional[AuditRegistry] = None


def get_audit_registry() -> AuditRegistry:
    """
    Get the global audit registry.

    Returns:
        The global AuditRegistry instance
    """
    global _AUDIT_REGISTRY
    if _AUDIT_REGISTRY is None:
        _AUDIT_REGISTRY = AuditRegistry()
    return _AUDIT_REGISTRY


def set_audit_registry(registry: AuditRegistry) -> None:
    """
    Set the global audit registry.

    Args:
        registry: The registry to use globally
    """
    global _AUDIT_REGISTRY
    _AUDIT_REGISTRY = registry


def register_audit(audit: BaseAudit, domain: Optional[str] = None) -> None:
    """
    Register an audit globally.

    Args:
        audit: The audit to register
        domain: Optional domain name
    """
    get_audit_registry().register(audit, domain)
