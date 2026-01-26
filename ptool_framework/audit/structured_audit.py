"""
Structured Audit DSL and pre-built audits.

This module provides:
1. AuditDSL: A domain-specific language for querying and asserting on trace DataFrames
2. @audit_rule decorator: For creating declarative audit rules
3. DeclarativeAudit: Audit class that uses decorated methods as rules
4. Pre-built audits: Common audits ready to use

The DSL enables writing audits like Python unit tests, as described in the SSRM paper:

    >>> dsl = AuditDSL(df, metadata)
    >>> assert dsl.has_at_least("extract_data", 1)
    >>> assert dsl.comes_before("extract_data", "analyze_data")
    >>> assert dsl.pattern_exists(["extract", "analyze", "format"])

Example of a declarative audit:

    >>> class MyAudit(DeclarativeAudit):
    ...     @audit_rule("has_extraction", "Must have extraction step")
    ...     def check_extraction(self, dsl):
    ...         return dsl.has_at_least("extract", 1)
    ...
    ...     @audit_rule("correct_order", "Extraction before analysis")
    ...     def check_order(self, dsl):
    ...         return dsl.comes_before("extract", "analyze")
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

from .base import (
    AuditResult,
    AuditViolation,
    AuditReport,
    StructuredAudit,
    register_audit,
)


class AuditDSL:
    """
    Domain-specific language for querying and asserting on trace DataFrames.

    The DSL provides methods for common audit patterns:
    - Counting steps by name
    - Checking step order
    - Pattern matching
    - Output validation
    - DataFrame querying

    Attributes:
        df: The trace DataFrame
        metadata: Trace metadata

    Example:
        >>> dsl = AuditDSL(df, metadata)
        >>>
        >>> # Count assertions
        >>> assert dsl.has_exactly("extract_data", 1)
        >>> assert dsl.has_at_least("validate", 2)
        >>> assert dsl.has_at_most("error_handler", 1)
        >>>
        >>> # Order assertions
        >>> assert dsl.comes_before("extract", "analyze")
        >>> assert dsl.pattern_exists(["extract", "analyze", "format"])
        >>>
        >>> # Query methods
        >>> extract_steps = dsl.steps_named("extract")
        >>> failed = dsl.steps_with_status("failed")
    """

    def __init__(self, df: "pd.DataFrame", metadata: Dict[str, Any]):
        """
        Initialize the DSL with a trace DataFrame.

        Args:
            df: DataFrame representation of the trace
            metadata: Trace metadata dictionary
        """
        self.df = df
        self.metadata = metadata

    # =========================================================================
    # Query Methods
    # =========================================================================

    def steps_named(self, fn_name: str, exact: bool = False) -> "pd.DataFrame":
        """
        Get steps with a specific function name.

        Args:
            fn_name: Function name to search for
            exact: If True, require exact match; if False, use contains

        Returns:
            DataFrame with matching steps

        Example:
            >>> extract_steps = dsl.steps_named("extract")  # Contains "extract"
            >>> exact_steps = dsl.steps_named("extract_data", exact=True)
        """
        if exact:
            return self.df[self.df["fn_name"] == fn_name]
        else:
            return self.df[self.df["fn_name"].str.contains(fn_name, na=False)]

    def steps_with_status(self, status: str) -> "pd.DataFrame":
        """
        Get steps with a specific status.

        Args:
            status: Status to filter by ("completed", "failed", "pending", etc.)

        Returns:
            DataFrame with matching steps

        Example:
            >>> failed_steps = dsl.steps_with_status("failed")
            >>> completed = dsl.steps_with_status("completed")
        """
        return self.df[self.df["status"] == status]

    def query(self, query_str: str) -> "pd.DataFrame":
        """
        Run a pandas query on the DataFrame.

        Args:
            query_str: Pandas query string

        Returns:
            Filtered DataFrame

        Example:
            >>> slow_steps = dsl.query("duration_ms > 1000")
            >>> extract_completed = dsl.query("fn_name.str.contains('extract') and status == 'completed'")
        """
        return self.df.query(query_str)

    def get_step(self, index: int) -> Optional["pd.Series"]:
        """
        Get a step by index.

        Args:
            index: Step index

        Returns:
            Step as Series, or None if index out of bounds

        Example:
            >>> first_step = dsl.get_step(0)
            >>> last_step = dsl.get_step(-1)
        """
        if index < 0:
            index = len(self.df) + index
        if 0 <= index < len(self.df):
            return self.df.iloc[index]
        return None

    def get_pattern(self) -> List[str]:
        """
        Get the sequence of function names (pattern) from the trace.

        Returns:
            List of function names in order

        Example:
            >>> pattern = dsl.get_pattern()
            >>> print(pattern)  # ['extract_data', 'analyze', 'format_output']
        """
        return self.df["fn_name"].tolist()

    # =========================================================================
    # Count Assertions
    # =========================================================================

    def has_exactly(self, fn_name: str, count: int, exact: bool = False) -> bool:
        """
        Check if trace has exactly N steps with given function name.

        Args:
            fn_name: Function name to search for
            count: Expected count
            exact: If True, require exact match on fn_name

        Returns:
            True if count matches exactly

        Example:
            >>> assert dsl.has_exactly("extract_data", 1)
        """
        matches = self.steps_named(fn_name, exact=exact)
        return len(matches) == count

    def has_at_least(self, fn_name: str, count: int, exact: bool = False) -> bool:
        """
        Check if trace has at least N steps with given function name.

        Args:
            fn_name: Function name to search for
            count: Minimum count
            exact: If True, require exact match on fn_name

        Returns:
            True if count is >= minimum

        Example:
            >>> assert dsl.has_at_least("validate", 1)
        """
        matches = self.steps_named(fn_name, exact=exact)
        return len(matches) >= count

    def has_at_most(self, fn_name: str, count: int, exact: bool = False) -> bool:
        """
        Check if trace has at most N steps with given function name.

        Args:
            fn_name: Function name to search for
            count: Maximum count
            exact: If True, require exact match on fn_name

        Returns:
            True if count is <= maximum

        Example:
            >>> assert dsl.has_at_most("error_handler", 1)
        """
        matches = self.steps_named(fn_name, exact=exact)
        return len(matches) <= count

    def is_non_empty(self) -> bool:
        """
        Check if trace has at least one step.

        Returns:
            True if trace has steps

        Example:
            >>> assert dsl.is_non_empty()
        """
        return len(self.df) > 0

    # =========================================================================
    # Order Assertions
    # =========================================================================

    def comes_before(self, fn_a: str, fn_b: str, exact: bool = False) -> bool:
        """
        Check if any step matching fn_a comes before any step matching fn_b.

        Args:
            fn_a: Function name that should come first
            fn_b: Function name that should come second
            exact: If True, require exact match on function names

        Returns:
            True if fn_a appears before fn_b

        Example:
            >>> assert dsl.comes_before("extract", "analyze")
        """
        steps_a = self.steps_named(fn_a, exact=exact)
        steps_b = self.steps_named(fn_b, exact=exact)

        if steps_a.empty or steps_b.empty:
            return False

        first_a = steps_a["step_idx"].min()
        first_b = steps_b["step_idx"].min()

        return first_a < first_b

    def comes_after(self, fn_a: str, fn_b: str, exact: bool = False) -> bool:
        """
        Check if any step matching fn_a comes after any step matching fn_b.

        Args:
            fn_a: Function name that should come second
            fn_b: Function name that should come first
            exact: If True, require exact match on function names

        Returns:
            True if fn_a appears after fn_b

        Example:
            >>> assert dsl.comes_after("format_output", "analyze")
        """
        return self.comes_before(fn_b, fn_a, exact=exact)

    def immediately_follows(self, fn_a: str, fn_b: str, exact: bool = False) -> bool:
        """
        Check if fn_b immediately follows fn_a (no steps in between).

        Args:
            fn_a: Function name that comes first
            fn_b: Function name that should immediately follow
            exact: If True, require exact match on function names

        Returns:
            True if fn_b immediately follows fn_a

        Example:
            >>> assert dsl.immediately_follows("extract", "validate")
        """
        steps_a = self.steps_named(fn_a, exact=exact)
        steps_b = self.steps_named(fn_b, exact=exact)

        if steps_a.empty or steps_b.empty:
            return False

        for _, step_a in steps_a.iterrows():
            idx_a = step_a["step_idx"]
            # Check if any fn_b step immediately follows
            next_steps = steps_b[steps_b["step_idx"] == idx_a + 1]
            if not next_steps.empty:
                return True

        return False

    def pattern_exists(
        self,
        pattern: List[str],
        exact: bool = False,
        contiguous: bool = False,
    ) -> bool:
        """
        Check if a pattern (sequence of function names) exists in the trace.

        Args:
            pattern: List of function names to search for
            exact: If True, require exact match on function names
            contiguous: If True, pattern must be contiguous (no gaps)

        Returns:
            True if pattern exists in trace

        Example:
            >>> assert dsl.pattern_exists(["extract", "analyze", "format"])
            >>> assert dsl.pattern_exists(["extract", "analyze"], contiguous=True)
        """
        if not pattern:
            return True

        fn_names = self.get_pattern()

        if exact:
            match_fn = lambda name, pat: name == pat
        else:
            match_fn = lambda name, pat: pat in name

        if contiguous:
            # Look for contiguous sequence
            for i in range(len(fn_names) - len(pattern) + 1):
                if all(
                    match_fn(fn_names[i + j], pattern[j])
                    for j in range(len(pattern))
                ):
                    return True
            return False
        else:
            # Look for subsequence (can have gaps)
            pattern_idx = 0
            for fn_name in fn_names:
                if match_fn(fn_name, pattern[pattern_idx]):
                    pattern_idx += 1
                    if pattern_idx == len(pattern):
                        return True
            return False

    # =========================================================================
    # Output Validation
    # =========================================================================

    def output_matches(
        self,
        fn_name: str,
        predicate: Callable[[Any], bool],
        all_matches: bool = True,
    ) -> bool:
        """
        Check if outputs of matching steps satisfy a predicate.

        Args:
            fn_name: Function name to filter by
            predicate: Function that takes output and returns bool
            all_matches: If True, all matching steps must satisfy predicate;
                        if False, at least one must satisfy

        Returns:
            True if predicate is satisfied

        Example:
            >>> # Check that all extract outputs are non-empty lists
            >>> assert dsl.output_matches("extract", lambda x: isinstance(x, list) and len(x) > 0)
        """
        steps = self.steps_named(fn_name)

        if steps.empty:
            return False

        results = []
        for _, step in steps.iterrows():
            output = step.get("output")
            # Try to parse JSON output
            if isinstance(output, str):
                try:
                    output = json.loads(output)
                except (json.JSONDecodeError, TypeError):
                    pass
            results.append(predicate(output))

        if all_matches:
            return all(results)
        else:
            return any(results)

    def verify_calculation(
        self,
        step_idx: int,
        expected_formula: Callable[[Dict[str, Any]], Any],
        tolerance: float = 1e-6,
    ) -> bool:
        """
        Verify that a step's output matches an expected calculation.

        Args:
            step_idx: Index of the step to verify
            expected_formula: Function that takes step's input dict and returns expected output
            tolerance: Tolerance for numeric comparison

        Returns:
            True if output matches expected formula

        Example:
            >>> # Verify BMI calculation
            >>> assert dsl.verify_calculation(
            ...     step_idx=2,
            ...     expected_formula=lambda inp: inp["weight"] / (inp["height"] ** 2)
            ... )
        """
        step = self.get_step(step_idx)
        if step is None:
            return False

        # Parse input
        input_data = step.get("input")
        if isinstance(input_data, str):
            try:
                input_data = json.loads(input_data)
            except (json.JSONDecodeError, TypeError):
                return False

        # Parse output
        output = step.get("output")
        if isinstance(output, str):
            try:
                output = json.loads(output)
            except (json.JSONDecodeError, TypeError):
                pass

        # Calculate expected
        try:
            expected = expected_formula(input_data)
        except Exception:
            return False

        # Compare
        if isinstance(output, (int, float)) and isinstance(expected, (int, float)):
            return abs(output - expected) <= tolerance
        else:
            return output == expected

    # =========================================================================
    # Status Assertions
    # =========================================================================

    def all_completed(self) -> bool:
        """
        Check if all steps completed successfully.

        Returns:
            True if all steps have status "completed"

        Example:
            >>> assert dsl.all_completed()
        """
        if self.df.empty:
            return True
        return (self.df["status"] == "completed").all()

    def no_failures(self) -> bool:
        """
        Check that no steps failed.

        Returns:
            True if no steps have status "failed"

        Example:
            >>> assert dsl.no_failures()
        """
        return self.steps_with_status("failed").empty

    def success_rate(self) -> float:
        """
        Calculate the success rate of steps.

        Returns:
            Ratio of completed steps to total steps (0.0 to 1.0)

        Example:
            >>> rate = dsl.success_rate()
            >>> assert rate >= 0.8  # At least 80% success
        """
        if self.df.empty:
            return 1.0
        completed = len(self.steps_with_status("completed"))
        return completed / len(self.df)


# =============================================================================
# Decorator for Declarative Audits
# =============================================================================

@dataclass
class AuditRuleInfo:
    """Metadata for an audit rule."""
    name: str
    error_message: str
    severity: str = "error"


def audit_rule(
    name: str,
    error_message: str = "",
    severity: str = "error",
) -> Callable:
    """
    Decorator to mark a method as an audit rule.

    The decorated method should take an AuditDSL instance and return bool.

    Args:
        name: Name of the rule
        error_message: Message to show if rule fails
        severity: Severity level ("error", "warning", "info")

    Returns:
        Decorator function

    Example:
        >>> class MyAudit(DeclarativeAudit):
        ...     @audit_rule("has_steps", "Trace must have at least one step")
        ...     def check_has_steps(self, dsl):
        ...         return dsl.is_non_empty()
        ...
        ...     @audit_rule("all_complete", "All steps must complete", severity="warning")
        ...     def check_complete(self, dsl):
        ...         return dsl.all_completed()
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        # Store rule info as attribute
        wrapper._audit_rule_info = AuditRuleInfo(
            name=name,
            error_message=error_message,
            severity=severity,
        )
        return wrapper
    return decorator


class DeclarativeAudit(StructuredAudit):
    """
    Base class for declarative audits using the @audit_rule decorator.

    Subclasses define rules as methods decorated with @audit_rule.
    Rules are automatically discovered and executed.

    Example:
        >>> class MyAudit(DeclarativeAudit):
        ...     def __init__(self):
        ...         super().__init__("my_audit", "My custom audit")
        ...
        ...     @audit_rule("has_extraction", "Must have extraction step")
        ...     def check_extraction(self, dsl):
        ...         return dsl.has_at_least("extract", 1)
        ...
        ...     @audit_rule("correct_order", "Extraction before analysis")
        ...     def check_order(self, dsl):
        ...         return dsl.comes_before("extract", "analyze")
        >>>
        >>> audit = MyAudit()
        >>> report = audit.run(df, metadata)
    """

    def __init__(self, name: str, description: str = ""):
        """
        Initialize the declarative audit.

        Args:
            name: Audit name
            description: Human-readable description
        """
        super().__init__(name, description)
        self._discover_rules()

    def _discover_rules(self) -> None:
        """Discover and register all @audit_rule decorated methods."""
        for attr_name in dir(self):
            if attr_name.startswith('_'):
                continue

            attr = getattr(self, attr_name)
            if callable(attr) and hasattr(attr, '_audit_rule_info'):
                info: AuditRuleInfo = attr._audit_rule_info
                self._register_rule_method(attr, info)

    def _register_rule_method(
        self,
        method: Callable,
        info: AuditRuleInfo,
    ) -> None:
        """Register a rule method."""
        def check_fn(df: "pd.DataFrame", metadata: Dict[str, Any]) -> bool:
            dsl = AuditDSL(df, metadata)
            return method(dsl)

        # Override severity in violations
        original_rules_len = len(self._rules)
        self.add_rule(info.name, check_fn, info.error_message)

        # Store severity for later use
        if not hasattr(self, '_rule_severities'):
            self._rule_severities = {}
        self._rule_severities[info.name] = info.severity

    def run(self, df: "pd.DataFrame", trace_metadata: Dict[str, Any]) -> AuditReport:
        """
        Run all rules and return report with correct severities.

        Args:
            df: DataFrame representation of the trace
            trace_metadata: Trace metadata

        Returns:
            AuditReport with results
        """
        # Run parent implementation
        report = super().run(df, trace_metadata)

        # Update severities based on stored info
        if hasattr(self, '_rule_severities'):
            for violation in report.violations:
                if violation.rule_name in self._rule_severities:
                    violation.severity = self._rule_severities[violation.rule_name]

        return report


# =============================================================================
# Pre-built Audits
# =============================================================================

class NonEmptyTraceAudit(StructuredAudit):
    """
    Audit that checks trace has at least one step.

    Example:
        >>> audit = NonEmptyTraceAudit()
        >>> report = audit.run(df, metadata)
    """

    def __init__(self):
        super().__init__(
            name="non_empty_trace",
            description="Checks that the trace has at least one step"
        )
        self.add_rule(
            "has_steps",
            lambda df, _: len(df) > 0,
            "Trace must have at least one step"
        )


class NoFailedStepsAudit(StructuredAudit):
    """
    Audit that checks no steps have failed.

    Example:
        >>> audit = NoFailedStepsAudit()
        >>> report = audit.run(df, metadata)
    """

    def __init__(self):
        super().__init__(
            name="no_failed_steps",
            description="Checks that no steps have failed status"
        )
        self.add_rule(
            "no_failures",
            lambda df, _: (df["status"] != "failed").all() if len(df) > 0 else True,
            "No steps should have failed status"
        )


class AllCompletedAudit(StructuredAudit):
    """
    Audit that checks all steps completed successfully.

    Example:
        >>> audit = AllCompletedAudit()
        >>> report = audit.run(df, metadata)
    """

    def __init__(self):
        super().__init__(
            name="all_completed",
            description="Checks that all steps completed successfully"
        )
        self.add_rule(
            "all_completed",
            lambda df, _: (df["status"] == "completed").all() if len(df) > 0 else True,
            "All steps must complete successfully"
        )


class RequiredStepsAudit(StructuredAudit):
    """
    Audit that checks required function names are present.

    Example:
        >>> audit = RequiredStepsAudit(["extract", "analyze", "format"])
        >>> report = audit.run(df, metadata)
    """

    def __init__(
        self,
        required_steps: List[str],
        exact_match: bool = False,
    ):
        """
        Initialize with required step names.

        Args:
            required_steps: List of function names that must appear
            exact_match: If True, require exact match; if False, use contains
        """
        super().__init__(
            name="required_steps",
            description=f"Checks that required steps are present: {required_steps}"
        )
        self.required_steps = required_steps
        self.exact_match = exact_match

        for step_name in required_steps:
            self._add_required_step_rule(step_name)

    def _add_required_step_rule(self, step_name: str) -> None:
        """Add a rule for a required step."""
        if self.exact_match:
            check_fn = lambda df, _, sn=step_name: (df["fn_name"] == sn).any()
        else:
            check_fn = lambda df, _, sn=step_name: df["fn_name"].str.contains(sn, na=False).any()

        self.add_rule(
            f"has_{step_name}",
            check_fn,
            f"Trace must contain step matching '{step_name}'"
        )


class StepOrderAudit(StructuredAudit):
    """
    Audit that checks steps appear in a specific order.

    Example:
        >>> # extract must come before analyze
        >>> audit = StepOrderAudit([("extract", "analyze"), ("analyze", "format")])
        >>> report = audit.run(df, metadata)
    """

    def __init__(
        self,
        order_constraints: List[Tuple[str, str]],
        exact_match: bool = False,
    ):
        """
        Initialize with order constraints.

        Args:
            order_constraints: List of (before, after) tuples
            exact_match: If True, require exact match on function names
        """
        super().__init__(
            name="step_order",
            description="Checks that steps appear in the correct order"
        )
        self.order_constraints = order_constraints
        self.exact_match = exact_match

        for before, after in order_constraints:
            self._add_order_rule(before, after)

    def _add_order_rule(self, before: str, after: str) -> None:
        """Add a rule for an order constraint."""
        def check_order(df: "pd.DataFrame", _: Dict, b=before, a=after) -> bool:
            dsl = AuditDSL(df, {})
            return dsl.comes_before(b, a, exact=self.exact_match)

        self.add_rule(
            f"order_{before}_before_{after}",
            check_order,
            f"'{before}' must come before '{after}'"
        )


class PatternAudit(StructuredAudit):
    """
    Audit that checks for required patterns in the trace.

    Example:
        >>> audit = PatternAudit(
        ...     required_patterns=[["extract", "validate", "process"]],
        ...     forbidden_patterns=[["error", "retry", "error"]]
        ... )
        >>> report = audit.run(df, metadata)
    """

    def __init__(
        self,
        required_patterns: Optional[List[List[str]]] = None,
        forbidden_patterns: Optional[List[List[str]]] = None,
        contiguous: bool = False,
    ):
        """
        Initialize with pattern constraints.

        Args:
            required_patterns: Patterns that must exist
            forbidden_patterns: Patterns that must not exist
            contiguous: If True, patterns must be contiguous
        """
        super().__init__(
            name="pattern_audit",
            description="Checks for required/forbidden patterns"
        )
        self.required_patterns = required_patterns or []
        self.forbidden_patterns = forbidden_patterns or []
        self.contiguous = contiguous

        # Add rules for required patterns
        for i, pattern in enumerate(self.required_patterns):
            self._add_required_pattern_rule(i, pattern)

        # Add rules for forbidden patterns
        for i, pattern in enumerate(self.forbidden_patterns):
            self._add_forbidden_pattern_rule(i, pattern)

    def _add_required_pattern_rule(self, idx: int, pattern: List[str]) -> None:
        """Add a rule for a required pattern."""
        def check_pattern(df: "pd.DataFrame", _: Dict, p=pattern) -> bool:
            dsl = AuditDSL(df, {})
            return dsl.pattern_exists(p, contiguous=self.contiguous)

        pattern_str = " -> ".join(pattern)
        self.add_rule(
            f"required_pattern_{idx}",
            check_pattern,
            f"Pattern must exist: {pattern_str}"
        )

    def _add_forbidden_pattern_rule(self, idx: int, pattern: List[str]) -> None:
        """Add a rule for a forbidden pattern."""
        def check_pattern(df: "pd.DataFrame", _: Dict, p=pattern) -> bool:
            dsl = AuditDSL(df, {})
            return not dsl.pattern_exists(p, contiguous=self.contiguous)

        pattern_str = " -> ".join(pattern)
        self.add_rule(
            f"forbidden_pattern_{idx}",
            check_pattern,
            f"Pattern must not exist: {pattern_str}"
        )


class PerformanceAudit(StructuredAudit):
    """
    Audit that checks performance constraints.

    Example:
        >>> audit = PerformanceAudit(
        ...     max_step_duration_ms=5000,
        ...     max_total_duration_ms=30000,
        ...     max_step_count=10
        ... )
        >>> report = audit.run(df, metadata)
    """

    def __init__(
        self,
        max_step_duration_ms: Optional[float] = None,
        max_total_duration_ms: Optional[float] = None,
        max_step_count: Optional[int] = None,
    ):
        """
        Initialize with performance constraints.

        Args:
            max_step_duration_ms: Maximum duration for any single step
            max_total_duration_ms: Maximum total duration for all steps
            max_step_count: Maximum number of steps
        """
        super().__init__(
            name="performance_audit",
            description="Checks performance constraints"
        )

        if max_step_duration_ms is not None:
            self.add_rule(
                "max_step_duration",
                lambda df, _, m=max_step_duration_ms: (
                    df["duration_ms"].max() <= m if df["duration_ms"].notna().any() else True
                ),
                f"No step should take longer than {max_step_duration_ms}ms"
            )

        if max_total_duration_ms is not None:
            self.add_rule(
                "max_total_duration",
                lambda df, _, m=max_total_duration_ms: (
                    df["duration_ms"].sum() <= m if df["duration_ms"].notna().any() else True
                ),
                f"Total duration should not exceed {max_total_duration_ms}ms"
            )

        if max_step_count is not None:
            self.add_rule(
                "max_step_count",
                lambda df, _, m=max_step_count: len(df) <= m,
                f"Trace should have at most {max_step_count} steps"
            )


# =============================================================================
# Factory Functions
# =============================================================================

def create_basic_audit(name: str = "basic_audit") -> StructuredAudit:
    """
    Create a basic audit with common checks.

    Returns an audit that checks:
    - Trace is non-empty
    - No steps have failed
    - All steps completed

    Args:
        name: Name for the audit

    Returns:
        StructuredAudit with basic rules

    Example:
        >>> audit = create_basic_audit()
        >>> report = audit.run(df, metadata)
    """
    audit = StructuredAudit(name, "Basic trace validation")
    audit.add_rule(
        "non_empty",
        lambda df, _: len(df) > 0,
        "Trace must have at least one step"
    )
    audit.add_rule(
        "no_failures",
        lambda df, _: (df["status"] != "failed").all() if len(df) > 0 else True,
        "No steps should have failed"
    )
    audit.add_rule(
        "all_completed",
        lambda df, _: (df["status"] == "completed").all() if len(df) > 0 else True,
        "All steps should be completed"
    )
    return audit


def create_workflow_audit(
    required_steps: List[str],
    step_order: Optional[List[Tuple[str, str]]] = None,
    name: str = "workflow_audit",
) -> StructuredAudit:
    """
    Create a workflow audit with required steps and order constraints.

    Args:
        required_steps: Steps that must be present
        step_order: Optional list of (before, after) order constraints
        name: Name for the audit

    Returns:
        StructuredAudit configured for workflow validation

    Example:
        >>> audit = create_workflow_audit(
        ...     required_steps=["extract", "analyze", "format"],
        ...     step_order=[("extract", "analyze"), ("analyze", "format")]
        ... )
        >>> report = audit.run(df, metadata)
    """
    audit = StructuredAudit(name, f"Workflow audit: {required_steps}")

    # Add required steps
    for step in required_steps:
        audit.add_rule(
            f"has_{step}",
            lambda df, _, s=step: df["fn_name"].str.contains(s, na=False).any(),
            f"Workflow must include step matching '{step}'"
        )

    # Add order constraints
    if step_order:
        for before, after in step_order:
            def check_order(df: "pd.DataFrame", _: Dict, b=before, a=after) -> bool:
                dsl = AuditDSL(df, {})
                return dsl.comes_before(b, a)

            audit.add_rule(
                f"order_{before}_before_{after}",
                check_order,
                f"'{before}' must come before '{after}'"
            )

    return audit


# Register pre-built audits globally
def register_prebuilt_audits() -> None:
    """Register all pre-built audits in the global registry."""
    register_audit(NonEmptyTraceAudit())
    register_audit(NoFailedStepsAudit())
    register_audit(AllCompletedAudit())
