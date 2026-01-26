"""
MedCalc domain-specific audits.

This module provides audits specifically designed for medical calculator tasks,
such as BMI calculation, GFR estimation, etc.

These audits demonstrate the pattern of inheriting from generic base classes
and adding domain-specific validation.

Audits:
    - MedCalcStructureAudit: Checks trace structure for medical calculations
    - MedCalcOutputValidationAudit: Validates outputs are appropriate
    - MedCalcCorrectnessAudit: Verifies calculation correctness
    - MedCalcBMIAudit: Specific audit for BMI calculations

Example:
    >>> from ptool_framework.audit.domains.medcalc import (
    ...     MedCalcStructureAudit,
    ...     MedCalcBMIAudit,
    ... )
    >>>
    >>> # Create combined audit
    >>> audit = MedCalcStructureAudit()
    >>> report = audit.run(df, metadata)
    >>>
    >>> # Or use BMI-specific audit
    >>> bmi_audit = MedCalcBMIAudit()
    >>> report = bmi_audit.run(df, metadata)
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

from ..base import (
    AuditResult,
    AuditReport,
    AuditViolation,
    register_audit,
)
from ..structured_audit import (
    AuditDSL,
    DeclarativeAudit,
    StructuredAudit,
    audit_rule,
)


# =============================================================================
# Dynamic Calculator Import (from calculators.py)
# =============================================================================

def get_known_calculators() -> Set[str]:
    """
    Dynamically get calculator names from calculators.py.

    This ensures audits stay in sync with available calculators.
    When calculators.py is updated (calculators added/removed),
    audits automatically adjust.

    Returns:
        Set of lowercase calculator names/patterns
    """
    try:
        from benchmark.experiments.calculators import CALCULATOR_PATTERNS
        # Return all pattern keys (already lowercase)
        return set(CALCULATOR_PATTERNS.keys())
    except ImportError:
        # Fallback to basic list if import fails
        return {
            # Core calculators
            "bmi", "body mass index",
            "gfr", "glomerular filtration rate",
            "creatinine clearance", "cockcroft-gault", "cockroft-gault",
            "map", "mean arterial pressure",
            "bsa", "body surface area",
            "ideal body weight", "adjusted body weight",
            # QTc variants
            "qtc bazett", "bazett",
            "qtc fridericia", "fridericia", "fredericia",
            "qtc framingham", "framingham qtc",
            "qtc hodges", "hodges",
            # Lab values
            "ckd-epi", "mdrd",
            "anion gap", "delta gap", "delta ratio",
            "serum osmolality", "osmolality",
            "free water deficit",
            "sodium correction", "calcium correction", "corrected calcium",
            "ldl", "ldl calculated",
            "fib-4", "fibrosis-4", "fib4",
            "albumin corrected anion gap",
            "target weight", "maintenance fluids",
        }


# Legacy static dict (kept for backwards compatibility)
# NOTE: Use get_known_calculators() for dynamic access
KNOWN_CALCULATORS = {
    "bmi": {
        "name": "Body Mass Index",
        "inputs": ["weight", "height"],
        "formula": lambda w, h: w / (h ** 2),
        "units": "kg/m²",
        "valid_range": (10, 60),  # Reasonable BMI range
    },
    "bsa": {
        "name": "Body Surface Area",
        "inputs": ["weight", "height"],
        "formula": lambda w, h: 0.007184 * (w ** 0.425) * ((h * 100) ** 0.725),  # DuBois
        "units": "m²",
        "valid_range": (0.5, 3.0),
    },
    "gfr": {
        "name": "Glomerular Filtration Rate",
        "inputs": ["creatinine", "age", "sex"],
        "units": "mL/min/1.73m²",
        "valid_range": (0, 200),
    },
    "creatinine_clearance": {
        "name": "Creatinine Clearance (Cockcroft-Gault)",
        "inputs": ["creatinine", "age", "weight", "sex"],
        "units": "mL/min",
        "valid_range": (0, 200),
    },
    "ibw": {
        "name": "Ideal Body Weight",
        "inputs": ["height", "sex"],
        "units": "kg",
        "valid_range": (30, 150),
    },
}

# Step patterns commonly seen in medical calculations
MEDCALC_STEP_PATTERNS = [
    "identify_calculator",
    "extract_values",
    "validate_inputs",
    "calculate",
    "format_result",
]


# =============================================================================
# MedCalc Structure Audit
# =============================================================================

class MedCalcStructureAudit(StructuredAudit):
    """
    Simple audit: did we attempt to call compute_calculation?

    This checks that the LLM tried to use the calculator, not whether it succeeded.
    Success/failure of the calculation is a separate concern.
    """

    def __init__(self):
        super().__init__(
            name="medcalc_structure",
            description="Checks that compute_calculation was attempted"
        )

    def run(self, df: "pd.DataFrame", trace_metadata: Dict[str, Any]) -> "AuditReport":
        from ..base import AuditReport, AuditResult, AuditViolation

        violations = []
        passed = []

        # Check: did we ATTEMPT to call compute_calculation?
        # We pass if the LLM tried, even if the calculation errored
        attempted_calculation = False
        got_result = False

        for idx, row in df.iterrows():
            fn_name = str(row.get("fn_name", "")).lower()
            if "compute" in fn_name or "calculate" in fn_name:
                # LLM attempted to use calculator - that's what we care about
                attempted_calculation = True

                output = self._parse_output(row.get("output"))
                # Check if we also got a result (for info, not required)
                if isinstance(output, dict):
                    if "result" in output and output["result"] is not None:
                        got_result = True
                elif isinstance(output, (int, float)):
                    got_result = True

        if attempted_calculation:
            passed.append("attempted_calculation")
            if got_result:
                passed.append("got_result")
        else:
            violations.append(AuditViolation(
                audit_name=self.name,
                rule_name="attempted_calculation",
                message="LLM must attempt to call compute_calculation",
                severity="error",
            ))

        result = AuditResult.PASS if not violations else AuditResult.FAIL
        return AuditReport(
            trace_id=trace_metadata.get("trace_id", "unknown"),
            result=result,
            violations=violations,
            passed_checks=passed,
            metadata={"audit_name": self.name},
        )

    def _parse_output(self, output: Any) -> Any:
        """Parse JSON string to dict if needed."""
        if output is None:
            return None
        if isinstance(output, str):
            try:
                return json.loads(output)
            except (json.JSONDecodeError, TypeError):
                return output
        return output


# =============================================================================
# MedCalc Output Validation Audit
# =============================================================================

class MedCalcOutputValidationAudit(StructuredAudit):
    """
    Simple audit: check that at least one compute_calculation succeeded.

    Allows retry behavior - if LLM tries, fails, then succeeds, that's fine.
    We only care that EVENTUALLY a valid result was obtained.
    """

    def __init__(self, allowed_calculators: Optional[Set[str]] = None):
        super().__init__(
            name="medcalc_output_validation",
            description="Checks at least one compute_calculation succeeded"
        )

    def run(self, df: "pd.DataFrame", trace_metadata: Dict[str, Any]) -> "AuditReport":
        from ..base import AuditReport, AuditResult, AuditViolation

        violations = []
        passed = []

        # Check: did ANY compute_calculation succeed?
        # We allow retries - LLM can try multiple times until it gets a valid result
        found_success = False
        last_error = None
        last_error_step = None

        for idx, row in df.iterrows():
            fn_name = str(row.get("fn_name", "")).lower()
            if "compute" in fn_name or "calculate" in fn_name:
                output = self._parse_output(row.get("output"))
                step_idx = int(row.get("step_idx", idx))

                if isinstance(output, dict):
                    if "result" in output and output["result"] is not None:
                        # Found a successful call - that's all we need
                        found_success = True
                        passed.append("valid_output")
                    elif "error" in output:
                        # Track error but don't fail yet - might retry
                        last_error = output.get("error", "unknown")
                        last_error_step = step_idx

        # Only fail if NO successful compute_calculation was found
        if not found_success and last_error:
            violations.append(AuditViolation(
                audit_name=self.name,
                rule_name="has_valid_result",
                message=f"No successful compute_calculation (last error: {last_error})",
                severity="error",
                location={"step_index": last_error_step} if last_error_step else None,
            ))

        result = AuditResult.PASS if not violations else AuditResult.FAIL
        return AuditReport(
            trace_id=trace_metadata.get("trace_id", "unknown"),
            result=result,
            violations=violations,
            passed_checks=passed,
            metadata={"audit_name": self.name},
        )

    def _parse_output(self, output: Any) -> Any:
        """
        Parse output from string or return as-is.

        Tries multiple parsing strategies:
        1. JSON parsing
        2. Python literal eval (for dicts, lists, tuples)
        3. Return as string if parsing fails
        """
        if output is None:
            return None

        if isinstance(output, str):
            # Strategy 1: JSON
            try:
                return json.loads(output)
            except (json.JSONDecodeError, TypeError):
                pass

            # Strategy 2: Python literal eval (safe subset)
            try:
                import ast
                return ast.literal_eval(output)
            except (ValueError, SyntaxError):
                pass

            # Strategy 3: Return as-is (keep string for text searching)
            return output

        return output


# =============================================================================
# MedCalc Correctness Audit (using calculators.py)
# =============================================================================

class MedCalcCorrectnessAudit(StructuredAudit):
    """
    Audit that verifies compute_calculation was attempted.

    This checks that the LLM tried to use the verified calculator.
    The calculator uses Python formulas internally, so if it's called,
    the formula application is guaranteed correct.

    We check for ATTEMPT, not success - if the calculator errors due to
    extraction issues, that's not the LLM's fault for trying to use it.
    """

    def __init__(self, tolerance: float = 0.05):
        super().__init__(
            name="medcalc_correctness",
            description="Verifies compute_calculation was attempted"
        )
        self.tolerance = tolerance

    def run(self, df: "pd.DataFrame", trace_metadata: Dict[str, Any]) -> "AuditReport":
        """Check that compute_calculation was attempted (not necessarily successful)."""
        from ..base import AuditReport, AuditResult, AuditViolation

        violations = []
        passed = []

        # Check: was compute_calculation ATTEMPTED?
        # We pass if the LLM tried, even if it errored
        attempted = False
        got_result = False

        for idx, row in df.iterrows():
            fn_name = str(row.get("fn_name", "")).lower()
            if "compute_calculation" in fn_name or "compute" in fn_name:
                attempted = True
                output = self._parse_output(row.get("output"))
                if isinstance(output, dict) and "result" in output and output["result"] is not None:
                    got_result = True
                    break
            elif "python_calculate" in fn_name:
                attempted = True
                output = self._parse_output(row.get("output"))
                if isinstance(output, dict) and "result" in output:
                    got_result = True
                    break

        if attempted:
            passed.append("attempted_calculator")
            if got_result:
                passed.append("got_result")
        else:
            violations.append(AuditViolation(
                audit_name=self.name,
                rule_name="attempted_calculator",
                message="LLM must attempt to use compute_calculation",
                severity="warning",
            ))

        result = AuditResult.PASS if not violations else AuditResult.FAIL
        return AuditReport(
            trace_id=trace_metadata.get("trace_id", "unknown"),
            result=result,
            violations=violations,
            passed_checks=passed,
            metadata={"audit_name": self.name},
        )

    def _parse_output(self, output: Any) -> Any:
        """Parse JSON string to dict if needed."""
        if output is None:
            return None
        if isinstance(output, str):
            try:
                return json.loads(output)
            except (json.JSONDecodeError, TypeError):
                return output
        return output

    def _check_calculation_with_details(
        self, df: "pd.DataFrame", metadata: Dict
    ) -> Tuple[bool, Optional[int], Optional[float], Optional[float]]:
        """
        Check if LLM calculation matches Python calculation.

        Returns:
            (is_correct, step_index, expected_value, actual_value)
        """
        calculators = self._get_calculators()
        if calculators is None:
            return True, None, None, None

        # Get patient note and question from metadata
        patient_note = metadata.get("patient_note", "")
        question = metadata.get("question", metadata.get("goal", ""))

        if not patient_note or not question:
            goal = metadata.get("goal", "")
            if "Patient Note:" in goal:
                parts = goal.split("Task:", 1)
                if len(parts) == 2:
                    patient_note = parts[0].replace("Patient Note:", "").strip()
                    question = parts[1].strip()

        if not patient_note:
            return True, None, None, None

        # Calculate expected result using Python
        try:
            expected_result = calculators.calculate(patient_note, question)
            if expected_result is None:
                return True, None, None, None
        except Exception:
            return True, None, None, None

        expected_value = expected_result.result

        # Extract LLM's result from trace with step index
        llm_result, step_idx = self._extract_llm_result_with_step(df)
        if llm_result is None:
            return True, None, None, None

        # Compare with tolerance
        if expected_value == 0:
            is_correct = abs(llm_result) < 0.01
        else:
            relative_error = abs(expected_value - llm_result) / abs(expected_value)
            is_correct = relative_error <= self.tolerance

        return is_correct, step_idx, expected_value, llm_result

    def _extract_llm_result_with_step(self, df: "pd.DataFrame") -> Tuple[Optional[float], Optional[int]]:
        """Extract numeric result and step index from the trace's calculation step."""
        # Look for calculation/compute steps in reverse order
        for idx in range(len(df) - 1, -1, -1):
            row = df.iloc[idx]
            fn_name = str(row.get("fn_name", "")).lower()

            if any(kw in fn_name for kw in ["calculat", "compute", "perform", "result"]):
                output = row.get("output")
                result = self._extract_number_from_output(output)
                if result is not None:
                    step_idx = row.get("step_idx", idx)
                    return result, int(step_idx) if step_idx is not None else idx

        # Fallback: check any step with numeric output
        for idx in range(len(df) - 1, -1, -1):
            row = df.iloc[idx]
            output = row.get("output")
            result = self._extract_number_from_output(output)
            if result is not None:
                step_idx = row.get("step_idx", idx)
                return result, int(step_idx) if step_idx is not None else idx

        return None, None

    def _extract_number_from_output(self, output: Any) -> Optional[float]:
        """Extract numeric value from various output formats."""
        if output is None:
            return None

        # Direct number
        if isinstance(output, (int, float)):
            return float(output)

        # String that might be JSON
        if isinstance(output, str):
            try:
                output = json.loads(output)
            except (json.JSONDecodeError, TypeError):
                # Try to extract number from string
                numbers = re.findall(r'-?\d+\.?\d*', output)
                if numbers:
                    return float(numbers[-1])  # Last number is often the result
                return None

        # Dict with result key
        if isinstance(output, dict):
            # Priority keys
            for key in ["result", "value", "answer", "output", "calculation"]:
                if key in output:
                    val = output[key]
                    if isinstance(val, (int, float)):
                        return float(val)
                    if isinstance(val, dict) and "result" in val:
                        inner = val["result"]
                        if isinstance(inner, (int, float)):
                            return float(inner)
                    if isinstance(val, str):
                        try:
                            return float(val)
                        except ValueError:
                            numbers = re.findall(r'-?\d+\.?\d*', val)
                            if numbers:
                                return float(numbers[-1])

            # Check nested result
            if "result" in output and isinstance(output["result"], dict):
                return self._extract_number_from_output(output["result"])

        return None


# =============================================================================
# Combined MedCalc BMI Audit
# =============================================================================

class MedCalcBMIAudit(DeclarativeAudit):
    """
    Combined audit specifically for BMI calculation tasks.

    Includes structure, validation, and correctness checks
    all in one audit.

    Example:
        >>> audit = MedCalcBMIAudit()
        >>> report = audit.run(df, metadata)
    """

    def __init__(self):
        super().__init__(
            name="medcalc_bmi",
            description="Complete audit for BMI calculation tasks"
        )

    @audit_rule(
        "has_value_extraction",
        "Must extract weight and height values",
        severity="error"
    )
    def check_extraction(self, dsl: AuditDSL) -> bool:
        """Check for value extraction."""
        return (
            dsl.has_at_least("extract", 1) or
            dsl.has_at_least("parse", 1) or
            dsl.has_at_least("get", 1)
        )

    @audit_rule(
        "has_bmi_calculation",
        "Must calculate BMI",
        severity="error"
    )
    def check_calculation(self, dsl: AuditDSL) -> bool:
        """Check for BMI calculation."""
        return (
            dsl.has_at_least("bmi", 1) or
            dsl.has_at_least("calculate", 1) or
            dsl.has_at_least("compute", 1)
        )

    @audit_rule(
        "extraction_before_calculation",
        "Must extract values before calculating",
        severity="warning"
    )
    def check_order(self, dsl: AuditDSL) -> bool:
        """Check operation order."""
        has_extract = dsl.has_at_least("extract", 1) or dsl.has_at_least("parse", 1)
        has_calc = dsl.has_at_least("bmi", 1) or dsl.has_at_least("calculate", 1)

        if has_extract and has_calc:
            return (
                dsl.comes_before("extract", "bmi") or
                dsl.comes_before("extract", "calculate") or
                dsl.comes_before("parse", "bmi") or
                dsl.comes_before("parse", "calculate")
            )
        return True

    @audit_rule(
        "no_failures",
        "All steps must succeed",
        severity="error"
    )
    def check_no_failures(self, dsl: AuditDSL) -> bool:
        """Check for failures."""
        return dsl.no_failures()

    @audit_rule(
        "reasonable_step_count",
        "BMI calculation should have 2-5 steps",
        severity="warning"
    )
    def check_step_count(self, dsl: AuditDSL) -> bool:
        """Check reasonable step count."""
        pattern = dsl.get_pattern()
        # Exclude start/end tokens
        actual_steps = [s for s in pattern if s not in ("<START>", "<END>")]
        return 2 <= len(actual_steps) <= 10


# =============================================================================
# Factory Functions
# =============================================================================

def create_medcalc_audit(
    calculator_type: Optional[str] = None,
    include_correctness: bool = True,
) -> StructuredAudit:
    """
    Create a combined MedCalc audit.

    Args:
        calculator_type: Optional specific calculator (e.g., "bmi", "gfr")
        include_correctness: Whether to include correctness checks

    Returns:
        Combined StructuredAudit

    Example:
        >>> audit = create_medcalc_audit("bmi")
        >>> report = audit.run(df, metadata)
    """
    if calculator_type == "bmi":
        return MedCalcBMIAudit()

    # Create combined audit
    audit = StructuredAudit(
        name="medcalc_combined",
        description="Combined audit for medical calculations"
    )

    # Add structure checks
    audit.add_rule(
        "has_extraction",
        lambda df, _: any(
            "extract" in fn.lower() or "parse" in fn.lower()
            for fn in df["fn_name"]
        ),
        "Must extract input values"
    )

    audit.add_rule(
        "has_calculation",
        lambda df, _: any(
            "calculate" in fn.lower() or "compute" in fn.lower()
            for fn in df["fn_name"]
        ),
        "Must perform calculation"
    )

    audit.add_rule(
        "no_failures",
        lambda df, _: (df["status"] != "failed").all() if len(df) > 0 else True,
        "No steps should fail"
    )

    return audit


def register_medcalc_audits(domain: str = "medcalc") -> None:
    """
    Register all MedCalc audits in the global registry.

    Args:
        domain: Domain name to register under

    Example:
        >>> register_medcalc_audits()
        >>> # Now available via registry.list_by_domain("medcalc")
    """
    register_audit(MedCalcStructureAudit(), domain=domain)
    register_audit(MedCalcOutputValidationAudit(), domain=domain)
    register_audit(MedCalcCorrectnessAudit(), domain=domain)
    register_audit(MedCalcBMIAudit(), domain=domain)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Constants
    "KNOWN_CALCULATORS",
    "MEDCALC_STEP_PATTERNS",
    # Functions
    "get_known_calculators",  # Dynamic calculator import
    # Audits
    "MedCalcStructureAudit",
    "MedCalcOutputValidationAudit",
    "MedCalcCorrectnessAudit",
    "MedCalcBMIAudit",
    # Factory functions
    "create_medcalc_audit",
    "register_medcalc_audits",
]
