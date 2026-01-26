# MedCalc Domain Example

This document describes the MedCalc domain implementation, which serves as a reference example for creating domain-specific audits.

## Overview

The MedCalc domain implements audits for medical calculator workflows. It demonstrates:

1. Domain-specific structured audits
2. Formula verification
3. Validation rules
4. How to create your own domain audits

## Quick Start

```python
from ptool_framework.audit.domains.medcalc import (
    MedCalcStructureAudit,
    MedCalcOutputValidationAudit,
    MedCalcBMIAudit,
    KNOWN_CALCULATORS,
)

# Use pre-built MedCalc audits
audits = [
    MedCalcStructureAudit(),       # Check workflow structure
    MedCalcOutputValidationAudit(), # Validate outputs
    MedCalcBMIAudit(),             # BMI-specific checks
]

# Create runner
from ptool_framework.audit import AuditRunner
runner = AuditRunner(audits=audits)

# Audit a MedCalc trace
report = runner.audit_trace(trace, {"goal": "Calculate BMI"})
```

## MedCalc Audits

### MedCalcStructureAudit

Validates workflow structure for medical calculations:

```python
from ptool_framework.audit.domains.medcalc import MedCalcStructureAudit

audit = MedCalcStructureAudit()

# Checks:
# - has_identification: Trace identifies a calculator
# - has_extraction: Trace extracts values
# - correct_order: Extraction comes before calculation
```

Rules implemented:

| Rule | Description |
|------|-------------|
| `has_identification` | Must have a step that identifies the calculator type |
| `has_extraction` | Must have a step that extracts values from input |
| `correct_order` | Extraction must come before calculation |

### MedCalcOutputValidationAudit

Validates outputs are reasonable for medical calculations:

```python
from ptool_framework.audit.domains.medcalc import MedCalcOutputValidationAudit

audit = MedCalcOutputValidationAudit()

# Checks:
# - valid_calculator: Calculator name is recognized
# - numeric_outputs: Extracted values are numeric
# - reasonable_values: Values are within expected ranges
```

### MedCalcBMIAudit

Specific audit for BMI calculations:

```python
from ptool_framework.audit.domains.medcalc import MedCalcBMIAudit

audit = MedCalcBMIAudit()

# Checks:
# - has_weight: Weight is extracted
# - has_height: Height is extracted
# - bmi_formula: BMI calculated correctly (weight / height^2)
```

### MedCalcCorrectnessAudit

Verifies LLM calculations against Python calculators:

```python
from ptool_framework.audit.domains.medcalc import MedCalcCorrectnessAudit

audit = MedCalcCorrectnessAudit(tolerance=0.05)  # 5% tolerance

# Checks:
# - calculation_correct: LLM result matches Python calculator within tolerance
```

This audit:
1. Extracts the goal to get patient note and question
2. Calls `calculators.calculate()` to get the expected result
3. Finds the calculation step in the trace
4. Compares LLM result with expected result
5. Creates violation with `step_index` if difference exceeds tolerance

```python
# Example violation generated:
AuditViolation(
    audit_name="medcalc_correctness",
    rule_name="calculation_correct",
    message="LLM result (22.39) differs from expected (25.02) by >5%",
    severity="error",
    location={"step_index": 3},  # Points to the calculation step
    context={"expected": 25.02, "actual": 22.39, "tolerance": 0.05},
)
```

The step_index enables the repair agent to regenerate the specific calculation step using the Python calculator fallback.

## Known Calculators

The domain includes formulas for common medical calculators:

```python
from ptool_framework.audit.domains.medcalc import KNOWN_CALCULATORS

# Available calculators:
KNOWN_CALCULATORS = {
    "bmi": {
        "name": "Body Mass Index",
        "formula": "weight / height^2",
        "unit": "kg/m²",
        "required_inputs": ["weight", "height"],
    },
    "bsa": {
        "name": "Body Surface Area",
        "formula": "sqrt((height * weight) / 3600)",
        "unit": "m²",
        "required_inputs": ["weight", "height"],
    },
    "gfr": {
        "name": "Glomerular Filtration Rate",
        "formula": "CKD-EPI equation",
        "unit": "mL/min/1.73m²",
        "required_inputs": ["creatinine", "age", "sex"],
    },
    # ... more calculators
}
```

## Creating Domain Audits

### Step 1: Create Domain Module

```python
# ptool_framework/audit/domains/my_domain.py

from ..base import StructuredAudit, AuditResult
from ..structured_audit import DeclarativeAudit, audit_rule, AuditDSL

# Domain-specific constants
VALID_OPERATIONS = ["fetch", "transform", "validate", "store"]
```

### Step 2: Create Structure Audit

```python
class MyDomainStructureAudit(DeclarativeAudit):
    """Validate workflow structure for my domain."""

    def __init__(self):
        super().__init__(
            "my_domain_structure",
            "My Domain Structure Validation"
        )

    @audit_rule("has_fetch", "Must have a fetch step")
    def check_fetch(self, dsl: AuditDSL) -> bool:
        return dsl.has_at_least("fetch", 1)

    @audit_rule("has_transform", "Must have a transform step")
    def check_transform(self, dsl: AuditDSL) -> bool:
        return dsl.has_at_least("transform", 1)

    @audit_rule("correct_order", "Fetch must come before transform")
    def check_order(self, dsl: AuditDSL) -> bool:
        return dsl.comes_before("fetch", "transform")
```

### Step 3: Create Validation Audit

```python
class MyDomainValidationAudit(StructuredAudit):
    """Validate outputs for my domain."""

    def __init__(self):
        super().__init__(
            "my_domain_validation",
            "My Domain Output Validation"
        )

        # Add rules programmatically
        self.add_rule(
            "valid_operation",
            self._check_valid_operation,
            "Operation must be recognized"
        )
        self.add_rule(
            "has_result",
            self._check_has_result,
            "Must produce a result"
        )

    def _check_valid_operation(self, df, metadata):
        """Check all operations are valid."""
        if len(df) == 0:
            return True
        return df["fn_name"].isin(VALID_OPERATIONS).all()

    def _check_has_result(self, df, metadata):
        """Check trace produces a result."""
        if len(df) == 0:
            return False
        # Check last step has output
        last_output = df.iloc[-1]["output"]
        return last_output is not None and last_output != "null"
```

### Step 4: Create Correctness Audit

```python
class MyDomainCorrectnessAudit(StructuredAudit):
    """Verify calculation correctness."""

    def __init__(self, expected_formula=None):
        super().__init__(
            "my_domain_correctness",
            "My Domain Correctness Check"
        )
        self.expected_formula = expected_formula

        if expected_formula:
            self.add_rule(
                "formula_correct",
                self._check_formula,
                f"Must use formula: {expected_formula}"
            )

    def _check_formula(self, df, metadata):
        """Verify the formula was applied correctly."""
        # Implementation depends on domain
        # Example: Check inputs and outputs match expected formula
        try:
            # Get inputs
            transform_step = df[df["fn_name"].str.contains("transform")]
            if len(transform_step) == 0:
                return True  # Abstain if no transform step

            inputs = json.loads(transform_step.iloc[0]["input"])
            output = json.loads(transform_step.iloc[0]["output"])

            # Verify formula (domain-specific logic)
            expected = self._apply_formula(inputs)
            return abs(output - expected) < 0.01

        except Exception:
            return False

    def _apply_formula(self, inputs):
        """Apply the expected formula."""
        # Implement based on self.expected_formula
        pass
```

### Step 5: Register Domain

```python
# In __init__.py
from .my_domain import (
    MyDomainStructureAudit,
    MyDomainValidationAudit,
    MyDomainCorrectnessAudit,
    VALID_OPERATIONS,
)

# Register with global registry
from ...base import register_audit

register_audit(MyDomainStructureAudit(), domain="my_domain")
register_audit(MyDomainValidationAudit(), domain="my_domain")
```

### Step 6: Create Factory Functions

```python
def create_my_domain_audit(
    check_structure=True,
    check_validation=True,
    expected_formula=None,
):
    """Create a comprehensive my_domain audit suite."""
    from ..base import CompositeAudit

    audits = []
    if check_structure:
        audits.append(MyDomainStructureAudit())
    if check_validation:
        audits.append(MyDomainValidationAudit())
    if expected_formula:
        audits.append(MyDomainCorrectnessAudit(expected_formula))

    return CompositeAudit("my_domain", audits)
```

## Example: Complete MedCalc Audit

```python
from ptool_framework.audit.domains.medcalc import (
    MedCalcStructureAudit,
    MedCalcOutputValidationAudit,
    MedCalcBMIAudit,
)
from ptool_framework.audit import AuditRunner
from ptool_framework.traces import WorkflowTrace, StepStatus
import json

# Create trace
trace = WorkflowTrace(goal="Calculate BMI for patient")
trace.add_step(
    "identify_calculator",
    {"text": "Calculate BMI for patient with weight 70kg, height 1.75m"},
    goal="Identify the calculator type"
)
trace.add_step(
    "extract_values",
    {"text": "weight 70kg, height 1.75m"},
    goal="Extract numeric values"
)
trace.add_step(
    "calculate_bmi",
    {"weight": 70, "height": 1.75},
    goal="Calculate BMI"
)

# Mark steps completed with results
trace.steps[0].status = StepStatus.COMPLETED
trace.steps[0].result = {"calculator": "bmi"}

trace.steps[1].status = StepStatus.COMPLETED
trace.steps[1].result = {"weight": 70, "height": 1.75}

trace.steps[2].status = StepStatus.COMPLETED
trace.steps[2].result = {"bmi": 22.86}

# Run audits
runner = AuditRunner(audits=[
    MedCalcStructureAudit(),
    MedCalcOutputValidationAudit(),
    MedCalcBMIAudit(),
])

report = runner.audit_trace(trace, {"goal": "Calculate BMI"})

print(f"Result: {report.result.value}")
print(f"Passed: {report.passed_checks}")
if report.violations:
    for v in report.violations:
        print(f"Failed: {v.rule_name} - {v.message}")
```

## Domain Audit Patterns

### Pattern 1: Required Steps

```python
@audit_rule("required_steps", "Must have all required steps")
def check_required_steps(self, dsl):
    required = ["authenticate", "fetch", "process", "respond"]
    return all(dsl.has_at_least(step, 1) for step in required)
```

### Pattern 2: Value Ranges

```python
def _check_value_ranges(self, df, metadata):
    """Check values are within acceptable ranges."""
    for _, row in df.iterrows():
        try:
            output = json.loads(row["output"])
            if isinstance(output, dict):
                for key, value in output.items():
                    if key in self.value_ranges:
                        min_val, max_val = self.value_ranges[key]
                        if not (min_val <= value <= max_val):
                            return False
        except:
            pass
    return True
```

### Pattern 3: Output Format Validation

```python
def _check_output_format(self, df, metadata):
    """Validate output format matches expected schema."""
    last_step = df.iloc[-1] if len(df) > 0 else None
    if last_step is None:
        return False

    try:
        output = json.loads(last_step["output"])
        return all(key in output for key in self.required_output_keys)
    except:
        return False
```

### Pattern 4: Dependency Chains

```python
@audit_rule("dependency_chain", "Steps must follow dependency order")
def check_dependencies(self, dsl):
    dependencies = [
        ("authenticate", "fetch"),
        ("fetch", "transform"),
        ("transform", "validate"),
        ("validate", "store"),
    ]
    return all(
        dsl.comes_before(a, b)
        for a, b in dependencies
    )
```

## Best Practices

1. **Start generic**: Create base domain audit, then add specific ones
2. **Fail early**: Check structure before content
3. **Provide clear messages**: Make violations actionable
4. **Test edge cases**: Empty traces, single steps, missing data
5. **Document formulas**: Explain validation logic

## See Also

- [Audit System](audit-system.md)
- [Critic and Repair](critic-repair.md)
- [API Reference](api-reference.md)
