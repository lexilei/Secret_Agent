# Audit System

The audit system provides comprehensive quality assurance for agent traces through structured audits and typicality models, following the SSRM (Semi-Structured Reasoning Models) approach.

## Overview

The audit system has three main components:

1. **Structured Audits**: Rule-based checks using a DataFrame DSL
2. **Typicality Models**: Probabilistic models for detecting atypical patterns
3. **Audit Runner**: Orchestration for running audits on traces

## Quick Start

```python
from ptool_framework.audit import (
    StructuredAudit,
    AuditDSL,
    NonEmptyTraceAudit,
    NoFailedStepsAudit,
    AuditRunner,
)
from ptool_framework.traces import WorkflowTrace, StepStatus

# Create a simple audit
audit = StructuredAudit("basic_check", "Basic trace validation")
audit.add_rule(
    "has_steps",
    lambda df, _: len(df) > 0,
    "Trace must have at least one step"
)
audit.add_rule(
    "no_failures",
    lambda df, _: (df["status"] != "failed").all(),
    "No failed steps allowed"
)

# Create a trace
trace = WorkflowTrace(goal="Test")
trace.add_step("extract", {"text": "hello"})
trace.steps[0].status = StepStatus.COMPLETED

# Run audit
runner = AuditRunner(audits=[audit])
report = runner.audit_trace(trace)
print(f"Result: {report.result.value}")  # "pass" or "fail"
```

## Structured Audits

### AuditDSL

The `AuditDSL` class provides a domain-specific language for querying trace DataFrames:

```python
from ptool_framework.audit import AuditDSL
import pandas as pd

# Create DSL from DataFrame
df = pd.DataFrame({
    "fn_name": ["extract", "validate", "calculate", "format"],
    "status": ["completed", "completed", "completed", "completed"],
    "step_idx": [0, 1, 2, 3],
    "input": ['{"text": "..."}', '{"data": []}', '{"values": {}}', '{"result": 42}'],
    "output": ['[1,2]', 'true', '42', '"Result: 42"'],
    "duration_ms": [100, 50, 200, 30],
})
metadata = {"trace_id": "test123", "goal": "Calculate result"}
dsl = AuditDSL(df, metadata)

# Query methods
extract_steps = dsl.steps_named("extract")  # Partial match
exact_match = dsl.steps_named("extract", exact=True)
completed = dsl.steps_with_status("completed")
first_step = dsl.get_step(0)
last_step = dsl.get_step(-1)
pattern = dsl.get_pattern()  # ["extract", "validate", "calculate", "format"]

# Count assertions
assert dsl.has_exactly("extract", 1)
assert dsl.has_at_least("validate", 1)
assert dsl.has_at_most("calculate", 2)
assert dsl.is_non_empty()

# Order assertions
assert dsl.comes_before("extract", "calculate")
assert dsl.comes_after("format", "validate")
assert dsl.pattern_exists(["extract", "calculate"])  # Subsequence
assert dsl.pattern_exists(["extract", "validate"], contiguous=True)  # Adjacent

# Status checks
assert dsl.all_completed()
assert dsl.no_failures()
assert dsl.success_rate() == 1.0
```

### Pre-built Audits

The framework provides several ready-to-use audits:

```python
from ptool_framework.audit import (
    NonEmptyTraceAudit,
    NoFailedStepsAudit,
    AllCompletedAudit,
    RequiredStepsAudit,
    StepOrderAudit,
    PatternAudit,
    PerformanceAudit,
    create_basic_audit,
    create_workflow_audit,
)

# Simple audits
non_empty = NonEmptyTraceAudit()
no_fails = NoFailedStepsAudit()
all_done = AllCompletedAudit()

# Configurable audits
required = RequiredStepsAudit(["extract", "validate"])
order = StepOrderAudit([("extract", "validate"), ("validate", "calculate")])
pattern = PatternAudit(
    required_patterns=[["extract", "calculate"]],
    forbidden_patterns=[["calculate", "extract"]],
)
perf = PerformanceAudit(max_step_duration_ms=1000, max_step_count=10)

# Factory functions
basic = create_basic_audit()  # NonEmpty + NoFailed + AllCompleted
workflow = create_workflow_audit(
    required_steps=["extract", "calculate"],
    step_order=[("extract", "calculate")],
)
```

### Declarative Audits

Create custom audits using the `@audit_rule` decorator:

```python
from ptool_framework.audit import DeclarativeAudit, audit_rule

class MyWorkflowAudit(DeclarativeAudit):
    def __init__(self):
        super().__init__("my_workflow", "Custom workflow validation")

    @audit_rule("has_extraction", "Must have extraction step")
    def check_extraction(self, dsl):
        return dsl.has_at_least("extract", 1)

    @audit_rule("has_validation", "Must have validation step")
    def check_validation(self, dsl):
        return dsl.has_at_least("validate", 1)

    @audit_rule("correct_order", "Extract must come before validate")
    def check_order(self, dsl):
        return dsl.comes_before("extract", "validate")

# Use the audit
audit = MyWorkflowAudit()
report = audit.run(df, metadata)
print(report.passed_checks)  # List of passed rule names
print(report.violations)     # List of failed rules with details
```

## Typicality Models

Typicality models learn patterns from historical traces and score new traces based on how typical they are.

### Available Models

```python
from ptool_framework.audit.typicality import (
    UnigramModel,
    BigramModel,
    TrigramModel,
    InterpolatedModel,
    HMMModel,
    create_typicality_model,
    train_ensemble_models,
    ensemble_score,
    extract_pattern,
    START_TOKEN,
    END_TOKEN,
)

# Training patterns
patterns = [
    [START_TOKEN, "extract", "validate", "calculate", END_TOKEN],
    [START_TOKEN, "extract", "calculate", END_TOKEN],
    [START_TOKEN, "extract", "validate", "calculate", "format", END_TOKEN],
]

# Unigram model - word frequency
unigram = UnigramModel()
unigram.fit(patterns)
score = unigram.score([START_TOKEN, "extract", "calculate", END_TOKEN])

# Bigram model - P(word | previous_word)
bigram = BigramModel()
bigram.fit(patterns)
score = bigram.score([START_TOKEN, "extract", "validate", END_TOKEN])

# Trigram model - P(word | previous_two_words)
trigram = TrigramModel()
trigram.fit(patterns)
score = trigram.score([START_TOKEN, "extract", "validate", "calculate", END_TOKEN])

# Interpolated model - weighted combination
interpolated = InterpolatedModel(lambdas=(0.2, 0.3, 0.5))
interpolated.fit(patterns)
score = interpolated.score(pattern)

# HMM model - Hidden Markov Model
hmm = HMMModel(n_states=3)
hmm.fit(patterns)
score = hmm.score(pattern)

# Factory functions
model = create_typicality_model("bigram")
model.fit(patterns)

# Ensemble scoring
models = train_ensemble_models(patterns)
score = ensemble_score(models, pattern, weights={"unigram": 0.2, "bigram": 0.5, "trigram": 0.3})
```

### Pattern Extraction

```python
from ptool_framework.audit.typicality import (
    extract_pattern,
    get_ngrams,
    PatternStats,
    normalize_step_names,
)
import pandas as pd

# Extract pattern from DataFrame
df = pd.DataFrame({"fn_name": ["extract", "validate", "calculate"]})
pattern = extract_pattern(df)  # ['<START>', 'extract', 'validate', 'calculate', '<END>']

# Get n-grams
bigrams = get_ngrams(pattern, 2)  # [('<START>', 'extract'), ('extract', 'validate'), ...]

# Pattern statistics
stats = PatternStats.from_patterns(patterns)
print(stats.summary())
print(stats.most_common_steps(3))
print(stats.transition_probability("<START>", "extract"))

# Normalize step names
normalized = normalize_step_names(patterns, "lowercase")  # Lowercase all
normalized = normalize_step_names(patterns, "strip_prefix")  # Remove common prefixes
```

## Audit Runner

The `AuditRunner` orchestrates running multiple audits on traces:

```python
from ptool_framework.audit import AuditRunner, TypicalityTrainer
from ptool_framework.audit.typicality import BigramModel

# Create runner with audits
runner = AuditRunner(
    audits=[NonEmptyTraceAudit(), NoFailedStepsAudit()],
    typicality_model=BigramModel(),
)

# Audit single trace
report = runner.audit_trace(trace, metadata={"goal": "Test"})

# Batch audit
results = runner.run_batch(traces, metadata_list=metadata_list)
print(f"Pass rate: {results.pass_rate:.1%}")
print(f"Failed: {len(results.failed_traces)}")

# Train typicality model from traces
trainer = TypicalityTrainer(model_type="bigram")
trainer.train(successful_traces)
typicality_model = trainer.model

# Use trained model in runner
runner = AuditRunner(
    audits=[NonEmptyTraceAudit()],
    typicality_model=typicality_model,
)
```

## DataFrame Schema

When traces are converted to DataFrames, they follow this schema:

| Column | Type | Description |
|--------|------|-------------|
| `step_idx` | int | Step index (0-based) |
| `fn_name` | str | Function/ptool name |
| `input` | str | JSON-serialized inputs |
| `output` | str | JSON-serialized output |
| `status` | str | "completed", "failed", "pending" |
| `duration_ms` | float | Execution time in ms |
| `goal` | str | Step goal/description |
| `error` | str/None | Error message if failed |
| `input_0`, `input_1`, ... | str | Individual input arguments |

## Audit Registry

Register and manage audits globally:

```python
from ptool_framework.audit import (
    AuditRegistry,
    get_audit_registry,
    register_audit,
)

# Global registry
registry = get_audit_registry()

# Register audit
register_audit(MyWorkflowAudit(), domain="medcalc")

# List audits
all_audits = registry.list_all()
medcalc_audits = registry.list_by_domain("medcalc")

# Get specific audit
audit = registry.get("my_workflow")
```

## Step Location Tracking

Audits can include step location information in violations for targeted repair:

```python
from ptool_framework.audit.base import AuditViolation

# When creating violations, include step location
violation = AuditViolation(
    audit_name="my_audit",
    rule_name="calculation_correct",
    message="LLM result (22.39) differs from expected (25.02)",
    severity="error",
    location={"step_index": 3},  # Points to the specific step
    context={"expected": 25.02, "actual": 22.39},
)
```

The `location` field enables the repair agent to:
1. Know which step to regenerate
2. Apply targeted fixes instead of general retries
3. Track which steps have been repaired

### Custom Audit with Step Tracking

```python
class MyAuditWithStepTracking(StructuredAudit):
    """Audit that tracks step locations in violations."""

    def run(self, df: pd.DataFrame, trace_metadata: Dict) -> AuditReport:
        violations = []

        for idx, row in df.iterrows():
            step_idx = int(row.get("step_idx", idx))

            if not self._is_valid(row):
                violations.append(AuditViolation(
                    audit_name=self.name,
                    rule_name="validation",
                    message="Step validation failed - regenerate step",
                    severity="error",
                    location={"step_index": step_idx},  # Track which step
                ))

        return AuditReport(
            trace_id=trace_metadata.get("trace_id", "unknown"),
            result=AuditResult.PASS if not violations else AuditResult.FAIL,
            violations=violations,
        )
```

## Best Practices

1. **Start simple**: Use pre-built audits before creating custom ones
2. **Layer audits**: Use basic audits first, then domain-specific ones
3. **Use typicality**: Train on successful traces to detect anomalies
4. **Test audits**: Write unit tests for custom audit rules
5. **Log violations**: Store audit reports for debugging
6. **Include step locations**: Add `location={"step_index": N}` to violations for targeted repair

## See Also

- [Critic and Repair System](critic-repair.md)
- [MedCalc Domain Example](medcalc-domain.md)
- [API Reference](api-reference.md)
