# Critic and Repair System

The critic and repair system evaluates trace quality and automatically fixes problematic traces based on audit results and repair suggestions.

## Overview

The system implements William's "Approach 3" for agent reliability:

1. **Critic**: Evaluates trace quality using audits and generates verdicts
2. **Repair Agent**: Fixes traces based on critic feedback
3. **Iterative Loop**: Repair-evaluate cycle until acceptance or max attempts

## Quick Start

```python
from ptool_framework.critic import TraceCritic, CriticVerdict, evaluate_trace
from ptool_framework.repair import RepairAgent, repair_trace
from ptool_framework.audit import NonEmptyTraceAudit, NoFailedStepsAudit

# Create critic with audits
critic = TraceCritic(
    audits=[NonEmptyTraceAudit(), NoFailedStepsAudit()],
    accept_threshold=0.8,
    repair_threshold=0.4,
)

# Evaluate trace
evaluation = critic.evaluate(trace, goal="Calculate BMI")

if evaluation.verdict == CriticVerdict.ACCEPT:
    print("Trace is good!")
elif evaluation.verdict == CriticVerdict.REPAIR_NEEDED:
    print("Attempting repair...")
    result = repair_trace(trace, goal="Calculate BMI", critic=critic)
    if result.success:
        print(f"Fixed in {result.iterations} iterations")
        fixed_trace = result.repaired_trace
else:
    print("Trace is unfixable")
```

## TraceCritic

The `TraceCritic` class evaluates trace quality and generates verdicts.

### Configuration

```python
from ptool_framework.critic import TraceCritic, CriticVerdict

critic = TraceCritic(
    audits=[...],              # List of audits to run
    accept_threshold=0.8,      # Confidence threshold for ACCEPT
    repair_threshold=0.4,      # Confidence threshold for REPAIR_NEEDED (below = REJECT)
)
```

### Evaluation

```python
# Full evaluation
evaluation = critic.evaluate(trace, goal="Calculate BMI")

# Access evaluation results
print(evaluation.verdict)           # CriticVerdict.ACCEPT, REPAIR_NEEDED, or REJECT
print(evaluation.confidence)        # 0.0 - 1.0
print(evaluation.is_acceptable)     # True if ACCEPT
print(evaluation.needs_repair)      # True if REPAIR_NEEDED
print(evaluation.should_reject)     # True if REJECT

# Scores
print(evaluation.completeness_score)  # How complete is the trace
print(evaluation.correctness_score)   # How correct are the results
print(evaluation.efficiency_score)    # How efficient is the trace

# Details
print(evaluation.failed_steps)        # List of failed step indices
print(evaluation.repair_suggestions)  # List of suggested repairs
print(evaluation.audit_reports)       # Individual audit reports
```

### CriticVerdict

```python
from ptool_framework.critic import CriticVerdict

# Verdicts
CriticVerdict.ACCEPT         # Trace is acceptable
CriticVerdict.REPAIR_NEEDED  # Trace can be fixed
CriticVerdict.REJECT         # Trace is unfixable
```

### CriticEvaluation

```python
from ptool_framework.critic import CriticEvaluation

@dataclass
class CriticEvaluation:
    verdict: CriticVerdict
    confidence: float
    trace_id: Optional[str] = None
    audit_reports: List[AuditReport] = field(default_factory=list)
    failed_steps: List[int] = field(default_factory=list)
    repair_suggestions: List[Dict[str, Any]] = field(default_factory=list)
    completeness_score: float = 0.0
    correctness_score: float = 0.0
    efficiency_score: float = 0.0
    reasoning_issues: List[str] = field(default_factory=list)

    @property
    def is_acceptable(self) -> bool: ...
    @property
    def needs_repair(self) -> bool: ...
    @property
    def should_reject(self) -> bool: ...

    def to_dict(self) -> Dict[str, Any]: ...
    def summary(self) -> str: ...
```

## RepairAgent

The `RepairAgent` fixes traces based on critic feedback.

### Configuration

```python
from ptool_framework.repair import RepairAgent

agent = RepairAgent(
    critic=critic,           # TraceCritic for re-evaluation
    max_attempts=3,          # Maximum repair iterations
    model="deepseek-v3",     # LLM for regeneration
    use_icl_examples=True,   # Use ICL examples from trace store
    trace_store=trace_store, # Optional trace store for examples
)
```

### Repair Actions

The repair agent can perform several types of repairs:

```python
from ptool_framework.repair import RepairActionType

RepairActionType.REGENERATE_STEP  # Re-run a failed step
RepairActionType.ADD_STEP         # Add a missing step
RepairActionType.REMOVE_STEP      # Remove a redundant step
RepairActionType.MODIFY_ARGS      # Change step arguments
RepairActionType.REORDER_STEPS    # Fix step ordering
```

### Repair Result

```python
from ptool_framework.repair import RepairResult

result = agent.repair(trace, evaluation, goal)

# Check result
if result.success:
    fixed_trace = result.repaired_trace
    print(f"Fixed in {result.iterations} iterations")
    print(f"Actions taken: {len(result.actions_taken)}")
else:
    print(f"Repair failed: {result.error}")

# Inspect actions
for action in result.actions_taken:
    print(f"{action.action_type.value}: {action.reason}")
```

### Repair Flow

```
┌──────────────────────┐
│   Receive Trace      │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Evaluate with       │
│  Critic              │
└──────────┬───────────┘
           │
           ▼
    ┌──────┴──────┐
    │  Verdict?   │
    └──────┬──────┘
           │
    ┌──────┼──────────────┐
    │      │              │
    ▼      ▼              ▼
 ACCEPT  REPAIR        REJECT
    │    NEEDED          │
    │      │             │
    │      ▼             │
    │  Get Suggestions   │
    │      │             │
    │      ▼             │
    │  Apply Repairs     │
    │      │             │
    │      ▼             │
    │  Re-evaluate       │◄─────┐
    │      │             │      │
    │  ┌───┴───┐        │      │
    │  │ACCEPT?│        │      │
    │  └───┬───┘        │      │
    │   No │ Yes         │      │
    │      │  │         │      │
    │      │  ▼         │      │
    │      │ Done       │      │
    │      │            │      │
    │   More  ──────────┼──────┘
    │   attempts?       │
    │      │            │
    ▼      ▼            ▼
   Done   Fail        Reject
```

## Convenience Functions

```python
from ptool_framework.critic import evaluate_trace, quick_evaluate
from ptool_framework.repair import repair_trace, auto_repair

# Quick evaluation (returns CriticEvaluation)
evaluation = evaluate_trace(trace, audits=[...])

# Very quick evaluation (returns just CriticVerdict)
verdict = quick_evaluate(trace)

# Full repair with custom critic
result = repair_trace(trace, goal, critic=critic, max_attempts=3)

# Auto-repair (returns fixed trace or None)
fixed = auto_repair(trace, goal)
if fixed:
    use_trace(fixed)
```

## Integration with ReActAgent

```python
from ptool_framework.react import ReActAgent
from ptool_framework.critic import TraceCritic
from ptool_framework.repair import RepairAgent

# Create components
critic = TraceCritic(audits=[...])
repair_agent = RepairAgent(critic=critic)

# Use with ReActAgent
agent = ReActAgent(
    ptools=[...],
    model="deepseek-v3",
)

# Run agent
trajectory = agent.run(goal="Calculate BMI for patient with weight 70kg, height 1.75m")

# Evaluate and repair if needed
if trajectory.success:
    evaluation = critic.evaluate(trajectory, goal=goal)

    if evaluation.verdict == CriticVerdict.REPAIR_NEEDED:
        result = repair_agent.repair(
            trajectory.generated_trace,
            evaluation,
            goal
        )
        if result.success:
            trajectory.repaired_trace = result.repaired_trace
```

## Repair Suggestions Format

Repair suggestions follow this format:

```python
{
    "action": "regenerate_step",     # Action type
    "step_index": 2,                 # Step to repair
    "reason": "Step failed: Missing height parameter",
    "priority": "high",              # "high", "medium", "low"
    "details": {                     # Additional info
        "error": "KeyError: height",
        "suggested_fix": "Add height parameter",
    }
}
```

## Python Calculator Fallback

For calculation errors, the repair agent can use deterministic Python functions instead of regenerating with the LLM. This is especially useful for domains like medical calculations where accuracy is critical:

```python
from ptool_framework.repair import RepairAgent

# The repair agent automatically detects calculation errors
# and uses Python calculators when available

agent = RepairAgent(
    critic=critic,
    max_attempts=2,
    verbose=True,  # Shows Python calculator usage
)

# During repair, if a calculation step has an error like:
# "LLM result (22.39) differs from expected (25.02)"
# The agent will:
# 1. Detect it's a calculation error (ptool name contains "calculation" or "perform")
# 2. Parse the goal to extract patient_note and question
# 3. Call the Python calculator directly
# 4. Update the step with the deterministic result
```

### How it Works

When `_regenerate_step` is called for a calculation step with an error containing "differs", "incorrect", or "wrong":

1. The agent calls `_use_python_calculator(trace, step_index, goal)`
2. This extracts patient note and question from the goal
3. Calls `calculators.calculate(patient_note, question)`
4. Returns the Python-calculated result

```python
# Example repair output:
# [regenerate_step] Regenerating step 3: perform_calculation
# [regenerate_step] Calculation error detected - using Python calculator
# [python_calc] Got result: 25.02 from Creatinine Clearance (Cockcroft-Gault Equation)
# [OK] Applied: regenerate_step
```

### Repaired Trace Always Returned

The repair agent now **always returns the repaired trace** when any actions were taken, even if the overall verdict is not ACCEPT. This allows extracting partial fixes:

```python
result = repair_agent.repair(trace, evaluation, goal)

# Even if result.success is False, we may have fixed the calculation
if repair_result.repaired_trace:
    # Check for repaired calculation results
    for step in reversed(repair_result.repaired_trace.steps):
        if step.result and isinstance(step.result, dict) and "result" in step.result:
            fixed_answer = step.result["result"]
            print(f"Extracted fixed answer: {fixed_answer}")
            break
```

## ICL Examples

The repair agent can use In-Context Learning examples from a trace store:

```python
from ptool_framework.repair import RepairAgent
from ptool_framework.trace_store import TraceStore

trace_store = TraceStore()
agent = RepairAgent(
    critic=critic,
    trace_store=trace_store,
    use_icl_examples=True,
)

# During repair, the agent retrieves:
# - Positive examples: Successful traces for the same ptool
# - Negative examples: Failed traces to avoid
```

## Best Practices

1. **Set appropriate thresholds**: Balance between too strict and too lenient
2. **Limit repair attempts**: Usually 2-3 is sufficient
3. **Log repair actions**: Track what repairs were applied
4. **Use ICL examples**: Helps regeneration quality
5. **Monitor repair success rate**: Track long-term effectiveness

## Example: Full Pipeline

```python
from ptool_framework.critic import TraceCritic, CriticVerdict
from ptool_framework.repair import RepairAgent
from ptool_framework.audit import (
    NonEmptyTraceAudit,
    NoFailedStepsAudit,
    RequiredStepsAudit,
)

# Setup
audits = [
    NonEmptyTraceAudit(),
    NoFailedStepsAudit(),
    RequiredStepsAudit(["extract", "calculate"]),
]

critic = TraceCritic(
    audits=audits,
    accept_threshold=0.8,
    repair_threshold=0.4,
)

repair_agent = RepairAgent(
    critic=critic,
    max_attempts=3,
)

def process_trace(trace, goal):
    """Process trace with quality assurance."""
    # Evaluate
    evaluation = critic.evaluate(trace, goal=goal)

    if evaluation.is_acceptable:
        return trace, "accepted"

    if evaluation.needs_repair:
        result = repair_agent.repair(trace, evaluation, goal)
        if result.success:
            return result.repaired_trace, "repaired"
        return None, "repair_failed"

    return None, "rejected"

# Use
fixed_trace, status = process_trace(trace, "Calculate BMI")
print(f"Status: {status}")
```

## See Also

- [Audit System](audit-system.md)
- [Model Selection](model-selection.md)
- [API Reference](api-reference.md)
