# Mode 3: ReAct Agent

**Let the agent reason about and execute ptools dynamically.**

The ReAct (Reasoning + Acting) agent iteratively thinks about what to do, executes ptools, and observes results until it achieves the goal.

---

## Quick Example

```python
from ptool_framework import ptool, ReActAgent, get_registry

# Define some ptools
@ptool(model="deepseek-v3")
def add(a: int, b: int) -> int:
    """Add two integers."""
    ...

@ptool(model="deepseek-v3")
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    ...

# Create agent with available ptools
agent = ReActAgent(
    available_ptools=[get_registry().get("add"), get_registry().get("multiply")],
    model="deepseek-v3",
    max_steps=10,
    echo=True  # Print reasoning
)

# Give it a goal
result = agent.run("Calculate (2 + 3) * 4")
print(result.answer)  # 20
```

**Output:**

```
============================================================
ReAct Agent: Calculate (2 + 3) * 4
============================================================

--- Step 1 ---
Thought: First, I need to add 2 and 3.
Action: add({'a': 2, 'b': 3})
Observation: 5

--- Step 2 ---
Thought: Now I multiply the result by 4.
Action: multiply({'a': 5, 'b': 4})
Observation: 20

--- Step 3 ---
Thought: The calculation is complete. The answer is 20.

Final Answer: 20
```

---

## How It Works

```
┌────────────────────────────────────────────────────────────────┐
│                         ReAct Loop                              │
│                                                                 │
│   Goal: "Calculate (2 + 3) * 4"                                 │
│          │                                                      │
│          ▼                                                      │
│   ┌─────────────────┐                                          │
│   │ Generate Thought │ ← LLM reasons about what to do          │
│   │ "I need to add   │                                          │
│   │  2 and 3 first"  │                                          │
│   └────────┬────────┘                                          │
│            │                                                    │
│            ▼                                                    │
│   ┌─────────────────┐                                          │
│   │ Parse Action    │                                          │
│   │ add(a=2, b=3)   │                                          │
│   └────────┬────────┘                                          │
│            │                                                    │
│            ▼                                                    │
│   ┌─────────────────┐                                          │
│   │ Execute Ptool   │ ← Separate LLM call for the ptool        │
│   │ → 5             │                                          │
│   └────────┬────────┘                                          │
│            │                                                    │
│            ▼                                                    │
│   ┌─────────────────┐                                          │
│   │ Observation: 5  │                                          │
│   └────────┬────────┘                                          │
│            │                                                    │
│            ▼                                                    │
│   ┌─────────────────┐                                          │
│   │ Check: Done?    │ ──No──► Back to Generate Thought         │
│   └────────┬────────┘                                          │
│            │ Yes                                                │
│            ▼                                                    │
│   Return Answer: 20                                             │
└────────────────────────────────────────────────────────────────┘
```

### Two Levels of LLM Calls

```
ReActAgent (reasoning LLM)
    │
    ├── "I should call add(2, 3)"
    │       │
    │       └── add() ptool (execution LLM) → 5
    │
    ├── "Now I should call multiply(5, 4)"
    │       │
    │       └── multiply() ptool (execution LLM) → 20
    │
    └── "The answer is 20"
```

- **ReAct LLM**: Decides *which* ptool to call and *why*
- **Ptool LLM**: Actually *executes* the function

---

## API Reference

### ReActAgent

```python
from ptool_framework import ReActAgent

agent = ReActAgent(
    available_ptools=None,       # List of PToolSpec, or None for all registered
    model="deepseek-v3",         # LLM for reasoning
    max_steps=10,                # Maximum iterations
    echo=False,                  # Print reasoning to console
    store_trajectories=True,     # Save for later distillation
    execution_mode="immediate",  # "immediate" or "deferred"
)

result = agent.run("Your goal here")
```

### ReActResult

```python
result = agent.run("Goal")

result.success      # bool: Did it complete successfully?
result.answer       # str: The final answer
result.trajectory   # ReActTrajectory: Full reasoning history
result.trace        # WorkflowTrace: For TraceExecutor (if successful)
```

### ReActTrajectory

```python
trajectory = result.trajectory

trajectory.trajectory_id     # Unique ID
trajectory.goal              # The original goal
trajectory.steps             # List[ReActStep]: All steps taken
trajectory.final_answer      # The answer (if found)
trajectory.success           # Did it succeed?
trajectory.termination_reason  # "answer_found", "max_steps", "error"
trajectory.total_time_ms     # Total execution time

# Convert to different formats
trajectory.to_ptp_trace()    # Program Trace Prompting format (text)
trajectory.to_dict()         # Dictionary for JSON serialization
```

---

## Convenience Functions

### react()

Quick one-liner:

```python
from ptool_framework import react

result = react("Calculate BMI for 70kg, 1.75m tall")
print(result.answer)
```

### react_and_execute()

Run ReAct and get an ExecutionResult:

```python
from ptool_framework import react_and_execute

result = react_and_execute(
    goal="Analyze this text for sentiment",
    model="deepseek-v3"
)

print(result.success)
print(result.final_result)
```

---

## Trace Formats

### 1. ReActTrajectory (Structured)

Full history of the reasoning process:

```python
trajectory = result.trajectory

for step in trajectory.steps:
    print(f"Thought: {step.thought.content}")
    if step.action:
        print(f"Action: {step.action.ptool_name}({step.action.args})")
    if step.observation:
        print(f"Observation: {step.observation.result}")
```

### 2. WorkflowTrace (For TraceExecutor)

The sequence of ptool calls, without the reasoning:

```python
trace = result.trace  # WorkflowTrace

for step in trace.steps:
    print(f"{step.ptool_name}({step.args}) -> {step.result}")
```

### 3. PTP Trace (Human-Readable)

Program Trace Prompting format:

```python
ptp = result.trajectory.to_ptp_trace()
print(ptp)
```

Output:
```
Calling add(a=2, b=3)...
...add returned 5

Calling multiply(a=5, b=4)...
...multiply returned 20

Final answer: 20
```

---

## Storing Trajectories for Distillation

ReAct trajectories are stored for later analysis and distillation:

```python
from ptool_framework import get_react_store

store = get_react_store()

# Get successful trajectories
trajectories = store.get_successful_trajectories(limit=100)

# Find distillation candidates (goals that work reliably)
candidates = store.get_distillation_candidates(
    min_trajectories=10,
    min_success_rate=0.9
)

for candidate in candidates:
    print(f"Goal pattern: {candidate['goal_pattern']}")
    print(f"Success rate: {candidate['success_rate']:.0%}")
    print(f"Avg steps: {candidate['avg_steps']}")
```

---

## Example: Sports Understanding

From William Cohen's demo:

```python
from ptool_framework import ptool, ReActAgent, get_registry

@ptool(model="deepseek-v3")
def analyze_sentence(sentence: str) -> tuple:
    """Extract player, action, and event from a sports sentence.

    Examples:
    >>> analyze_sentence("Bam Adebayo scored a reverse layup.")
    ('Bam Adebayo', 'scored a reverse layup', '')
    """
    ...

@ptool(model="deepseek-v3")
def sport_for(x: str) -> str:
    """Return the sport associated with a player, action, or event.

    Examples:
    >>> sport_for('Bam Adebayo')
    'basketball'
    >>> sport_for('scored a touchdown')
    'American football'
    """
    ...

@ptool(model="deepseek-v3")
def consistent_sports(sport1: str, sport2: str) -> bool:
    """Check if two sport descriptions are consistent."""
    ...

# Create agent
agent = ReActAgent(
    available_ptools=[
        get_registry().get("analyze_sentence"),
        get_registry().get("sport_for"),
        get_registry().get("consistent_sports"),
    ],
    echo=True
)

# Run
result = agent.run(
    "Does this sentence make sense? 'Santi Cazorla scored a touchdown.'"
)
print(f"Answer: {result.answer}")  # No (soccer player, football action)
```

---

## Execution Modes

### Immediate (Default)

Ptools are executed as soon as they're called:

```python
agent = ReActAgent(execution_mode="immediate")
result = agent.run("Goal")
# Ptools executed during run()
```

### Deferred

Ptools are planned but not executed - returns a WorkflowTrace:

```python
agent = ReActAgent(execution_mode="deferred")
result = agent.run("Goal")

# Get the planned trace
trace = result.trace

# Execute later with TraceExecutor
from ptool_framework import TraceExecutor
executor = TraceExecutor()
exec_result = executor.execute(trace)
```

---

## Error Handling

```python
from ptool_framework import ReActAgent, ReActError, ActionParseError

agent = ReActAgent(max_steps=5)

try:
    result = agent.run("Impossible goal")
except ReActError as e:
    print(f"ReAct failed: {e}")
except ActionParseError as e:
    print(f"Couldn't parse action: {e}")

# Or check result
result = agent.run("Goal")
if not result.success:
    print(f"Failed: {result.trajectory.termination_reason}")
```

---

## Configuration Options

```python
agent = ReActAgent(
    # Ptools
    available_ptools=None,       # None = all registered ptools

    # Model
    model="deepseek-v3",         # Reasoning model

    # Limits
    max_steps=10,                # Stop after N steps

    # Output
    echo=False,                  # Print to console

    # Storage
    store_trajectories=True,     # Save for distillation

    # Execution
    execution_mode="immediate",  # "immediate" or "deferred"
)
```

---

## When to Use Mode 3

**Good for:**
- Open-ended goals where the steps aren't known in advance
- Exploratory tasks
- When the workflow depends on intermediate results
- Research and experimentation

**Consider Mode 1 when:**
- You know the exact workflow
- Performance is critical (ReAct has LLM overhead)
- The task is well-defined

**Consider Mode 2 when:**
- You want a fixed program generated once
- The workflow is always the same for a task type

---

## The Distillation Pipeline

ReAct enables William Cohen's "Behavior Distillation Approach 2":

```
1. Run ReAct on many goals
   └── Collect trajectories

2. Cluster similar trajectories
   └── Find common patterns

3. Generalize to Python workflows
   └── mode-1-ptool-decorator.md patterns

4. Create new ptools from patterns
   └── Recursive improvement
```

---

## Tips

1. **Define focused ptools**: ReAct works better with specific, well-documented ptools
2. **Use echo=True for debugging**: See the agent's reasoning
3. **Set reasonable max_steps**: Prevents infinite loops
4. **Check termination_reason**: Understand why the agent stopped
5. **Store trajectories**: Enable distillation later

---

## Next Steps

- **Use ptools directly**: See [Mode 1: @ptool Decorator](mode-1-ptool-decorator.md)
- **Generate programs**: See [Mode 2: Program Generator](mode-2-program-generator.md)
- **Technical details**: See [Architecture](architecture.md)
