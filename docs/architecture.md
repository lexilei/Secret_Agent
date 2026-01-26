# ptool_framework Architecture

Technical design and implementation details.

---

## Core Philosophy

### The Paradigm Shift

| Traditional Agents | ptool_framework |
|-------------------|-----------------|
| LLM decides what tool to call | Python code decides what ptool to call |
| Control flow is implicit in LLM reasoning | Control flow is explicit Python code |
| Hard to test and predict | Easy to test, predictable behavior |
| "LLM calling tools" | "Python calling LLMs" |

### Key Insight

> "ptools are glorified prompt templates with type signatures"

A **ptool** (pseudo-tool) is:
1. A Python function stub with typed parameters and return type
2. A docstring that serves as the prompt template
3. Optional in-context learning examples
4. Executed by calling an LLM, NOT by running Python code

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         THREE MODES OF USE                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Mode 1: @ptool        Mode 2: Generator      Mode 3: ReAct         │
│  ┌────────────────┐    ┌────────────────┐    ┌────────────────┐     │
│  │ Direct use in  │    │ Task → Program │    │ Goal → Agent   │     │
│  │ existing code  │    │ generation     │    │ reasoning loop │     │
│  └───────┬────────┘    └───────┬────────┘    └───────┬────────┘     │
│          │                     │                     │               │
│          └─────────────────────┴─────────────────────┘               │
│                                │                                     │
└────────────────────────────────┼─────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          EXECUTION LAYER                             │
├─────────────────────────────────────────────────────────────────────┤
│  @ptool (ptool.py)               │  @distilled (distilled.py)       │
│  • Decorator for LLM functions   │  • Pure Python with LLM fallback │
│  • Typed parameters & returns    │  • Raises DistillationFallback   │
│  • Registered in PToolRegistry   │  • Tracks success/fallback stats │
│                                  │                                   │
│  TraceExecutor (executor.py)     │  llm_backend.py                  │
│  • Executes WorkflowTraces       │  • Multi-provider LLM calls      │
│  • Validation & retry logic      │  • Together, OpenAI, Anthropic   │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        OBSERVABILITY LAYER                           │
├─────────────────────────────────────────────────────────────────────┤
│  TraceStore (trace_store.py)     │  ReActStore (react.py)           │
│  • Persistent JSON storage       │  • Trajectory storage            │
│  • ExecutionTrace records        │  • For distillation              │
│  • Query by ptool, time, status  │  • Success/failure tracking      │
│                                  │                                   │
│  Dashboard (dashboard/)          │  PTP Traces                      │
│  • Real-time execution view      │  • Human-readable format         │
│  • WebSocket updates             │  • ICL demonstrations            │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        OPTIMIZATION LAYER                            │
├─────────────────────────────────────────────────────────────────────┤
│  BehaviorDistiller (distiller.py)    │  CodeRefactorer (refactorer.py)  │
│  • Analyzes traces for patterns      │  • AST-based code transformation │
│  • Generates Python from patterns    │  • Distill: @ptool → @distilled  │
│  • Validates against held-out data   │  • Expand: Python → @ptool       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### @ptool Decorator (ptool.py)

The core abstraction:

```python
@ptool(model="deepseek-v3", output_mode="structured")
def classify_sentiment(text: str) -> Literal["positive", "negative", "neutral"]:
    """Classify the emotional tone of the given text."""
    ...  # Body ignored - LLM executes
```

**Key Classes:**
- `PToolSpec`: Dataclass storing all ptool metadata (name, params, return type, docstring)
- `PToolRegistry`: Global singleton registry of all defined ptools
- `PToolWrapper`: Callable wrapper that intercepts calls and routes to LLM

**Key Methods:**
- `spec.format_prompt(**kwargs)`: Generate prompt from inputs
- `spec.get_signature_str()`: Human-readable signature
- `get_registry()`: Get global registry

### LLM Backend (llm_backend.py)

Multi-provider LLM interface:

```python
config = LLMConfig.load("LLMS.json")
response = call_llm(prompt, model="deepseek-v3")
result = execute_ptool(spec, inputs)
```

**Provider Support:**
- Anthropic (Claude)
- OpenAI (GPT)
- Together.ai (DeepSeek, Llama, Mixtral)
- Groq (fast inference)
- Local (Ollama, vLLM)

### Workflow Traces (traces.py)

Data structures for execution sequences:

```python
@dataclass
class TraceStep:
    ptool_name: str              # Which ptool
    args: Dict[str, Any]         # Arguments
    expected_type: Type          # Expected return type
    goal: Optional[str]          # Why this step
    status: StepStatus           # PENDING, COMPLETED, FAILED
    result: Optional[Any]        # Actual result

@dataclass
class WorkflowTrace:
    goal: str                    # Overall goal
    steps: List[TraceStep]       # Sequence of steps
    trace_id: str                # Unique identifier
```

### ReAct Agent (react.py)

Reasoning + Acting loop:

```python
@dataclass
class Thought:
    content: str                 # Reasoning text
    step_number: int

@dataclass
class Action:
    ptool_name: str              # Which ptool to call
    args: Dict[str, Any]         # With what arguments

@dataclass
class Observation:
    result: Any                  # What the ptool returned
    success: bool

@dataclass
class ReActStep:
    thought: Thought
    action: Optional[Action]
    observation: Optional[Observation]

@dataclass
class ReActTrajectory:
    goal: str
    steps: List[ReActStep]
    final_answer: Optional[str]
    success: bool
```

### @distilled Decorator (distilled.py)

Python with LLM fallback:

```python
@distilled(fallback_ptool="extract_items")
def extract_items_fast(text: str) -> List[str]:
    """Python first, LLM fallback."""
    pattern = r'\b(pizza|salad|burger)\b'
    matches = re.findall(pattern, text.lower())
    if not matches:
        raise DistillationFallback("No patterns matched")
    return matches
```

**How it works:**
1. Call the decorated function
2. If it returns normally → use Python result
3. If it raises `DistillationFallback` → call the fallback ptool
4. Log the fallback for analysis

---

## File Structure

```
AgentProject/
├── README.md                    # Getting started
├── LLMS.json                    # LLM provider configuration
├── CONTEXT.md                   # Full context for AI assistants
│
├── docs/
│   ├── mode-1-ptool-decorator.md
│   ├── mode-2-program-generator.md
│   ├── mode-3-react-agent.md
│   └── architecture.md          # This file
│
├── ptool_framework/
│   ├── __init__.py              # Package exports
│   ├── ptool.py                 # @ptool decorator
│   ├── llm_backend.py           # LLM API calls
│   ├── traces.py                # TraceStep, WorkflowTrace
│   ├── executor.py              # TraceExecutor
│   ├── trace_builder.py         # NL → trace
│   │
│   ├── react.py                 # ReAct agent
│   ├── distilled.py             # @distilled decorator
│   ├── trace_store.py           # Persistent storage
│   ├── distiller.py             # BehaviorDistiller
│   ├── refactorer.py            # CodeRefactorer
│   ├── program_generator.py     # ProgramGenerator
│   │
│   ├── cli.py                   # CLI commands
│   ├── prompts/                 # Prompt templates
│   ├── examples/                # Example programs
│   └── dashboard/               # Web dashboard
│
└── tests/
    ├── test_ptool.py
    ├── test_react.py
    └── test_react_real.py
```

---

## LLM Configuration (LLMS.json)

```json
{
  "default_model": "deepseek-v3",
  "models": {
    "deepseek-v3": {
      "provider": "together",
      "model_id": "deepseek-ai/DeepSeek-V3",
      "cost": {"input": 0.0005, "output": 0.001}
    },
    "gpt-4o": {
      "provider": "openai",
      "model_id": "gpt-4o",
      "cost": {"input": 0.005, "output": 0.015}
    }
  },
  "providers": {
    "together": {
      "api_base": "https://api.together.xyz/v1",
      "env_key": "TOGETHER_API_KEY"
    }
  }
}
```

---

## Data Flow Examples

### Mode 1: Direct ptool Use

```
extract_sentiment("I love it!")
    │
    ▼
┌─────────────────────────────────────────┐
│ PToolWrapper.__call__()                 │
│   1. Format prompt from docstring + args│
│   2. Call LLM via llm_backend           │
│   3. Parse response to return type      │
│   4. Log to TraceStore (if enabled)     │
└─────────────────────────────────────────┘
    │
    ▼
{"sentiment": "positive", "confidence": 0.95}
```

### Mode 3: ReAct Reasoning

```
agent.run("Calculate (2+3)*4")
    │
    ▼
┌─────────────────────────────────────────┐
│ ReActAgent.run()                        │
│   Loop:                                 │
│     1. _generate_thought() [LLM]        │
│        → "I need to add 2 and 3"        │
│     2. _parse_action()                  │
│        → Action(add, {a:2, b:3})        │
│     3. _execute_action() [ptool LLM]    │
│        → Observation(result=5)          │
│     4. Check for <answer> tag           │
│        → Continue or return             │
└─────────────────────────────────────────┘
    │
    ▼
ReActResult(answer="20", trajectory=...)
```

### Distillation Pipeline

```
After 50+ ReAct trajectories:
    │
    ▼
┌─────────────────────────────────────────┐
│ BehaviorDistiller.distill()             │
│   1. Load traces from TraceStore        │
│   2. analyze_trace_patterns() [LLM]     │
│   3. generate_python_implementation()   │
│   4. Compile-test the code              │
│   5. Validate against held-out data     │
│   6. Wrap in @distilled decorator       │
└─────────────────────────────────────────┘
    │
    ▼
@distilled(fallback_ptool="original")
def optimized_function(...): ...
```

---

## Design Decisions

### Why "Python calling LLMs"?

**Traditional approach problems:**
- LLM decides what to call and when → unpredictable
- State management is complex → bugs
- Hard to test → unreliable

**Our approach benefits:**
- Python controls flow → predictable
- Explicit state in variables → debuggable
- Easy to mock LLM calls → testable

### Why Typed ptools?

```python
@ptool()
def extract_items(text: str) -> List[str]:  # Typed!
    ...
```

- **Validation**: Check LLM output matches expected type
- **Documentation**: Signature is self-documenting
- **Distillation**: Types guide Python generation
- **Testing**: Easy to create test cases

### Why @distilled with Fallback?

Instead of binary "LLM or Python":
```
@ptool (100% LLM)
    ↓ collect traces
@distilled (Python + fallback)
    ↓ improve Python
Pure Python (0% LLM)
```

Enables incremental optimization without breaking production.

### Why ReAct for Agentic Behavior?

- Contained agency: Reasoning is visible
- Trace generation: Every run produces a trace
- Distillation-ready: Traces can become workflows

---

## The Spectrum of Autonomy

```
L0: Fixed Python workflow calling LLMs          ← Mode 1
L1: LLMs as routers (modify workflow)           ← Mode 2
L2: State graphs (LangGraph-style)
L3: ReAct agents (thought-action loops)         ← Mode 3
L4: Multi-agent systems
L5: Self-modifying systems
```

ptool_framework operates at L0-L1 (Mode 1-2) with L3 (Mode 3) for exploration.
The key insight: Use L3 to discover workflows, then distill to L0 for production.

---

## Extension Points

### Adding a New LLM Provider

1. Add to `LLMS.json`:
```json
"providers": {
  "new_provider": {
    "api_base": "https://api.example.com/v1",
    "env_key": "NEW_PROVIDER_KEY"
  }
}
```

2. If non-OpenAI-compatible, add handler in `llm_backend.py`

### Adding a New CLI Command

```python
# In cli.py
@cli.command()
@click.argument("...")
def new_command(...):
    """Description."""
    ...
```

### Adding Dashboard Features

1. Add tab in `dashboard/static/index.html`
2. Add API endpoint in `dashboard/server.py`
3. Add WebSocket event type if needed

---

## Debugging Guide

### "ptool not being called"
```python
from ptool_framework import get_registry
print(get_registry().list_ptools())  # Is it registered?
```

### "Distillation failing"
```python
from ptool_framework import get_trace_store
store = get_trace_store()
print(store.get_trace_count("my_ptool"))  # Enough traces?
```

### "ReAct not finding answer"
```python
agent = ReActAgent(echo=True)  # See reasoning
result = agent.run("goal")
print(result.trajectory.to_ptp_trace())  # See what happened
```

---

## Glossary

| Term | Definition |
|------|------------|
| **ptool** | "Pseudo-tool" - function executed by LLM prompting |
| **distilled** | Pure Python function with LLM fallback |
| **trace** | Record of a ptool execution (inputs, outputs, timing) |
| **WorkflowTrace** | Sequence of ptool calls to achieve a goal |
| **ReActTrajectory** | Full reasoning history from ReAct agent |
| **PTP trace** | Program Trace Prompting format (human-readable) |
| **distillation** | Converting LLM calls to pure Python |
| **expansion** | Converting pure Python to LLM calls |
