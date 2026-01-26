# ptool_framework: Complete Context Document

**Last Updated**: 2024-12-24
**Version**: 0.3.0
**Author**: Claude (for future Claude sessions)

---

## Executive Summary

`ptool_framework` is a Python framework that inverts the traditional agent paradigm: instead of "LLMs calling tools," we have "Python programs calling LLMs." This enables:

1. **Predictable control flow** - Python handles loops, conditions, orchestration
2. **Testable components** - Each ptool has typed inputs/outputs
3. **Gradual optimization** - LLM calls can be "distilled" to pure Python over time
4. **Full observability** - Every LLM call is traced and stored

**Core Philosophy** (from William Cohen's research):
> "Python does maximum heavy lifting before handing to LLMs"

---

## Three Modes of Use

### Mode 1: @ptool Decorator
Add LLM-powered functions to existing Python code:
```python
@ptool(model="deepseek-v3")
def summarize(text: str) -> str:
    """Summarize the text."""
    ...  # LLM executes this
```

### Mode 2: Program Generator
Generate complete programs from task descriptions:
```python
result = generate_program("Analyze customer reviews", output_path="analyzer.py")
```

### Mode 3: ReAct Agent
Let an agent reason about and execute ptools dynamically:
```python
result = react("Calculate (2 + 3) * 4 using add and multiply tools")
# Agent: Think → Act → Observe → Think → ... → Answer
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         THREE MODES OF USE                           │
├─────────────────────────────────────────────────────────────────────┤
│  Mode 1: @ptool        Mode 2: Generator      Mode 3: ReAct         │
│  Direct use in code    Task → Program         Goal → Agent loop     │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          EXECUTION LAYER                             │
├─────────────────────────────────────────────────────────────────────┤
│  @ptool (ptool.py)               │  @distilled (distilled.py)       │
│  • LLM-executed functions        │  • Python with LLM fallback      │
│  • Typed I/O                     │  • Fallback tracking             │
│                                  │                                   │
│  TraceExecutor (executor.py)     │  llm_backend.py                  │
│  • Executes WorkflowTraces       │  • Multi-provider LLM calls      │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        OBSERVABILITY LAYER                           │
├─────────────────────────────────────────────────────────────────────┤
│  TraceStore              │  ReActStore              │  Dashboard     │
│  • Execution traces      │  • ReAct trajectories    │  • Real-time   │
│  • For distillation      │  • PTP traces            │  • WebSocket   │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        OPTIMIZATION LAYER                            │
├─────────────────────────────────────────────────────────────────────┤
│  BehaviorDistiller               │  CodeRefactorer                  │
│  • Trace → Python                │  • @ptool ↔ @distilled           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
AgentProject/
├── README.md                    # Getting started guide
├── CONTEXT.md                   # This file (AI assistant context)
├── LLMS.json                    # LLM provider configuration
├── environment.yml              # Conda environment
│
├── docs/
│   ├── mode-1-ptool-decorator.md   # @ptool usage guide
│   ├── mode-2-program-generator.md # ProgramGenerator guide
│   ├── mode-3-react-agent.md       # ReAct agent guide
│   └── architecture.md             # Technical architecture
│
├── ptool_framework/
│   ├── __init__.py              # Package exports (v0.3.0)
│   ├── ptool.py                 # @ptool decorator, PToolSpec, registry
│   ├── llm_backend.py           # LLM API calls, multi-provider
│   ├── traces.py                # TraceStep, WorkflowTrace
│   ├── executor.py              # TraceExecutor
│   ├── trace_builder.py         # NL → trace generation
│   │
│   ├── react.py                 # ReAct agent (NEW)
│   ├── distilled.py             # @distilled decorator
│   ├── trace_store.py           # Persistent trace storage
│   ├── distiller.py             # BehaviorDistiller
│   ├── refactorer.py            # CodeRefactorer
│   ├── program_generator.py     # ProgramGenerator
│   ├── cli.py                   # CLI commands
│   │
│   ├── prompts/                 # Prompt templates
│   │   └── ptp_prompt.txt       # PTP format template
│   ├── examples/                # Example programs
│   │   └── sports_understanding.py
│   └── dashboard/               # Web dashboard
│       ├── events.py
│       ├── server.py
│       └── static/
│
├── tests/
│   ├── __init__.py
│   ├── test_ptool.py            # @ptool unit tests
│   ├── test_react.py            # ReAct unit tests (mock)
│   ├── test_react_integration.py # ReAct integration tests (real LLM)
│   └── test_deepseek.py         # DeepSeek V3 tests
│
└── examples/
    └── medcalc_agent.py         # Medical calculation example
```

---

## Key Components

### @ptool Decorator (ptool.py)

```python
@ptool(model="deepseek-v3", output_mode="structured")
def classify(text: str) -> Literal["A", "B", "C"]:
    """Classify text into categories."""
    ...  # Body ignored - LLM executes
```

**Classes:**
- `PToolSpec`: Stores function metadata (name, params, return type, docstring)
- `PToolRegistry`: Global singleton registry
- `get_registry()`: Access the registry

### ReAct Agent (react.py)

**Data Structures:**
```python
@dataclass
class Thought:
    content: str
    step_number: int

@dataclass
class Action:
    ptool_name: str
    args: Dict[str, Any]

@dataclass
class Observation:
    result: Any
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

**Usage:**
```python
agent = ReActAgent(
    available_ptools=[...],
    model="deepseek-v3",
    max_steps=10,
    echo=True,
)
result = agent.run("Your goal here")
print(result.answer)
print(result.trajectory.to_ptp_trace())  # Human-readable trace
```

### @distilled Decorator (distilled.py)

Python implementation with LLM fallback:
```python
@distilled(fallback_ptool="extract_items")
def extract_items_fast(text: str) -> List[str]:
    if "," in text:
        return text.split(",")
    raise DistillationFallback("No pattern matched")
```

### LLM Backend (llm_backend.py)

Multi-provider support via LLMS.json:
- Together.ai (DeepSeek, Llama, Mixtral)
- OpenAI (GPT models)
- Anthropic (Claude)
- Local (Ollama, vLLM)

```python
response = call_llm(prompt, model="deepseek-v3")
result = execute_ptool(spec, inputs)
```

---

## Configuration

### LLMS.json

```json
{
  "default_model": "deepseek-v3",
  "models": {
    "deepseek-v3": {
      "provider": "together",
      "model_id": "deepseek-ai/DeepSeek-V3",
      "api_key_env": "TOGETHER_API_KEY"
    }
  }
}
```

### Environment Variables

```bash
export TOGETHER_API_KEY="..."
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
```

---

## Program Trace Prompting (PTP)

Human-readable trace format from William's research:

```
Calling add(a=2, b=3)...
...add returned 5

Calling multiply(a=5, b=4)...
...multiply returned 20

Final answer: 20
```

Generated via:
```python
trajectory = result.trajectory
print(trajectory.to_ptp_trace())
```

---

## ReAct Execution Flow

```
agent.run("Calculate (2+3)*4")
    │
    ▼
┌─────────────────────────────────────────┐
│ ReAct Loop                              │
│   1. Generate thought (LLM)             │
│      → "I need to add 2 and 3"          │
│   2. Parse action                       │
│      → add(a=2, b=3)                    │
│   3. Execute ptool (separate LLM call)  │
│      → 5                                │
│   4. Observe result                     │
│   5. Check for <answer> tag             │
│   6. Repeat or return                   │
└─────────────────────────────────────────┘
    │
    ▼
ReActResult(
    success=True,
    answer="20",
    trajectory=ReActTrajectory(...),
    trace=WorkflowTrace(...)  # For TraceExecutor
)
```

---

## Testing

```bash
# Unit tests (mock LLM)
python -m pytest tests/test_react.py -v

# Integration tests (real LLM - needs API key)
export TOGETHER_API_KEY="..."
python tests/test_react_integration.py

# All tests
python -m pytest tests/ -v
```

---

## CLI Commands

```bash
# Generate program
ptool generate "Task description" -o output.py

# Run with tracing
ptool run program.py --input "data" --trace

# Analyze for distillation
ptool analyze program.py

# Distill to Python
ptool refactor program.py --mode distill -o program_v2.py

# View traces
ptool traces --ptool my_function

# Start dashboard
ptool dashboard
```

---

## Relationship to William's Research

| Concept | Implementation |
|---------|----------------|
| ptools | @ptool decorator |
| Program Trace Prompting | ReActTrajectory.to_ptp_trace() |
| Behavior Distillation | BehaviorDistiller, ReActStore |
| Trace-based learning | TraceStore, trajectory storage |

**Distillation Pipeline:**
1. Run ReAct on many goals → collect trajectories
2. Cluster similar trajectories
3. Generalize to Python workflows
4. Create new ptools from patterns

---

## Common Patterns

### Direct ptool composition
```python
def workflow(text):
    entities = extract_entities(text)
    sentiment = analyze_sentiment(text)
    return {"entities": entities, "sentiment": sentiment}
```

### ReAct for exploration
```python
result = react("Analyze this data and find patterns")
# Agent figures out which ptools to use
```

### Gradual optimization
```python
# Start with @ptool (100% LLM)
@ptool()
def classify(text): ...

# After traces, distill to @distilled (Python + fallback)
@distilled(fallback_ptool="classify")
def classify_fast(text):
    if "spam" in text.lower():
        return "spam"
    raise DistillationFallback()
```

---

## Debugging

### ptool not executing
```python
from ptool_framework import get_registry
print(get_registry().list_ptools())  # Is it registered?
```

### ReAct not finding answer
```python
agent = ReActAgent(echo=True)  # See reasoning
result = agent.run("goal")
print(result.trajectory.to_ptp_trace())
```

### Check traces
```python
from ptool_framework import get_trace_store
store = get_trace_store()
print(store.get_stats())
```

---

## Version History

- **0.3.0** (2024-12-24): Added ReAct agent, PTP traces, reorganized docs
- **0.2.0** (2024-12-17): Added distillation, refactoring, dashboard
- **0.1.0**: Initial @ptool, traces, executor

---

*This document should be read by AI assistants to understand the codebase before making modifications.*
