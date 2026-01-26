# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**ptool_framework** - "Python programs calling LLMs, not LLMs calling tools."

This research framework from William Cohen's group at CMU inverts the traditional LLM-agent paradigm: Python controls workflow while LLMs handle "thinking" parts. This enables predictable, testable, and optimizable LLM-powered applications.

## Environment Setup

```bash
# Activate the conda environment
conda activate ptool_framework

# API keys are stored in .env file - load them
source .env
# Or export manually:
# export TOGETHER_API_KEY="..."

# Use DeepSeek V3 from Together.ai for testing
```

## Build & Test Commands

```bash
# Run all tests
pytest tests/ -v

# Run unit tests (no API calls - mocked)
pytest tests/test_ptool.py tests/test_react.py -v

# Run integration tests (requires TOGETHER_API_KEY)
pytest tests/test_react_integration.py -v

# Run single test
pytest tests/test_ptool.py::test_ptool_decorator -v

# Run tests with coverage
pytest tests/ --cov=ptool_framework --cov-report=html
```

## CLI Commands

```bash
ptool generate "task description" -o output.py  # Generate program from description
ptool run program.py --input "data"             # Run with trace collection
ptool refactor file.py --mode distill -o out.py # Convert @ptool to @distilled
ptool refactor file.py --mode expand -o out.py  # Convert Python to @ptool
ptool analyze                                   # Analyze distillation opportunities
ptool traces --ptool name --limit 20            # View execution traces
ptool models                                    # List available LLM models
ptool dashboard --port 8080                     # Launch web dashboard
```

## Architecture

### Three Modes of Use

1. **Mode 1 (@ptool)**: Add LLM functions to existing Python code
   ```python
   @ptool(model="deepseek-v3-0324")
   def classify(text: str) -> Literal["pos", "neg"]:
       """Classify sentiment."""
       ...  # LLM executes this, not Python
   ```

2. **Mode 2 (Program Generator)**: Generate complete programs from task descriptions
   ```python
   result = generate_program("Analyze reviews", output_path="analyzer.py")
   ```

3. **Mode 3 (ReAct Agent)**: Goal-driven reasoning with dynamic ptool execution
   ```python
   agent = ReActAgent(available_ptools=[...], echo=True)
   result = agent.run("Calculate (2+3)*4")
   ```

### The Distillation Pipeline

The framework's key workflow: Start with flexible LLM agents (Mode 3), progressively optimize to deterministic code (Mode 1).

```
@ptool (100% LLM) → collect traces → @distilled (Python + LLM fallback) → Pure Python
```

### Layered Architecture

- **Execution Layer**: `ptool.py` (@ptool decorator), `llm_backend.py` (multi-provider LLM), `executor.py` (trace execution)
- **Observability Layer**: `trace_store.py` (persistent storage), `dashboard/` (web UI)
- **Optimization Layer**: `distiller.py` (trace→Python), `refactorer.py` (AST transformation)
- **Agent Layer**: `react.py` (ReAct reasoning), `orchestrator.py` (multi-agent), `self_improving.py` (learning agents)
- **Validation Layer**: `audit/` (structured audits with step tracking), `critic.py` (output validation), `repair.py` (error fixing with Python calculator fallback)

### Key Data Structures

- `PToolSpec`: Ptool metadata (name, parameters, return_type, docstring, model)
- `WorkflowTrace`: Sequence of `TraceStep` objects with goal
- `ReActTrajectory`: Full reasoning history (Thought → Action → Observation)
- `ExecutionTrace`: Single ptool execution record for storage

## Key Files

| File | Purpose |
|------|---------|
| `ptool.py` | @ptool decorator, PToolSpec, PToolRegistry |
| `llm_backend.py` | Multi-provider LLM calls (Together, OpenAI, Anthropic, Groq) |
| `react.py` | ReActAgent, Thought, Action, Observation, ReActTrajectory |
| `distilled.py` | @distilled decorator, DistillationFallback |
| `traces.py` | TraceStep, WorkflowTrace |
| `critic.py` | TraceCritic, CriticVerdict, CriticEvaluation |
| `repair.py` | RepairAgent, RepairAction, Python calculator fallback |
| `audit/domains/medcalc.py` | MedCalc audits with step location tracking |
| `LLMS.json` | LLM model configuration (providers, costs, capabilities) |
| `.env` | API keys (TOGETHER_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY) |

## Configuration

Models configured in `LLMS.json`. API keys stored in `.env` file:
- `TOGETHER_API_KEY` - Primary provider (DeepSeek V3 for testing)
- `OPENAI_API_KEY` - GPT models
- `ANTHROPIC_API_KEY` - Claude models

## Code Patterns

### Adding a ptool
```python
from ptool_framework import ptool

@ptool(model="deepseek-v3-0324", output_mode="structured")
def my_func(text: str) -> dict:
    """Return {"key": value} from the text."""
    ...
```

### Python with LLM fallback
```python
from ptool_framework import distilled, DistillationFallback

@distilled(fallback_ptool="my_func")
def my_func_fast(text: str) -> dict:
    if "keyword" in text:
        return {"key": "pattern_match"}
    raise DistillationFallback("No pattern matched")
```

### Using the registry
```python
from ptool_framework import get_registry
registry = get_registry()
spec = registry.get("my_ptool")
all_ptools = registry.list_ptools()
```

## Documentation

- `docs/architecture.md` - Technical design and data flow
- `docs/CONTEXT.md` - Full context for AI assistants
- `docs/mode-1-ptool-decorator.md` - @ptool usage guide
- `docs/mode-2-program-generator.md` - Program generation
- `docs/mode-3-react-agent.md` - ReAct agent usage
- `docs/audit-system.md` - Structured audits with step location tracking
- `docs/critic-repair.md` - Critic evaluation and repair with Python calculator fallback
- `docs/medcalc-domain.md` - MedCalc domain audits example
- `docs/TODO.md` - Roadmap and research tasks
