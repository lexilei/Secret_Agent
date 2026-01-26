# ptool_framework

**Python programs calling LLMs, not LLMs calling tools.**

A framework for building predictable, testable, and optimizable LLM-powered applications.

---

## The Idea

Traditional agent frameworks have LLMs decide what tools to call. This is unpredictable and hard to test.

**ptool_framework inverts this**: Python controls the workflow, LLMs handle the "thinking" parts.

```python
from ptool_framework import ptool

@ptool()
def summarize(text: str) -> str:
    """Summarize the text in one sentence."""
    ...  # LLM executes this

# Python controls the flow
for doc in documents:
    summary = summarize(doc)  # LLM does the thinking
    save_to_db(summary)       # Python handles the rest
```

---

## Three Ways to Use It

### Mode 1: Add LLM to Existing Code

Just decorate functions with `@ptool`:

```python
from ptool_framework import ptool

@ptool(model="deepseek-v3")
def extract_entities(text: str) -> list[str]:
    """Extract named entities from text."""
    ...

entities = extract_entities("Apple CEO Tim Cook announced...")
# Returns: ["Apple", "Tim Cook"]
```

### Mode 2: Generate Programs from Descriptions

Describe what you want, get a complete program:

```python
from ptool_framework import generate_program

result = generate_program(
    task_description="Analyze customer reviews for sentiment",
    output_path="analyzer.py"
)
```

### Mode 3: Let an Agent Figure It Out

Give a goal, let the agent reason step by step:

```python
from ptool_framework import react

result = react("Calculate (2 + 3) * 4 using the add and multiply tools")
print(result.answer)  # 20
```

---

## Quick Start

### 1. Setup

```bash
# Clone and setup
git clone <repo>
cd AgentProject

# Create environment
conda env create -f environment.yml
conda activate ptool_env

# Set API key
export TOGETHER_API_KEY="your-key-here"
```

### 2. Try It

```python
from ptool_framework import ptool

@ptool()
def analyze_sentiment(text: str) -> dict:
    """Return {"sentiment": "positive/negative/neutral", "confidence": 0-1}"""
    ...

result = analyze_sentiment("I love this product!")
print(result)  # {"sentiment": "positive", "confidence": 0.95}
```

### 3. Collect Traces

```python
from ptool_framework import enable_tracing

enable_tracing(True)  # Now all calls are logged

# Run your code normally
for text in my_texts:
    analyze_sentiment(text)
```

### 4. Optimize with Python

After collecting traces, add fast paths:

```python
from ptool_framework import distilled, DistillationFallback

@distilled(fallback_ptool="analyze_sentiment")
def analyze_sentiment_fast(text: str) -> dict:
    """Python first, LLM fallback."""
    if any(w in text.lower() for w in ["love", "great", "awesome"]):
        return {"sentiment": "positive", "confidence": 0.8}
    raise DistillationFallback("No pattern matched")
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [Mode 1: @ptool Decorator](docs/mode-1-ptool-decorator.md) | Add LLM functions to existing code |
| [Mode 2: Program Generator](docs/mode-2-program-generator.md) | Generate programs from descriptions |
| [Mode 3: ReAct Agent](docs/mode-3-react-agent.md) | Goal-driven agent reasoning |
| [Architecture](docs/architecture.md) | Technical design and internals |
| [Context](docs/CONTEXT.md) | Full context for AI assistants |
| [Research Tasks](docs/RESEARCH_TASKS.md) | Research planning and tasks |
| [TODO](docs/TODO.md) | Roadmap to complete the vision |

---

## CLI Commands

```bash
# Generate a program
ptool generate "Analyze customer reviews" -o analyzer.py

# Run with tracing
ptool run analyzer.py --input "Great product!" --trace

# View traces
ptool traces --ptool analyze_sentiment

# Distill to Python
ptool refactor analyzer.py --mode distill -o analyzer_v2.py

# Start dashboard
ptool dashboard
```

---

## Key Features

- **Predictable**: Python controls flow, LLMs just think
- **Testable**: Mock LLM calls, test your workflows
- **Typed**: Full type annotations, validated outputs
- **Traceable**: Every LLM call is logged
- **Optimizable**: Distill LLM calls to Python over time

---

## Configuration

Models are configured in `LLMS.json`:

```json
{
  "default_model": "deepseek-v3",
  "models": {
    "deepseek-v3": {
      "provider": "together",
      "model_id": "deepseek-ai/DeepSeek-V3"
    },
    "gpt-4o": {
      "provider": "openai",
      "model_id": "gpt-4o"
    }
  }
}
```

Set API keys via environment variables:
```bash
export TOGETHER_API_KEY="..."
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
```

---

## Examples

See `ptool_framework/examples/` for complete examples:

- `sports_understanding.py` - Analyze sports sentences for consistency
- `medcalc_agent.py` - Medical calculation assistant

---

## Troubleshooting

**"API key not found"**
```bash
export TOGETHER_API_KEY="your-key"
```

**"Module not found"**
```bash
cd /path/to/AgentProject
python -c "from ptool_framework import ptool"
```

**"LLM returns wrong type"**
- Make return type more specific
- Add examples in docstring
- Try a more capable model

---

## Research Context

This framework implements ideas from William Cohen's agent research:

- **ptools**: Prompt templates with type signatures
- **Program Trace Prompting**: Observable execution traces
- **Behavior Distillation**: Converting LLM behavior to Python

The goal: Start with flexible LLM agents, progressively optimize to deterministic code.

---

## License

MIT
