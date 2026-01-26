# Mode 1: The @ptool Decorator

**Use ptools directly in your existing Python code.**

This is the simplest way to use the ptool_framework. You define functions that are executed by LLMs instead of Python code.

---

## Quick Example

```python
from ptool_framework import ptool

@ptool(model="deepseek-v3")
def summarize(text: str, max_sentences: int = 3) -> str:
    """Summarize the given text into the specified number of sentences.

    Keep the main ideas and key points.
    """
    ...  # Body is ignored - LLM executes this

# Use it like any normal function
result = summarize("Long article text here...", max_sentences=2)
print(result)  # "Summary of the article..."
```

---

## How It Works

When you call a `@ptool` function:

```
┌────────────────────────────────────────────────────────────┐
│  summarize("Long article...")                              │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────────────────────────────────┐                   │
│  │ 1. Build prompt from:               │                   │
│  │    - Function signature             │                   │
│  │    - Type annotations               │                   │
│  │    - Docstring                      │                   │
│  │    - Provided arguments             │                   │
│  └─────────────────────────────────────┘                   │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────────────────────────────────┐                   │
│  │ 2. Call LLM (deepseek-v3)           │                   │
│  └─────────────────────────────────────┘                   │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────────────────────────────────┐                   │
│  │ 3. Parse response to match          │                   │
│  │    return type (str)                │                   │
│  └─────────────────────────────────────┘                   │
│       │                                                     │
│       ▼                                                     │
│  Return: "Summary of the article..."                        │
└────────────────────────────────────────────────────────────┘
```

---

## Key Principles

### 1. The Docstring IS the Prompt

```python
@ptool()
def classify_sentiment(text: str) -> Literal["positive", "negative", "neutral"]:
    """Classify the emotional tone of the given text.

    Consider:
    - Word choice and language
    - Punctuation and emphasis
    - Overall context and tone

    Return exactly one of: positive, negative, or neutral.
    """
    ...
```

The more detailed your docstring, the better the LLM output.

### 2. Type Everything

```python
# Good - specific types
@ptool()
def extract_entities(text: str) -> List[Dict[str, str]]:
    """Extract named entities.
    Return [{"name": "...", "type": "person/org/location"}]
    """
    ...

# Bad - vague types
@ptool()
def extract_entities(text) -> list:  # No type info!
    """Extract entities."""
    ...
```

### 3. Use `...` as the Body

The function body is completely ignored. Use `...` (ellipsis) to make this clear.

```python
@ptool()
def my_function(x: str) -> str:
    """Description."""
    ...  # <-- Always use this
```

---

## Decorator Options

```python
@ptool(
    model="deepseek-v3",        # Which LLM to use (see LLMS.json)
    output_mode="structured",    # "structured" (JSON) or "freeform" (text)
)
```

### Output Modes

**Structured (default)** - Returns typed Python objects:

```python
@ptool(output_mode="structured")
def extract_data(text: str) -> Dict[str, Any]:
    """Extract structured data."""
    ...

result = extract_data("John is 25 years old")
# Returns: {"name": "John", "age": 25}
```

**Freeform** - Returns raw string:

```python
@ptool(output_mode="freeform")
def write_poem(topic: str) -> str:
    """Write a creative poem about the topic."""
    ...

result = write_poem("sunset")
# Returns: "Golden rays descend..."
```

---

## Choosing a Model

Configure models in `LLMS.json`:

```python
# Use default model (from LLMS.json)
@ptool()
def simple_task(x: str) -> str: ...

# Use a specific model
@ptool(model="gpt-4o")
def complex_reasoning(problem: str) -> Dict: ...

# Use a fast/cheap model
@ptool(model="gpt-4o-mini")
def simple_classification(text: str) -> Literal["A", "B", "C"]: ...

# Use local model
@ptool(model="local-llama")
def private_task(data: str) -> str: ...
```

---

## Common Patterns

### Classification

```python
from typing import Literal

@ptool()
def classify_email(subject: str, body: str) -> Literal["spam", "important", "normal"]:
    """Classify an email by importance.

    spam: Unsolicited promotional content
    important: Requires immediate attention
    normal: Regular correspondence
    """
    ...
```

### Extraction

```python
from typing import List, Tuple

@ptool()
def extract_dates(text: str) -> List[str]:
    """Extract all dates mentioned in the text.

    Return dates in ISO format (YYYY-MM-DD).
    """
    ...

@ptool()
def parse_address(text: str) -> Tuple[str, str, str, str]:
    """Parse an address into components.

    Return (street, city, state, zip_code).
    """
    ...
```

### Analysis

```python
@ptool()
def analyze_sentiment(review: str) -> Dict[str, Any]:
    """Analyze the sentiment of a customer review.

    Return:
    {
        "sentiment": "positive" | "negative" | "neutral",
        "confidence": float between 0 and 1,
        "key_phrases": list of important phrases,
        "issues": list of problems mentioned (empty if none)
    }
    """
    ...
```

### Text Generation

```python
@ptool(output_mode="freeform")
def generate_response(query: str, context: str) -> str:
    """Generate a helpful response to the user query.

    Use the provided context to inform your answer.
    Be concise but thorough.
    """
    ...
```

---

## Using with Docstring Examples

Add examples to guide the LLM:

```python
@ptool()
def normalize_name(name: str) -> str:
    """Normalize a person's name to "First Last" format.

    Examples:
    >>> normalize_name("JOHN DOE")
    'John Doe'
    >>> normalize_name("doe, john")
    'John Doe'
    >>> normalize_name("Dr. John Q. Doe III")
    'John Doe'
    """
    ...
```

---

## Composing ptools in Workflows

Python controls the flow, ptools do the thinking:

```python
from ptool_framework import ptool

@ptool()
def extract_topics(text: str) -> List[str]:
    """Extract main topics from text."""
    ...

@ptool()
def summarize_topic(text: str, topic: str) -> str:
    """Summarize content related to a specific topic."""
    ...

@ptool()
def combine_summaries(summaries: List[str]) -> str:
    """Combine multiple summaries into a coherent overview."""
    ...

# Python workflow using ptools
def analyze_document(document: str) -> str:
    """Analyze a document by topic."""

    # Step 1: Find topics (LLM)
    topics = extract_topics(document)

    # Step 2: Summarize each topic (LLM, in a Python loop)
    summaries = []
    for topic in topics:
        summary = summarize_topic(document, topic)
        summaries.append(f"{topic}: {summary}")

    # Step 3: Combine (LLM)
    return combine_summaries(summaries)
```

---

## Error Handling

```python
from ptool_framework import ptool
from ptool_framework.llm_backend import LLMError

@ptool()
def risky_analysis(data: str) -> Dict:
    """Analyze potentially malformed data."""
    ...

try:
    result = risky_analysis(user_input)
except LLMError as e:
    print(f"LLM call failed: {e}")
    result = {"error": str(e)}
```

---

## Enabling Tracing

Collect execution data for analysis and distillation:

```python
from ptool_framework import ptool, enable_tracing, get_trace_store

# Enable tracing
enable_tracing(True)

@ptool()
def my_function(x: str) -> str:
    """Do something."""
    ...

# Run your code - traces are collected automatically
result = my_function("input")

# View collected traces
store = get_trace_store()
traces = store.get_traces(ptool_name="my_function")
for trace in traces:
    print(f"Input: {trace.inputs}, Output: {trace.output}")
```

---

## Accessing the Registry

See all registered ptools:

```python
from ptool_framework import get_registry

registry = get_registry()

# List all ptools
for name in registry.list_ptools():
    spec = registry.get(name)
    print(f"{name}: {spec.get_signature_str()}")
    print(f"  {spec.docstring[:100]}...")
```

---

## When to Use Mode 1

**Good for:**
- Adding LLM capabilities to existing code
- Well-defined input/output requirements
- Tasks where Python controls the flow
- Building blocks for larger workflows

**Consider Mode 2 or 3 when:**
- You don't know what ptools you need yet
- The task is exploratory or open-ended
- You want the LLM to decide the workflow

---

## Next Steps

- **Add Python fast-paths**: See [The @distilled Decorator](architecture.md#distilled-decorator)
- **Generate programs automatically**: See [Mode 2: Program Generator](mode-2-program-generator.md)
- **Let the agent decide**: See [Mode 3: ReAct Agent](mode-3-react-agent.md)
