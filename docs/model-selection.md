# Intelligent Model Selection

The model selection system automatically routes ptool calls to the most appropriate LLM based on task complexity, historical performance, and cost considerations.

## Overview

Key features:

1. **Complexity Estimation**: Automatically estimate task difficulty
2. **Experience Learning**: Learn from past executions to improve routing
3. **Fallback Chains**: Try multiple models in order for robustness
4. **Cost/Latency Optimization**: Balance quality, speed, and cost

## Quick Start

```python
from ptool_framework.model_selector import (
    ModelSelector,
    TaskComplexity,
    ExperienceStore,
)

# Create selector
selector = ModelSelector(
    default_model="deepseek-v3",
    enable_learning=True,
)

# Select model for a task
model = selector.select(ptool_spec, inputs)

# Get fallback chain
chain = selector.get_fallback_chain(ptool_spec, inputs)
for model in chain:
    try:
        result = execute_with_model(model, inputs)
        selector.record_execution(ptool_spec.name, model, inputs, True, latency_ms)
        break
    except Exception:
        selector.record_execution(ptool_spec.name, model, inputs, False, latency_ms)
        continue
```

## TaskComplexity

Task complexity levels guide model selection:

```python
from ptool_framework.model_selector import TaskComplexity

TaskComplexity.TRIVIAL   # Very simple, any model works
TaskComplexity.SIMPLE    # Basic reasoning, fast models work
TaskComplexity.MODERATE  # Some complexity, mid-tier models
TaskComplexity.COMPLEX   # Significant reasoning, capable models
TaskComplexity.EXPERT    # Maximum difficulty, best models only
```

## ModelSelector

### Configuration

```python
from ptool_framework.model_selector import (
    ModelSelector,
    ModelConfig,
    ExperienceStore,
)

# Custom model configurations
models = {
    "my-model": ModelConfig(
        name="my-model",
        provider="together",
        cost_per_1k_input=0.001,
        cost_per_1k_output=0.002,
        capabilities=["reasoning", "code"],
        quality_score=0.85,
        latency_ms_avg=500,
    ),
}

# Create selector
selector = ModelSelector(
    experience_store=ExperienceStore(),  # Persistent storage
    models=models,                        # Model configs
    default_model="deepseek-v3",         # Fallback
    enable_learning=True,                # Learn from executions
)
```

### Selection

```python
# Simple selection - returns model name
model = selector.select(ptool_spec, inputs)

# Detailed selection - returns SelectionResult
result = selector.select_with_details(ptool_spec, inputs)
print(result.selected_model)   # "deepseek-v3"
print(result.reason)           # "Selected for moderate task (score: 0.85)"
print(result.confidence)       # 0.9
print(result.fallback_chain)   # ["llama-3.1-70b", "gpt-4o"]
print(result.estimated_cost)   # 0.001

# With criteria
from ptool_framework.model_selector import SelectionCriteria

criteria = SelectionCriteria(
    required_capabilities=["reasoning", "code"],
    max_cost_per_1k_tokens=0.01,
    max_latency_ms=2000,
    min_quality_threshold=0.7,
)
result = selector.select_with_details(ptool_spec, inputs, criteria)
```

### Fallback Chains

```python
# Get ordered list of models to try
chain = selector.get_fallback_chain(ptool_spec, inputs, max_length=3)

# Use the chain
for model in chain:
    try:
        result = execute_ptool(spec, inputs, model_override=model)
        selector.record_execution(spec.name, model, inputs, True, latency)
        break
    except Exception as e:
        selector.record_execution(spec.name, model, inputs, False, latency)
        continue
else:
    raise RuntimeError("All models failed")
```

### Recording Executions

```python
# Record execution for learning
selector.record_execution(
    ptool_name="extract_values",
    model="deepseek-v3",
    inputs={"text": "..."},
    success=True,
    latency_ms=500,
    cost=0.001,
)
```

## ExperienceStore

Persistent storage for performance history:

```python
from ptool_framework.model_selector import ExperienceStore

# Create store (defaults to ~/.ptool_experiences)
store = ExperienceStore(path="~/.ptool_experiences")

# Record execution
store.record_execution(
    model_name="deepseek-v3",
    ptool_name="extract_values",
    success=True,
    latency_ms=500,
    cost=0.001,
)

# Query performance
perf = store.get_performance("deepseek-v3", "extract_values")
print(perf.success_rate)   # 0.95
print(perf.total_calls)    # 100
print(perf.avg_latency_ms) # 480.5

# Get best model for ptool
best = store.get_best_model_for_ptool("extract_values", min_calls=5)
print(best)  # "deepseek-v3"

# Get ranked list
ranking = store.get_model_ranking("extract_values", min_calls=3)
for model, score in ranking:
    print(f"{model}: {score:.1%}")
```

## Complexity Estimation

### Heuristic Estimator

Fast, no-LLM estimation based on:
- Input size
- Number of inputs
- Ptool name patterns
- Docstring keywords

```python
from ptool_framework.model_selector import heuristic_complexity_estimator

complexity = heuristic_complexity_estimator(ptool_spec, inputs)
print(complexity)  # TaskComplexity.MODERATE
```

### LLM Estimator

More accurate but slower:

```python
from ptool_framework.model_selector import estimate_task_complexity_llm

complexity = estimate_task_complexity_llm(
    ptool_spec,
    inputs,
    model="gpt-4o-mini",  # Fast model for estimation
)
```

### Custom Estimator

```python
def my_complexity_estimator(ptool_spec, inputs):
    # Custom logic
    if "analyze" in ptool_spec.name:
        return TaskComplexity.COMPLEX
    return TaskComplexity.SIMPLE

selector = ModelSelector(complexity_estimator=my_complexity_estimator)
```

## Default Models

Pre-configured model settings:

```python
from ptool_framework.model_selector import DEFAULT_MODELS

# Available models
# - deepseek-v3: High quality, low cost (Together API)
# - llama-3.1-70b: Good quality, fast (Together API)
# - llama-3.1-8b: Fast, cheap (Together API)
# - gpt-4o: Highest quality (OpenAI)
# - gpt-4o-mini: Fast, cheap (OpenAI)
# - claude-3-sonnet: High quality (Anthropic)

for name, config in DEFAULT_MODELS.items():
    print(f"{name}: quality={config.quality_score}, cost=${config.cost_per_1k_input}/1k")
```

## Convenience Functions

```python
from ptool_framework.model_selector import (
    select_model,
    get_model_for_complexity,
)

# Quick selection by ptool name
model = select_model("extract_values", {"text": "..."})

# Get model for complexity level
model = get_model_for_complexity(TaskComplexity.COMPLEX)
```

## Integration with Executor

```python
from ptool_framework.executor import TraceExecutor
from ptool_framework.model_selector import ModelSelector

selector = ModelSelector()

executor = TraceExecutor(
    model_selector=selector,
    use_fallback_chain=True,
)

# Executor will automatically:
# 1. Estimate complexity for each step
# 2. Select appropriate model
# 3. Use fallback chain on failures
# 4. Record executions for learning
```

## Best Practices

1. **Start with defaults**: Use default models initially
2. **Enable learning**: Let the system learn from executions
3. **Set min_calls threshold**: Require enough data before trusting rankings
4. **Use fallback chains**: Always have backup models
5. **Monitor costs**: Track total costs over time
6. **Review rankings periodically**: Check if routing makes sense

## Example: Cost-Optimized Routing

```python
from ptool_framework.model_selector import (
    ModelSelector,
    SelectionCriteria,
    TaskComplexity,
)

# Cost-conscious criteria
cheap_criteria = SelectionCriteria(
    max_cost_per_1k_tokens=0.001,
    max_latency_ms=3000,
    min_quality_threshold=0.6,
)

# Quality-first criteria
quality_criteria = SelectionCriteria(
    min_quality_threshold=0.9,
    max_latency_ms=5000,
)

selector = ModelSelector()

def route_task(ptool_spec, inputs, priority="balanced"):
    """Route based on priority."""
    if priority == "cheap":
        return selector.select(ptool_spec, inputs, cheap_criteria)
    elif priority == "quality":
        return selector.select(ptool_spec, inputs, quality_criteria)
    else:
        return selector.select(ptool_spec, inputs)

# Usage
model = route_task(spec, inputs, priority="cheap")
```

## Example: Logging and Monitoring

```python
import logging
from ptool_framework.model_selector import ModelSelector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model_selection")

selector = ModelSelector()

def select_and_log(ptool_spec, inputs):
    """Select model with logging."""
    result = selector.select_with_details(ptool_spec, inputs)

    logger.info(
        f"Selected {result.selected_model} for {ptool_spec.name} "
        f"(confidence={result.confidence:.2f}, reason={result.reason})"
    )

    return result.selected_model

def record_and_log(ptool_name, model, inputs, success, latency_ms, cost):
    """Record execution with logging."""
    selector.record_execution(ptool_name, model, inputs, success, latency_ms, cost)

    status = "SUCCESS" if success else "FAILURE"
    logger.info(
        f"{status}: {ptool_name} with {model} "
        f"(latency={latency_ms:.0f}ms, cost=${cost:.4f})"
    )
```

## See Also

- [Audit System](audit-system.md)
- [Critic and Repair](critic-repair.md)
- [API Reference](api-reference.md)
