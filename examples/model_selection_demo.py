"""
Demo: Intelligent Model Selection

This example demonstrates:
1. Selecting models based on task complexity
2. Using fallback chains for robustness
3. Learning from execution history
4. Custom selection criteria

Run:
    python examples/model_selection_demo.py
"""

import tempfile
from dataclasses import dataclass
from ptool_framework.model_selector import (
    TaskComplexity,
    ModelConfig,
    ModelSelector,
    ExperienceStore,
    SelectionCriteria,
    FallbackChain,
    DEFAULT_MODELS,
    heuristic_complexity_estimator,
    select_model,
    get_model_for_complexity,
)


@dataclass
class MockPToolSpec:
    """Mock ptool spec for testing."""
    name: str
    docstring: str = ""


def demo_task_complexity():
    """Demo task complexity levels."""
    print("=" * 60)
    print("Demo 1: Task Complexity Levels")
    print("=" * 60)

    print("\nTaskComplexity values:")
    for complexity in TaskComplexity:
        print(f"  {complexity.name}: '{complexity.value}'")

    print("\nRecommended models by complexity:")
    for complexity in TaskComplexity:
        model = get_model_for_complexity(complexity)
        print(f"  {complexity.name}: {model}")
    print()


def demo_heuristic_estimation():
    """Demo heuristic complexity estimation."""
    print("=" * 60)
    print("Demo 2: Heuristic Complexity Estimation")
    print("=" * 60)

    test_cases = [
        ("format_text", "Simple formatting", {"text": "hello"}),
        ("extract_values", "Extract values from text", {"text": "weight 70kg"}),
        ("analyze_reasoning_chain", "Complex multi-step synthesis", {"data": "x" * 2000}),
        ("synthesize_complex_output", "Expert-level task", {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6}),
    ]

    print("\nEstimated complexities:")
    for name, docstring, inputs in test_cases:
        spec = MockPToolSpec(name=name, docstring=docstring)
        complexity = heuristic_complexity_estimator(spec, inputs)
        print(f"  {name}:")
        print(f"    Complexity: {complexity.value}")
        print(f"    Input size: {sum(len(str(v)) for v in inputs.values())} chars")
    print()


def demo_model_selection():
    """Demo basic model selection."""
    print("=" * 60)
    print("Demo 3: Model Selection")
    print("=" * 60)

    # Create selector
    selector = ModelSelector(default_model="deepseek-v3")

    # Test different tasks
    tasks = [
        MockPToolSpec("format_text", "Format text"),
        MockPToolSpec("extract_values", "Extract values"),
        MockPToolSpec("analyze_complex_reasoning", "Complex analysis"),
    ]

    print("\nModel selection results:")
    for spec in tasks:
        result = selector.select_with_details(spec, {"input": "test"})
        print(f"\n  Task: {spec.name}")
        print(f"    Selected: {result.selected_model}")
        print(f"    Reason: {result.reason}")
        print(f"    Confidence: {result.confidence:.2f}")
        print(f"    Fallbacks: {result.fallback_chain}")
    print()


def demo_selection_criteria():
    """Demo selection with custom criteria."""
    print("=" * 60)
    print("Demo 4: Selection Criteria")
    print("=" * 60)

    selector = ModelSelector()
    spec = MockPToolSpec("my_task", "Some task")
    inputs = {"data": "test input"}

    # Different criteria
    criteria_sets = [
        ("Cheap", SelectionCriteria(
            max_cost_per_1k_tokens=0.001,
            min_quality_threshold=0.5,
        )),
        ("Fast", SelectionCriteria(
            max_latency_ms=500,
            min_quality_threshold=0.5,
        )),
        ("Quality", SelectionCriteria(
            min_quality_threshold=0.9,
        )),
        ("Reasoning", SelectionCriteria(
            required_capabilities=["reasoning", "code"],
            min_quality_threshold=0.7,
        )),
    ]

    print("\nSelection with different criteria:")
    for name, criteria in criteria_sets:
        result = selector.select_with_details(spec, inputs, criteria)
        print(f"\n  {name} criteria:")
        print(f"    Selected: {result.selected_model}")
        config = result.config
        if config:
            print(f"    Cost: ${config.cost_per_1k_input}/1k input")
            print(f"    Latency: {config.latency_ms_avg}ms")
            print(f"    Quality: {config.quality_score}")
    print()


def demo_fallback_chains():
    """Demo fallback chain usage."""
    print("=" * 60)
    print("Demo 5: Fallback Chains")
    print("=" * 60)

    selector = ModelSelector()
    spec = MockPToolSpec("my_task", "Some task")

    chain = selector.get_fallback_chain(spec, {"input": "test"}, max_length=4)

    print(f"\nFallback chain for '{spec.name}':")
    print(f"  Reason: {chain.reason}")
    print(f"  Models ({len(chain)}):")
    for i, model in enumerate(chain):
        config = DEFAULT_MODELS.get(model)
        quality = config.quality_score if config else "?"
        print(f"    {i+1}. {model} (quality: {quality})")

    print("\nSimulated execution with fallbacks:")
    for i, model in enumerate(chain):
        if i < 2:
            print(f"  Try {model}: FAILED (simulated)")
        else:
            print(f"  Try {model}: SUCCESS")
            break
    print()


def demo_experience_store():
    """Demo experience store for learning."""
    print("=" * 60)
    print("Demo 6: Experience Store")
    print("=" * 60)

    # Create temporary store
    with tempfile.TemporaryDirectory() as temp_dir:
        store = ExperienceStore(path=temp_dir)

        # Simulate executions
        executions = [
            ("deepseek-v3", "extract", True, 450),
            ("deepseek-v3", "extract", True, 520),
            ("deepseek-v3", "extract", True, 480),
            ("deepseek-v3", "extract", False, 600),  # 1 failure
            ("llama-3.1-70b", "extract", True, 300),
            ("llama-3.1-70b", "extract", True, 320),
            ("llama-3.1-70b", "extract", True, 350),
            ("llama-3.1-70b", "extract", True, 310),
            ("llama-3.1-70b", "extract", True, 340),
        ]

        print("\nRecording executions...")
        for model, ptool, success, latency in executions:
            store.record_execution(model, ptool, success, latency, 0.001)

        # Query performance
        print("\nPerformance statistics:")
        for model in ["deepseek-v3", "llama-3.1-70b"]:
            perf = store.get_performance(model, "extract")
            if perf:
                print(f"\n  {model}:")
                print(f"    Total calls: {perf.total_calls}")
                print(f"    Success rate: {perf.success_rate:.1%}")
                print(f"    Avg latency: {perf.avg_latency_ms:.0f}ms")

        # Get best model
        best = store.get_best_model_for_ptool("extract", min_calls=3)
        print(f"\nBest model for 'extract': {best}")

        # Get ranking
        ranking = store.get_model_ranking("extract", min_calls=3)
        print("\nModel ranking:")
        for model, score in ranking:
            print(f"  {model}: {score:.1%}")
    print()


def demo_learning_selection():
    """Demo experience-based model selection."""
    print("=" * 60)
    print("Demo 7: Learning-Based Selection")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as temp_dir:
        store = ExperienceStore(path=temp_dir)
        selector = ModelSelector(
            experience_store=store,
            enable_learning=True,
        )

        spec = MockPToolSpec("extract_values", "Extract values")

        # Initial selection (no experience)
        result1 = selector.select_with_details(spec, {"text": "test"})
        print(f"\nBefore learning:")
        print(f"  Selected: {result1.selected_model}")
        print(f"  Reason: {result1.reason}")

        # Record some executions
        print("\nSimulating executions...")
        for _ in range(10):
            selector.record_execution("llama-3.1-70b", "extract_values", {"text": "x"}, True, 300, 0.001)
        for _ in range(10):
            selector.record_execution("deepseek-v3", "extract_values", {"text": "x"}, False, 500, 0.002)

        # Selection after learning
        result2 = selector.select_with_details(spec, {"text": "test"})
        print(f"\nAfter learning:")
        print(f"  Selected: {result2.selected_model}")
        print(f"  Reason: {result2.reason}")
    print()


def demo_default_models():
    """Demo default model configurations."""
    print("=" * 60)
    print("Demo 8: Default Models")
    print("=" * 60)

    print("\nAvailable models:")
    for name, config in DEFAULT_MODELS.items():
        print(f"\n  {name}:")
        print(f"    Provider: {config.provider}")
        print(f"    Quality: {config.quality_score}")
        print(f"    Input cost: ${config.cost_per_1k_input}/1k tokens")
        print(f"    Output cost: ${config.cost_per_1k_output}/1k tokens")
        print(f"    Latency: {config.latency_ms_avg}ms avg")
        print(f"    Capabilities: {config.capabilities}")
    print()


def demo_convenience_functions():
    """Demo convenience functions."""
    print("=" * 60)
    print("Demo 9: Convenience Functions")
    print("=" * 60)

    # Quick selection by name
    print("\nQuick model selection:")
    model = select_model("extract_values", {"text": "test"})
    print(f"  select_model('extract_values', ...): {model}")

    # Get model by complexity
    print("\nGet model for complexity:")
    for complexity in [TaskComplexity.SIMPLE, TaskComplexity.COMPLEX, TaskComplexity.EXPERT]:
        model = get_model_for_complexity(complexity)
        print(f"  {complexity.name}: {model}")
    print()


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("MODEL SELECTION DEMO")
    print("=" * 60 + "\n")

    demo_task_complexity()
    demo_heuristic_estimation()
    demo_model_selection()
    demo_selection_criteria()
    demo_fallback_chains()
    demo_experience_store()
    demo_learning_selection()
    demo_default_models()
    demo_convenience_functions()

    print("=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
