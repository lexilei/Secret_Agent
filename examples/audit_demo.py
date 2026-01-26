"""
Demo: Structured and Typicality Audits

This example demonstrates:
1. Creating structured audits with the DSL
2. Training typicality models on patterns
3. Running audits on traces
4. Using pre-built and custom audits

Run:
    python examples/audit_demo.py
"""

import pandas as pd
from ptool_framework.traces import WorkflowTrace, StepStatus
from ptool_framework.audit import (
    # Base classes
    StructuredAudit,
    AuditResult,
    AuditRunner,
    AuditDSL,
    # Pre-built audits
    NonEmptyTraceAudit,
    NoFailedStepsAudit,
    RequiredStepsAudit,
    StepOrderAudit,
    create_basic_audit,
    create_workflow_audit,
    # Typicality
    DeclarativeAudit,
    audit_rule,
)
from ptool_framework.audit.typicality import (
    UnigramModel,
    BigramModel,
    InterpolatedModel,
    extract_pattern,
    PatternStats,
    START_TOKEN,
    END_TOKEN,
    train_ensemble_models,
    ensemble_score,
)


def create_sample_trace(success=True):
    """Create a sample workflow trace."""
    trace = WorkflowTrace(goal="Calculate BMI for patient")

    # Add steps
    trace.add_step("identify_calculator", {"text": "weight 70kg, height 1.75m"})
    trace.add_step("extract_values", {"text": "weight 70kg, height 1.75m"})
    trace.add_step("calculate_bmi", {"weight": 70, "height": 1.75})
    trace.add_step("format_result", {"bmi": 22.86})

    # Mark steps as completed (or failed)
    for i, step in enumerate(trace.steps):
        if success or i < 2:
            step.status = StepStatus.COMPLETED
            step.result = {"output": f"result_{i}"}
        else:
            step.status = StepStatus.FAILED
            step.error = "Calculation failed"

    return trace


def demo_basic_structured_audit():
    """Demo basic structured audit creation."""
    print("=" * 60)
    print("Demo 1: Basic Structured Audit")
    print("=" * 60)

    # Create a simple audit
    audit = StructuredAudit("my_basic_audit", "Basic validation checks")

    # Add rules using lambda functions
    audit.add_rule(
        "has_steps",
        lambda df, _: len(df) > 0,
        "Trace must have at least one step"
    )
    audit.add_rule(
        "no_failures",
        lambda df, _: (df["status"] != "failed").all() if len(df) > 0 else True,
        "No failed steps allowed"
    )
    audit.add_rule(
        "max_10_steps",
        lambda df, _: len(df) <= 10,
        "Maximum 10 steps allowed"
    )

    # Create and audit a trace
    trace = create_sample_trace(success=True)
    runner = AuditRunner(audits=[audit])
    report = runner.audit_trace(trace)

    print(f"Trace: {trace.goal}")
    print(f"Result: {report.result.value}")
    print(f"Passed checks: {report.passed_checks}")
    print()


def demo_prebuilt_audits():
    """Demo pre-built audits."""
    print("=" * 60)
    print("Demo 2: Pre-built Audits")
    print("=" * 60)

    # Create audits
    audits = [
        NonEmptyTraceAudit(),
        NoFailedStepsAudit(),
        RequiredStepsAudit(["extract", "calculate"]),
        StepOrderAudit([("extract", "calculate")]),
    ]

    # Create traces
    good_trace = create_sample_trace(success=True)
    bad_trace = create_sample_trace(success=False)

    runner = AuditRunner(audits=audits)

    # Audit good trace
    print("Good trace:")
    report = runner.audit_trace(good_trace)
    print(f"  Result: {report.result.value}")
    print(f"  Passed: {len(report.passed_checks)} checks")

    # Audit bad trace
    print("\nBad trace:")
    report = runner.audit_trace(bad_trace)
    print(f"  Result: {report.result.value}")
    print(f"  Violations: {len(report.violations)}")
    for v in report.violations:
        print(f"    - {v.rule_name}: {v.message}")
    print()


def demo_declarative_audit():
    """Demo declarative audit with @audit_rule decorator."""
    print("=" * 60)
    print("Demo 3: Declarative Audit")
    print("=" * 60)

    class MedCalcAudit(DeclarativeAudit):
        """Custom audit for medical calculations."""

        def __init__(self):
            super().__init__("medcalc_audit", "Medical Calculator Validation")

        @audit_rule("has_identification", "Must identify calculator type")
        def check_identification(self, dsl):
            return dsl.has_at_least("identify", 1)

        @audit_rule("has_extraction", "Must extract values from input")
        def check_extraction(self, dsl):
            return dsl.has_at_least("extract", 1)

        @audit_rule("has_calculation", "Must perform calculation")
        def check_calculation(self, dsl):
            return dsl.has_at_least("calculate", 1)

        @audit_rule("correct_order", "Extraction must come before calculation")
        def check_order(self, dsl):
            return dsl.comes_before("extract", "calculate")

    audit = MedCalcAudit()
    trace = create_sample_trace(success=True)
    runner = AuditRunner(audits=[audit])
    report = runner.audit_trace(trace)

    print(f"Result: {report.result.value}")
    print(f"Passed checks: {report.passed_checks}")
    print()


def demo_audit_dsl():
    """Demo AuditDSL query capabilities."""
    print("=" * 60)
    print("Demo 4: AuditDSL Queries")
    print("=" * 60)

    # Create DataFrame directly for demo
    df = pd.DataFrame({
        "step_idx": [0, 1, 2, 3],
        "fn_name": ["identify_calculator", "extract_values", "calculate_bmi", "format_result"],
        "status": ["completed", "completed", "completed", "completed"],
        "input": ['{"text": "..."}', '{"text": "..."}', '{"weight": 70}', '{"bmi": 22.86}'],
        "output": ['{"calc": "bmi"}', '{"values": [70, 1.75]}', '{"bmi": 22.86}', '"BMI: 22.86"'],
        "duration_ms": [100, 150, 200, 50],
    })
    metadata = {"trace_id": "demo123", "goal": "Calculate BMI"}

    dsl = AuditDSL(df, metadata)

    # Demonstrate query methods
    print("Query demonstrations:")
    print(f"  is_non_empty(): {dsl.is_non_empty()}")
    print(f"  has_exactly('extract', 1): {dsl.has_exactly('extract', 1)}")
    print(f"  has_at_least('calculate', 1): {dsl.has_at_least('calculate', 1)}")
    print(f"  comes_before('extract', 'calculate'): {dsl.comes_before('extract', 'calculate')}")
    print(f"  pattern_exists(['extract', 'calculate']): {dsl.pattern_exists(['extract', 'calculate'])}")
    print(f"  all_completed(): {dsl.all_completed()}")
    print(f"  success_rate(): {dsl.success_rate():.1%}")
    print(f"  get_pattern(): {dsl.get_pattern()}")
    print()


def demo_typicality_models():
    """Demo typicality models for pattern scoring."""
    print("=" * 60)
    print("Demo 5: Typicality Models")
    print("=" * 60)

    # Training patterns (from successful traces)
    training_patterns = [
        [START_TOKEN, "identify", "extract", "calculate", "format", END_TOKEN],
        [START_TOKEN, "identify", "extract", "validate", "calculate", "format", END_TOKEN],
        [START_TOKEN, "identify", "extract", "calculate", END_TOKEN],
        [START_TOKEN, "identify", "extract", "validate", "calculate", END_TOKEN],
        [START_TOKEN, "identify", "extract", "calculate", "format", END_TOKEN],
    ]

    # Train models
    unigram = UnigramModel()
    unigram.fit(training_patterns)

    bigram = BigramModel()
    bigram.fit(training_patterns)

    interpolated = InterpolatedModel(lambdas=(0.1, 0.3, 0.6))
    interpolated.fit(training_patterns)

    # Test patterns
    typical_pattern = [START_TOKEN, "identify", "extract", "calculate", "format", END_TOKEN]
    atypical_pattern = [START_TOKEN, "format", "calculate", "identify", END_TOKEN]

    print("Pattern scores (higher = more typical):")
    print(f"\nTypical pattern: {typical_pattern[1:-1]}")
    print(f"  Unigram score: {unigram.score(typical_pattern):.4f}")
    print(f"  Bigram score: {bigram.score(typical_pattern):.4f}")
    print(f"  Interpolated score: {interpolated.score(typical_pattern):.4f}")

    print(f"\nAtypical pattern: {atypical_pattern[1:-1]}")
    print(f"  Unigram score: {unigram.score(atypical_pattern):.4f}")
    print(f"  Bigram score: {bigram.score(atypical_pattern):.4f}")
    print(f"  Interpolated score: {interpolated.score(atypical_pattern):.4f}")
    print()


def demo_ensemble_scoring():
    """Demo ensemble typicality scoring."""
    print("=" * 60)
    print("Demo 6: Ensemble Typicality Scoring")
    print("=" * 60)

    # Training patterns
    patterns = [
        [START_TOKEN, "extract", "validate", "calculate", END_TOKEN],
        [START_TOKEN, "extract", "calculate", END_TOKEN],
        [START_TOKEN, "extract", "validate", "calculate", "format", END_TOKEN],
    ] * 5  # Repeat for better statistics

    # Train ensemble
    models = train_ensemble_models(patterns)
    print(f"Trained models: {list(models.keys())}")

    # Test patterns
    test_patterns = [
        [START_TOKEN, "extract", "calculate", END_TOKEN],
        [START_TOKEN, "extract", "validate", "calculate", END_TOKEN],
        [START_TOKEN, "unknown_step", END_TOKEN],
    ]

    print("\nEnsemble scores:")
    for pattern in test_patterns:
        score = ensemble_score(models, pattern)
        print(f"  {pattern[1:-1]}: {score:.4f}")
    print()


def demo_pattern_statistics():
    """Demo pattern statistics."""
    print("=" * 60)
    print("Demo 7: Pattern Statistics")
    print("=" * 60)

    patterns = [
        [START_TOKEN, "extract", "validate", "calculate", END_TOKEN],
        [START_TOKEN, "extract", "calculate", END_TOKEN],
        [START_TOKEN, "extract", "validate", "calculate", "format", END_TOKEN],
        [START_TOKEN, "extract", "calculate", "format", END_TOKEN],
        [START_TOKEN, "extract", "validate", "calculate", END_TOKEN],
    ]

    stats = PatternStats.from_patterns(patterns)
    print(stats.summary())

    print("\nMost common steps:")
    for step, count in stats.most_common_steps(5):
        print(f"  {step}: {count}")

    print("\nTransition probabilities from <START>:")
    for next_step in ["extract", "calculate", "format"]:
        prob = stats.transition_probability(START_TOKEN, next_step)
        print(f"  P({next_step} | <START>) = {prob:.2f}")
    print()


def demo_factory_functions():
    """Demo factory functions for quick audit creation."""
    print("=" * 60)
    print("Demo 8: Factory Functions")
    print("=" * 60)

    # Basic audit (non-empty, no failures, all completed)
    basic = create_basic_audit()
    print(f"Basic audit: {basic.name}")

    # Workflow audit with custom requirements
    workflow = create_workflow_audit(
        required_steps=["extract", "calculate"],
        step_order=[("extract", "calculate")],
    )
    print(f"Workflow audit: {workflow.name}")

    # Test on trace
    trace = create_sample_trace(success=True)
    runner = AuditRunner(audits=[basic, workflow])
    report = runner.audit_trace(trace)

    print(f"\nCombined result: {report.result.value}")
    print(f"All passed checks: {report.passed_checks}")
    print()


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("AUDIT SYSTEM DEMO")
    print("=" * 60 + "\n")

    demo_basic_structured_audit()
    demo_prebuilt_audits()
    demo_declarative_audit()
    demo_audit_dsl()
    demo_typicality_models()
    demo_ensemble_scoring()
    demo_pattern_statistics()
    demo_factory_functions()

    print("=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
