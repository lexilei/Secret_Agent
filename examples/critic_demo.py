"""
Demo: Critic and Repair System

This example demonstrates:
1. Evaluating traces with the critic
2. Understanding critic verdicts
3. Repairing traces with the repair agent
4. Complete evaluation-repair workflow

Run:
    python examples/critic_demo.py
"""

from ptool_framework.traces import WorkflowTrace, StepStatus
from ptool_framework.critic import (
    TraceCritic,
    CriticVerdict,
    CriticEvaluation,
    evaluate_trace,
    quick_evaluate,
)
from ptool_framework.repair import (
    RepairAgent,
    RepairActionType,
    RepairResult,
    repair_trace,
    auto_repair,
)
from ptool_framework.audit import (
    NonEmptyTraceAudit,
    NoFailedStepsAudit,
    RequiredStepsAudit,
    StepOrderAudit,
    StructuredAudit,
)


def create_good_trace():
    """Create a successful trace."""
    trace = WorkflowTrace(goal="Calculate BMI")
    trace.add_step("identify_calculator", {"text": "Calculate BMI for patient"})
    trace.add_step("extract_values", {"text": "weight 70kg, height 1.75m"})
    trace.add_step("calculate_bmi", {"weight": 70, "height": 1.75})

    for step in trace.steps:
        step.status = StepStatus.COMPLETED
        step.result = {"success": True}

    return trace


def create_bad_trace():
    """Create a trace with failures."""
    trace = WorkflowTrace(goal="Calculate BMI")
    trace.add_step("extract_values", {"text": "weight 70kg"})  # Missing height
    trace.add_step("calculate_bmi", {"weight": 70})  # Will fail

    trace.steps[0].status = StepStatus.COMPLETED
    trace.steps[0].result = {"weight": 70}

    trace.steps[1].status = StepStatus.FAILED
    trace.steps[1].error = "Missing height parameter"

    return trace


def create_empty_trace():
    """Create an empty trace."""
    return WorkflowTrace(goal="Nothing")


def demo_basic_critic():
    """Demo basic critic evaluation."""
    print("=" * 60)
    print("Demo 1: Basic Critic Evaluation")
    print("=" * 60)

    # Create critic with audits
    critic = TraceCritic(
        audits=[
            NonEmptyTraceAudit(),
            NoFailedStepsAudit(),
        ],
        accept_threshold=0.8,
        repair_threshold=0.4,
    )

    # Evaluate different traces
    traces = [
        ("Good trace", create_good_trace()),
        ("Bad trace", create_bad_trace()),
        ("Empty trace", create_empty_trace()),
    ]

    for name, trace in traces:
        evaluation = critic.evaluate(trace, goal=trace.goal)
        print(f"\n{name}:")
        print(f"  Verdict: {evaluation.verdict.value}")
        print(f"  Confidence: {evaluation.confidence:.2f}")
        print(f"  Acceptable: {evaluation.is_acceptable}")
        print(f"  Needs repair: {evaluation.needs_repair}")
        print(f"  Should reject: {evaluation.should_reject}")

    print()


def demo_evaluation_details():
    """Demo detailed evaluation results."""
    print("=" * 60)
    print("Demo 2: Evaluation Details")
    print("=" * 60)

    critic = TraceCritic(
        audits=[
            NonEmptyTraceAudit(),
            NoFailedStepsAudit(),
            RequiredStepsAudit(["extract", "calculate"]),
        ]
    )

    trace = create_bad_trace()
    evaluation = critic.evaluate(trace, goal="Calculate BMI")

    print(f"Verdict: {evaluation.verdict.value}")
    print(f"Confidence: {evaluation.confidence:.2f}")

    print(f"\nScores:")
    print(f"  Completeness: {evaluation.completeness_score:.2f}")
    print(f"  Correctness: {evaluation.correctness_score:.2f}")
    print(f"  Efficiency: {evaluation.efficiency_score:.2f}")

    print(f"\nFailed steps: {evaluation.failed_steps}")

    print(f"\nRepair suggestions:")
    for suggestion in evaluation.repair_suggestions:
        print(f"  - Action: {suggestion.get('action')}")
        print(f"    Reason: {suggestion.get('reason')}")
        print(f"    Priority: {suggestion.get('priority')}")

    print(f"\nSummary:\n{evaluation.summary()}")
    print()


def demo_repair_agent():
    """Demo repair agent functionality."""
    print("=" * 60)
    print("Demo 3: Repair Agent")
    print("=" * 60)

    # Create critic and repair agent
    critic = TraceCritic(
        audits=[
            NonEmptyTraceAudit(),
            NoFailedStepsAudit(),
        ],
        accept_threshold=0.8,
    )

    repair_agent = RepairAgent(
        critic=critic,
        max_attempts=3,
        model="deepseek-v3",
    )

    # Try to repair bad trace
    trace = create_bad_trace()
    evaluation = critic.evaluate(trace, goal="Calculate BMI")

    print(f"Initial verdict: {evaluation.verdict.value}")
    print(f"Attempting repair...")

    result = repair_agent.repair(trace, evaluation, goal="Calculate BMI")

    print(f"\nRepair result:")
    print(f"  Success: {result.success}")
    print(f"  Iterations: {result.iterations}")
    print(f"  Actions taken: {len(result.actions_taken)}")

    if result.actions_taken:
        print("\n  Actions:")
        for action in result.actions_taken:
            print(f"    - {action.action_type.value}: {action.reason}")

    if result.error:
        print(f"  Error: {result.error}")

    print(f"\n{result.summary()}")
    print()


def demo_convenience_functions():
    """Demo convenience functions."""
    print("=" * 60)
    print("Demo 4: Convenience Functions")
    print("=" * 60)

    good_trace = create_good_trace()
    bad_trace = create_bad_trace()

    # Quick evaluation
    print("Quick evaluation (returns verdict only):")
    verdict = quick_evaluate(good_trace)
    print(f"  Good trace: {verdict.value}")
    verdict = quick_evaluate(bad_trace)
    print(f"  Bad trace: {verdict.value}")

    # Full evaluation with custom audits
    print("\nFull evaluation:")
    evaluation = evaluate_trace(
        good_trace,
        audits=[NonEmptyTraceAudit(), NoFailedStepsAudit()]
    )
    print(f"  Verdict: {evaluation.verdict.value}")
    print(f"  Confidence: {evaluation.confidence:.2f}")

    # Auto repair
    print("\nAuto repair (returns fixed trace or None):")
    fixed = auto_repair(good_trace, "Calculate BMI")
    print(f"  Good trace repair result: {'Trace returned' if fixed else 'None'}")
    print()


def demo_custom_thresholds():
    """Demo custom threshold configuration."""
    print("=" * 60)
    print("Demo 5: Custom Thresholds")
    print("=" * 60)

    # Very strict critic
    strict_critic = TraceCritic(
        audits=[NonEmptyTraceAudit(), NoFailedStepsAudit()],
        accept_threshold=0.95,  # Very high bar
        repair_threshold=0.6,
    )

    # Lenient critic
    lenient_critic = TraceCritic(
        audits=[NonEmptyTraceAudit(), NoFailedStepsAudit()],
        accept_threshold=0.5,  # Lower bar
        repair_threshold=0.2,
    )

    trace = create_good_trace()

    strict_eval = strict_critic.evaluate(trace)
    lenient_eval = lenient_critic.evaluate(trace)

    print(f"Same trace with different thresholds:")
    print(f"  Strict critic: {strict_eval.verdict.value} (confidence: {strict_eval.confidence:.2f})")
    print(f"  Lenient critic: {lenient_eval.verdict.value} (confidence: {lenient_eval.confidence:.2f})")
    print()


def demo_full_pipeline():
    """Demo complete evaluation-repair pipeline."""
    print("=" * 60)
    print("Demo 6: Full Pipeline")
    print("=" * 60)

    # Setup
    audits = [
        NonEmptyTraceAudit(),
        NoFailedStepsAudit(),
        RequiredStepsAudit(["extract", "calculate"]),
        StepOrderAudit([("extract", "calculate")]),
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
        """Complete trace processing pipeline."""
        print(f"\nProcessing: {goal}")

        # Step 1: Evaluate
        evaluation = critic.evaluate(trace, goal=goal)
        print(f"  Initial verdict: {evaluation.verdict.value}")

        # Step 2: Handle based on verdict
        if evaluation.is_acceptable:
            print(f"  Status: ACCEPTED")
            return trace, "accepted"

        if evaluation.needs_repair:
            print(f"  Status: Attempting repair...")
            result = repair_agent.repair(trace, evaluation, goal)

            if result.success:
                print(f"  Status: REPAIRED in {result.iterations} iterations")
                return result.repaired_trace, "repaired"
            else:
                print(f"  Status: REPAIR FAILED - {result.error}")
                return None, "repair_failed"

        print(f"  Status: REJECTED")
        return None, "rejected"

    # Process different traces
    traces = [
        ("Calculate BMI", create_good_trace()),
        ("Calculate BMI", create_bad_trace()),
        ("Empty goal", create_empty_trace()),
    ]

    results = []
    for goal, trace in traces:
        fixed_trace, status = process_trace(trace, goal)
        results.append((goal, status))

    print("\n\nPipeline Summary:")
    for goal, status in results:
        print(f"  {goal}: {status}")
    print()


def demo_verdict_properties():
    """Demo verdict enum properties."""
    print("=" * 60)
    print("Demo 7: Verdict Properties")
    print("=" * 60)

    print("CriticVerdict values:")
    for verdict in CriticVerdict:
        print(f"  {verdict.name}: '{verdict.value}'")

    print("\nCreating evaluations with different verdicts:")

    # ACCEPT
    accept_eval = CriticEvaluation(
        verdict=CriticVerdict.ACCEPT,
        confidence=0.9,
    )
    print(f"\nACCEPT evaluation:")
    print(f"  is_acceptable: {accept_eval.is_acceptable}")
    print(f"  needs_repair: {accept_eval.needs_repair}")
    print(f"  should_reject: {accept_eval.should_reject}")

    # REPAIR_NEEDED
    repair_eval = CriticEvaluation(
        verdict=CriticVerdict.REPAIR_NEEDED,
        confidence=0.6,
    )
    print(f"\nREPAIR_NEEDED evaluation:")
    print(f"  is_acceptable: {repair_eval.is_acceptable}")
    print(f"  needs_repair: {repair_eval.needs_repair}")
    print(f"  should_reject: {repair_eval.should_reject}")

    # REJECT
    reject_eval = CriticEvaluation(
        verdict=CriticVerdict.REJECT,
        confidence=0.3,
    )
    print(f"\nREJECT evaluation:")
    print(f"  is_acceptable: {reject_eval.is_acceptable}")
    print(f"  needs_repair: {reject_eval.needs_repair}")
    print(f"  should_reject: {reject_eval.should_reject}")
    print()


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("CRITIC AND REPAIR SYSTEM DEMO")
    print("=" * 60 + "\n")

    demo_basic_critic()
    demo_evaluation_details()
    demo_repair_agent()
    demo_convenience_functions()
    demo_custom_thresholds()
    demo_full_pipeline()
    demo_verdict_properties()

    print("=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
