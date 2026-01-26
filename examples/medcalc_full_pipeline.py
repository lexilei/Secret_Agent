"""
Demo: Full MedCalc Pipeline

This example demonstrates the complete pipeline:
1. Create MedCalc workflow traces
2. Audit with domain-specific checks
3. Evaluate with critic
4. Repair if needed
5. Use intelligent model selection

Run:
    python examples/medcalc_full_pipeline.py
"""

import json
from ptool_framework.traces import WorkflowTrace, StepStatus
from ptool_framework.audit import (
    AuditRunner,
    NonEmptyTraceAudit,
    NoFailedStepsAudit,
    RequiredStepsAudit,
    StepOrderAudit,
)
from ptool_framework.audit.domains.medcalc import (
    MedCalcStructureAudit,
    MedCalcOutputValidationAudit,
    MedCalcBMIAudit,
    KNOWN_CALCULATORS,
)
from ptool_framework.audit.typicality import (
    BigramModel,
    START_TOKEN,
    END_TOKEN,
)
from ptool_framework.critic import TraceCritic, CriticVerdict
from ptool_framework.repair import RepairAgent
from ptool_framework.model_selector import (
    ModelSelector,
    TaskComplexity,
)


def create_bmi_trace(weight=70, height=1.75, complete=True):
    """Create a BMI calculation trace."""
    trace = WorkflowTrace(goal=f"Calculate BMI for patient (weight={weight}kg, height={height}m)")

    # Step 1: Identify calculator
    trace.add_step(
        "identify_calculator",
        {"text": f"Calculate BMI for patient with weight {weight}kg, height {height}m"},
        goal="Identify calculator type"
    )

    # Step 2: Extract values
    trace.add_step(
        "extract_values",
        {"text": f"weight {weight}kg, height {height}m"},
        goal="Extract numeric values"
    )

    # Step 3: Calculate BMI
    trace.add_step(
        "calculate_bmi",
        {"weight": weight, "height": height},
        goal="Calculate BMI"
    )

    # Step 4: Format result
    bmi = weight / (height ** 2)
    trace.add_step(
        "format_result",
        {"bmi": round(bmi, 2)},
        goal="Format output"
    )

    # Mark completion status
    for i, step in enumerate(trace.steps):
        if complete or i < 2:
            step.status = StepStatus.COMPLETED

            # Add results
            if i == 0:
                step.result = {"calculator": "bmi"}
            elif i == 1:
                step.result = {"weight": weight, "height": height}
            elif i == 2:
                step.result = {"bmi": round(bmi, 2)}
            elif i == 3:
                step.result = {"formatted": f"BMI: {round(bmi, 2)} kg/m²"}
        else:
            step.status = StepStatus.FAILED
            step.error = "Calculation failed"

    return trace


def create_incomplete_bmi_trace():
    """Create an incomplete BMI trace (missing extraction)."""
    trace = WorkflowTrace(goal="Calculate BMI")

    # Skip identification and extraction - go straight to calculation
    trace.add_step(
        "calculate_bmi",
        {"weight": 70},  # Missing height!
        goal="Calculate BMI"
    )

    trace.steps[0].status = StepStatus.FAILED
    trace.steps[0].error = "Missing height parameter"

    return trace


def demo_known_calculators():
    """Show available medical calculators."""
    print("=" * 60)
    print("Demo 1: Known Medical Calculators")
    print("=" * 60)

    print("\nAvailable calculators:")
    for calc_id, info in KNOWN_CALCULATORS.items():
        print(f"\n  {calc_id}:")
        print(f"    Name: {info['name']}")
        print(f"    Units: {info.get('units', 'N/A')}")
        print(f"    Inputs: {info.get('inputs', [])}")
        print(f"    Valid range: {info.get('valid_range', 'N/A')}")
    print()


def demo_medcalc_audits():
    """Demo MedCalc-specific audits."""
    print("=" * 60)
    print("Demo 2: MedCalc Audits")
    print("=" * 60)

    # Create audits
    audits = [
        MedCalcStructureAudit(),
        MedCalcOutputValidationAudit(),
        MedCalcBMIAudit(),
    ]

    runner = AuditRunner(audits=audits)

    # Test good trace
    good_trace = create_bmi_trace(weight=70, height=1.75, complete=True)
    print("\nGood BMI trace:")
    report = runner.audit_trace(good_trace)
    print(f"  Result: {report.result.value}")
    print(f"  Passed checks: {len(report.passed_checks)}")

    # Test incomplete trace
    bad_trace = create_incomplete_bmi_trace()
    print("\nIncomplete BMI trace:")
    report = runner.audit_trace(bad_trace)
    print(f"  Result: {report.result.value}")
    print(f"  Violations: {len(report.violations)}")
    for v in report.violations:
        print(f"    - {v.rule_name}: {v.message}")
    print()


def demo_typicality_for_medcalc():
    """Demo typicality model for MedCalc patterns."""
    print("=" * 60)
    print("Demo 3: Typicality Model for MedCalc")
    print("=" * 60)

    # Training patterns from successful MedCalc traces
    training_patterns = [
        [START_TOKEN, "identify", "extract", "calculate", "format", END_TOKEN],
        [START_TOKEN, "identify", "extract", "validate", "calculate", "format", END_TOKEN],
        [START_TOKEN, "identify", "extract", "calculate", END_TOKEN],
        [START_TOKEN, "identify", "extract", "validate", "calculate", END_TOKEN],
        [START_TOKEN, "identify", "extract", "calculate", "format", END_TOKEN],
    ] * 10  # Repeat for better statistics

    model = BigramModel()
    model.fit(training_patterns)

    # Test patterns
    test_cases = [
        ("Typical pattern", [START_TOKEN, "identify", "extract", "calculate", "format", END_TOKEN]),
        ("Missing identify", [START_TOKEN, "extract", "calculate", END_TOKEN]),
        ("Wrong order", [START_TOKEN, "calculate", "extract", END_TOKEN]),
    ]

    print("\nPattern typicality scores:")
    for name, pattern in test_cases:
        score = model.score(pattern)
        steps = pattern[1:-1]
        print(f"\n  {name}:")
        print(f"    Steps: {steps}")
        print(f"    Score: {score:.4f}")
    print()


def demo_critic_for_medcalc():
    """Demo critic system for MedCalc."""
    print("=" * 60)
    print("Demo 4: Critic for MedCalc")
    print("=" * 60)

    # Create critic with MedCalc audits
    audits = [
        NonEmptyTraceAudit(),
        NoFailedStepsAudit(),
        MedCalcStructureAudit(),
        MedCalcBMIAudit(),
    ]

    critic = TraceCritic(
        audits=audits,
        accept_threshold=0.8,
        repair_threshold=0.4,
    )

    # Evaluate traces
    traces = [
        ("Good BMI trace", create_bmi_trace(70, 1.75, True)),
        ("Failed BMI trace", create_bmi_trace(70, 1.75, False)),
        ("Incomplete trace", create_incomplete_bmi_trace()),
    ]

    print("\nCritic evaluations:")
    for name, trace in traces:
        evaluation = critic.evaluate(trace, goal=trace.goal)
        print(f"\n  {name}:")
        print(f"    Verdict: {evaluation.verdict.value}")
        print(f"    Confidence: {evaluation.confidence:.2f}")
        print(f"    Completeness: {evaluation.completeness_score:.2f}")
        print(f"    Correctness: {evaluation.correctness_score:.2f}")
    print()


def demo_repair_for_medcalc():
    """Demo repair agent for MedCalc."""
    print("=" * 60)
    print("Demo 5: Repair for MedCalc")
    print("=" * 60)

    audits = [
        NonEmptyTraceAudit(),
        NoFailedStepsAudit(),
        MedCalcStructureAudit(),
    ]

    critic = TraceCritic(audits=audits)
    repair_agent = RepairAgent(critic=critic, max_attempts=3)

    # Try to repair incomplete trace
    trace = create_incomplete_bmi_trace()
    evaluation = critic.evaluate(trace, goal=trace.goal)

    print(f"\nOriginal trace:")
    print(f"  Steps: {[s.ptool_name for s in trace.steps]}")
    print(f"  Verdict: {evaluation.verdict.value}")

    if evaluation.verdict == CriticVerdict.REPAIR_NEEDED:
        print(f"\nAttempting repair...")
        result = repair_agent.repair(trace, evaluation, trace.goal)
        print(f"\nRepair result:")
        print(f"  Success: {result.success}")
        print(f"  Iterations: {result.iterations}")
        print(f"  Actions: {len(result.actions_taken)}")

        for action in result.actions_taken:
            print(f"    - {action.action_type.value}: {action.reason}")
    print()


def demo_model_selection_for_medcalc():
    """Demo model selection for MedCalc tasks."""
    print("=" * 60)
    print("Demo 6: Model Selection for MedCalc")
    print("=" * 60)

    from dataclasses import dataclass

    @dataclass
    class MockSpec:
        name: str
        docstring: str = ""

    selector = ModelSelector()

    # Different MedCalc tasks
    tasks = [
        MockSpec("identify_calculator", "Identify the medical calculator type"),
        MockSpec("extract_values", "Extract numeric values from text"),
        MockSpec("calculate_bmi", "Calculate BMI from weight and height"),
        MockSpec("analyze_complex_case", "Analyze complex multi-step medical reasoning"),
    ]

    print("\nModel selection for MedCalc tasks:")
    for spec in tasks:
        result = selector.select_with_details(spec, {"input": "test"})
        print(f"\n  {spec.name}:")
        print(f"    Model: {result.selected_model}")
        print(f"    Confidence: {result.confidence:.2f}")
    print()


def demo_full_pipeline():
    """Demo complete MedCalc pipeline."""
    print("=" * 60)
    print("Demo 7: Full MedCalc Pipeline")
    print("=" * 60)

    # Setup components
    audits = [
        NonEmptyTraceAudit(),
        NoFailedStepsAudit(),
        MedCalcStructureAudit(),
        MedCalcBMIAudit(),
    ]

    critic = TraceCritic(
        audits=audits,
        accept_threshold=0.8,
    )

    repair_agent = RepairAgent(
        critic=critic,
        max_attempts=2,
    )

    selector = ModelSelector()

    def process_bmi_request(weight, height):
        """Complete BMI processing pipeline."""
        print(f"\n{'='*40}")
        print(f"Processing: BMI for weight={weight}kg, height={height}m")
        print(f"{'='*40}")

        # Step 1: Create trace
        trace = create_bmi_trace(weight, height, complete=True)
        print(f"1. Created trace with {len(trace.steps)} steps")

        # Step 2: Select model (for demo)
        from dataclasses import dataclass
        @dataclass
        class Spec:
            name: str = "calculate_bmi"
            docstring: str = ""

        model = selector.select(Spec(), {"weight": weight, "height": height})
        print(f"2. Selected model: {model}")

        # Step 3: Evaluate trace
        evaluation = critic.evaluate(trace, goal=trace.goal)
        print(f"3. Evaluation: {evaluation.verdict.value} (confidence: {evaluation.confidence:.2f})")

        # Step 4: Handle based on verdict
        if evaluation.is_acceptable:
            print(f"4. Trace ACCEPTED")
            # Get result
            last_step = trace.steps[-1]
            result = last_step.result
            print(f"   Result: {result}")
            return trace, result

        elif evaluation.needs_repair:
            print(f"4. Attempting repair...")
            repair_result = repair_agent.repair(trace, evaluation, trace.goal)
            if repair_result.success:
                print(f"   Repair SUCCESS in {repair_result.iterations} iterations")
                return repair_result.repaired_trace, None
            else:
                print(f"   Repair FAILED: {repair_result.error}")
                return None, None

        else:
            print(f"4. Trace REJECTED")
            return None, None

    # Process several BMI calculations
    test_cases = [
        (70, 1.75),   # Normal BMI
        (100, 1.80),  # High BMI
        (55, 1.65),   # Normal BMI
    ]

    results = []
    for weight, height in test_cases:
        trace, result = process_bmi_request(weight, height)
        if trace and result:
            bmi = weight / (height ** 2)
            results.append((weight, height, round(bmi, 2)))

    print(f"\n{'='*60}")
    print("Pipeline Summary")
    print(f"{'='*60}")
    print(f"\nProcessed {len(results)} BMI calculations:")
    for weight, height, bmi in results:
        print(f"  Weight: {weight}kg, Height: {height}m -> BMI: {bmi}")
    print()


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("FULL MEDCALC PIPELINE DEMO")
    print("=" * 60 + "\n")

    demo_known_calculators()
    demo_medcalc_audits()
    demo_typicality_for_medcalc()
    demo_critic_for_medcalc()
    demo_repair_for_medcalc()
    demo_model_selection_for_medcalc()
    demo_full_pipeline()

    print("=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
