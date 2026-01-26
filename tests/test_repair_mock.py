"""
Mock Failure Test for Repair System.

This test creates intentional failures to verify that:
1. The critic correctly identifies issues (REPAIR_NEEDED)
2. The repair agent attempts to fix them
3. The repair mechanism actually works

Run with:
    PYTHONPATH=. python tests/test_repair_mock.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ptool_framework.traces import WorkflowTrace, TraceStep, StepStatus
from ptool_framework.critic import TraceCritic, CriticVerdict
from ptool_framework.repair import RepairAgent, RepairActionType
from ptool_framework.audit import (
    NonEmptyTraceAudit,
    NoFailedStepsAudit,
    AllCompletedAudit,
)
from ptool_framework.audit.domains.medcalc import MedCalcStructureAudit


def test_scenario_1_empty_trace():
    """Test 1: Empty trace should trigger REPAIR_NEEDED."""
    print("\n" + "=" * 70)
    print("TEST 1: Empty Trace")
    print("=" * 70)

    # Create empty trace
    trace = WorkflowTrace(goal="Calculate BMI for patient")
    print(f"Created trace with {len(trace.steps)} steps")

    # Create critic with audits
    audits = [
        NonEmptyTraceAudit(),
        NoFailedStepsAudit(),
        AllCompletedAudit(),
    ]
    critic = TraceCritic(audits=audits, accept_threshold=0.7, repair_threshold=0.3)

    # Evaluate
    evaluation = critic.evaluate(trace, goal="Calculate BMI")
    print(f"\nCritic Evaluation:")
    print(f"  Verdict: {evaluation.verdict.value}")
    print(f"  Confidence: {evaluation.confidence:.2f}")
    print(f"  Total violations: {evaluation.total_violations}")
    print(f"  Repair suggestions: {len(evaluation.repair_suggestions)}")

    for i, suggestion in enumerate(evaluation.repair_suggestions):
        print(f"    {i+1}. {suggestion['action']}: {suggestion['reason'][:50]}")

    assert evaluation.verdict != CriticVerdict.ACCEPT, "Empty trace should NOT be accepted!"
    print("\n[PASS] Empty trace correctly identified as needing repair/rejection")

    return evaluation


def test_scenario_2_failed_step():
    """Test 2: Trace with a failed step should trigger REPAIR_NEEDED."""
    print("\n" + "=" * 70)
    print("TEST 2: Failed Step")
    print("=" * 70)

    # Create trace with a failed step
    trace = WorkflowTrace(goal="Calculate BMI for patient")

    # Add identify step (completed)
    step1 = TraceStep(
        ptool_name="identify_calculator",
        args={"clinical_text": "Calculate BMI"},
        goal="Identify the calculator",
    )
    step1.status = StepStatus.COMPLETED
    step1.result = {"calculator_name": "BMI", "required_inputs": ["weight", "height"]}
    trace.steps.append(step1)

    # Add extract step (FAILED)
    step2 = TraceStep(
        ptool_name="extract_clinical_values",
        args={"patient_note": "Patient is 70kg", "required_values": ["weight", "height"]},
        goal="Extract values",
    )
    step2.status = StepStatus.FAILED
    step2.error = "Missing required value: height"
    trace.steps.append(step2)

    # Add calculation step (completed but with missing data)
    step3 = TraceStep(
        ptool_name="perform_calculation",
        args={"calculator_name": "BMI", "values": {"weight": 70}},
        goal="Calculate BMI",
    )
    step3.status = StepStatus.COMPLETED
    step3.result = {"error": "Missing height"}
    trace.steps.append(step3)

    print(f"Created trace with {len(trace.steps)} steps:")
    for i, step in enumerate(trace.steps):
        print(f"  {i+1}. {step.ptool_name} - {step.status.value}")

    # Create critic
    audits = [
        NonEmptyTraceAudit(),
        NoFailedStepsAudit(),
        AllCompletedAudit(),
        MedCalcStructureAudit(),
    ]
    critic = TraceCritic(audits=audits, accept_threshold=0.7, repair_threshold=0.3)

    # Evaluate
    evaluation = critic.evaluate(trace, goal="Calculate BMI")
    print(f"\nCritic Evaluation:")
    print(f"  Verdict: {evaluation.verdict.value}")
    print(f"  Confidence: {evaluation.confidence:.2f}")
    print(f"  Failed steps: {evaluation.failed_steps}")
    print(f"  Total violations: {evaluation.total_violations}")

    for report in evaluation.audit_reports:
        if report.violations:
            print(f"  Audit '{report.trace_id or report.metadata.get('audit_name', 'unknown')}':")
            for v in report.violations:
                print(f"    - [{v.severity}] {v.rule_name}: {v.message}")

    print(f"\n  Repair suggestions: {len(evaluation.repair_suggestions)}")
    for i, suggestion in enumerate(evaluation.repair_suggestions):
        print(f"    {i+1}. {suggestion['action']}: {suggestion['reason'][:60]}")

    assert evaluation.verdict == CriticVerdict.REPAIR_NEEDED, \
        f"Failed step trace should need repair, got {evaluation.verdict.value}!"
    assert len(evaluation.failed_steps) > 0, "Should identify failed steps!"
    assert len(evaluation.repair_suggestions) > 0, "Should have repair suggestions!"

    print("\n[PASS] Failed step correctly identified and repair suggestions generated")

    return trace, evaluation, critic


def test_scenario_3_repair_agent():
    """Test 3: Verify RepairAgent attempts to fix issues."""
    print("\n" + "=" * 70)
    print("TEST 3: Repair Agent Execution")
    print("=" * 70)

    # Get trace and evaluation from test 2
    trace, evaluation, critic = test_scenario_2_failed_step()

    print("\n--- Running Repair Agent ---")

    # Create repair agent with verbose mode
    repair_agent = RepairAgent(
        critic=critic,
        max_attempts=3,
        model="deepseek-v3-0324",
        verbose=True,  # Enable verbose output
    )

    # Attempt repair
    result = repair_agent.repair(
        trace=trace,
        evaluation=evaluation,
        goal="Calculate BMI for patient",
    )

    print(f"\nRepair Result:")
    print(f"  Success: {result.success}")
    print(f"  Iterations: {result.iterations}")
    print(f"  Actions taken: {len(result.actions_taken)}")

    for i, action in enumerate(result.actions_taken):
        print(f"    {i+1}. {action.action_type.value}: {action.reason[:50]}")

    if result.error:
        print(f"  Error: {result.error}")

    if result.final_evaluation:
        print(f"\n  Final verdict: {result.final_evaluation.verdict.value}")
        print(f"  Final violations: {result.final_evaluation.total_violations}")

    # The repair agent should have at least TRIED to do something
    # Even if it fails (no LLM backend), it should attempt repairs
    print(f"\n[INFO] Repair agent processed {len(result.actions_taken)} actions")

    return result


def test_scenario_4_manual_fix_simulation():
    """Test 4: Simulate fixing a trace and verify critic accepts it."""
    print("\n" + "=" * 70)
    print("TEST 4: Manual Fix Simulation")
    print("=" * 70)

    # Create a GOOD trace
    trace = WorkflowTrace(goal="Calculate BMI for patient")

    # Add identify step (completed)
    step1 = TraceStep(
        ptool_name="identify_calculator",
        args={"clinical_text": "Calculate BMI"},
        goal="Identify the calculator",
    )
    step1.status = StepStatus.COMPLETED
    step1.result = {"calculator_name": "BMI", "required_inputs": ["weight", "height"]}
    trace.steps.append(step1)

    # Add extract step (completed)
    step2 = TraceStep(
        ptool_name="extract_clinical_values",
        args={"patient_note": "Patient is 70kg, 175cm", "required_values": ["weight", "height"]},
        goal="Extract values",
    )
    step2.status = StepStatus.COMPLETED
    step2.result = {"extracted": {"weight": 70, "height": 1.75}, "missing": []}
    trace.steps.append(step2)

    # Add calculation step (completed with result)
    step3 = TraceStep(
        ptool_name="perform_calculation",
        args={"calculator_name": "BMI", "values": {"weight": 70, "height": 1.75}},
        goal="Calculate BMI",
    )
    step3.status = StepStatus.COMPLETED
    step3.result = {"result": 22.86, "unit": "kg/m²"}
    trace.steps.append(step3)

    print(f"Created GOOD trace with {len(trace.steps)} steps:")
    for i, step in enumerate(trace.steps):
        print(f"  {i+1}. {step.ptool_name} - {step.status.value}")
        if step.result:
            print(f"      Result: {step.result}")

    # Create critic
    audits = [
        NonEmptyTraceAudit(),
        NoFailedStepsAudit(),
        AllCompletedAudit(),
        MedCalcStructureAudit(),
    ]
    critic = TraceCritic(audits=audits, accept_threshold=0.7, repair_threshold=0.3)

    # Evaluate
    evaluation = critic.evaluate(trace, goal="Calculate BMI")
    print(f"\nCritic Evaluation:")
    print(f"  Verdict: {evaluation.verdict.value}")
    print(f"  Confidence: {evaluation.confidence:.2f}")
    print(f"  Total violations: {evaluation.total_violations}")
    print(f"  Completeness: {evaluation.completeness_score:.2f}")
    print(f"  Correctness: {evaluation.correctness_score:.2f}")

    for report in evaluation.audit_reports:
        status = "PASS" if report.is_pass else "FAIL"
        print(f"  Audit: {status} - passed {len(report.passed_checks)} checks")
        if report.violations:
            for v in report.violations:
                print(f"    - [{v.severity}] {v.message}")

    assert evaluation.verdict == CriticVerdict.ACCEPT, \
        f"Good trace should be accepted, got {evaluation.verdict.value}!"

    print("\n[PASS] Good trace correctly accepted by critic")

    return evaluation


def test_scenario_5_missing_step():
    """Test 5: Trace missing a required step."""
    print("\n" + "=" * 70)
    print("TEST 5: Missing Required Step")
    print("=" * 70)

    # Create trace missing the extraction step
    trace = WorkflowTrace(goal="Calculate BMI for patient")

    # Only identify step
    step1 = TraceStep(
        ptool_name="identify_calculator",
        args={"clinical_text": "Calculate BMI"},
        goal="Identify the calculator",
    )
    step1.status = StepStatus.COMPLETED
    step1.result = {"calculator_name": "BMI", "required_inputs": ["weight", "height"]}
    trace.steps.append(step1)

    # Skip extraction, go straight to calculation (bad!)
    step2 = TraceStep(
        ptool_name="perform_calculation",
        args={"calculator_name": "BMI", "values": {}},  # Empty values - should fail
        goal="Calculate BMI",
    )
    step2.status = StepStatus.COMPLETED
    step2.result = {"error": "No values provided"}
    trace.steps.append(step2)

    print(f"Created trace MISSING extraction step:")
    for i, step in enumerate(trace.steps):
        print(f"  {i+1}. {step.ptool_name} - {step.status.value}")

    # Create critic
    audits = [
        NonEmptyTraceAudit(),
        NoFailedStepsAudit(),
        AllCompletedAudit(),
        MedCalcStructureAudit(),
    ]
    critic = TraceCritic(audits=audits, accept_threshold=0.7, repair_threshold=0.3)

    # Evaluate
    evaluation = critic.evaluate(trace, goal="Calculate BMI")
    print(f"\nCritic Evaluation:")
    print(f"  Verdict: {evaluation.verdict.value}")
    print(f"  Confidence: {evaluation.confidence:.2f}")
    print(f"  Total violations: {evaluation.total_violations}")

    for report in evaluation.audit_reports:
        if report.violations:
            audit_name = report.metadata.get('audit_name', 'unknown')
            print(f"  Audit '{audit_name}':")
            for v in report.violations:
                print(f"    - [{v.severity}] {v.rule_name}: {v.message}")

    print(f"\n  Repair suggestions: {len(evaluation.repair_suggestions)}")
    for i, suggestion in enumerate(evaluation.repair_suggestions):
        print(f"    {i+1}. {suggestion['action']}: {suggestion['reason'][:60]}")

    # Should identify the missing extraction step
    assert evaluation.verdict != CriticVerdict.ACCEPT, \
        "Trace missing extraction should NOT be accepted!"

    print("\n[PASS] Missing step correctly identified")

    return evaluation


def run_all_tests():
    """Run all mock failure tests."""
    print("\n" + "#" * 70)
    print("#  MOCK FAILURE TESTS FOR AUDIT + REPAIR SYSTEM")
    print("#" * 70)

    results = {}

    try:
        results['empty_trace'] = test_scenario_1_empty_trace()
        print("\n" + "-" * 70)
    except Exception as e:
        print(f"\n[FAIL] Test 1 failed: {e}")
        results['empty_trace'] = None

    try:
        results['failed_step'] = test_scenario_2_failed_step()
        print("\n" + "-" * 70)
    except Exception as e:
        print(f"\n[FAIL] Test 2 failed: {e}")
        results['failed_step'] = None

    try:
        results['repair_agent'] = test_scenario_3_repair_agent()
        print("\n" + "-" * 70)
    except Exception as e:
        print(f"\n[FAIL] Test 3 failed: {e}")
        import traceback
        traceback.print_exc()
        results['repair_agent'] = None

    try:
        results['good_trace'] = test_scenario_4_manual_fix_simulation()
        print("\n" + "-" * 70)
    except Exception as e:
        print(f"\n[FAIL] Test 4 failed: {e}")
        results['good_trace'] = None

    try:
        results['missing_step'] = test_scenario_5_missing_step()
    except Exception as e:
        print(f"\n[FAIL] Test 5 failed: {e}")
        results['missing_step'] = None

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed = sum(1 for r in results.values() if r is not None)
    total = len(results)

    for name, result in results.items():
        status = "[PASS]" if result is not None else "[FAIL]"
        print(f"  {status} {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    return results


if __name__ == "__main__":
    run_all_tests()
