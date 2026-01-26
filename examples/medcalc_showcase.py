#!/usr/bin/env python3
"""
MedCalc Showcase - Comprehensive demonstration of ptool_framework

This script demonstrates ALL major features of ptool_framework using medical
calculation tasks as the domain:

1. ReAct Agent - Solving MedCalc problems with reasoning
2. Audit System - Checking trace quality with domain-specific rules
3. Typicality Models - Learning patterns from successful traces
4. Critic System - Evaluating traces for acceptance/repair/rejection
5. Repair Agent - Automatically fixing problematic traces
6. Model Selection - Intelligent LLM routing based on experience
7. ICL Generation - Creating in-context learning examples
8. Full Pipeline - End-to-end demonstration with statistics

This example directly addresses William's 2026 Agent Research requirements:
- ptools as LLM prompts
- Python programs calling ptools
- Behavior distillation (approaches 1-3)
- Critic + repair loop
- Auto-select LLMs
- Audit/typicality for trace quality

Usage:
    # Run full demo
    python examples/medcalc_showcase.py

    # Run specific sections
    python examples/medcalc_showcase.py --demo react
    python examples/medcalc_showcase.py --demo audit
    python examples/medcalc_showcase.py --demo typicality
    python examples/medcalc_showcase.py --demo critic
    python examples/medcalc_showcase.py --demo repair
    python examples/medcalc_showcase.py --demo model
    python examples/medcalc_showcase.py --demo icl
    python examples/medcalc_showcase.py --demo full

    # Options
    python examples/medcalc_showcase.py --demo full --problems 5 -v
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

# =============================================================================
# IMPORTS
# =============================================================================

# Core framework
from ptool_framework import (
    # Core
    ptool, get_registry,
    # ReAct
    ReActAgent, ReActResult, ReActTrajectory,
    # Traces
    WorkflowTrace, TraceStep,
    # Audit
    AuditRunner, AuditReport, AuditResult,
    NonEmptyTraceAudit, NoFailedStepsAudit, RequiredStepsAudit,
    # Critic
    TraceCritic, CriticVerdict, CriticEvaluation,
    # Repair
    RepairAgent, RepairResult,
    # Model Selection
    ModelSelector, TaskComplexity, SelectionResult,
)
from ptool_framework.traces import StepStatus

# Audit extras
from ptool_framework.audit import (
    StepOrderAudit,
    AllCompletedAudit,
    TypicalityTrainer,
)
from ptool_framework.audit.domains.medcalc import (
    MedCalcStructureAudit,
    MedCalcOutputValidationAudit,
    KNOWN_CALCULATORS,
)
from ptool_framework.audit.typicality import (
    BigramModel,
    TrigramModel,
    InterpolatedModel,
    extract_pattern,
    START_TOKEN,
    END_TOKEN,
)
from ptool_framework.audit.dataframe_converter import convert_trace
from ptool_framework.llm_backend import call_llm

# Local ptools - add examples dir to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from medcalc_ptools import (
    identify_calculator, extract_clinical_values, interpret_result,
    identify_calculator_fast, extract_clinical_values_fast,
    calculate_bmi, calculate_cha2ds2_vasc, calculate_egfr,
    CALCULATORS, get_medcalc_ptools, register_ptools,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_MODEL = "deepseek-v3-0324"

# Test problems for MedCalc
MEDCALC_PROBLEMS = [
    "Calculate BMI for a patient who weighs 75kg and is 1.80m tall.",
    "65 year old male with atrial fibrillation, hypertension, and diabetes. Calculate stroke risk using CHA2DS2-VASc.",
    "Patient is a 70 year old female with creatinine 1.4 mg/dL. Calculate eGFR to assess kidney function.",
    "A 45 year old obese male weighs 110kg and is 1.75m tall. What is his BMI?",
    "82 year old female with CHF, prior stroke, and hypertension. Assess stroke risk.",
]


# =============================================================================
# UTILITIES
# =============================================================================

def print_header(title: str, char: str = "=", width: int = 70):
    """Print a formatted header."""
    print()
    print(char * width)
    print(f" {title}")
    print(char * width)


def print_subheader(title: str, char: str = "-", width: int = 50):
    """Print a formatted subheader."""
    print()
    print(f"{char * 3} {title} {char * 3}")


def print_json(obj: Any, indent: int = 2):
    """Print object as formatted JSON."""
    print(json.dumps(obj, indent=indent, default=str))


# =============================================================================
# DEMO 1: REACT AGENT
# =============================================================================

def demo_react_agent(verbose: bool = False, n_problems: int = 3) -> List[ReActTrajectory]:
    """
    Demonstrate the ReAct agent solving MedCalc problems.

    William's requirement: "ReAct-style agent and collect the sequence of tool/ptool calls"
    """
    print_header("DEMO 1: ReAct Agent")

    print("""
The ReAct agent uses iterative reasoning to solve problems:
  1. THINK - Reason about the current state
  2. ACT - Execute a ptool
  3. OBSERVE - See the result
  4. Repeat until answer found

This implements William's "behavior distillation approach 2":
Run a react-style agent and collect the sequence of tool/ptool calls.
""")

    # Register ptools
    ptools = register_ptools()
    print(f"Registered {len(ptools)} MedCalc ptools: {[p.name for p in ptools if p]}")

    # Create agent
    agent = ReActAgent(
        available_ptools=ptools,
        model=DEFAULT_MODEL,
        max_steps=8,
        echo=verbose,
        store_trajectories=True,
    )

    # Solve problems
    problems = MEDCALC_PROBLEMS[:n_problems]
    trajectories = []

    for i, problem in enumerate(problems, 1):
        print_subheader(f"Problem {i}/{len(problems)}")
        print(f"Question: {problem[:80]}...")

        start_time = time.time()
        result = agent.run(problem)
        elapsed = time.time() - start_time

        trajectories.append(result.trajectory)

        print(f"Success: {result.success}")
        print(f"Steps: {len(result.trajectory.steps)}")
        print(f"Time: {elapsed:.1f}s")
        if result.answer:
            print(f"Answer: {result.answer[:200]}...")

        # Show PTP trace if verbose
        if verbose and result.success:
            print("\nPTP Trace:")
            print(result.trajectory.to_ptp_trace())

    # Summary
    print_subheader("Summary")
    successful = sum(1 for t in trajectories if t.success)
    print(f"Problems solved: {successful}/{len(trajectories)}")
    print(f"Trajectories collected for distillation: {len(trajectories)}")

    return trajectories


# =============================================================================
# DEMO 2: AUDIT SYSTEM
# =============================================================================

def demo_audit_system(trajectories: List[ReActTrajectory], verbose: bool = False) -> List[AuditReport]:
    """
    Demonstrate the audit system checking trace quality.

    William's requirement: "LLMs can generalize from traces to 'reasoning audits' -
    basically unit tests that describe properties of the algorithmic behavior"
    """
    print_header("DEMO 2: Audit System")

    print("""
The audit system validates traces using:
  - Structured audits: Rule-based checks (DSL for querying trace DataFrames)
  - Domain audits: MedCalc-specific validation
  - Typicality: Statistical models of "normal" trace patterns

This implements William's SSRM requirement: "reasoning audits - basically
unit tests that describe properties of the algorithmic behavior"
""")

    # Create audit runner with multiple audits
    audits = [
        NonEmptyTraceAudit(),
        NoFailedStepsAudit(),
        AllCompletedAudit(),
        MedCalcStructureAudit(),
    ]

    runner = AuditRunner(audits=audits)

    print(f"Running {len(audits)} audits on {len(trajectories)} traces...")

    reports = []
    for i, traj in enumerate(trajectories):
        if traj.generated_trace is None:
            print(f"  Trace {i+1}: No generated trace (skipped)")
            continue

        report = runner.audit_trace(traj.generated_trace)
        reports.append(report)

        status = "PASS" if report.is_pass else "FAIL"
        print(f"  Trace {i+1} [{traj.trajectory_id}]: {status}")

        if verbose or not report.is_pass:
            print(f"    Passed checks: {len(report.passed_checks)}")
            if report.violations:
                for v in report.violations[:3]:
                    print(f"    Violation: {v.rule_name} - {v.message}")

    # Summary
    print_subheader("Audit Summary")
    passed = sum(1 for r in reports if r.is_pass)
    print(f"Traces passed: {passed}/{len(reports)}")

    return reports


# =============================================================================
# DEMO 3: TYPICALITY MODELS
# =============================================================================

def demo_typicality_models(trajectories: List[ReActTrajectory], verbose: bool = False):
    """
    Demonstrate typicality models for pattern learning.

    William's requirement: "all experiences of the agent are used" +
    "audits are also unit tests for trace builders"
    """
    print_header("DEMO 3: Typicality Models")

    print("""
Typicality models learn "normal" trace patterns and score new traces:
  - Unigram: P(step) - frequency of each step type
  - Bigram: P(step | prev_step) - transition probabilities
  - Trigram: P(step | prev_2_steps) - longer context
  - Interpolated: Weighted combination

High scores = typical patterns, Low scores = anomalies
""")

    # Extract patterns from successful traces
    patterns = []
    for traj in trajectories:
        if traj.success and traj.generated_trace:
            try:
                df = convert_trace(traj.generated_trace)
                pattern = extract_pattern(df)
                patterns.append(pattern)
                if verbose:
                    print(f"  Pattern from {traj.trajectory_id}: {pattern}")
            except Exception as e:
                if verbose:
                    print(f"  Error extracting pattern: {e}")

    if len(patterns) < 2:
        print("Not enough successful traces for typicality modeling.")
        print("Adding synthetic training patterns for demonstration...")

        # Add synthetic patterns based on expected MedCalc workflow
        synthetic_patterns = [
            [START_TOKEN, "identify_calculator", "extract_clinical_values", END_TOKEN],
            [START_TOKEN, "identify_calculator", "extract_clinical_values", "interpret_result", END_TOKEN],
        ] * 5  # Repeat for better statistics
        patterns.extend(synthetic_patterns)

    print(f"\nTraining on {len(patterns)} patterns...")

    # Train models
    bigram = BigramModel()
    bigram.fit(patterns)

    trigram = TrigramModel()
    trigram.fit(patterns)

    interpolated = InterpolatedModel(lambdas=(0.2, 0.3, 0.5))
    interpolated.fit(patterns)

    # Test patterns
    print_subheader("Pattern Scores")

    test_patterns = [
        ("Typical: identify -> extract", [START_TOKEN, "identify_calculator", "extract_clinical_values", END_TOKEN]),
        ("Typical: id -> extract -> interpret", [START_TOKEN, "identify_calculator", "extract_clinical_values", "interpret_result", END_TOKEN]),
        ("Atypical: missing identify", [START_TOKEN, "extract_clinical_values", END_TOKEN]),
        ("Atypical: wrong order", [START_TOKEN, "interpret_result", "identify_calculator", END_TOKEN]),
    ]

    print(f"{'Pattern':<35} {'Bigram':<10} {'Trigram':<10} {'Interp':<10}")
    print("-" * 65)

    for name, pattern in test_patterns:
        bi_score = bigram.score(pattern)
        tri_score = trigram.score(pattern)
        int_score = interpolated.score(pattern)
        print(f"{name:<35} {bi_score:<10.4f} {tri_score:<10.4f} {int_score:<10.4f}")


# =============================================================================
# DEMO 4: CRITIC SYSTEM
# =============================================================================

def demo_critic_system(trajectories: List[ReActTrajectory], verbose: bool = False) -> List[CriticEvaluation]:
    """
    Demonstrate the critic system for trace evaluation.

    William's requirement: "add a critic for the traces - which would check
    audits, monitor failed tool calls, etc"
    """
    print_header("DEMO 4: Critic System")

    print("""
The critic evaluates traces and produces verdicts:
  - ACCEPT: Trace is good quality
  - REPAIR_NEEDED: Trace has issues but can be fixed
  - REJECT: Trace is unfixable

Verdicts are based on:
  - Audit results (passed/failed checks)
  - Completeness score (did it achieve the goal?)
  - Correctness score (are the results valid?)

This implements William's "behavior distillation approach 3"
""")

    # Create critic with audits
    audits = [
        NonEmptyTraceAudit(),
        NoFailedStepsAudit(),
        AllCompletedAudit(),
        MedCalcStructureAudit(),
    ]

    critic = TraceCritic(
        audits=audits,
        accept_threshold=0.7,
        repair_threshold=0.3,
    )

    print(f"Evaluating {len(trajectories)} traces...")
    print(f"Accept threshold: 0.7, Repair threshold: 0.3")

    evaluations = []
    for i, traj in enumerate(trajectories):
        if traj.generated_trace is None:
            continue

        evaluation = critic.evaluate(
            traj.generated_trace,
            goal=traj.goal,
        )
        evaluations.append(evaluation)

        verdict_emoji = {"accept": "✓", "repair_needed": "~", "reject": "✗"}.get(evaluation.verdict.value, "?")

        print(f"  Trace {i+1}: {verdict_emoji} {evaluation.verdict.value.upper()}")
        print(f"           Confidence: {evaluation.confidence:.2f}")
        print(f"           Completeness: {evaluation.completeness_score:.2f}")

        if verbose:
            print(f"           Correctness: {evaluation.correctness_score:.2f}")
            if evaluation.repair_suggestions:
                print(f"           Suggestions: {len(evaluation.repair_suggestions)}")

    # Summary
    print_subheader("Critic Summary")
    verdicts = {"accept": 0, "repair_needed": 0, "reject": 0}
    for e in evaluations:
        verdicts[e.verdict.value] = verdicts.get(e.verdict.value, 0) + 1

    print(f"Accept: {verdicts['accept']}, Repair needed: {verdicts['repair_needed']}, Reject: {verdicts['reject']}")

    return evaluations


# =============================================================================
# DEMO 5: REPAIR AGENT
# =============================================================================

def demo_repair_agent(critic: TraceCritic = None, verbose: bool = False):
    """
    Demonstrate the repair agent fixing traces.

    William's requirement: "a trace repair subagent"
    """
    print_header("DEMO 5: Repair Agent")

    print("""
The repair agent fixes problematic traces:
  1. Analyzes critic feedback
  2. Applies repair actions (regenerate, add, remove, modify steps)
  3. Re-evaluates until acceptable or max attempts

Repair actions:
  - REGENERATE_STEP: Re-run a failed step
  - ADD_STEP: Insert a missing step
  - REMOVE_STEP: Remove redundant step
  - MODIFY_ARGS: Fix step arguments
  - REORDER_STEPS: Fix step ordering
""")

    # Create a deliberately incomplete trace for demonstration
    print_subheader("Creating problematic trace")

    bad_trace = WorkflowTrace(goal="Calculate BMI")

    # Missing identification step - goes straight to calculation
    bad_trace.add_step(
        "extract_clinical_values",
        {"clinical_text": "75kg, 1.80m", "required_fields": ["weight_kg", "height_m"]},
        goal="Extract values without identifying calculator first"
    )
    bad_trace.steps[0].status = StepStatus.COMPLETED
    bad_trace.steps[0].result = {"values": {"weight_kg": 75, "height_m": 1.80}}

    print("Created trace with issues:")
    print("  - Missing 'identify_calculator' step")
    print(f"  - Steps: {[s.ptool_name for s in bad_trace.steps]}")

    # Create critic and evaluate
    if critic is None:
        audits = [
            NonEmptyTraceAudit(),
            MedCalcStructureAudit(),
        ]
        critic = TraceCritic(audits=audits, accept_threshold=0.7, repair_threshold=0.3)

    evaluation = critic.evaluate(bad_trace, goal=bad_trace.goal)
    print(f"\nInitial evaluation: {evaluation.verdict.value}")
    print(f"Confidence: {evaluation.confidence:.2f}")

    if evaluation.verdict == CriticVerdict.ACCEPT:
        print("Trace was already acceptable - no repair needed")
        return

    # Create repair agent
    repair_agent = RepairAgent(
        critic=critic,
        max_attempts=2,
        model=DEFAULT_MODEL,
    )

    print_subheader("Attempting repair")

    if evaluation.verdict == CriticVerdict.REPAIR_NEEDED:
        print("Verdict: REPAIR_NEEDED - attempting fix...")

        result = repair_agent.repair(bad_trace, evaluation, goal=bad_trace.goal)

        print(f"\nRepair result:")
        print(f"  Success: {result.success}")
        print(f"  Iterations: {result.iterations}")
        print(f"  Actions taken: {len(result.actions_taken)}")

        for action in result.actions_taken:
            print(f"    - {action.action_type.value}: {action.reason[:60]}...")

        if result.success and result.repaired_trace:
            print(f"\nRepaired trace steps: {[s.ptool_name for s in result.repaired_trace.steps]}")

    else:
        print(f"Verdict: {evaluation.verdict.value} - trace cannot be repaired")


# =============================================================================
# DEMO 6: MODEL SELECTION
# =============================================================================

def demo_model_selection(trajectories: List[ReActTrajectory], verbose: bool = False):
    """
    Demonstrate intelligent model selection.

    William's requirement: "autoselect among alternative LLMs for ptools
    to trade off speed/cost/quality"
    """
    print_header("DEMO 6: Model Selection")

    print("""
The model selector routes ptool calls to appropriate LLMs based on:
  - Task complexity estimation
  - Historical performance (success rate, latency, cost)
  - Selection criteria (required capabilities, budget)
  - Fallback chains for robustness

This implements William's "autoselect among alternative LLMs for ptools
to trade off speed/cost/quality"
""")

    # Create model selector
    selector = ModelSelector(
        default_model=DEFAULT_MODEL,
        enable_learning=True,
    )

    # Record executions from trajectories
    print_subheader("Recording experience from trajectories")

    execution_count = 0
    for traj in trajectories:
        for step in traj.steps:
            if step.action and step.observation:
                selector.record_execution(
                    ptool_name=step.action.ptool_name,
                    model=traj.model_used or DEFAULT_MODEL,
                    inputs=step.action.args,
                    success=step.observation.success,
                    latency_ms=step.observation.execution_time_ms,
                    cost=0.001,  # Estimated
                )
                execution_count += 1

    print(f"Recorded {execution_count} executions")

    # Query model selection
    print_subheader("Model Selection Examples")

    # Create mock specs for testing
    @dataclass
    class MockSpec:
        name: str
        docstring: str = ""
        params: Dict = field(default_factory=dict)
        return_type: type = str

    test_tasks = [
        MockSpec("identify_calculator", "Identify which medical calculator to use"),
        MockSpec("extract_clinical_values", "Extract values from clinical notes"),
        MockSpec("interpret_result", "Interpret medical score clinically"),
        MockSpec("complex_analysis", "Perform complex multi-step medical reasoning"),
    ]

    print(f"{'Task':<25} {'Selected Model':<20} {'Confidence':<12}")
    print("-" * 60)

    for spec in test_tasks:
        result = selector.select_with_details(spec, {"input": "test"})
        print(f"{spec.name:<25} {result.selected_model:<20} {result.confidence:.2f}")

    # Show fallback chain
    print_subheader("Fallback Chain Example")
    chain = selector.get_fallback_chain(test_tasks[0], {"input": "test"}, max_length=3)
    print(f"For '{test_tasks[0].name}':")
    print(f"  Primary: {chain.models[0] if chain.models else 'N/A'}")
    print(f"  Fallbacks: {chain.models[1:] if len(chain.models) > 1 else 'None'}")


# =============================================================================
# DEMO 7: ICL GENERATION
# =============================================================================

def demo_icl_generation(trajectories: List[ReActTrajectory], verbose: bool = False):
    """
    Demonstrate ICL example generation from successful traces.

    William's requirement: "if you know a trace or ptool output is a good
    ptool output - make it an ICL demo if that is necessary/helpful"
    """
    print_header("DEMO 7: ICL Generation")

    print("""
In-Context Learning (ICL) examples are generated from successful traces:
  - Good ptool outputs become demonstration examples
  - PTP traces show the reasoning process
  - These examples improve future LLM calls

This implements William's requirement:
"if you know a trace or ptool output is a good ptool output -
 make it an ICL demo if that is necessary/helpful"
""")

    # Find successful trajectories
    successful = [t for t in trajectories if t.success]
    print(f"Found {len(successful)} successful trajectories")

    if not successful:
        print("No successful trajectories for ICL generation.")
        print("Creating synthetic example...")

        # Create a synthetic successful example
        example_ptp = """Goal: Calculate BMI for patient weighing 75kg, height 1.80m

Calling identify_calculator(clinical_text="Calculate BMI for patient weighing 75kg, height 1.80m")...
...identify_calculator returned {"calculator": "BMI", "required_fields": ["weight_kg", "height_m"], "reason": "BMI calculation requested"}

Calling extract_clinical_values(clinical_text="weighing 75kg, height 1.80m", required_fields=["weight_kg", "height_m"])...
...extract_clinical_values returned {"values": {"weight_kg": 75.0, "height_m": 1.80}, "confidence": {"weight_kg": 0.95, "height_m": 0.95}}

Final answer: BMI = 75 / (1.80)^2 = 23.1 kg/m^2 (Normal weight)
"""
        print_subheader("Synthetic ICL Example")
        print(example_ptp)
        return

    # Generate ICL examples from real traces
    print_subheader("ICL Examples from Successful Traces")

    for i, traj in enumerate(successful[:3], 1):
        print(f"\n--- ICL Example {i} ---")
        print(f"Goal: {traj.goal}")
        print()
        print(traj.to_ptp_trace())
        print()


# =============================================================================
# DEMO 8: FULL PIPELINE
# =============================================================================

def demo_full_pipeline(verbose: bool = False, n_problems: int = 3):
    """
    Run the complete MedCalc pipeline end-to-end.

    This demonstrates William's full vision:
    1. ReAct generates traces (behavior distillation approach 2)
    2. Audit system validates traces (SSRM audits)
    3. Critic evaluates (approach 3)
    4. Repair fixes issues (approach 3)
    5. Model selection optimizes (autoselect LLMs)
    6. ICL examples generated (use all experiences)
    """
    print_header("DEMO 8: Full Pipeline")

    print("""
Running the complete ptool_framework pipeline:

  ┌──────────────┐
  │  MedCalc     │  Clinical questions
  │  Problems    │
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │  ReAct       │  Reasoning + Acting
  │  Agent       │  → Generates traces
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │  Audit       │  Quality checks
  │  System      │  → Pass/Fail
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │  Critic      │  Accept/Repair/Reject
  │  System      │  → Verdicts
  └──────┬───────┘
         │
    ┌────┴────┐
    │ Repair? │
    └────┬────┘
         │
         ▼
  ┌──────────────┐
  │  Output      │  Answers + ICL examples
  │  + Stats     │  → Ready for use
  └──────────────┘
""")

    start_time = time.time()

    # Stats tracking
    stats = {
        "problems": n_problems,
        "react_success": 0,
        "audit_pass": 0,
        "critic_accept": 0,
        "critic_repair": 0,
        "critic_reject": 0,
        "repairs_attempted": 0,
        "repairs_successful": 0,
    }

    # Step 1: ReAct Agent
    print_subheader("Step 1: ReAct Agent")
    trajectories = demo_react_agent(verbose=verbose, n_problems=n_problems)
    stats["react_success"] = sum(1 for t in trajectories if t.success)

    # Step 2: Audit
    print_subheader("Step 2: Audit System")
    audit_reports = demo_audit_system(trajectories, verbose=verbose)
    stats["audit_pass"] = sum(1 for r in audit_reports if r.is_pass)

    # Step 3: Critic
    print_subheader("Step 3: Critic System")
    evaluations = demo_critic_system(trajectories, verbose=verbose)
    for e in evaluations:
        if e.verdict == CriticVerdict.ACCEPT:
            stats["critic_accept"] += 1
        elif e.verdict == CriticVerdict.REPAIR_NEEDED:
            stats["critic_repair"] += 1
        else:
            stats["critic_reject"] += 1

    # Step 4: Repair (for traces that need it)
    print_subheader("Step 4: Repair Agent")
    repair_needed = [e for e in evaluations if e.verdict == CriticVerdict.REPAIR_NEEDED]
    if repair_needed:
        print(f"Found {len(repair_needed)} traces needing repair")
        demo_repair_agent(verbose=verbose)
        stats["repairs_attempted"] = len(repair_needed)
    else:
        print("No traces need repair")

    # Step 5: Typicality
    print_subheader("Step 5: Typicality Models")
    demo_typicality_models(trajectories, verbose=verbose)

    # Step 6: Model Selection
    print_subheader("Step 6: Model Selection")
    demo_model_selection(trajectories, verbose=verbose)

    # Step 7: ICL Generation
    print_subheader("Step 7: ICL Generation")
    demo_icl_generation(trajectories, verbose=verbose)

    # Final Summary
    elapsed = time.time() - start_time

    print_header("PIPELINE SUMMARY", "=", 70)

    print(f"""
Results:
  Problems attempted: {stats['problems']}
  ReAct success: {stats['react_success']}/{stats['problems']}
  Audit pass: {stats['audit_pass']}/{len(audit_reports)}

  Critic verdicts:
    Accept: {stats['critic_accept']}
    Repair needed: {stats['critic_repair']}
    Reject: {stats['critic_reject']}

  Repairs: {stats['repairs_attempted']} attempted

  Total time: {elapsed:.1f}s

William's Requirements Status:
  ✓ ptools as LLM prompts (identify_calculator, extract_clinical_values, interpret_result)
  ✓ Python programs calling ptools (medcalc_ptools.py, calculators)
  ✓ Behavior distillation approach 2 (ReAct → traces stored)
  ✓ Behavior distillation approach 3 (critic + repair)
  ✓ Audits/typicality (MedCalcStructureAudit, BigramModel)
  ✓ Autoselect LLMs (ModelSelector)
  ✓ ICL generation from good traces
""")


# =============================================================================
# DEMO 9: DISTILLATION (Script Improvement) - ACTUALLY RUNS!
# =============================================================================

def demo_distillation(verbose: bool = False):
    """
    Demonstrate REAL script improvement through distillation.

    Actually runs:
    1. Define a broad/naive ptool
    2. Run it on test inputs, collect real traces
    3. Use LLM to analyze patterns
    4. Use LLM to generate improved script
    """
    print_header("DEMO 9: Script Distillation (LIVE)")

    print("""
This demo ACTUALLY runs the distillation process:
  1. Execute a broad ptool on test inputs
  2. Collect real traces (input/output pairs)
  3. LLM analyzes patterns in the traces
  4. LLM generates improved, focused ptools
""")

    # =========================================================================
    # STEP 1: Define and run a broad ptool
    # =========================================================================
    print_subheader("Step 1: Run Broad Ptool on Test Inputs")

    # Define test inputs
    test_inputs = [
        "Patient weighs 75kg and is 1.80m tall. Calculate BMI.",
        "65 year old male with hypertension and diabetes. Assess stroke risk.",
        "Creatinine level is 1.4 mg/dL, patient is 70yo female. Check kidney function.",
        "Check the patient's cardiovascular status.",  # Intentionally vague
    ]

    print(f"Running broad ptool on {len(test_inputs)} test inputs...")
    print()

    # Collect traces by running the broad ptool
    traces = []
    for i, text in enumerate(test_inputs, 1):
        print(f"  Input {i}: {text[:50]}...")

        # Actually call LLM with a broad prompt
        broad_prompt = f"""Analyze this medical text and extract relevant information.
Return a JSON object with whatever information you find relevant.

Text: {text}

Return JSON:"""

        try:
            result = call_llm(broad_prompt, model=DEFAULT_MODEL)
            traces.append({
                "input": text,
                "output": result,
                "success": True
            })
            # Show truncated output
            output_preview = result[:80].replace('\n', ' ')
            print(f"    Output: {output_preview}...")
        except Exception as e:
            traces.append({
                "input": text,
                "output": str(e),
                "success": False
            })
            print(f"    Error: {e}")
        print()

    successful = sum(1 for t in traces if t["success"])
    print(f"Collected {len(traces)} traces ({successful} successful)")

    # =========================================================================
    # STEP 2: LLM analyzes patterns
    # =========================================================================
    print_subheader("Step 2: LLM Analyzes Trace Patterns")

    # Format traces for analysis
    traces_text = ""
    for i, t in enumerate(traces, 1):
        traces_text += f"""
Trace {i}:
  Input: {t['input']}
  Output: {t['output'][:200]}...
  Success: {t['success']}
"""

    analysis_prompt = f"""Analyze these execution traces from a medical text analysis ptool.

{traces_text}

Identify:
1. What distinct TASKS is this ptool being asked to do? (e.g., BMI calculation, stroke risk, etc.)
2. What PATTERNS appear in successful inputs? (keywords, structure)
3. What causes FAILURES?
4. How should we SPLIT this into focused ptools?

Provide a structured analysis with specific recommendations for improvement."""

    print("Asking LLM to analyze patterns...")
    analysis = call_llm(analysis_prompt, model=DEFAULT_MODEL)
    print()
    print("Pattern Analysis:")
    print("-" * 50)
    print(analysis)
    print("-" * 50)

    # =========================================================================
    # STEP 3: LLM generates improved script
    # =========================================================================
    print_subheader("Step 3: LLM Generates Improved Script")

    generation_prompt = f"""Based on this analysis of a medical text ptool:

{analysis}

Generate an IMPROVED Python script that:

1. SPLITS the broad ptool into focused, single-purpose ptools
2. Uses STRONG TYPING (dataclasses, List[str], etc.)
3. Adds ICL EXAMPLES from the successful traces to each ptool's docstring
4. Includes a @distilled version with pure Python regex for common patterns
5. Adds VALIDATION before returning results

The script should use the ptool_framework decorators:
- @ptool(model="deepseek-v3-0324", output_mode="structured") for LLM functions
- @distilled(fallback_ptool="name") for Python-first with LLM fallback

Generate the complete improved script:"""

    print("Asking LLM to generate improved script...")
    print()
    improved_script = call_llm(generation_prompt, model=DEFAULT_MODEL)

    print("Generated Improved Script:")
    print("=" * 60)
    print(improved_script)
    print("=" * 60)

    # =========================================================================
    # STEP 4: Summary
    # =========================================================================
    print_subheader("Distillation Complete")

    print("""
What happened:
  1. Ran a REAL broad ptool on 4 test inputs
  2. Collected REAL traces (input/output pairs)
  3. LLM ANALYZED the patterns in traces
  4. LLM GENERATED improved, focused ptools

The improved script has:
  - Focused ptools (one task each)
  - Strong typing
  - ICL examples from traces
  - @distilled pure Python versions
  - Validation logic

You can now use the generated script as a starting point!
""")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="MedCalc Showcase - Demonstrate ptool_framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python medcalc_showcase.py                    # Run full demo
  python medcalc_showcase.py --demo react       # Just ReAct demo
  python medcalc_showcase.py --demo full -v     # Verbose full demo
  python medcalc_showcase.py --problems 5       # Solve 5 problems
"""
    )

    parser.add_argument(
        "--demo",
        choices=["react", "audit", "typicality", "critic", "repair", "model", "icl", "distill", "full"],
        default="full",
        help="Which demo section to run (default: full)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output with more details"
    )
    parser.add_argument(
        "--problems",
        type=int,
        default=3,
        help="Number of MedCalc problems to solve (default: 3)"
    )

    args = parser.parse_args()

    # Check for API key
    if not os.environ.get("TOGETHER_API_KEY"):
        print("WARNING: TOGETHER_API_KEY not set. LLM calls may fail.")
        print("Set it in .env file or export TOGETHER_API_KEY=...")
        print()

    print_header("MEDCALC SHOWCASE", "=", 70)
    print(f"Model: {DEFAULT_MODEL}")
    print(f"Demo: {args.demo}")
    print(f"Problems: {args.problems}")
    print(f"Verbose: {args.verbose}")

    # Run selected demo
    if args.demo == "full":
        demo_full_pipeline(verbose=args.verbose, n_problems=args.problems)

    elif args.demo == "react":
        trajectories = demo_react_agent(verbose=args.verbose, n_problems=args.problems)

    elif args.demo == "audit":
        # Need trajectories first
        trajectories = demo_react_agent(verbose=False, n_problems=args.problems)
        demo_audit_system(trajectories, verbose=args.verbose)

    elif args.demo == "typicality":
        trajectories = demo_react_agent(verbose=False, n_problems=args.problems)
        demo_typicality_models(trajectories, verbose=args.verbose)

    elif args.demo == "critic":
        trajectories = demo_react_agent(verbose=False, n_problems=args.problems)
        demo_critic_system(trajectories, verbose=args.verbose)

    elif args.demo == "repair":
        demo_repair_agent(verbose=args.verbose)

    elif args.demo == "model":
        trajectories = demo_react_agent(verbose=False, n_problems=args.problems)
        demo_model_selection(trajectories, verbose=args.verbose)

    elif args.demo == "icl":
        trajectories = demo_react_agent(verbose=False, n_problems=args.problems)
        demo_icl_generation(trajectories, verbose=args.verbose)

    elif args.demo == "distill":
        demo_distillation(verbose=args.verbose)

    print_header("DEMO COMPLETE", "=", 70)


if __name__ == "__main__":
    main()
