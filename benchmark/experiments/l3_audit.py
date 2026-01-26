"""
L3+ Experiment: ReAct + TraceCritic + RepairAgent.

Adds error detection and automatic repair to the L3 ReAct experiment.
Uses the critic to evaluate traces and the repair agent to fix issues.

V2: Uses direct structured function calls (compute_calculation) instead of
    text-based python_calculate to avoid format mismatch issues.
"""

import time
from typing import Dict, Any, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ..config import ExperimentConfig
from ..dataset.loader import MedCalcInstance
from ..metrics.cost import calculate_cost
from ..metrics.accuracy import calculate_accuracy
from .base import BaseExperiment, ExperimentResult
from .l3_react import get_medcalc_ptools, extract_clinical_values, build_goal_prompt

from ptool_framework import ReActAgent
from ptool_framework.critic import TraceCritic, CriticVerdict
from ptool_framework.repair import RepairAgent
from ptool_framework.audit import (
    AuditRunner,
    NonEmptyTraceAudit,
    NoFailedStepsAudit,
    AllCompletedAudit,
)
from ptool_framework.audit.domains.medcalc import (
    MedCalcStructureAudit,
    MedCalcCorrectnessAudit,
)


class L3AuditExperiment(BaseExperiment):
    """
    L3+ Experiment: ReAct + TraceCritic + RepairAgent.

    Runs the ReAct agent, then evaluates the trace with a critic.
    If the critic identifies issues, attempts automatic repair.
    """

    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        self.model = config.model
        self.max_steps = config.max_steps
        self.max_repair_attempts = config.max_repair_attempts
        self.agent: Optional[ReActAgent] = None
        self.critic: Optional[TraceCritic] = None
        self.repair_agent: Optional[RepairAgent] = None

        # Stats
        self.repair_attempts = 0
        self.repair_successes = 0

    def setup(self):
        """Initialize agent, critic, and repair components."""
        # Ensure ptools are registered
        _ = extract_clinical_values

        # Get ptools - use direct compute_calculation (structured values, no format mismatch)
        ptools = get_medcalc_ptools(use_direct_compute=True)

        # Create ReAct agent
        self.agent = ReActAgent(
            available_ptools=ptools,
            model=self.model,
            max_steps=self.max_steps,
            echo=False,
        )

        # Create audits for the critic
        # Structural audits (check trace structure)
        # Note: We check for ATTEMPT, not success - if LLM tried to use calculator, that's enough
        audits = [
            NonEmptyTraceAudit(),
            NoFailedStepsAudit(),
            AllCompletedAudit(),
            MedCalcStructureAudit(),  # Checks LLM attempted compute_calculation
            MedCalcCorrectnessAudit(tolerance=0.05),  # Checks LLM attempted calculator
        ]

        # Create critic
        self.critic = TraceCritic(
            audits=audits,
            model=self.model,
            accept_threshold=0.7,
            repair_threshold=0.3,
        )

        # Create repair agent with verbose mode
        # Pass react_agent to enable rewind-and-regenerate repair
        self.repair_agent = RepairAgent(
            critic=self.critic,
            react_agent=self.agent,  # Enable rewind-and-regenerate repair
            max_attempts=self.max_repair_attempts,
            model=self.model,
            verbose=True,  # Show repair attempts
        )

        self._setup_complete = True

    def run_instance(self, instance: MedCalcInstance) -> ExperimentResult:
        """
        Run ReAct with critic evaluation and repair.
        """
        if self.agent is None:
            self.setup()

        # Use the same goal prompt as l3_react with calculator signatures
        goal = build_goal_prompt(instance)

        start_time = time.time()
        repair_attempts = 0
        repair_success = False
        audit_info = {}  # Store audit details for logging
        repair_actions_taken = []

        try:
            # Run initial ReAct agent
            result = self.agent.run(goal)

            # Evaluate with critic
            if result.trajectory:
                evaluation = self.critic.evaluate(
                    trace=result.trajectory.generated_trace,
                    goal=goal,
                )

                # Log critic verdict
                print(f"  [Critic] Verdict: {evaluation.verdict.value}")
                print(f"  [Critic] Violations: {evaluation.total_violations}, Failed steps: {evaluation.failed_steps}")
                if evaluation.repair_suggestions:
                    print(f"  [Critic] Repair suggestions: {len(evaluation.repair_suggestions)}")

                # Store evaluation in audit_info
                audit_info['evaluation'] = {
                    'verdict': evaluation.verdict.value,
                    'confidence': evaluation.confidence,
                    'total_violations': evaluation.total_violations,
                    'failed_steps': evaluation.failed_steps,
                    'audit_reports': [r.to_dict() for r in evaluation.audit_reports],
                    'repair_suggestions': evaluation.repair_suggestions,
                }

                # Attempt repair if needed
                if evaluation.verdict == CriticVerdict.REPAIR_NEEDED:
                    self.repair_attempts += 1
                    print(f"  [Repair] Starting repair (attempt #{self.repair_attempts})...")

                    repair_result = None
                    try:
                        # Pass the full trajectory for rewind-and-regenerate repair
                        repair_result = self.repair_agent.repair(
                            trace=result.trajectory,  # Pass trajectory, not just trace
                            evaluation=evaluation,
                            goal=goal,
                        )

                        repair_attempts = repair_result.iterations or 0
                        repair_actions_taken = repair_result.actions_taken or []
                        print(f"  [Repair] Result: success={repair_result.success}, actions={len(repair_result.actions_taken)}")
                        if repair_result.error:
                            print(f"  [Repair] Error: {repair_result.error}")
                    except Exception as e:
                        print(f"  [Repair] Exception: {e}")
                        import traceback
                        traceback.print_exc()

                    if repair_result and repair_result.success:
                        self.repair_successes += 1
                        repair_success = True

                    # IMPORTANT: Extract repaired answer even if overall success=False
                    # The repair may have fixed the calculation even if other audits still fail
                    if repair_result and repair_result.repaired_trace:
                        for step in reversed(repair_result.repaired_trace.steps):
                            if hasattr(step, 'result') and step.result:
                                step_result = step.result
                                # Check for Python calculator result format
                                if isinstance(step_result, dict) and "result" in step_result:
                                    repaired_answer = step_result["result"]
                                    if repaired_answer is not None and isinstance(repaired_answer, (int, float)):
                                        result.answer = repaired_answer
                                        print(f"  [Repair] Extracted answer: {result.answer}")
                                        break

                    # NOTE: Removed Python calculator fallback - L3 should stay LLM-based
                    # If repair fails, we accept the LLM's answer (even if audit found issues)
                    # Python calculator belongs in L2 (distilled), not as L3 fallback

                # Store repair info
                audit_info['repair_attempts'] = repair_attempts
                audit_info['repair_success'] = repair_success
                audit_info['repair_actions'] = repair_actions_taken

            latency_ms = (time.time() - start_time) * 1000

            # Extract answer
            predicted = None
            if result.answer:
                predicted = self.extract_number(str(result.answer))

            # If no final answer, try to extract from last successful observation
            # Look for observations with a "result" key containing a numeric value
            if predicted is None and result.trajectory:
                for step in reversed(result.trajectory.steps):
                    if (step.observation and
                        step.observation.success and
                        step.observation.result):
                        obs_result = step.observation.result
                        # Handle nested {"result": {...}} from structured output
                        if isinstance(obs_result, dict) and "result" in obs_result:
                            inner = obs_result["result"]
                            # If inner is also a dict with "result", use that
                            if isinstance(inner, dict) and "result" in inner:
                                predicted = inner["result"]
                            elif isinstance(inner, (int, float)):
                                predicted = inner
                            elif isinstance(inner, dict):
                                # Skip dicts that don't have a numeric result
                                continue
                            else:
                                predicted = inner
                        elif isinstance(obs_result, (int, float)):
                            predicted = obs_result
                        else:
                            continue

                        if predicted is not None:
                            try:
                                predicted = float(predicted)
                                break
                            except (ValueError, TypeError):
                                predicted = self.extract_number(str(predicted))
                                if predicted is not None:
                                    break

            # Estimate tokens
            total_input = len(goal) // 4 + 500
            total_output = 0
            for step in result.trajectory.steps if result.trajectory else []:
                if step.thought:
                    total_input += len(step.thought.content) // 4
                if step.observation and step.observation.result:
                    total_output += len(str(step.observation.result)) // 4

            # Add critic and repair tokens
            total_input += 200 * (1 + repair_attempts)  # Critic evaluations
            total_output += 100 * (1 + repair_attempts)

            cost_metrics = calculate_cost(
                input_tokens=total_input,
                output_tokens=total_output,
                model=self.model,
            )

            accuracy = calculate_accuracy(
                predicted=predicted,
                ground_truth=instance.ground_truth_answer,
                lower_limit=instance.lower_limit,
                upper_limit=instance.upper_limit,
                output_type=instance.output_type,
                category=instance.category,
            )

            num_steps = len(result.trajectory.steps) if result.trajectory else 0

            return ExperimentResult(
                instance_id=instance.row_number,
                calculator_name=instance.calculator_name,
                category=instance.category,
                predicted_answer=predicted,
                ground_truth=instance.ground_truth_answer,
                is_correct_exact=accuracy.is_exact_match,
                is_correct_tolerance=accuracy.is_within_tolerance,
                is_within_limits=accuracy.is_within_limits,
                latency_ms=latency_ms,
                input_tokens=total_input,
                output_tokens=total_output,
                total_tokens=total_input + total_output,
                cost_usd=cost_metrics.cost_usd,
                num_steps=num_steps,
                repair_attempts=repair_attempts,
                repair_success=repair_success,
                raw_response=str(result.answer),
                trace=result.trajectory.to_dict() if result.trajectory else None,
                audit_info=audit_info if audit_info else None,
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return ExperimentResult(
                instance_id=instance.row_number,
                calculator_name=instance.calculator_name,
                category=instance.category,
                predicted_answer=None,
                ground_truth=instance.ground_truth_answer,
                is_correct_exact=False,
                is_correct_tolerance=False,
                is_within_limits=False,
                latency_ms=latency_ms,
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                cost_usd=0.0,
                repair_attempts=repair_attempts,
                error=str(e),
                error_type=type(e).__name__,
            )

    def get_summary(self) -> Dict[str, Any]:
        """Get summary with repair stats."""
        summary = super().get_summary()
        summary["total_repair_attempts"] = self.repair_attempts
        summary["repair_successes"] = self.repair_successes
        summary["repair_success_rate"] = (
            self.repair_successes / self.repair_attempts
            if self.repair_attempts > 0 else 0
        )
        return summary


# ============================================================================
# Standalone execution
# ============================================================================

if __name__ == "__main__":
    """
    Run L3+ Audit experiment standalone.

    Usage:
        # Set API key first
        export TOGETHER_API_KEY="your-key"

        # Run with default settings (first 5 instances)
        python -m benchmark.experiments.l3_audit

        # Run with custom number of instances
        python -m benchmark.experiments.l3_audit --n 10

        # Run on a specific instance
        python -m benchmark.experiments.l3_audit --instance 42
    """
    import argparse

    parser = argparse.ArgumentParser(description="Run L3+ Audit Experiment")
    parser.add_argument("-n", "--num", type=int, default=5, help="Number of instances to run")
    parser.add_argument("--instance", type=int, help="Run a specific instance by ID")
    args = parser.parse_args()

    # Load dataset
    print("Loading MedCalc-Bench dataset...")
    from benchmark.dataset.loader import MedCalcDataset
    from benchmark.config import ABLATION_CONFIGS

    dataset = MedCalcDataset()

    if args.instance is not None:
        # Run specific instance
        all_instances = dataset.load("test")
        instances = [inst for inst in all_instances if inst.row_number == args.instance]
        if not instances:
            print(f"Error: Instance {args.instance} not found")
            sys.exit(1)
    else:
        instances = dataset.get_debug_subset(args.num)

    print(f"Running L3+ Audit on {len(instances)} instances...\n")

    # Create and run experiment using pre-defined config
    config = ABLATION_CONFIGS["l3_audit"]
    experiment = L3AuditExperiment(config)
    experiment.setup()

    # Run each instance
    correct_exact = 0
    correct_tolerance = 0
    total = 0
    errors = 0
    repair_attempts = 0
    repair_successes = 0

    for i, instance in enumerate(instances):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(instances)}] {instance.calculator_name}")
        print(f"  Question: {instance.question[:80]}...")

        result = experiment.run_instance(instance)
        total += 1

        if result.error:
            errors += 1
            status = f"ERROR: {result.error}"
        elif result.is_correct_tolerance:
            correct_tolerance += 1
            if result.is_correct_exact:
                correct_exact += 1
                status = "EXACT MATCH"
            else:
                status = "WITHIN 5%"
        else:
            status = f"WRONG (predicted={result.predicted_answer}, expected={result.ground_truth})"

        print(f"  {status}")
        print(f"  Steps: {result.num_steps}, Latency: {result.latency_ms:.0f}ms")

        if result.repair_attempts:
            print(f"  Repair attempts: {result.repair_attempts}, Success: {result.repair_success}")
            repair_attempts += result.repair_attempts
            if result.repair_success:
                repair_successes += 1

    # Summary
    print("\n" + "=" * 60)
    print(f"RESULTS: L3+ Audit on {total} instances")
    print("=" * 60)
    print(f"  Exact match:    {correct_exact}/{total} ({correct_exact/total*100:.1f}%)")
    print(f"  Within 5%:      {correct_tolerance}/{total} ({correct_tolerance/total*100:.1f}%)")
    print(f"  Errors:         {errors}/{total} ({errors/total*100:.1f}%)")
    print(f"\n  Repair attempts: {repair_attempts}")
    print(f"  Repair successes: {repair_successes}")
    print("=" * 60)
