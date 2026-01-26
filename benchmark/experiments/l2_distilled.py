"""
L2 Experiment: @distilled with Python-first execution.

Uses pure Python implementations of all MedCalc calculators,
falling back to LLM only when Python extraction/calculation fails.

Key improvement over L1:
- Zero API cost for calculators that Python can handle
- Instant response (milliseconds vs seconds)
- Deterministic results (no LLM variability)
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

from ptool_framework import distilled, DistillationFallback

# IMPORTANT: Import L1 ptool to ensure it's registered for fallback
from .l1_ptool import calculate_medical_value  # noqa: F401

# Import our comprehensive calculator implementations
from .calculators import calculate as python_calculate, CalcResult


# ============================================================================
# Distilled function using comprehensive calculators
# ============================================================================


@distilled(fallback_ptool="_calculate_medical_value")
def calculate_medical_value_distilled(
    patient_note: str,
    question: str,
) -> Dict[str, Any]:
    """
    Calculate medical value using Python-first approach with STRICT parsing.

    Tries pure Python extraction and calculation first using
    comprehensive implementations of all MedCalc calculators.

    STRICT approach:
    - Python only returns results when extraction is certain
    - Returns None (triggers LLM fallback) when uncertain
    - No confidence thresholds - parsing either succeeds or fails
    """
    # Try Python calculation with STRICT parsing
    result = python_calculate(patient_note, question)

    if result is not None:
        # Python extraction succeeded with high confidence
        return {
            "calculator_type": result.calculator_name,
            "extracted_values": result.extracted_values,
            "result": result.result,
            "method": "python",
            "formula": result.formula_used,
        }

    # Python couldn't handle it (strict parsing returned None) - use LLM
    raise DistillationFallback("Strict parsing returned None - using LLM fallback")


class L2DistilledExperiment(BaseExperiment):
    """
    L2 Experiment: @distilled Python-first with LLM fallback.

    Tries pure Python extraction and calculation first,
    uses LLM only when Python can't handle the case.
    """

    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        self.model = config.model
        self.python_success_count = 0
        self.llm_fallback_count = 0

    def run_instance(self, instance: MedCalcInstance) -> ExperimentResult:
        """
        Run L2 distilled experiment on a single instance.
        """
        start_time = time.time()

        try:
            # Try Python-first calculation
            result = calculate_medical_value_distilled(
                patient_note=instance.patient_note,
                question=instance.question,
            )

            latency_ms = (time.time() - start_time) * 1000

            # Determine if we used Python or LLM fallback
            # Python returns dict with "method": "python"
            # LLM fallback returns float directly
            if isinstance(result, dict):
                used_python = result.get("method") == "python"
            else:
                used_python = False  # LLM fallback returns float directly

            if used_python:
                self.python_success_count += 1
                input_tokens = 0
                output_tokens = 0
                cost = 0.0
            else:
                self.llm_fallback_count += 1
                # Estimate tokens for LLM call
                prompt_text = f"{instance.patient_note} {instance.question}"
                input_tokens = len(prompt_text) // 4 + 200
                output_tokens = len(str(result)) // 4
                cost_metrics = calculate_cost(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    model=self.model,
                )
                cost = cost_metrics.cost_usd

            # Extract numeric answer
            # Python success returns dict with "result" key
            # LLM fallback returns float directly
            predicted = None
            if isinstance(result, dict):
                predicted = result.get("result")
                if predicted is not None:
                    predicted = float(predicted)
            elif isinstance(result, (int, float)):
                predicted = float(result)

            # Evaluate accuracy
            accuracy = calculate_accuracy(
                predicted=predicted,
                ground_truth=instance.ground_truth_answer,
                lower_limit=instance.lower_limit,
                upper_limit=instance.upper_limit,
                output_type=instance.output_type,
                category=instance.category,
            )

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
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                cost_usd=cost,
                raw_response=str(result),
                trace={
                    "method": "python" if used_python else "llm",
                    "result": result,
                },
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
                error=str(e),
                error_type=type(e).__name__,
            )

    def get_summary(self) -> Dict[str, Any]:
        """Get summary with distillation stats."""
        summary = super().get_summary()
        total = self.python_success_count + self.llm_fallback_count
        summary["python_success_count"] = self.python_success_count
        summary["llm_fallback_count"] = self.llm_fallback_count
        summary["python_success_rate"] = (
            self.python_success_count / total if total > 0 else 0
        )
        return summary


# ============================================================================
# Standalone execution
# ============================================================================

if __name__ == "__main__":
    """
    Run L2 experiment standalone.

    Usage:
        # Set API key first (only needed for LLM fallback)
        export TOGETHER_API_KEY="your-key"

        # Run with default settings (first 5 instances)
        python -m benchmark.experiments.l2_distilled

        # Run with custom number of instances
        python -m benchmark.experiments.l2_distilled --n 10

        # Run on a specific instance
        python -m benchmark.experiments.l2_distilled --instance 42
    """
    import argparse

    parser = argparse.ArgumentParser(description="Run L2 Distilled Experiment")
    parser.add_argument("-n", "--num", type=int, default=5, help="Number of instances to run")
    parser.add_argument("--instance", type=int, help="Run a specific instance by ID")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    args = parser.parse_args()

    # Load dataset
    print("Loading MedCalc-Bench dataset...")
    from ..dataset.loader import MedCalcDataset
    from ..config import ABLATION_CONFIGS

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

    print(f"Running L2 Distilled on {len(instances)} instances...")
    print("(Python-first with comprehensive calculators, LLM fallback for failures)")
    print()

    # Create and run experiment
    config = ABLATION_CONFIGS["l2_distilled"]
    experiment = L2DistilledExperiment(config)
    experiment.setup()

    # Run each instance
    correct = 0
    total = 0
    python_count = 0
    llm_count = 0

    for i, instance in enumerate(instances):
        print(f"[{i+1}/{len(instances)}] {instance.calculator_name}")
        print(f"  Question: {instance.question[:80]}...")

        result = experiment.run_instance(instance)
        total += 1

        # Track method used
        trace = result.trace or {}
        method = trace.get("method", "unknown")
        if method == "python":
            python_count += 1
            method_str = "Python"
        else:
            llm_count += 1
            method_str = "LLM"

        if result.is_correct_tolerance:
            correct += 1
            status = f"✓ CORRECT ({method_str})"
        elif result.error:
            status = f"✗ ERROR: {result.error}"
        else:
            status = f"✗ WRONG (predicted={result.predicted_answer}, expected={result.ground_truth}) [{method_str}]"

        print(f"  {status}")

        if args.verbose and result.raw_response:
            print(f"  Response: {result.raw_response[:100]}...")

        print()

    # Summary
    print("=" * 60)
    print(f"RESULTS: {correct}/{total} correct ({correct/total*100:.1f}%)")
    print(f"  Python: {python_count} instances ($0.00)")
    print(f"  LLM:    {llm_count} instances")
    print("=" * 60)
