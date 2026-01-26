"""
L1 Experiment: @ptool with structured output.

Uses a single @ptool call with typed return value for structured extraction
and calculation. The LLM is guided by the type signature.
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

from ptool_framework import ptool
from ptool_framework.llm_backend import call_llm


# =============================================================================
# Formula Reference (dynamically extracted from calculators.py)
# =============================================================================

def get_formula_reference() -> str:
    """Extract formulas from calculators.py dynamically."""
    import inspect
    import re
    from . import calculators

    formulas = {}

    # Extract formula_used from each calculate_ function's source
    for name, func in vars(calculators).items():
        if name.startswith('calculate_') and callable(func):
            try:
                source = inspect.getsource(func)
                # Use regex to extract the quoted string after formula_used=
                match = re.search(r'formula_used\s*=\s*["\']([^"\']+)["\']', source)
                if match:
                    formula = match.group(1)
                    calc_name = name.replace('calculate_', '').replace('_', ' ').title()
                    if calc_name not in formulas:
                        formulas[calc_name] = formula
            except Exception:
                pass

    # Build reference string
    lines = ["FORMULAS (use these exact formulas):"]
    for name, formula in sorted(formulas.items()):
        lines.append(f"- {name}: {formula}")

    return "\n".join(lines)


# Cache the formula reference at module load time
FORMULA_REFERENCE = get_formula_reference()

# Build the full docstring with formulas
_DOCSTRING = f"""
Calculate a medical value from a patient note.

Given a patient note and a calculation question:
1. Carefully read the patient note to extract all relevant clinical values
2. Identify what medical calculation/score is needed
3. Apply the appropriate formula from the reference below
4. Show your calculation step by step

{FORMULA_REFERENCE}

Important: Be precise with extracted values. Double-check your arithmetic.

ANSWER: <the numeric result as a number>
"""


# Define the function first, set docstring, then apply decorator
def _calculate_medical_value(patient_note: str, question: str) -> float:
    ...

# Set the docstring before decorating
_calculate_medical_value.__doc__ = _DOCSTRING

# Apply the ptool decorator
calculate_medical_value = ptool(model="deepseek-v3-0324", output_mode="freeform")(_calculate_medical_value)


class L1PToolExperiment(BaseExperiment):
    """
    L1 Experiment: Single @ptool call with structured output.

    Uses the @ptool decorator to get structured JSON output,
    guiding the LLM to extract values and calculate in one step.
    """

    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        self.model = config.model

    def run_instance(self, instance: MedCalcInstance) -> ExperimentResult:
        """
        Run L1 ptool experiment on a single instance.

        Args:
            instance: MedCalc problem instance

        Returns:
            ExperimentResult with prediction and metrics
        """
        start_time = time.time()

        try:
            # Call the ptool
            result = calculate_medical_value(
                patient_note=instance.patient_note,
                question=instance.question,
            )

            latency_ms = (time.time() - start_time) * 1000

            # Extract numeric answer from result
            predicted = None
            if result is not None:
                try:
                    predicted = float(result)
                except (ValueError, TypeError):
                    predicted = self.extract_number(str(result))

            # Estimate tokens
            prompt_text = f"{instance.patient_note} {instance.question}"
            input_tokens = len(prompt_text) // 4 + 200  # +200 for system prompt
            output_tokens = len(str(result)) // 4 if result else 0

            # Calculate cost
            cost_metrics = calculate_cost(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model=self.model,
            )

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
                cost_usd=cost_metrics.cost_usd,
                raw_response=str(result),
                trace={"ptool_result": result},
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


# ============================================================================
# Standalone execution
# ============================================================================

if __name__ == "__main__":
    """
    Run L1 experiment standalone.

    Usage:
        # Set API key first
        export TOGETHER_API_KEY="your-key"

        # Run with default settings (first 5 instances)
        python -m benchmark.experiments.l1_ptool

        # Run with custom number of instances
        python -m benchmark.experiments.l1_ptool --n 10

        # Run on a specific instance
        python -m benchmark.experiments.l1_ptool --instance 42
    """
    import argparse

    parser = argparse.ArgumentParser(description="Run L1 PTools Experiment")
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

    print(f"Running L1 PTools on {len(instances)} instances...\n")

    # Create and run experiment
    config = ABLATION_CONFIGS["l1_ptool"]
    experiment = L1PToolExperiment(config)
    experiment.setup()

    # Run each instance
    correct = 0
    total = 0

    for i, instance in enumerate(instances):
        print(f"[{i+1}/{len(instances)}] {instance.calculator_name}")
        print(f"  Question: {instance.question[:80]}...")

        result = experiment.run_instance(instance)
        total += 1

        if result.is_correct_tolerance:
            correct += 1
            status = "✓ CORRECT"
        elif result.error:
            status = f"✗ ERROR: {result.error}"
        else:
            status = f"✗ WRONG (predicted={result.predicted_answer}, expected={result.ground_truth})"

        print(f"  {status}")

        if args.verbose and result.raw_response:
            print(f"  Response: {result.raw_response[:100]}...")

        print()

    # Summary
    print("=" * 60)
    print(f"RESULTS: {correct}/{total} correct ({correct/total*100:.1f}%)")
    print("=" * 60)
