"""
L1O Experiment: LLM extraction + Official Python calculators.

Uses a single @ptool call to identify the calculator and extract values,
then uses the official Python implementations from calculator_simple.py
to perform the actual computation.

Key difference from L1:
- L1: LLM does extraction AND calculation
- L1O: LLM extracts, Python computes using official implementations
"""

import time
import json
import re
from typing import Dict, Any, Optional, List
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

# Import official calculator infrastructure
from .calculator_simple import (
    CALCULATOR_REGISTRY,
    get_calculator_names,
    get_calculator_signatures,
    compute as compute_structured,
)


# =============================================================================
# Build Calculator Reference for LLM
# =============================================================================

def build_calculator_reference() -> str:
    """Build a reference string of all calculators with their signatures."""
    signatures = get_calculator_signatures()

    lines = ["AVAILABLE CALCULATORS (use exact name):"]
    for name, sig in sorted(signatures.items()):
        required = ", ".join(sig["required"]) if sig["required"] else "none"
        optional = ", ".join(sig["optional"]) if sig["optional"] else "none"
        formula = sig.get("formula", "")

        lines.append(f"\n{name}:")
        lines.append(f"  Required: {required}")
        if sig["optional"]:
            lines.append(f"  Optional: {optional}")
        if formula:
            lines.append(f"  Formula: {formula}")

    return "\n".join(lines)


# Cache at module load time
CALCULATOR_REFERENCE = build_calculator_reference()
AVAILABLE_CALCULATORS = get_calculator_names()

# Build the docstring for the extraction ptool
_EXTRACTION_DOCSTRING = f"""
Extract values for a medical calculator from a patient note.

Given a patient note and a calculation question:
1. Identify which calculator is needed from the available list
2. Extract all required (and any optional) values from the patient note
3. Convert units as needed (e.g., lbs to kg, feet/inches to cm)

{CALCULATOR_REFERENCE}

IMPORTANT EXTRACTION GUIDELINES:
- For sex/gender: "male" or "female" (infer from pronouns like he/she)
- For boolean values: use true/false
- For numeric values: extract as numbers (float or int)
- For dates: use YYYY-MM-DD format
- Convert units to match what the calculator expects

Return a JSON object with:
{{
    "calculator_name": "exact name from the list above",
    "extracted_values": {{
        "param1": value1,
        "param2": value2,
        ...
    }},
    "reasoning": "brief explanation of extraction"
}}
"""


def _extract_for_calculator(patient_note: str, question: str) -> Dict[str, Any]:
    ...

# Set the docstring before decorating
_extract_for_calculator.__doc__ = _EXTRACTION_DOCSTRING

# Apply the ptool decorator
extract_for_calculator = ptool(
    model="deepseek-v3-0324",
    output_mode="structured"
)(_extract_for_calculator)


class L1OOfficialExperiment(BaseExperiment):
    """
    L1O Experiment: LLM extraction + Official Python calculators.

    Uses a single @ptool call to extract calculator name and values,
    then computes using the official Python implementations.
    """

    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        self.model = config.model

    def run_instance(self, instance: MedCalcInstance) -> ExperimentResult:
        """
        Run L1O experiment on a single instance.

        Args:
            instance: MedCalc problem instance

        Returns:
            ExperimentResult with prediction and metrics
        """
        start_time = time.time()

        try:
            # Step 1: Extract calculator name and values using LLM
            extraction = extract_for_calculator(
                patient_note=instance.patient_note,
                question=instance.question,
            )

            extraction_time = time.time() - start_time

            # Parse the extraction result
            if isinstance(extraction, str):
                # Try to parse as JSON if it's a string
                try:
                    extraction = json.loads(extraction)
                except json.JSONDecodeError:
                    # Try to extract JSON from the string
                    match = re.search(r'\{[\s\S]*\}', extraction)
                    if match:
                        extraction = json.loads(match.group())
                    else:
                        raise ValueError(f"Could not parse extraction: {extraction}")

            # Handle nested "result" wrapper from ptool structured output
            if isinstance(extraction, dict) and "result" in extraction and isinstance(extraction["result"], dict):
                extraction = extraction["result"]

            calculator_name = extraction.get("calculator_name", "")
            extracted_values = extraction.get("extracted_values", {})

            # Step 2: Compute using official Python implementation
            predicted = compute_structured(calculator_name, extracted_values)

            latency_ms = (time.time() - start_time) * 1000

            # Estimate tokens (only for extraction, compute is free)
            prompt_text = f"{instance.patient_note} {instance.question}"
            input_tokens = len(prompt_text) // 4 + 500  # +500 for calculator reference
            output_tokens = len(str(extraction)) // 4 if extraction else 0

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
                raw_response=str(extraction),
                trace={
                    "extraction": extraction,
                    "identified_calculator": calculator_name,
                    "extracted_values": extracted_values,
                    "computed_result": predicted,
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


# ============================================================================
# Standalone execution
# ============================================================================

if __name__ == "__main__":
    """
    Run L1O experiment standalone.

    Usage:
        # Set API key first
        export TOGETHER_API_KEY="your-key"

        # Run with default settings (first 5 instances)
        python -m benchmark.experiments.l1o_official

        # Run with custom number of instances
        python -m benchmark.experiments.l1o_official --n 10

        # Run on a specific instance
        python -m benchmark.experiments.l1o_official --instance 42
    """
    import argparse

    parser = argparse.ArgumentParser(description="Run L1O Official Experiment")
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

    print(f"Running L1O Official on {len(instances)} instances...\n")

    # Create and run experiment
    # Use l1_ptool config as base since l1o_official might not exist yet
    config = ABLATION_CONFIGS.get("l1o_official", ABLATION_CONFIGS["l1_ptool"])
    experiment = L1OOfficialExperiment(config)
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
            status = "CORRECT"
        elif result.error:
            status = f"ERROR: {result.error}"
        else:
            status = f"WRONG (predicted={result.predicted_answer}, expected={result.ground_truth})"

        print(f"  {status}")

        if args.verbose and result.trace:
            trace = result.trace
            print(f"  Identified: {trace.get('identified_calculator', 'N/A')}")
            print(f"  Extracted: {trace.get('extracted_values', {})}")
            print(f"  Computed: {trace.get('computed_result', 'N/A')}")

        print()

    # Summary
    print("=" * 60)
    print(f"RESULTS: {correct}/{total} correct ({correct/total*100:.1f}%)")
    print("=" * 60)
