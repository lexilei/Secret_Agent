"""
L0 Baseline with Formulae: Direct LLM call with formula reference.

Like the vanilla baseline, but includes the available medical formulae
in the prompt. No ptools, no ReAct, no structure - just a better prompt.
"""

import time
from typing import Optional
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ..config import ExperimentConfig
from ..dataset.loader import MedCalcInstance
from ..metrics.cost import calculate_cost
from ..metrics.accuracy import calculate_accuracy
from .base import BaseExperiment, ExperimentResult

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
    lines = ["Available Medical Formulas:"]
    for name, formula in sorted(formulas.items()):
        lines.append(f"- {name}: {formula}")

    return "\n".join(lines)


# Cache the formula reference at module load time
FORMULA_REFERENCE = get_formula_reference()


class BaselineWithFormulaeExperiment(BaseExperiment):
    """
    L0 Baseline with Formulae: Direct LLM call with formula reference.

    Simply prompts the LLM with the patient note, question, and available
    medical formulae, asking for a numeric answer. No structure, no multi-step reasoning.
    """

    PROMPT_TEMPLATE = """You are a medical calculation assistant.

{formula_reference}

Patient Note:
{patient_note}

Question: {question}

Instructions:
1. Read the patient note carefully
2. Extract the relevant values needed for the calculation
3. Use the appropriate formula from the list above
4. Perform the calculation
5. Return ONLY the final numeric answer

Important: Return just the number, nothing else. No units, no explanation.

Answer:"""

    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        self.model = config.model

    def run_instance(self, instance: MedCalcInstance) -> ExperimentResult:
        """
        Run baseline with formulae on a single instance.

        Args:
            instance: MedCalc problem instance

        Returns:
            ExperimentResult with prediction and metrics
        """
        # Build prompt
        prompt = self.PROMPT_TEMPLATE.format(
            formula_reference=FORMULA_REFERENCE,
            patient_note=instance.patient_note,
            question=instance.question,
        )

        # Call LLM
        start_time = time.time()
        try:
            response = call_llm(
                prompt=prompt,
                model=self.model,
            )
            latency_ms = (time.time() - start_time) * 1000

            # Extract answer
            predicted = self.extract_number(response)

            # Estimate tokens (rough)
            input_tokens = len(prompt) // 4
            output_tokens = len(response) // 4 if response else 0

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
                raw_response=response,
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
