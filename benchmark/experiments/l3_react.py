"""
L3 Experiment: ReActAgent with multi-step reasoning.

Uses the ReActAgent to solve problems through a think-act-observe loop,
breaking down complex calculations into steps.

V2: Uses direct structured function calls instead of text-based python_calculate.
    - compute_calculation(calculator_name, values) accepts pre-extracted values
    - Dynamic calculator signatures in prompt
    - Explicit <answer> tags for final output
"""

import re
import time
from typing import Dict, Any, List, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ..config import ExperimentConfig
from ..dataset.loader import MedCalcInstance
from ..metrics.cost import calculate_cost
from ..metrics.accuracy import calculate_accuracy
from .base import BaseExperiment, ExperimentResult

from ptool_framework import ptool, ReActAgent, PToolSpec

# Import formula reference from L1 for calculation accuracy
from .l1_ptool import FORMULA_REFERENCE

# Import direct computation interface from calculators
from .calculators import get_signatures, compute_direct


# ============================================================================
# MedCalc ptools for ReAct agent
# ============================================================================

@ptool(model="deepseek-v3-0324", output_mode="structured")
def identify_calculator(clinical_text: str) -> Dict[str, Any]:
    """
    Identify which medical calculator is needed based on the clinical question.

    Analyze the text and determine:
    - What type of calculation is requested (BMI, eGFR, CHA2DS2-VASc, etc.)
    - What input values will be needed
    - What formula or rules apply

    Return:
    {
        "calculator_name": "name of the calculator",
        "required_inputs": ["list", "of", "required", "values"],
        "category": "equation-based" or "rule-based"
    }
    """
    ...


@ptool(model="deepseek-v3-0324", output_mode="structured")
def extract_clinical_values(
    patient_note: str,
    required_values: List[str],
) -> Dict[str, Any]:
    """
    Extract specific clinical values from a patient note.

    Given the patient note and list of required values, find and extract each one.
    Convert all values to standard units (kg for weight, m for height, etc.).

    IMPORTANT for sex/gender:
    - "man", "male", "he", "his" → sex = "male"
    - "woman", "female", "she", "her" → sex = "female"
    - Always infer sex from these terms - do NOT mark as missing!

    Return:
    {
        "extracted": {"value_name": numeric_value, ...},
        "missing": ["list of values not found"],
        "notes": "any relevant observations"
    }
    """
    ...


# Build docstring for perform_calculation with formula reference
_PERFORM_CALCULATION_DOCSTRING = f"""
Perform the medical calculation using the extracted values.

{FORMULA_REFERENCE}

Use the appropriate formula from the list above for the given calculator.
Show your work step by step.

Return:
{{
    "calculation_steps": ["step 1: ...", "step 2: ...", ...],
    "result": numeric_answer,
    "unit": "unit of the result",
    "interpretation": "brief clinical interpretation"
}}
"""


def _perform_calculation(
    calculator_name: str,
    values: Dict[str, Any],
) -> Dict[str, Any]:
    ...


# Set docstring before applying decorator
_perform_calculation.__doc__ = _PERFORM_CALCULATION_DOCSTRING
perform_calculation = ptool(model="deepseek-v3-0324", output_mode="structured")(_perform_calculation)


def python_calculate(patient_note: str, question: str) -> Dict[str, Any]:
    """
    Execute calculation using Python formulas (no LLM).

    This tool uses verified Python implementations for ALL 55 medical calculators.
    It is deterministic and accurate - use this for the actual calculation.

    SUPPORTED CALCULATORS (55 total):

    Physical/Anthropometric (7):
    - Body Mass Index (BMI), Ideal Body Weight, Adjusted Body Weight
    - Body Surface Area, Mean Arterial Pressure (MAP)
    - Target Weight, Maintenance Fluids

    Renal Function (3):
    - Creatinine Clearance (Cockcroft-Gault), CKD-EPI GFR, MDRD GFR

    Electrolytes/Metabolic (10):
    - Anion Gap, Albumin Corrected Anion Gap
    - Delta Gap, Albumin Corrected Delta Gap
    - Delta Ratio, Albumin Corrected Delta Ratio
    - Serum Osmolality, Free Water Deficit
    - Sodium Correction, Calcium Correction, LDL Calculated
    - Fractional Excretion of Sodium (FENa)

    Cardiac (9):
    - QTc Bazett, QTc Fridericia, QTc Framingham, QTc Hodges, QTc Rautaharju
    - CHA2DS2-VASc Score, HEART Score
    - Revised Cardiac Risk Index (RCRI), Wells' Criteria for PE

    Hepatic (4):
    - Fibrosis-4 (FIB-4), MELD-Na, Child-Pugh Score
    - Steroid Conversion Calculator

    Pulmonary (4):
    - CURB-65 Score, PSI/PORT Score, PERC Rule, SOFA Score

    Infectious/Inflammatory (4):
    - Centor Score (McIsaac), FeverPAIN Score, SIRS Criteria
    - Glasgow-Blatchford Bleeding Score (GBS)

    Hematologic/Coagulation (4):
    - HAS-BLED Score, Wells' Criteria for DVT
    - Caprini VTE Score, Morphine Milligram Equivalents (MME)

    ICU Scoring (2):
    - APACHE II Score, Charlson Comorbidity Index (CCI)

    Obstetric (3):
    - Gestational Age, Estimated Due Date, Date of Conception

    Other (5):
    - Glasgow Coma Score (GCS), HOMA-IR, Framingham Risk Score

    Args:
        patient_note: The full patient note text
        question: The calculation question

    Returns:
        {
            "calculator_name": "name of calculator used",
            "result": numeric_answer,
            "extracted_values": {"value_name": value, ...},
            "formula_used": "formula description"
        }
    """
    from . import calculators

    calc_result = calculators.calculate(patient_note, question)
    if calc_result is None:
        return {
            "error": "Could not identify calculator or extract required values",
            "result": None
        }

    return {
        "calculator_name": calc_result.calculator_name,
        "result": calc_result.result,
        "extracted_values": calc_result.extracted_values,
        "formula_used": calc_result.formula_used,
    }


def compute_calculation(calculator_name: str, values: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute a medical calculation using pre-extracted values directly.

    This tool accepts structured input - NO text parsing needed.
    Call this AFTER you have extracted the required values from the patient note.

    Args:
        calculator_name: The exact calculator name from the available list
        values: Dictionary of parameter names to values

    Examples:
        compute_calculation(
            "Creatinine Clearance (Cockcroft-Gault Equation)",
            {"age": 71, "sex": "male", "weight_kg": 59, "height_cm": 155, "creatinine_mg_dl": 1.42}
        )

        compute_calculation(
            "CKD-EPI Equations for Glomerular Filtration Rate",
            {"age": 65, "sex": "female", "creatinine_mg_dl": 0.9}
        )

        compute_calculation(
            "CHA2DS2-VASc Score for Atrial Fibrillation Stroke Risk",
            {"age": 72, "sex": "female", "has_chf": False, "has_hypertension": True,
             "has_diabetes": True, "has_stroke_tia": False, "has_vascular_disease": False}
        )

    Returns:
        {"calculator_name": str, "result": numeric_answer, "formula_used": str}
        OR {"error": str, "result": None} if calculation fails
    """
    result = compute_direct(calculator_name, values)
    if result is None:
        return {
            "error": f"Calculation failed. Check calculator name and required values. Name: {calculator_name}, Values: {values}",
            "result": None
        }

    return {
        "calculator_name": result.calculator_name,
        "result": result.result,
        "extracted_values": result.extracted_values,
        "formula_used": result.formula_used,
    }


def build_goal_prompt(instance) -> str:
    """Build goal prompt with dynamic calculator signatures."""
    # Get calculator signatures for prompt
    sigs = get_signatures()

    # Build calculator reference
    calc_lines = []
    for name, sig in sigs.items():
        required = ", ".join(sig["required"])
        optional = ", ".join(sig.get("optional", []))
        calc_lines.append(f"• {name}")
        calc_lines.append(f"  Required: {required}")
        if optional:
            calc_lines.append(f"  Optional: {optional}")
        calc_lines.append(f"  Formula: {sig['formula']}")
    calc_reference = "\n".join(calc_lines)

    return f"""Patient Note:
{instance.patient_note}

Task: {instance.question}

## Available Calculators and Required Parameters:
{calc_reference}

## Instructions:
1. Read the task to identify which calculator is needed
2. Extract ALL required parameters from the patient note (pay attention to units!)
3. Call compute_calculation(calculator_name, {{"param": value, ...}}) with the extracted values
4. The compute_calculation tool returns the exact numeric result - use it as your final answer

## CRITICAL Output Rules:
- Show your reasoning in your response
- When you have the FINAL numeric result, output it as: <answer>X.XX</answer>
- Do NOT put intermediate calculations (BMI, IBW) in <answer> tags
- The <answer> tag must contain ONLY the final result for the task question
- Example: If task asks for Creatinine Clearance and result is 35.33, output: <answer>35.33</answer>
"""


def get_medcalc_ptools(use_direct_compute: bool = True) -> List[PToolSpec]:
    """
    Get list of MedCalc ptools for the ReAct agent.

    Args:
        use_direct_compute: If True (default), use compute_calculation which accepts
            structured values directly. If False, use legacy python_calculate.
    """
    from ptool_framework.ptool import get_registry, PToolSpec

    registry = get_registry()
    ptools = []

    # LLM-based ptools for extraction (still useful)
    for name in ["extract_clinical_values"]:
        if name in registry:
            spec = registry.get(name)
            if spec:
                ptools.append(spec)

    if use_direct_compute:
        # NEW: Direct computation with structured values (no text parsing)
        compute_spec = PToolSpec(
            name="compute_calculation",
            func=compute_calculation,
            docstring=compute_calculation.__doc__ or "",
            parameters={
                "calculator_name": str,
                "values": Dict[str, Any],
            },
            return_type=Dict[str, Any],
            model="python",  # Direct Python execution
            output_mode="structured",
        )
        ptools.append(compute_spec)
    else:
        # Legacy: text-based python_calculate (prone to format mismatch)
        python_calc_spec = PToolSpec(
            name="python_calculate",
            func=python_calculate,
            docstring=python_calculate.__doc__ or "",
            parameters={
                "patient_note": str,
                "question": str,
            },
            return_type=Dict[str, Any],
            model="python",
            output_mode="structured",
        )
        ptools.append(python_calc_spec)

    return ptools


class L3ReactExperiment(BaseExperiment):
    """
    L3 Experiment: ReActAgent with multi-step reasoning.

    Uses think-act-observe loop to break down medical calculations.
    """

    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        self.model = config.model
        self.max_steps = config.max_steps
        self.agent: Optional[ReActAgent] = None

    def setup(self):
        """Initialize the ReAct agent with MedCalc ptools."""
        # Ensure ptools are registered by importing them
        _ = extract_clinical_values

        # Get ptools - use direct compute_calculation (structured values)
        ptools = get_medcalc_ptools(use_direct_compute=True)

        self.agent = ReActAgent(
            available_ptools=ptools,
            model=self.model,
            max_steps=self.max_steps,
            echo=False,  # Disable verbose output
        )
        self._setup_complete = True

    def _extract_answer_from_tags(self, result) -> Optional[float]:
        """Extract final answer from explicit <answer> tags."""
        # Check result.answer first
        if result.answer:
            match = re.search(r'<answer>([\d.]+)</answer>', str(result.answer))
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    pass

        # Check trajectory thoughts for <answer> tags
        if result.trajectory:
            for step in reversed(result.trajectory.steps):
                if step.thought and step.thought.content:
                    match = re.search(r'<answer>([\d.]+)</answer>', step.thought.content)
                    if match:
                        try:
                            return float(match.group(1))
                        except ValueError:
                            continue

        # Final fallback: try to extract plain number from result.answer
        if result.answer:
            try:
                return float(result.answer)
            except (ValueError, TypeError):
                # Try extract_number from base class
                return self.extract_number(str(result.answer))

        return None

    def run_instance(self, instance: MedCalcInstance) -> ExperimentResult:
        """
        Run ReAct agent on a single instance.
        """
        if self.agent is None:
            self.setup()

        # Construct goal with dynamic calculator signatures
        goal = build_goal_prompt(instance)

        start_time = time.time()

        try:
            # Run ReAct agent
            result = self.agent.run(goal)
            latency_ms = (time.time() - start_time) * 1000

            # Extract answer using explicit <answer> tags (primary method)
            predicted = self._extract_answer_from_tags(result)

            # Fallback: extract from last successful compute_calculation result
            if predicted is None and result.trajectory:
                for step in reversed(result.trajectory.steps):
                    if (step.observation and
                        step.observation.success and
                        step.observation.result):
                        obs_result = step.observation.result
                        # Only use results from compute_calculation (not intermediate extractions)
                        if isinstance(obs_result, dict):
                            calc_name = obs_result.get("calculator_name", "")
                            # Skip intermediate calculations like BMI or IBW
                            if calc_name and "BMI" not in calc_name and "Ideal Body Weight" not in calc_name:
                                if "result" in obs_result and obs_result["result"] is not None:
                                    try:
                                        predicted = float(obs_result["result"])
                                        break
                                    except (ValueError, TypeError):
                                        continue

            # Estimate tokens from steps
            total_input = 0
            total_output = 0
            for step in result.trajectory.steps if result.trajectory else []:
                if step.thought:
                    total_input += len(step.thought.content) // 4
                if step.observation and step.observation.result:
                    total_output += len(str(step.observation.result)) // 4

            # Add base prompt tokens
            total_input += len(goal) // 4 + 500

            cost_metrics = calculate_cost(
                input_tokens=total_input,
                output_tokens=total_output,
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
                raw_response=str(result.answer),
                trace=result.trajectory.to_dict() if result.trajectory else None,
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
    Run L3 ReAct experiment standalone.

    Usage:
        # Set API key first
        export TOGETHER_API_KEY="your-key"

        # Run with default settings (first 5 instances)
        python -m benchmark.experiments.l3_react

        # Run with custom number of instances
        python -m benchmark.experiments.l3_react --n 10

        # Run on a specific instance
        python -m benchmark.experiments.l3_react --instance 42

        # Enable verbose mode to see ReAct reasoning
        python -m benchmark.experiments.l3_react --verbose
    """
    import argparse

    parser = argparse.ArgumentParser(description="Run L3 ReAct Experiment")
    parser.add_argument("-n", "--num", type=int, default=5, help="Number of instances to run")
    parser.add_argument("--instance", type=int, help="Run a specific instance by ID")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show ReAct reasoning steps")
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

    print(f"Running L3 ReAct on {len(instances)} instances...\n")

    # Create and run experiment
    config = ABLATION_CONFIGS["l3_react"]
    experiment = L3ReactExperiment(config)
    experiment.setup()

    # Override echo setting if verbose
    if args.verbose and experiment.agent:
        experiment.agent.echo = True

    # Run each instance
    correct_exact = 0
    correct_tolerance = 0
    total = 0
    errors = 0

    for i, instance in enumerate(instances):
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

        if args.verbose and result.trace:
            print(f"  Trace: {len(result.trace.get('steps', []))} steps")

        print()

    # Summary
    print("=" * 60)
    print(f"RESULTS: L3 ReAct on {total} instances")
    print("=" * 60)
    print(f"  Exact match:    {correct_exact}/{total} ({correct_exact/total*100:.1f}%)")
    print(f"  Within 5%:      {correct_tolerance}/{total} ({correct_tolerance/total*100:.1f}%)")
    print(f"  Errors:         {errors}/{total} ({errors/total*100:.1f}%)")
    print("=" * 60)
