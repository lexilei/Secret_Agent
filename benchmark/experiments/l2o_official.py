"""
L2O Experiment: @distilled using Official Calculator Implementations.

Uses the official Python calculator implementations from calculator_implementations/,
with LLM-based parameter extraction. The LLM extracts structured parameters
from patient notes, which are then passed to the official calculators.

Key difference from L2:
- L2 uses our own calculator implementations in calculators.py
- L2O uses the original/official implementations from calculator_implementations/

The official calculators expect structured input with specific formats:
- height: [value, "unit"] or [feet, "ft", inches, "in"]
- weight: [value, "unit"]
- lab values: (value, "unit") tuples
- age: numeric (handled by age_conversion module)
- sex: "Male" or "Female"
"""

import json
import os
import sys
import time
import importlib.util
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ..config import ExperimentConfig
from ..dataset.loader import MedCalcInstance
from ..metrics.cost import calculate_cost
from ..metrics.accuracy import calculate_accuracy
from .base import BaseExperiment, ExperimentResult

from ptool_framework import ptool
from ptool_framework.llm_backend import call_llm

# Path to official calculator implementations
CALC_IMPL_DIR = Path(__file__).parent / "calculator_implementations"


# =============================================================================
# Load calculator mappings
# =============================================================================

def load_calculator_mappings() -> Dict[str, Dict[str, Any]]:
    """Load calculator name to file path mappings from calc_path.json."""
    json_path = CALC_IMPL_DIR / "calc_path.json"
    with open(json_path) as f:
        return json.load(f)


CALCULATOR_MAPPINGS = load_calculator_mappings()


# =============================================================================
# Calculator loader and registry
# =============================================================================

class OfficialCalculatorRegistry:
    """Registry for dynamically loading official calculator implementations."""

    def __init__(self):
        self._modules = {}
        self._functions = {}
        self._load_all_calculators()

    def _load_all_calculators(self):
        """Pre-load all calculator modules."""
        # Add calculator_implementations to path for relative imports
        calc_impl_path = str(CALC_IMPL_DIR)
        if calc_impl_path not in sys.path:
            sys.path.insert(0, calc_impl_path)

        for calc_name, info in CALCULATOR_MAPPINGS.items():
            file_path = info["File Path"]
            # Extract just the filename without directory
            filename = os.path.basename(file_path)
            module_name = filename.replace(".py", "")
            full_path = CALC_IMPL_DIR / filename

            if full_path.exists():
                try:
                    spec = importlib.util.spec_from_file_location(module_name, full_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    self._modules[calc_name] = module

                    # Find the main calculation function
                    func = self._find_calculator_function(module, calc_name)
                    if func:
                        self._functions[calc_name] = func
                except Exception as e:
                    pass  # Skip modules that fail to load

    def _find_calculator_function(self, module, calc_name: str):
        """Find the main calculator function in a module."""
        # Common function name patterns
        patterns = [
            "_explanation",
            "_calculator",
            "compute_",
            "calculate_",
            "generate_",
        ]

        for name in dir(module):
            if name.startswith("_") and not any(p in name for p in patterns):
                continue

            obj = getattr(module, name)
            if callable(obj) and not name.startswith("__"):
                # Check if it looks like a calculator function
                for pattern in patterns:
                    if pattern in name.lower():
                        return obj

        # Fallback: return any function that takes input_parameters/input_variables
        for name in dir(module):
            obj = getattr(module, name)
            if callable(obj) and not name.startswith("_"):
                return obj

        return None

    def get_function(self, calc_name: str):
        """Get the calculator function for a given calculator name."""
        return self._functions.get(calc_name)

    def get_module(self, calc_name: str):
        """Get the calculator module for a given calculator name."""
        return self._modules.get(calc_name)

    def list_calculators(self) -> List[str]:
        """List all available calculator names."""
        return list(self._functions.keys())


# Global registry instance
OFFICIAL_REGISTRY = None


def get_official_registry() -> OfficialCalculatorRegistry:
    """Get the global official calculator registry."""
    global OFFICIAL_REGISTRY
    if OFFICIAL_REGISTRY is None:
        OFFICIAL_REGISTRY = OfficialCalculatorRegistry()
    return OFFICIAL_REGISTRY


# =============================================================================
# Parameter extraction signatures for each calculator type
# =============================================================================

# Maps calculator names to their expected parameter schemas
CALCULATOR_SIGNATURES = {
    "Body Mass Index (BMI)": {
        "required": ["height", "weight"],
        "format": "height: [value, unit] or [ft, 'ft', in, 'in'], weight: [value, unit]",
    },
    "Creatinine Clearance (Cockcroft-Gault Equation)": {
        "required": ["age", "sex", "weight", "height", "creatinine"],
        "format": "age: number, sex: 'Male'/'Female', weight: [value, unit], height: [value, unit], creatinine: [value, unit]",
    },
    "CKD-EPI Equations for Glomerular Filtration Rate": {
        "required": ["age", "sex", "creatinine"],
        "format": "age: number, sex: 'Male'/'Female', creatinine: [value, unit]",
    },
    "Mean Arterial Pressure (MAP)": {
        "required": ["systolic_bp", "diastolic_bp"],
        "format": "systolic_bp: number (mmHg), diastolic_bp: number (mmHg)",
    },
    "Ideal Body Weight": {
        "required": ["height", "sex"],
        "format": "height: [value, unit], sex: 'Male'/'Female'",
    },
    "Anion Gap": {
        "required": ["sodium", "chloride", "bicarbonate"],
        "format": "sodium: [value, unit], chloride: [value, unit], bicarbonate: [value, unit]",
    },
    # Add more as needed - the LLM will adapt based on the question
}


# =============================================================================
# LLM-based parameter extraction
# =============================================================================

def extract_parameters_for_calculator(
    patient_note: str,
    question: str,
    calculator_name: str,
) -> Dict[str, Any]:
    """
    Extract structured parameters from patient note for a specific calculator.

    Uses LLM to extract values in the exact format expected by official calculators.
    """
    # Get signature if available
    sig = CALCULATOR_SIGNATURES.get(calculator_name, {})
    required = sig.get("required", [])
    format_hint = sig.get("format", "")

    prompt = f"""Extract parameters from this patient note for the "{calculator_name}" calculator.

PATIENT NOTE:
{patient_note}

QUESTION:
{question}

CRITICAL: You MUST follow these exact formats for the official calculator to work:

1. For age: Return as [value, "unit"] - ALWAYS include the unit!
   - Example: 65 years old -> [65, "years"]
   - Example: 10 months old -> [10, "months"]
   - NEVER return just a number like 65

2. For height: Return as [value, "unit"] where unit is "cm", "m", "in", or "ft"
   - If feet and inches: [feet_value, "ft", inches_value, "in"]
   - Example: 5'10" -> [5, "ft", 10, "in"]
   - Example: 170 cm -> [170, "cm"]
   - Example: 1.75 m -> [1.75, "m"]

3. For weight: Return as [value, "unit"] where unit is "kg" or "lbs"
   - Example: 70 kg -> [70, "kg"]
   - Example: 154 lbs -> [154, "lbs"]

4. For lab values (creatinine, sodium, chloride, bicarbonate, etc.): Return as [value, "unit"]
   - Example: creatinine 1.2 mg/dL -> [1.2, "mg/dL"]
   - Example: sodium 140 mEq/L -> [140, "mEq/L"]

5. For sex/gender: Return exactly "Male" or "Female"
   - "man", "male", "he", "gentleman" -> "Male"
   - "woman", "female", "she", "lady" -> "Female"

6. For blood pressure: Return numeric values in mmHg
   - systolic_bp: number, diastolic_bp: number

7. For boolean conditions (has_diabetes, has_hypertension, etc.): Return true/false

{f"REQUIRED PARAMETERS: {', '.join(required)}" if required else ""}
{f"FORMAT: {format_hint}" if format_hint else ""}

Return ONLY a JSON object with the extracted parameters. Example:
{{"age": [65, "years"], "sex": "Male", "height": [5, "ft", 10, "in"], "weight": [80, "kg"], "creatinine": [1.2, "mg/dL"]}}
"""

    try:
        response = call_llm(prompt=prompt, model="deepseek-v3-0324", max_tokens=500)

        # Parse JSON from response
        import re
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            params = json.loads(json_match.group())
            # Post-process to ensure correct formats
            params = normalize_official_params(params)
            return params
        return {}
    except Exception as e:
        return {"error": str(e)}


def normalize_unit_string(unit: str) -> str:
    """
    Normalize unit strings to match official calculator expectations.

    Handles common Unicode issues like Greek mu vs micro sign.
    """
    if not isinstance(unit, str):
        return unit

    # Replace Greek letter mu (U+03BC) with micro sign (U+00B5)
    unit = unit.replace('μ', 'µ')

    # Common unit aliases
    aliases = {
        'umol/L': 'µmol/L',
        'umol': 'µmol',
        'ug': 'µg',
        'uL': 'µL',
    }

    return aliases.get(unit, unit)


def normalize_official_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize extracted parameters to match official calculator expectations.

    The official calculators expect specific formats:
    - age: [value, "years"] or [value, "months"]
    - height: [value, "unit"] or [ft, "ft", in, "in"]
    - weight: [value, "unit"]
    - lab values: [value, "unit"] (but used as tuple internally)
    """
    normalized = {}

    for key, value in params.items():
        # Handle age - must be [value, "unit"]
        if key == "age":
            if isinstance(value, (int, float)):
                normalized[key] = [value, "years"]
            elif isinstance(value, list) and len(value) == 2:
                normalized[key] = [value[0], normalize_unit_string(value[1])]
            else:
                normalized[key] = value

        # Handle sex - must be "Male" or "Female"
        elif key == "sex":
            if isinstance(value, str):
                val_lower = value.lower()
                if val_lower in ("male", "man", "m"):
                    normalized[key] = "Male"
                elif val_lower in ("female", "woman", "f"):
                    normalized[key] = "Female"
                else:
                    normalized[key] = value
            else:
                normalized[key] = value

        # Handle height and weight - must be [value, "unit"]
        elif key in ("height", "weight"):
            if isinstance(value, list):
                # Normalize units in the list
                normalized[key] = [
                    normalize_unit_string(v) if isinstance(v, str) else v
                    for v in value
                ]
            elif isinstance(value, (int, float)):
                # Assume default units
                if key == "height":
                    normalized[key] = [value, "cm"]
                else:
                    normalized[key] = [value, "kg"]
            else:
                normalized[key] = value

        # Handle lab values - must be [value, "unit"] or (value, "unit")
        elif key in ("creatinine", "sodium", "chloride", "bicarbonate", "potassium",
                     "albumin", "bilirubin", "platelets", "inr", "glucose"):
            if isinstance(value, list) and len(value) == 2:
                normalized[key] = [value[0], normalize_unit_string(value[1])]
            elif isinstance(value, tuple) and len(value) == 2:
                normalized[key] = [value[0], normalize_unit_string(value[1])]
            elif isinstance(value, (int, float)):
                # Try to infer units
                default_units = {
                    "creatinine": "mg/dL",
                    "sodium": "mEq/L",
                    "chloride": "mEq/L",
                    "bicarbonate": "mEq/L",
                    "potassium": "mEq/L",
                    "albumin": "g/dL",
                    "bilirubin": "mg/dL",
                    "platelets": "10^9/L",
                    "inr": "",
                    "glucose": "mg/dL",
                }
                normalized[key] = [value, default_units.get(key, "")]
            else:
                normalized[key] = value

        else:
            normalized[key] = value

    return normalized


def identify_calculator(question: str) -> Optional[str]:
    """Identify which calculator is needed based on the question."""
    available = list(CALCULATOR_MAPPINGS.keys())

    prompt = f"""Identify which medical calculator is needed for this question.

QUESTION: {question}

AVAILABLE CALCULATORS:
{chr(10).join(f'- {name}' for name in available)}

Return ONLY the exact calculator name from the list above, nothing else.
"""

    try:
        response = call_llm(prompt=prompt, model="deepseek-v3-0324", max_tokens=100)
        response = response.strip().strip('"').strip("'")

        # Try exact match first
        if response in available:
            return response

        # Try fuzzy match
        response_lower = response.lower()
        for name in available:
            if response_lower in name.lower() or name.lower() in response_lower:
                return name

        return None
    except Exception:
        return None


# =============================================================================
# Main calculation function using official implementations
# =============================================================================

def calculate_with_official(
    patient_note: str,
    question: str,
) -> Optional[Dict[str, Any]]:
    """
    Calculate medical value using official calculator implementations.

    1. Identify which calculator is needed
    2. Extract parameters using LLM
    3. Call the official calculator function
    4. Return the result
    """
    registry = get_official_registry()

    # Step 1: Identify calculator
    calc_name = identify_calculator(question)
    if not calc_name:
        return None

    # Step 2: Get the official calculator function
    calc_func = registry.get_function(calc_name)
    if not calc_func:
        return None

    # Step 3: Extract parameters
    params = extract_parameters_for_calculator(patient_note, question, calc_name)
    if not params or "error" in params:
        return None

    # Step 4: Call the official calculator
    try:
        result = calc_func(params)

        if isinstance(result, dict):
            return {
                "calculator_name": calc_name,
                "answer": result.get("Answer"),
                "explanation": result.get("Explanation", ""),
                "extracted_params": params,
                "method": "official",
            }
        return None
    except Exception as e:
        return None


# =============================================================================
# Experiment class
# =============================================================================

class L2OOfficialExperiment(BaseExperiment):
    """
    L2O Experiment: Official Calculator Implementations.

    Uses the original calculator implementations from calculator_implementations/
    with LLM-based parameter extraction.
    """

    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        self.model = config.model
        self.official_success_count = 0
        self.llm_fallback_count = 0
        self.registry = get_official_registry()

    def setup(self):
        """Initialize the experiment."""
        # Ensure registry is loaded
        _ = self.registry.list_calculators()
        self._setup_complete = True

    def run_instance(self, instance: MedCalcInstance) -> ExperimentResult:
        """Run L2O experiment on a single instance."""
        start_time = time.time()
        input_tokens = 0
        output_tokens = 0

        try:
            # Try official calculator first
            result = calculate_with_official(
                patient_note=instance.patient_note,
                question=instance.question,
            )

            latency_ms = (time.time() - start_time) * 1000

            if result and result.get("method") == "official":
                self.official_success_count += 1
                predicted = result.get("answer")

                # Estimate tokens for extraction calls
                input_tokens = 800  # extraction prompt
                output_tokens = 100  # JSON response

                cost_metrics = calculate_cost(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    model=self.model,
                )

                if predicted is not None:
                    try:
                        predicted = float(predicted)
                    except (ValueError, TypeError):
                        predicted = None

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
                    trace={
                        "method": "official",
                        "calculator": result.get("calculator_name"),
                        "params": result.get("extracted_params"),
                        "explanation": result.get("explanation", "")[:500],
                    },
                )

            # Fallback: Use L1 ptool if official fails
            self.llm_fallback_count += 1

            from .l1_ptool import calculate_medical_value as l1_calculate

            l1_result = l1_calculate(
                patient_note=instance.patient_note,
                question=instance.question,
            )

            latency_ms = (time.time() - start_time) * 1000

            # Estimate tokens for L1 call
            input_tokens = 1000
            output_tokens = 200

            cost_metrics = calculate_cost(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model=self.model,
            )

            # Extract numeric result
            predicted = None
            if isinstance(l1_result, (int, float)):
                predicted = float(l1_result)
            elif isinstance(l1_result, dict):
                inner = l1_result.get("result", l1_result)
                if isinstance(inner, (int, float)):
                    predicted = float(inner)

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
                raw_response=str(l1_result),
                trace={
                    "method": "llm_fallback",
                    "result": l1_result,
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
        """Get summary with official vs fallback stats."""
        summary = super().get_summary()
        total = self.official_success_count + self.llm_fallback_count
        summary["official_success_count"] = self.official_success_count
        summary["llm_fallback_count"] = self.llm_fallback_count
        summary["official_success_rate"] = (
            self.official_success_count / total if total > 0 else 0
        )
        return summary


# =============================================================================
# Standalone execution
# =============================================================================

if __name__ == "__main__":
    """
    Run L2O experiment standalone.

    Usage:
        export TOGETHER_API_KEY="your-key"
        python -m benchmark.experiments.l2o_official
        python -m benchmark.experiments.l2o_official --n 10
        python -m benchmark.experiments.l2o_official --instance 42
    """
    import argparse

    parser = argparse.ArgumentParser(description="Run L2O Official Experiment")
    parser.add_argument("-n", "--num", type=int, default=5, help="Number of instances")
    parser.add_argument("--instance", type=int, help="Run specific instance")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    # Load dataset
    print("Loading MedCalc-Bench dataset...")
    from ..dataset.loader import MedCalcDataset
    from ..config import ExperimentConfig, ExperimentLevel

    dataset = MedCalcDataset()

    if args.instance is not None:
        all_instances = dataset.load("test")
        instances = [i for i in all_instances if i.row_number == args.instance]
        if not instances:
            print(f"Instance {args.instance} not found")
            sys.exit(1)
    else:
        instances = dataset.get_debug_subset(args.num)

    print(f"Running L2O Official on {len(instances)} instances...")
    print("(Using official calculator implementations with LLM extraction)")
    print()

    # Create experiment with minimal config
    config = ExperimentConfig(
        name="l2o_official",
        description="@distilled with official calculator implementations",
        level=ExperimentLevel.L2,
        model="deepseek-v3-0324",
    )

    experiment = L2OOfficialExperiment(config)
    experiment.setup()

    # Show available calculators
    print(f"Loaded {len(experiment.registry.list_calculators())} official calculators")
    print()

    # Run instances
    correct = 0
    total = 0
    official_count = 0
    fallback_count = 0

    for i, instance in enumerate(instances):
        print(f"[{i+1}/{len(instances)}] {instance.calculator_name}")
        print(f"  Q: {instance.question[:60]}...")

        result = experiment.run_instance(instance)
        total += 1

        trace = result.trace or {}
        method = trace.get("method", "unknown")

        if method == "official":
            official_count += 1
            method_str = "Official"
        else:
            fallback_count += 1
            method_str = "LLM Fallback"

        if result.is_correct_tolerance:
            correct += 1
            status = f"CORRECT ({method_str})"
        elif result.error:
            status = f"ERROR: {result.error}"
        else:
            status = f"WRONG (pred={result.predicted_answer}, exp={result.ground_truth}) [{method_str}]"

        print(f"  {status}")

        if args.verbose and trace:
            if method == "official":
                print(f"    Calculator: {trace.get('calculator')}")
                print(f"    Params: {trace.get('params')}")

        print()

    # Summary
    print("=" * 60)
    print(f"L2O OFFICIAL RESULTS: {correct}/{total} ({correct/total*100:.1f}%)")
    print(f"  Official: {official_count} instances")
    print(f"  Fallback: {fallback_count} instances")
    print("=" * 60)
