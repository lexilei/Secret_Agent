#!/usr/bin/env python3
"""
MedCalc Agent: Calculate medical scores from clinical notes.

This demonstrates William's vision:
- Python controls the workflow (L0/L1)
- ptools handle reasoning (extraction, interpretation)
- Distilled versions handle common patterns
- Fallback to LLM for edge cases

Run:
    python examples/medcalc_agent.py "65 year old male with CHF, hypertension, diabetes. Calculate stroke risk."
    python examples/medcalc_agent.py "Patient weighs 70kg, height 1.75m. Calculate BMI."
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Literal, Optional, Any
from dataclasses import dataclass

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("TOGETHER_API_KEY", "your-key-here")

from ptool_framework import ptool, distilled, DistillationFallback, enable_tracing, get_trace_store

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ClinicalValues:
    """Extracted clinical values."""
    values: Dict[str, Optional[float]]
    confidence: Dict[str, float]
    raw_mentions: Dict[str, str]

# =============================================================================
# PTOOLS - LLM handles reasoning
# =============================================================================

@ptool(model="deepseek-v3", output_mode="structured")
def identify_calculator(task_description: str) -> Dict[str, str]:
    """Identify which medical calculator to use from a task description.

    Analyze the task and determine:
    1. Which calculator is needed
    2. What values need to be extracted

    Available calculators:
    - CHA2DS2-VASc: Stroke risk in atrial fibrillation
    - BMI: Body Mass Index
    - GFR: Glomerular Filtration Rate (kidney function)
    - MELD: Model for End-Stage Liver Disease
    - Wells: DVT/PE probability
    - CURB-65: Pneumonia severity

    Return:
    {
        "calculator": "calculator name",
        "required_fields": ["field1", "field2", ...],
        "reason": "Why this calculator was chosen"
    }
    """
    ...


@ptool(model="deepseek-v3", output_mode="structured")
def extract_clinical_values(
    clinical_note: str,
    required_fields: List[str]
) -> Dict[str, Any]:
    """Extract clinical values from a medical note.

    Parse the clinical note and extract numeric values for each required field.
    Handle various formats and medical abbreviations.

    Examples of formats to handle:
    - "65 year old" → age: 65
    - "BP 120/80" → systolic_bp: 120, diastolic_bp: 80
    - "Cr 1.2" or "creatinine 1.2 mg/dL" → creatinine: 1.2
    - "BMI 25" or "weighs 70kg, 1.75m tall" → weight_kg: 70, height_m: 1.75
    - "has CHF" or "congestive heart failure" → chf: 1
    - "no diabetes" or "non-diabetic" → diabetes: 0

    Return:
    {
        "values": {"field1": value_or_null, ...},
        "confidence": {"field1": 0.0-1.0, ...},
        "raw_mentions": {"field1": "original text that mentioned this", ...}
    }
    """
    ...


@ptool(model="deepseek-v3", output_mode="structured")
def interpret_score(
    calculator_name: str,
    score_value: float,
    clinical_context: Dict[str, Any]
) -> Dict[str, str]:
    """Interpret a medical score in clinical context.

    Provide:
    1. Risk stratification
    2. Plain English interpretation
    3. Clinical recommendations

    Return:
    {
        "risk_level": "low" | "moderate" | "high" | "critical",
        "interpretation": "Plain English explanation of what this score means",
        "recommendations": "Clinical recommendations based on the score",
        "caveats": "Any limitations or caveats to consider"
    }
    """
    ...


# =============================================================================
# DISTILLED VERSIONS - Python first, LLM fallback
# =============================================================================

@distilled(fallback_ptool="identify_calculator")
def identify_calculator_fast(task_description: str) -> Dict[str, str]:
    """Fast calculator identification for common patterns."""
    task_lower = task_description.lower()

    # CHA2DS2-VASc patterns
    if any(kw in task_lower for kw in ["stroke risk", "afib", "atrial fibrillation", "cha2ds2"]):
        return {
            "calculator": "CHA2DS2-VASc",
            "required_fields": ["age", "female", "chf", "hypertension", "diabetes",
                              "stroke_history", "vascular_disease"],
            "reason": "Stroke risk assessment requested"
        }

    # BMI patterns
    if any(kw in task_lower for kw in ["bmi", "body mass", "weight", "obesity"]):
        return {
            "calculator": "BMI",
            "required_fields": ["weight_kg", "height_m"],
            "reason": "BMI calculation requested"
        }

    # GFR patterns
    if any(kw in task_lower for kw in ["gfr", "kidney", "renal", "creatinine clearance"]):
        return {
            "calculator": "GFR",
            "required_fields": ["creatinine", "age", "female"],
            "reason": "Kidney function assessment requested"
        }

    # MELD patterns
    if any(kw in task_lower for kw in ["meld", "liver", "hepatic", "cirrhosis"]):
        return {
            "calculator": "MELD",
            "required_fields": ["bilirubin", "inr", "creatinine", "sodium"],
            "reason": "Liver disease severity requested"
        }

    # Wells DVT patterns
    if any(kw in task_lower for kw in ["wells", "dvt", "deep vein", "pulmonary embolism", "pe"]):
        return {
            "calculator": "Wells",
            "required_fields": ["active_cancer", "paralysis", "bedridden", "tenderness",
                              "leg_swelling", "pitting_edema", "collateral_veins", "previous_dvt"],
            "reason": "DVT/PE probability assessment requested"
        }

    # CURB-65 patterns
    if any(kw in task_lower for kw in ["curb", "pneumonia", "cap"]):
        return {
            "calculator": "CURB-65",
            "required_fields": ["confusion", "urea", "respiratory_rate", "systolic_bp",
                              "diastolic_bp", "age"],
            "reason": "Pneumonia severity assessment requested"
        }

    # Can't determine - fall back to LLM
    raise DistillationFallback(f"Unknown calculator pattern in: {task_description[:50]}")


@distilled(fallback_ptool="extract_clinical_values")
def extract_clinical_values_fast(
    clinical_note: str,
    required_fields: List[str]
) -> Dict[str, Any]:
    """Fast extraction for common clinical patterns."""
    import re

    values = {}
    confidence = {}
    raw_mentions = {}
    note_lower = clinical_note.lower()

    # Define extraction patterns
    patterns = {
        # Demographics
        "age": [
            (r"(\d+)\s*(?:year|yr|y/?o|years?\s*old)", 0.95),
            (r"age[:\s]+(\d+)", 0.95),
        ],
        "female": [
            (r"\b(female|woman|f)\b", 0.9, 1),
            (r"\b(male|man|m)\b", 0.9, 0),
        ],

        # Vitals
        "weight_kg": [
            (r"(\d+(?:\.\d+)?)\s*kg", 0.95),
            (r"weight[:\s]+(\d+(?:\.\d+)?)", 0.8),
        ],
        "height_m": [
            (r"(\d+(?:\.\d+)?)\s*m(?:eter)?s?\b(?!\w)", 0.95),
            (r"height[:\s]+(\d+(?:\.\d+)?)", 0.8),
        ],
        "systolic_bp": [
            (r"(?:bp|blood pressure)[:\s]*(\d+)/\d+", 0.95),
            (r"systolic[:\s]+(\d+)", 0.9),
        ],
        "diastolic_bp": [
            (r"(?:bp|blood pressure)[:\s]*\d+/(\d+)", 0.95),
            (r"diastolic[:\s]+(\d+)", 0.9),
        ],

        # Labs
        "creatinine": [
            (r"(?:cr|creatinine)[:\s]+(\d+(?:\.\d+)?)", 0.9),
        ],
        "bilirubin": [
            (r"(?:bili|bilirubin)[:\s]+(\d+(?:\.\d+)?)", 0.9),
        ],
        "inr": [
            (r"inr[:\s]+(\d+(?:\.\d+)?)", 0.95),
        ],

        # Conditions (binary)
        "chf": [
            (r"\b(?:chf|congestive heart failure|heart failure)\b", 0.9, 1),
            (r"\bno\s+(?:chf|heart failure)\b", 0.9, 0),
        ],
        "hypertension": [
            (r"\b(?:htn|hypertension|high blood pressure)\b", 0.9, 1),
            (r"\bno\s+(?:htn|hypertension)\b", 0.9, 0),
        ],
        "diabetes": [
            (r"\b(?:dm|diabetes|diabetic)\b", 0.9, 1),
            (r"\b(?:no diabetes|non-diabetic)\b", 0.9, 0),
        ],
        "stroke_history": [
            (r"\b(?:stroke|tia|cva)\s*(?:history|hx)?\b", 0.85, 1),
            (r"\bprevious\s+(?:stroke|tia|cva)\b", 0.9, 1),
        ],
        "vascular_disease": [
            (r"\b(?:mi|myocardial infarction|pad|peripheral arterial|aortic plaque)\b", 0.85, 1),
        ],
    }

    extracted_count = 0
    for field in required_fields:
        values[field] = None
        confidence[field] = 0.0
        raw_mentions[field] = ""

        if field in patterns:
            for pattern_tuple in patterns[field]:
                if len(pattern_tuple) == 2:
                    pattern, conf = pattern_tuple
                    fixed_value = None
                else:
                    pattern, conf, fixed_value = pattern_tuple

                match = re.search(pattern, note_lower)
                if match:
                    if fixed_value is not None:
                        values[field] = fixed_value
                    else:
                        try:
                            values[field] = float(match.group(1))
                        except (ValueError, IndexError):
                            continue
                    confidence[field] = conf
                    raw_mentions[field] = match.group(0)
                    extracted_count += 1
                    break

    # If we couldn't extract enough required fields, fall back to LLM
    missing = [f for f in required_fields if values.get(f) is None]
    if len(missing) > len(required_fields) * 0.5:
        raise DistillationFallback(f"Too many missing fields: {missing}")

    return {
        "values": values,
        "confidence": confidence,
        "raw_mentions": raw_mentions
    }


# =============================================================================
# CALCULATORS - Pure Python (deterministic)
# =============================================================================

def calculate_cha2ds2_vasc(values: Dict[str, Optional[float]]) -> Dict[str, Any]:
    """CHA2DS2-VASc score for stroke risk in atrial fibrillation."""
    score = 0
    breakdown = []

    age = values.get("age", 0) or 0
    if age >= 75:
        score += 2
        breakdown.append("Age ≥75: +2")
    elif age >= 65:
        score += 1
        breakdown.append("Age 65-74: +1")

    if values.get("female"):
        score += 1
        breakdown.append("Female: +1")

    if values.get("chf"):
        score += 1
        breakdown.append("CHF: +1")

    if values.get("hypertension"):
        score += 1
        breakdown.append("Hypertension: +1")

    if values.get("diabetes"):
        score += 1
        breakdown.append("Diabetes: +1")

    if values.get("stroke_history"):
        score += 2
        breakdown.append("Stroke/TIA history: +2")

    if values.get("vascular_disease"):
        score += 1
        breakdown.append("Vascular disease: +1")

    return {
        "score": score,
        "max_score": 9,
        "breakdown": breakdown,
        "unit": "points"
    }


def calculate_bmi(values: Dict[str, Optional[float]]) -> Dict[str, Any]:
    """Body Mass Index calculation."""
    weight = values.get("weight_kg")
    height = values.get("height_m")

    if not weight or not height:
        raise ValueError("Missing weight or height for BMI calculation")

    bmi = weight / (height ** 2)

    return {
        "score": round(bmi, 1),
        "max_score": None,
        "breakdown": [f"BMI = {weight}kg / ({height}m)² = {bmi:.1f}"],
        "unit": "kg/m²"
    }


def calculate_gfr(values: Dict[str, Optional[float]]) -> Dict[str, Any]:
    """eGFR using CKD-EPI equation (simplified)."""
    creatinine = values.get("creatinine")
    age = values.get("age")
    female = values.get("female", 0)

    if not creatinine or not age:
        raise ValueError("Missing creatinine or age for GFR calculation")

    # Simplified CKD-EPI
    if female:
        if creatinine <= 0.7:
            gfr = 144 * (creatinine/0.7)**(-0.329) * (0.993)**age
        else:
            gfr = 144 * (creatinine/0.7)**(-1.209) * (0.993)**age
    else:
        if creatinine <= 0.9:
            gfr = 141 * (creatinine/0.9)**(-0.411) * (0.993)**age
        else:
            gfr = 141 * (creatinine/0.9)**(-1.209) * (0.993)**age

    return {
        "score": round(gfr, 1),
        "max_score": None,
        "breakdown": [f"eGFR (CKD-EPI) = {gfr:.1f} mL/min/1.73m²"],
        "unit": "mL/min/1.73m²"
    }


CALCULATORS = {
    "CHA2DS2-VASc": calculate_cha2ds2_vasc,
    "BMI": calculate_bmi,
    "GFR": calculate_gfr,
}


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def medcalc_workflow(
    clinical_note: str,
    use_distilled: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Main MedCalc workflow.

    This is William's L0/L1 architecture:
    - Python controls the flow
    - ptools (or distilled versions) handle reasoning
    - Pure Python does calculations
    """
    results = {
        "input": clinical_note,
        "steps": [],
        "method_used": {}
    }

    # Step 1: Identify calculator
    if verbose:
        print("\n🔍 Step 1: Identifying calculator...")

    if use_distilled:
        try:
            calc_info = identify_calculator_fast(clinical_note)
            results["method_used"]["identify"] = "distilled"
        except DistillationFallback:
            calc_info = identify_calculator(clinical_note)
            results["method_used"]["identify"] = "llm"
    else:
        calc_info = identify_calculator(clinical_note)
        results["method_used"]["identify"] = "llm"

    calculator_name = calc_info["calculator"]
    required_fields = calc_info["required_fields"]
    results["calculator"] = calculator_name
    results["required_fields"] = required_fields

    if verbose:
        print(f"   Calculator: {calculator_name}")
        print(f"   Required fields: {required_fields}")

    # Step 2: Extract clinical values
    if verbose:
        print("\n📋 Step 2: Extracting clinical values...")

    if use_distilled:
        try:
            extraction = extract_clinical_values_fast(clinical_note, required_fields)
            results["method_used"]["extract"] = "distilled"
        except DistillationFallback as e:
            if verbose:
                print(f"   (Falling back to LLM: {e})")
            extraction = extract_clinical_values(clinical_note, required_fields)
            results["method_used"]["extract"] = "llm"
    else:
        extraction = extract_clinical_values(clinical_note, required_fields)
        results["method_used"]["extract"] = "llm"

    values = extraction["values"]
    results["extracted_values"] = extraction

    if verbose:
        print(f"   Extracted: {values}")

    # Step 3: Calculate score (pure Python)
    if verbose:
        print("\n🧮 Step 3: Calculating score...")

    calculator_fn = CALCULATORS.get(calculator_name)
    if not calculator_fn:
        raise ValueError(f"Calculator '{calculator_name}' not implemented")

    try:
        calculation = calculator_fn(values)
        results["calculation"] = calculation
        results["score"] = calculation["score"]

        if verbose:
            print(f"   Score: {calculation['score']} {calculation.get('unit', '')}")
            for line in calculation.get("breakdown", []):
                print(f"      {line}")

    except ValueError as e:
        results["error"] = str(e)
        if verbose:
            print(f"   Error: {e}")
        return results

    # Step 4: Interpret score (always LLM - needs clinical reasoning)
    if verbose:
        print("\n💡 Step 4: Interpreting score...")

    interpretation = interpret_score(
        calculator_name=calculator_name,
        score_value=calculation["score"],
        clinical_context=values
    )
    results["interpretation"] = interpretation
    results["method_used"]["interpret"] = "llm"

    if verbose:
        print(f"   Risk Level: {interpretation['risk_level']}")
        print(f"   Interpretation: {interpretation['interpretation']}")
        print(f"   Recommendations: {interpretation['recommendations']}")

    return results


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="MedCalc Agent")
    parser.add_argument("note", nargs="?", help="Clinical note or task description")
    parser.add_argument("--no-distill", action="store_true", help="Disable distilled functions")
    parser.add_argument("--trace", action="store_true", help="Enable trace collection")
    parser.add_argument("--stats", action="store_true", help="Show distillation stats")
    args = parser.parse_args()

    if args.trace:
        enable_tracing(True)

    # Example cases if no input provided
    if not args.note:
        examples = [
            "65 year old male with CHF, hypertension, and diabetes. History of stroke. Calculate stroke risk.",
            "Patient weighs 85kg and is 1.80m tall. Calculate BMI.",
            "72 year old female, creatinine 1.4. Assess kidney function.",
        ]
        print("Running example cases:\n")

        for note in examples:
            print("=" * 70)
            result = medcalc_workflow(note, use_distilled=not args.no_distill)
            print()

    else:
        result = medcalc_workflow(args.note, use_distilled=not args.no_distill)
        print("\n" + "=" * 70)
        print(f"Final Result: {result.get('calculator')} = {result.get('score')}")

    if args.stats:
        from ptool_framework import get_distilled_stats
        print("\n📊 Distillation Statistics:")
        for func in [identify_calculator_fast, extract_clinical_values_fast]:
            stats = get_distilled_stats(func)
            if stats and stats["total_calls"] > 0:
                print(f"\n{func.__name__}:")
                print(f"  Total calls: {stats['total_calls']}")
                print(f"  Python success: {stats['python_success']} ({stats['success_rate']:.0%})")
                print(f"  LLM fallbacks: {stats['fallback_count']} ({stats['fallback_rate']:.0%})")

    if args.trace:
        store = get_trace_store()
        stats = store.get_stats()
        print(f"\n📝 Traces collected: {stats['total_traces']}")


if __name__ == "__main__":
    main()
