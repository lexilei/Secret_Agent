#!/usr/bin/env python3
"""
MedCalc ptools - Medical Calculator pseudo-tools for ptool_framework

This module provides LLM-powered functions (ptools) for medical calculations:
- identify_calculator: Determine which calculator to use
- extract_clinical_values: Parse values from clinical notes
- interpret_result: Clinical interpretation of scores

Also includes @distilled versions with Python-first execution and LLM fallback.

Supported calculators:
- BMI (Body Mass Index)
- CHA2DS2-VASc (Stroke risk in atrial fibrillation)
- eGFR (Estimated glomerular filtration rate - kidney function)

Usage:
    from medcalc_ptools import (
        identify_calculator, extract_clinical_values, interpret_result,
        calculate_bmi, calculate_cha2ds2_vasc, calculate_egfr,
        MEDCALC_PTOOLS,
    )
"""

import re
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))
from ptool_framework import ptool, distilled, DistillationFallback, get_registry


# =============================================================================
# PTOOLS - LLM-powered functions
# =============================================================================

@ptool(model="deepseek-v3-0324", output_mode="structured")
def identify_calculator(clinical_text: str) -> Dict[str, Any]:
    """Identify which medical calculator to use from clinical text.

    Analyze the text and determine:
    1. Which calculator is needed
    2. What values need to be extracted

    Available calculators:
    - BMI: Body Mass Index (requires: weight_kg, height_m)
    - CHA2DS2-VASc: Stroke risk in atrial fibrillation (requires: age, sex, chf, hypertension, diabetes, stroke_history, vascular_disease)
    - eGFR: Estimated glomerular filtration rate (requires: creatinine, age, sex)

    Return JSON with:
    {
        "calculator": "BMI" | "CHA2DS2-VASc" | "eGFR",
        "required_fields": ["field1", "field2", ...],
        "reason": "Brief explanation of why this calculator was chosen"
    }
    """
    ...


@ptool(model="deepseek-v3-0324", output_mode="structured")
def extract_clinical_values(
    clinical_text: str,
    required_fields: List[str]
) -> Dict[str, Any]:
    """Extract clinical values from a medical note.

    Parse the clinical note and extract numeric values for each required field.
    Handle various formats and medical abbreviations.

    Common formats:
    - "65 year old" -> age: 65
    - "BP 120/80" -> systolic_bp: 120, diastolic_bp: 80
    - "Cr 1.2" or "creatinine 1.2 mg/dL" -> creatinine: 1.2
    - "weighs 70kg, 1.75m tall" -> weight_kg: 70, height_m: 1.75
    - "has CHF" -> chf: 1 (present)
    - "no diabetes" -> diabetes: 0 (absent)
    - "male" / "female" -> sex: "male" / "female"

    Return JSON with:
    {
        "values": {"field1": value_or_null, ...},
        "confidence": {"field1": 0.0-1.0, ...},
        "missing": ["fields that could not be extracted"]
    }
    """
    ...


@ptool(model="deepseek-v3-0324", output_mode="structured")
def interpret_result(
    calculator_name: str,
    score: float,
    context: Dict[str, Any]
) -> Dict[str, str]:
    """Interpret a medical score in clinical context.

    Provide:
    1. Risk stratification
    2. Plain English interpretation
    3. Clinical recommendations (educational only)

    Return JSON with:
    {
        "risk_level": "low" | "moderate" | "high" | "very_high",
        "interpretation": "Plain English explanation",
        "recommendations": "General clinical guidance",
        "caveats": "Important limitations to note"
    }
    """
    ...


# =============================================================================
# DISTILLED VERSIONS - Python first, LLM fallback
# =============================================================================

@distilled(fallback_ptool="identify_calculator")
def identify_calculator_fast(clinical_text: str) -> Dict[str, Any]:
    """Fast calculator identification using pattern matching."""
    text_lower = clinical_text.lower()

    # BMI patterns
    if any(kw in text_lower for kw in ["bmi", "body mass", "weight", "height", "obesity", "overweight"]):
        if any(kw in text_lower for kw in ["kg", "pound", "lb", "meter", "cm", "feet", "tall"]):
            return {
                "calculator": "BMI",
                "required_fields": ["weight_kg", "height_m"],
                "reason": "Weight/height mentioned - BMI calculation"
            }

    # CHA2DS2-VASc patterns
    if any(kw in text_lower for kw in ["stroke risk", "afib", "atrial fibrillation", "cha2ds2", "anticoagulation"]):
        return {
            "calculator": "CHA2DS2-VASc",
            "required_fields": ["age", "sex", "chf", "hypertension", "diabetes", "stroke_history", "vascular_disease"],
            "reason": "Stroke risk assessment requested"
        }

    # eGFR patterns
    if any(kw in text_lower for kw in ["gfr", "egfr", "kidney", "renal function", "creatinine clearance"]):
        return {
            "calculator": "eGFR",
            "required_fields": ["creatinine", "age", "sex"],
            "reason": "Kidney function assessment requested"
        }

    # Can't determine - fall back to LLM
    raise DistillationFallback(f"No clear calculator pattern found")


@distilled(fallback_ptool="extract_clinical_values")
def extract_clinical_values_fast(
    clinical_text: str,
    required_fields: List[str]
) -> Dict[str, Any]:
    """Fast value extraction using regex patterns."""
    values = {}
    confidence = {}
    missing = []
    text_lower = clinical_text.lower()

    # Define extraction patterns
    patterns = {
        # Age
        "age": [
            (r"(\d+)\s*(?:year|yr|y/?o|years?\s*old)", 0.95),
            (r"age[:\s]+(\d+)", 0.95),
            (r"(\d+)\s*yo\b", 0.9),
        ],
        # Sex
        "sex": [
            (r"\b(male|man|gentleman)\b", 0.9, "male"),
            (r"\b(female|woman|lady)\b", 0.9, "female"),
            (r"\b(m)\b(?![a-z])", 0.7, "male"),
            (r"\b(f)\b(?![a-z])", 0.7, "female"),
        ],
        # Weight
        "weight_kg": [
            (r"(\d+(?:\.\d+)?)\s*kg", 0.95),
            (r"weight[:\s]+(\d+(?:\.\d+)?)\s*(?:kg)?", 0.85),
            (r"weighs?\s+(\d+(?:\.\d+)?)\s*kg", 0.95),
        ],
        # Height
        "height_m": [
            (r"(\d+(?:\.\d+)?)\s*m(?:eter)?s?\b(?!g|l)", 0.95),
            (r"height[:\s]+(\d+(?:\.\d+)?)\s*(?:m)?", 0.85),
            (r"(\d+)\s*cm", 0.9, "cm"),  # Will convert
        ],
        # Creatinine
        "creatinine": [
            (r"(?:cr|creatinine)[:\s]+(\d+(?:\.\d+)?)", 0.9),
            (r"creatinine\s+(?:of\s+)?(\d+(?:\.\d+)?)", 0.9),
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
            (r"\b(?:dm|diabetes|diabetic|dm2|t2dm)\b", 0.9, 1),
            (r"\b(?:no diabetes|non-diabetic)\b", 0.9, 0),
        ],
        "stroke_history": [
            (r"\b(?:prior\s+)?(?:stroke|tia|cva)\b", 0.85, 1),
            (r"\bh/o\s+(?:stroke|tia|cva)\b", 0.9, 1),
            (r"\bno\s+(?:prior\s+)?(?:stroke|tia)\b", 0.9, 0),
        ],
        "vascular_disease": [
            (r"\b(?:mi|myocardial infarction|pad|peripheral arterial|pvd|aortic plaque|cad)\b", 0.85, 1),
            (r"\bno\s+(?:cad|pad|vascular disease)\b", 0.9, 0),
        ],
    }

    for field in required_fields:
        values[field] = None
        confidence[field] = 0.0

        if field not in patterns:
            missing.append(field)
            continue

        for pattern_tuple in patterns[field]:
            if len(pattern_tuple) == 2:
                pattern, conf = pattern_tuple
                fixed_value = None
            elif len(pattern_tuple) == 3:
                pattern, conf, fixed_value = pattern_tuple
            else:
                continue

            match = re.search(pattern, text_lower)
            if match:
                if fixed_value is not None:
                    if fixed_value == "cm":
                        # Convert cm to m
                        values[field] = float(match.group(1)) / 100
                    else:
                        values[field] = fixed_value
                else:
                    try:
                        values[field] = float(match.group(1))
                    except (ValueError, IndexError):
                        continue
                confidence[field] = conf
                break

        if values[field] is None:
            missing.append(field)

    # If too many fields missing, fall back to LLM
    if len(missing) > len(required_fields) * 0.5:
        raise DistillationFallback(f"Too many missing fields: {missing}")

    return {
        "values": values,
        "confidence": confidence,
        "missing": missing
    }


# =============================================================================
# PURE PYTHON CALCULATORS - Deterministic
# =============================================================================

def calculate_bmi(weight_kg: float, height_m: float) -> Dict[str, Any]:
    """Calculate Body Mass Index.

    BMI = weight(kg) / height(m)^2

    Categories:
    - <18.5: Underweight
    - 18.5-24.9: Normal
    - 25-29.9: Overweight
    - 30+: Obese
    """
    if weight_kg <= 0 or height_m <= 0:
        raise ValueError("Weight and height must be positive")

    bmi = weight_kg / (height_m ** 2)

    if bmi < 18.5:
        category = "Underweight"
    elif bmi < 25:
        category = "Normal"
    elif bmi < 30:
        category = "Overweight"
    else:
        category = "Obese"

    return {
        "score": round(bmi, 1),
        "category": category,
        "unit": "kg/m^2",
        "calculation": f"{weight_kg} / ({height_m})^2 = {bmi:.1f}"
    }


def calculate_cha2ds2_vasc(
    age: float,
    sex: str,
    chf: int,
    hypertension: int,
    diabetes: int,
    stroke_history: int,
    vascular_disease: int
) -> Dict[str, Any]:
    """Calculate CHA2DS2-VASc stroke risk score.

    Points:
    - C: CHF = 1
    - H: Hypertension = 1
    - A2: Age >=75 = 2, Age 65-74 = 1
    - D: Diabetes = 1
    - S2: Prior Stroke/TIA = 2
    - V: Vascular disease = 1
    - A: Age (already counted above)
    - Sc: Sex category (female = 1)

    Annual stroke risk by score:
    0: 0%, 1: 1.3%, 2: 2.2%, 3: 3.2%, 4: 4.0%, 5: 6.7%, 6: 9.8%, 7: 9.6%, 8: 6.7%, 9: 15.2%
    """
    score = 0
    breakdown = []

    # Age
    if age >= 75:
        score += 2
        breakdown.append("Age >= 75: +2")
    elif age >= 65:
        score += 1
        breakdown.append("Age 65-74: +1")

    # Sex
    is_female = sex.lower() in ["female", "f", "woman"]
    if is_female:
        score += 1
        breakdown.append("Female: +1")

    # Conditions
    if chf:
        score += 1
        breakdown.append("CHF: +1")
    if hypertension:
        score += 1
        breakdown.append("Hypertension: +1")
    if diabetes:
        score += 1
        breakdown.append("Diabetes: +1")
    if stroke_history:
        score += 2
        breakdown.append("Stroke/TIA history: +2")
    if vascular_disease:
        score += 1
        breakdown.append("Vascular disease: +1")

    # Annual stroke risk approximation
    risk_table = {0: 0, 1: 1.3, 2: 2.2, 3: 3.2, 4: 4.0, 5: 6.7, 6: 9.8, 7: 9.6, 8: 12.2, 9: 15.2}
    annual_risk = risk_table.get(score, 15.2)

    return {
        "score": score,
        "max_score": 9,
        "annual_stroke_risk_percent": annual_risk,
        "breakdown": breakdown,
        "unit": "points"
    }


def calculate_egfr(creatinine: float, age: float, sex: str) -> Dict[str, Any]:
    """Calculate eGFR using CKD-EPI equation (2021).

    Simplified formula (without race):
    Male: 142 * min(Cr/0.9, 1)^alpha * max(Cr/0.9, 1)^(-1.200) * 0.9938^age
    Female: 142 * min(Cr/0.7, 1)^alpha * max(Cr/0.7, 1)^(-1.200) * 0.9938^age * 1.012

    Categories:
    - >90: Normal or high
    - 60-89: Mildly decreased
    - 45-59: Mildly to moderately decreased
    - 30-44: Moderately to severely decreased
    - 15-29: Severely decreased
    - <15: Kidney failure
    """
    is_female = sex.lower() in ["female", "f", "woman"]

    if is_female:
        kappa = 0.7
        alpha = -0.241
        multiplier = 1.012
    else:
        kappa = 0.9
        alpha = -0.302
        multiplier = 1.0

    cr_ratio = creatinine / kappa
    term1 = min(cr_ratio, 1) ** alpha
    term2 = max(cr_ratio, 1) ** (-1.200)
    age_factor = 0.9938 ** age

    egfr = 142 * term1 * term2 * age_factor * multiplier

    if egfr >= 90:
        stage = "G1 (Normal)"
    elif egfr >= 60:
        stage = "G2 (Mildly decreased)"
    elif egfr >= 45:
        stage = "G3a (Mild-moderate decrease)"
    elif egfr >= 30:
        stage = "G3b (Moderate-severe decrease)"
    elif egfr >= 15:
        stage = "G4 (Severely decreased)"
    else:
        stage = "G5 (Kidney failure)"

    return {
        "score": round(egfr, 1),
        "stage": stage,
        "unit": "mL/min/1.73m^2",
        "calculation": f"CKD-EPI 2021 equation"
    }


# =============================================================================
# CALCULATOR REGISTRY
# =============================================================================

CALCULATORS = {
    "BMI": {
        "function": calculate_bmi,
        "required_fields": ["weight_kg", "height_m"],
        "description": "Body Mass Index"
    },
    "CHA2DS2-VASc": {
        "function": calculate_cha2ds2_vasc,
        "required_fields": ["age", "sex", "chf", "hypertension", "diabetes", "stroke_history", "vascular_disease"],
        "description": "Stroke risk in atrial fibrillation"
    },
    "eGFR": {
        "function": calculate_egfr,
        "required_fields": ["creatinine", "age", "sex"],
        "description": "Estimated glomerular filtration rate"
    }
}


def get_calculator(name: str):
    """Get calculator info by name."""
    return CALCULATORS.get(name)


def list_calculators() -> List[str]:
    """List available calculators."""
    return list(CALCULATORS.keys())


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def get_medcalc_ptools() -> List:
    """Get list of MedCalc PToolSpecs for use with ReActAgent."""
    registry = get_registry()
    ptool_names = ["identify_calculator", "extract_clinical_values", "interpret_result"]
    return [registry.get(name) for name in ptool_names if registry.get(name)]


# Export list for convenience
MEDCALC_PTOOLS = None  # Will be populated when ptools are registered


def register_ptools():
    """Ensure ptools are registered and return the list."""
    global MEDCALC_PTOOLS
    # Force registration by accessing the functions
    _ = identify_calculator
    _ = extract_clinical_values
    _ = interpret_result
    MEDCALC_PTOOLS = get_medcalc_ptools()
    return MEDCALC_PTOOLS


# =============================================================================
# MAIN - Test the ptools
# =============================================================================

if __name__ == "__main__":
    print("MedCalc ptools module")
    print("=" * 40)

    # Test distilled versions
    test_cases = [
        "Patient weighs 75kg and is 1.80m tall. Calculate BMI.",
        "65 year old male with AFib, HTN, diabetes. Calculate stroke risk.",
        "Patient has creatinine 1.4, age 70, female. Check kidney function.",
    ]

    for text in test_cases:
        print(f"\nInput: {text}")
        try:
            result = identify_calculator_fast(text)
            print(f"Calculator: {result['calculator']}")
            print(f"Required: {result['required_fields']}")
        except DistillationFallback as e:
            print(f"Would fall back to LLM: {e}")

    # Test calculators
    print("\n" + "=" * 40)
    print("Calculator tests:")

    bmi = calculate_bmi(75, 1.80)
    print(f"\nBMI: {bmi['score']} {bmi['unit']} ({bmi['category']})")

    cha2ds2 = calculate_cha2ds2_vasc(
        age=65, sex="male", chf=0, hypertension=1, diabetes=1,
        stroke_history=0, vascular_disease=0
    )
    print(f"CHA2DS2-VASc: {cha2ds2['score']}/9 (Annual stroke risk: {cha2ds2['annual_stroke_risk_percent']}%)")

    egfr = calculate_egfr(creatinine=1.4, age=70, sex="female")
    print(f"eGFR: {egfr['score']} {egfr['unit']} ({egfr['stage']})")
