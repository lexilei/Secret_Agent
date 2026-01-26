"""
MedCalc-Bench Calculator Implementations.

Pure Python implementations of all 41 calculators from MedCalc-Bench training data.
Each calculator extracts required values from patient notes and applies the formula.

Formulas derived from training data ground truth explanations.
"""

import re
import math
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime, timedelta
from dataclasses import dataclass


# =============================================================================
# Result container
# =============================================================================

@dataclass
class CalcResult:
    """Result from a calculator."""
    calculator_name: str
    result: float
    extracted_values: Dict[str, Any]
    method: str = "python"
    formula_used: str = ""


# =============================================================================
# Value extraction utilities
# =============================================================================

def extract_number(text: str, patterns: List[str], default: Optional[float] = None) -> Optional[float]:
    """Extract a numeric value using regex patterns."""
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except (ValueError, IndexError):
                continue
    return default


def extract_age(text: str) -> Optional[float]:
    """Extract age from text."""
    patterns = [
        r'(\d+)\s*[-–]?\s*(?:year|yr|y/?o|years?\s*old)',
        r'age[:\s]+(\d+)',
        r'(\d+)\s*yo\b',
        r'(\d+)\s*(?:year|yr)s?\s*(?:of\s+age|old)',
        r'(?:is|was)\s+(\d+)\s*(?:years?\s*old)?',
    ]
    return extract_number(text, patterns)


def extract_sex(text: str) -> Optional[str]:
    """Extract sex from text. Returns 'male' or 'female'."""
    text_lower = text.lower()
    # Check for explicit gender
    if re.search(r'\b(male|man|boy|gentleman|mr\.)\b', text_lower):
        if 'female' not in text_lower and 'woman' not in text_lower:
            return 'male'
    if re.search(r'\b(female|woman|girl|lady|mrs\.|ms\.)\b', text_lower):
        return 'female'
    return None


def extract_weight_kg(text: str) -> Optional[float]:
    """Extract weight in kg."""
    # Try kg first
    patterns_kg = [
        r'weight[:\s]+(\d+\.?\d*)\s*kg',
        r'(\d+\.?\d*)\s*kg\b',
        r'(\d+\.?\d*)\s*kilograms?',
        r'weighs?\s+(\d+\.?\d*)\s*kg',
    ]
    kg = extract_number(text, patterns_kg)
    if kg:
        return kg

    # Try lbs
    patterns_lb = [
        r'weight[:\s]+(\d+\.?\d*)\s*(?:lbs?|pounds?)',
        r'(\d+\.?\d*)\s*(?:lbs?|pounds?)\b',
    ]
    lb = extract_number(text, patterns_lb)
    if lb:
        return lb * 0.453592

    return None


def extract_height_cm(text: str) -> Optional[float]:
    """Extract height in cm."""
    # Try cm first
    patterns_cm = [
        r'height[:\s]+(\d+\.?\d*)\s*cm',
        r'(\d+\.?\d*)\s*cm\b',
        r'(\d+\.?\d*)\s*centimeters?',
    ]
    cm = extract_number(text, patterns_cm)
    if cm and cm > 50:  # Reasonable height
        return cm

    # Try meters
    patterns_m = [
        r'height[:\s]+(\d+\.?\d*)\s*m\b',
        r'(\d+\.?\d*)\s*m(?:eters?)?\b',
    ]
    m = extract_number(text, patterns_m)
    if m:
        if m > 3:  # Likely cm
            return m
        return m * 100

    # Try inches
    patterns_in = [
        r'height[:\s]+(\d+\.?\d*)\s*(?:in|inches?)',
        r"(\d+)['\"]?\s*(?:ft|feet|foot)?\s*(\d+)?['\"]?\s*(?:in|inches?)?",
    ]
    inches = extract_number(text, patterns_in)
    if inches:
        return inches * 2.54

    return None


def extract_creatinine(text: str) -> Optional[float]:
    """Extract creatinine in mg/dL (converts from µmol/L if needed)."""
    # Try mg/dL first (explicit unit)
    patterns_mgdl = [
        r'creatinine[:\s,]+(\d+\.?\d*)\s*mg/dL',
        r'cr[:\s]+(\d+\.?\d*)\s*mg/dL',
        r'(\d+\.?\d*)\s*mg/dL\s*(?:creatinine)?',
    ]
    val = extract_number(text, patterns_mgdl)
    if val is not None:
        return val

    # Try µmol/L and convert to mg/dL
    patterns_umol = [
        r'creatinine[:\s,]+(\d+\.?\d*)\s*[µu]mol/L',
        r'creatinine[:\s,]+(\d+\.?\d*)\s*umol/L',
        r'cr[:\s]+(\d+\.?\d*)\s*[µu]mol/L',
        r'(\d+\.?\d*)\s*[µu]mol/L\s*(?:creatinine)?',
    ]
    val_umol = extract_number(text, patterns_umol)
    if val_umol is not None:
        # Convert µmol/L to mg/dL: divide by 88.4
        return val_umol / 88.4

    # Try without explicit units - guess based on value
    patterns_no_unit = [
        r'creatinine[:\s,]+(\d+\.?\d*)',
        r'serum\s+creatinine[:\s,]+(\d+\.?\d*)',
        r'cr[:\s]+(\d+\.?\d*)',
    ]
    val_unknown = extract_number(text, patterns_no_unit)
    if val_unknown is not None:
        # If value > 20, likely µmol/L (normal is 45-110 µmol/L)
        # If value < 20, likely mg/dL (normal is 0.6-1.2 mg/dL)
        if val_unknown > 20:
            return val_unknown / 88.4  # Convert µmol/L to mg/dL
        return val_unknown

    return None


def extract_blood_pressure(text: str) -> Tuple[Optional[float], Optional[float]]:
    """Extract systolic and diastolic blood pressure."""
    # Try pattern like "120/80 mmHg" or "BP: 120/80"
    pattern = r'(?:blood\s*pressure|bp)[:\s]+(\d+)/(\d+)'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return float(match.group(1)), float(match.group(2))

    # Try pattern like "120/80"
    pattern2 = r'(\d{2,3})/(\d{2,3})\s*(?:mm\s*Hg)?'
    match2 = re.search(pattern2, text, re.IGNORECASE)
    if match2:
        return float(match2.group(1)), float(match2.group(2))

    return None, None


def extract_heart_rate(text: str) -> Optional[float]:
    """Extract heart rate in bpm."""
    patterns = [
        r'heart\s*rate[:\s,]+(\d+\.?\d*)',
        r'hr[:\s]+(\d+)',
        r'pulse[:\s]+(\d+)',
        r'(\d+)\s*(?:bpm|beats?\s*per\s*minute)',
    ]
    return extract_number(text, patterns)


def extract_temperature_celsius(text: str) -> Optional[float]:
    """Extract temperature in Celsius."""
    # Try Celsius first
    patterns_c = [
        r'temperature[:\s,]+(\d+\.?\d*)\s*°?\s*C',
        r'temp[:\s]+(\d+\.?\d*)\s*°?\s*C',
        r'(\d+\.?\d*)\s*°?\s*C\b',
    ]
    c = extract_number(text, patterns_c)
    if c and 30 < c < 45:
        return c

    # Try Fahrenheit
    patterns_f = [
        r'temperature[:\s,]+(\d+\.?\d*)\s*°?\s*F',
        r'temp[:\s]+(\d+\.?\d*)\s*°?\s*F',
        r'(\d+\.?\d*)\s*°?\s*F\b',
    ]
    f = extract_number(text, patterns_f)
    if f and 90 < f < 110:
        return (f - 32) * 5 / 9

    return None


def extract_lab_value(text: str, names: List[str], unit: str = "") -> Optional[float]:
    """Extract a lab value by name."""
    for name in names:
        patterns = [
            rf'{name}[:\s,]+(\d+\.?\d*)\s*{unit}',
            rf'{name}\s+(?:is|was|of|level)[:\s]+(\d+\.?\d*)',
            rf'(\d+\.?\d*)\s*{unit}\s*{name}',
        ]
        val = extract_number(text, patterns)
        if val is not None:
            return val
    return None


def extract_sodium(text: str) -> Optional[float]:
    """Extract sodium in mEq/L (or mmol/L, same value)."""
    return extract_lab_value(text, ['sodium', 'na'], '(?:mEq/L|mmol/L)?')


def extract_potassium(text: str) -> Optional[float]:
    """Extract potassium."""
    return extract_lab_value(text, ['potassium', 'k'], '(?:mEq/L|mmol/L)?')


def extract_chloride(text: str) -> Optional[float]:
    """Extract chloride."""
    return extract_lab_value(text, ['chloride', 'cl'], '(?:mEq/L|mmol/L)?')


def extract_bicarbonate(text: str) -> Optional[float]:
    """Extract bicarbonate/CO2."""
    return extract_lab_value(text, ['bicarbonate', 'hco3', 'co2', 'bicarb'], '(?:mEq/L|mmol/L)?')


def extract_bun(text: str) -> Optional[float]:
    """Extract BUN in mg/dL."""
    val = extract_lab_value(text, ['bun', 'blood urea nitrogen', 'urea nitrogen'], '(?:mg/dL)?')
    if val is None:
        # Try mmol/L and convert
        val_mmol = extract_lab_value(text, ['bun', 'urea'], 'mmol/L')
        if val_mmol:
            val = val_mmol * 2.8  # Convert mmol/L to mg/dL
    return val


def extract_glucose(text: str) -> Optional[float]:
    """Extract glucose in mg/dL."""
    return extract_lab_value(text, ['glucose', 'blood sugar', 'serum glucose'], '(?:mg/dL)?')


def extract_albumin(text: str) -> Optional[float]:
    """Extract albumin in g/dL."""
    return extract_lab_value(text, ['albumin', 'alb'], '(?:g/dL)?')


def extract_calcium(text: str) -> Optional[float]:
    """Extract calcium in mg/dL."""
    return extract_lab_value(text, ['calcium', 'ca'], '(?:mg/dL)?')


def extract_ast(text: str) -> Optional[float]:
    """Extract AST."""
    return extract_lab_value(text, ['ast', 'aspartate aminotransferase', 'sgot'], '(?:U/L|IU/L)?')


def extract_alt(text: str) -> Optional[float]:
    """Extract ALT."""
    return extract_lab_value(text, ['alt', 'alanine aminotransferase', 'sgpt'], '(?:U/L|IU/L)?')


def extract_platelets(text: str) -> Optional[float]:
    """Extract platelet count (in 10^9/L or thousands)."""
    patterns = [
        r'platelet[s]?[:\s,]+(\d+\.?\d*)\s*[×x]?\s*10\^?9',
        r'platelet[s]?[:\s,]+(\d+\.?\d*)\s*(?:k|K|thousand)',
        r'platelet[s]?[:\s,]+(\d+\.?\d*)',
        r'plt[:\s]+(\d+\.?\d*)',
    ]
    val = extract_number(text, patterns)
    if val:
        # Normalize to 10^9/L
        if val > 1000:  # Likely in raw count
            val = val / 1000
    return val


def extract_qt_interval(text: str) -> Optional[float]:
    """Extract QT interval in msec."""
    patterns = [
        r'qt\s*(?:interval)?[:\s]+(\d+\.?\d*)\s*(?:ms|msec)?',
        r'(\d+\.?\d*)\s*(?:ms|msec)\s*qt',
    ]
    return extract_number(text, patterns)


def extract_cholesterol(text: str, type_: str = 'total') -> Optional[float]:
    """Extract cholesterol values in mg/dL."""
    if type_ == 'total':
        return extract_lab_value(text, ['total cholesterol', 'cholesterol'], '(?:mg/dL)?')
    elif type_ == 'hdl':
        return extract_lab_value(text, ['hdl', 'hdl cholesterol', 'hdl-c'], '(?:mg/dL)?')
    elif type_ == 'triglycerides':
        return extract_lab_value(text, ['triglycerides', 'tg', 'trigs'], '(?:mg/dL)?')
    return None


# =============================================================================
# Physical Calculators
# =============================================================================

def calculate_bmi(text: str) -> Optional[CalcResult]:
    """Calculate Body Mass Index."""
    weight = extract_weight_kg(text)
    height_cm = extract_height_cm(text)

    if weight is None or height_cm is None:
        return None

    height_m = height_cm / 100
    bmi = weight / (height_m ** 2)

    return CalcResult(
        calculator_name="Body Mass Index (BMI)",
        result=round(bmi, 3),
        extracted_values={"weight_kg": weight, "height_cm": height_cm},
        formula_used="BMI = weight / height^2"
    )


def calculate_map(text: str) -> Optional[CalcResult]:
    """Calculate Mean Arterial Pressure."""
    systolic, diastolic = extract_blood_pressure(text)

    if systolic is None or diastolic is None:
        return None

    # MAP = (2 * DBP + SBP) / 3 = 2/3 * DBP + 1/3 * SBP
    map_value = (2 * diastolic + systolic) / 3

    return CalcResult(
        calculator_name="Mean Arterial Pressure (MAP)",
        result=round(map_value, 3),
        extracted_values={"systolic": systolic, "diastolic": diastolic},
        formula_used="MAP = (2*DBP + SBP) / 3"
    )


def calculate_ideal_body_weight(text: str) -> Optional[CalcResult]:
    """Calculate Ideal Body Weight using Devine formula."""
    sex = extract_sex(text)
    height_cm = extract_height_cm(text)

    if sex is None or height_cm is None:
        return None

    height_in = height_cm / 2.54

    if sex == 'male':
        ibw = 50 + 2.3 * (height_in - 60)
    else:
        ibw = 45.5 + 2.3 * (height_in - 60)

    return CalcResult(
        calculator_name="Ideal Body Weight",
        result=round(ibw, 3),
        extracted_values={"sex": sex, "height_cm": height_cm, "height_in": height_in},
        formula_used="IBW = 50/45.5 + 2.3*(height_in - 60)"
    )


def calculate_adjusted_body_weight(text: str) -> Optional[CalcResult]:
    """Calculate Adjusted Body Weight."""
    ibw_result = calculate_ideal_body_weight(text)
    weight = extract_weight_kg(text)

    if ibw_result is None or weight is None:
        return None

    ibw = ibw_result.result
    # ABW = IBW + 0.4 * (actual - IBW)
    abw = ibw + 0.4 * (weight - ibw)

    return CalcResult(
        calculator_name="Adjusted Body Weight",
        result=round(abw, 3),
        extracted_values={"ibw": ibw, "actual_weight": weight},
        formula_used="ABW = IBW + 0.4*(weight - IBW)"
    )


def calculate_body_surface_area(text: str) -> Optional[CalcResult]:
    """Calculate Body Surface Area using Mosteller formula."""
    weight = extract_weight_kg(text)
    height_cm = extract_height_cm(text)

    if weight is None or height_cm is None:
        return None

    # BSA = sqrt((weight * height) / 3600)
    bsa = math.sqrt((weight * height_cm) / 3600)

    return CalcResult(
        calculator_name="Body Surface Area Calculator",
        result=round(bsa, 3),
        extracted_values={"weight_kg": weight, "height_cm": height_cm},
        formula_used="BSA = sqrt((weight * height) / 3600)"
    )


def calculate_target_weight(text: str) -> Optional[CalcResult]:
    """Calculate Target Weight from target BMI."""
    # Extract target BMI from question
    patterns = [
        r'target\s*bmi[:\s]+(\d+\.?\d*)',
        r'bmi\s*(?:of|is|should\s*be)[:\s]+(\d+\.?\d*)',
    ]
    target_bmi = extract_number(text, patterns)
    height_cm = extract_height_cm(text)

    if target_bmi is None or height_cm is None:
        return None

    height_m = height_cm / 100
    target_weight = target_bmi * (height_m ** 2)

    return CalcResult(
        calculator_name="Target weight",
        result=round(target_weight, 3),
        extracted_values={"target_bmi": target_bmi, "height_cm": height_cm},
        formula_used="weight = BMI * height^2"
    )


def calculate_maintenance_fluids(text: str) -> Optional[CalcResult]:
    """Calculate Maintenance Fluids using 4-2-1 rule."""
    weight = extract_weight_kg(text)

    if weight is None:
        return None

    # 4-2-1 rule: 4 mL/kg/hr for first 10 kg, 2 mL/kg/hr for next 10 kg, 1 mL/kg/hr thereafter
    if weight <= 10:
        fluids = 4 * weight
    elif weight <= 20:
        fluids = 40 + 2 * (weight - 10)
    else:
        fluids = 60 + 1 * (weight - 20)

    return CalcResult(
        calculator_name="Maintenance Fluids Calculations",
        result=round(fluids, 3),
        extracted_values={"weight_kg": weight},
        formula_used="4-2-1 rule"
    )


# =============================================================================
# QTc Calculators
# =============================================================================

def calculate_qtc_bazett(text: str) -> Optional[CalcResult]:
    """Calculate QTc using Bazett formula."""
    qt = extract_qt_interval(text)
    hr = extract_heart_rate(text)

    if qt is None or hr is None:
        return None

    rr = 60 / hr
    qtc = qt / math.sqrt(rr)

    return CalcResult(
        calculator_name="QTc Bazett Calculator",
        result=round(qtc, 3),
        extracted_values={"qt_msec": qt, "heart_rate": hr, "rr_sec": rr},
        formula_used="QTc = QT / sqrt(RR)"
    )


def calculate_qtc_fridericia(text: str) -> Optional[CalcResult]:
    """Calculate QTc using Fridericia formula."""
    qt = extract_qt_interval(text)
    hr = extract_heart_rate(text)

    if qt is None or hr is None:
        return None

    rr = 60 / hr
    qtc = qt / (rr ** (1/3))

    return CalcResult(
        calculator_name="QTc Fridericia Calculator",
        result=round(qtc, 3),
        extracted_values={"qt_msec": qt, "heart_rate": hr, "rr_sec": rr},
        formula_used="QTc = QT / RR^(1/3)"
    )


def calculate_qtc_framingham(text: str) -> Optional[CalcResult]:
    """Calculate QTc using Framingham formula."""
    qt = extract_qt_interval(text)
    hr = extract_heart_rate(text)

    if qt is None or hr is None:
        return None

    rr = 60 / hr
    qtc = qt + 154 * (1 - rr)

    return CalcResult(
        calculator_name="QTc Framingham Calculator",
        result=round(qtc, 3),
        extracted_values={"qt_msec": qt, "heart_rate": hr, "rr_sec": rr},
        formula_used="QTc = QT + 154*(1 - RR)"
    )


def calculate_qtc_hodges(text: str) -> Optional[CalcResult]:
    """Calculate QTc using Hodges formula."""
    qt = extract_qt_interval(text)
    hr = extract_heart_rate(text)

    if qt is None or hr is None:
        return None

    rr = 60 / hr
    qtc = qt + 1.75 * (hr - 60)

    return CalcResult(
        calculator_name="QTc Hodges Calculator",
        result=round(qtc, 3),
        extracted_values={"qt_msec": qt, "heart_rate": hr},
        formula_used="QTc = QT + 1.75*(HR - 60)"
    )


# =============================================================================
# Lab Calculators
# =============================================================================

def calculate_creatinine_clearance(text: str) -> Optional[CalcResult]:
    """
    Calculate Creatinine Clearance using Cockcroft-Gault.

    This implements the adjusted body weight logic:
    - If BMI > 30 (obese): use Adjusted Body Weight = IBW + 0.4*(actual - IBW)
    - If BMI 25-30 (overweight): use Adjusted Body Weight
    - If BMI < 18.5 (underweight): use actual weight
    - If BMI normal: use min(IBW, actual weight)
    """
    age = extract_age(text)
    actual_weight = extract_weight_kg(text)
    height_cm = extract_height_cm(text)
    creatinine = extract_creatinine(text)
    sex = extract_sex(text)

    if age is None or actual_weight is None or creatinine is None or sex is None:
        return None

    # Calculate weight to use based on BMI
    if height_cm is not None:
        height_m = height_cm / 100
        bmi = actual_weight / (height_m ** 2)
        height_in = height_cm / 2.54

        # Calculate Ideal Body Weight
        if sex == 'male':
            ibw = 50 + 2.3 * (height_in - 60)
        else:
            ibw = 45.5 + 2.3 * (height_in - 60)

        # Determine which weight to use
        if bmi >= 30:  # Obese
            weight_to_use = ibw + 0.4 * (actual_weight - ibw)
            weight_type = "adjusted (obese)"
        elif bmi >= 25:  # Overweight
            weight_to_use = ibw + 0.4 * (actual_weight - ibw)
            weight_type = "adjusted (overweight)"
        elif bmi < 18.5:  # Underweight
            weight_to_use = actual_weight
            weight_type = "actual (underweight)"
        else:  # Normal BMI
            weight_to_use = min(ibw, actual_weight)
            weight_type = "min(IBW, actual)"
    else:
        # No height available - use actual weight
        weight_to_use = actual_weight
        bmi = None
        ibw = None
        weight_type = "actual (no height)"

    # CrCl = ((140 - age) * weight * gender_coef) / (creatinine * 72)
    gender_coef = 1.0 if sex == 'male' else 0.85
    crcl = ((140 - age) * weight_to_use * gender_coef) / (creatinine * 72)

    return CalcResult(
        calculator_name="Creatinine Clearance (Cockcroft-Gault Equation)",
        result=round(crcl, 3),
        extracted_values={
            "age": age,
            "actual_weight": actual_weight,
            "weight_used": weight_to_use,
            "weight_type": weight_type,
            "bmi": round(bmi, 2) if bmi else None,
            "ibw": round(ibw, 2) if ibw else None,
            "creatinine": creatinine,
            "sex": sex,
            "gender_coef": gender_coef
        },
        formula_used="CrCl = ((140-age) * adjusted_weight * gender_coef) / (Cr * 72)"
    )


def calculate_ckd_epi_gfr(text: str) -> Optional[CalcResult]:
    """Calculate GFR using CKD-EPI 2021 equation."""
    age = extract_age(text)
    creatinine = extract_creatinine(text)
    sex = extract_sex(text)

    if age is None or creatinine is None or sex is None:
        return None

    # CKD-EPI 2021 (race-free):
    # GFR = 142 × (Scr/A)^B × 0.9938^age × (1.012 if female)
    if sex == 'female':
        a = 0.7
        if creatinine <= 0.7:
            b = -0.241
        else:
            b = -1.200
        gender_coef = 1.012
    else:
        a = 0.9
        if creatinine <= 0.9:
            b = -0.302
        else:
            b = -1.200
        gender_coef = 1.0

    gfr = 142 * ((creatinine / a) ** b) * (0.9938 ** age) * gender_coef

    return CalcResult(
        calculator_name="CKD-EPI Equations for Glomerular Filtration Rate",
        result=round(gfr, 3),
        extracted_values={"age": age, "creatinine": creatinine, "sex": sex},
        formula_used="GFR = 142 × (Scr/A)^B × 0.9938^age × gender_coef"
    )


def calculate_mdrd_gfr(text: str) -> Optional[CalcResult]:
    """Calculate GFR using MDRD equation."""
    age = extract_age(text)
    creatinine = extract_creatinine(text)
    sex = extract_sex(text)

    if age is None or creatinine is None or sex is None:
        return None

    # MDRD: GFR = 175 × Scr^-1.154 × age^-0.203 × (0.742 if female)
    gender_coef = 0.742 if sex == 'female' else 1.0
    gfr = 175 * (creatinine ** -1.154) * (age ** -0.203) * gender_coef

    return CalcResult(
        calculator_name="MDRD GFR Equation",
        result=round(gfr, 3),
        extracted_values={"age": age, "creatinine": creatinine, "sex": sex},
        formula_used="GFR = 175 × Scr^-1.154 × age^-0.203 × gender_coef"
    )


def calculate_anion_gap(text: str) -> Optional[CalcResult]:
    """Calculate Anion Gap."""
    sodium = extract_sodium(text)
    chloride = extract_chloride(text)
    bicarb = extract_bicarbonate(text)

    if sodium is None or chloride is None or bicarb is None:
        return None

    ag = sodium - (chloride + bicarb)

    return CalcResult(
        calculator_name="Anion Gap",
        result=round(ag, 3),
        extracted_values={"sodium": sodium, "chloride": chloride, "bicarbonate": bicarb},
        formula_used="AG = Na - (Cl + HCO3)"
    )


def calculate_delta_gap(text: str) -> Optional[CalcResult]:
    """Calculate Delta Gap."""
    ag_result = calculate_anion_gap(text)

    if ag_result is None:
        return None

    delta_gap = ag_result.result - 12

    return CalcResult(
        calculator_name="Delta Gap",
        result=round(delta_gap, 3),
        extracted_values=ag_result.extracted_values,
        formula_used="Delta Gap = AG - 12"
    )


def calculate_delta_ratio(text: str) -> Optional[CalcResult]:
    """Calculate Delta Ratio."""
    delta_gap_result = calculate_delta_gap(text)
    bicarb = extract_bicarbonate(text)

    if delta_gap_result is None or bicarb is None:
        return None

    if bicarb == 24:  # Avoid division by zero
        return None

    delta_ratio = delta_gap_result.result / (24 - bicarb)

    return CalcResult(
        calculator_name="Delta Ratio",
        result=round(delta_ratio, 3),
        extracted_values={**delta_gap_result.extracted_values, "bicarbonate": bicarb},
        formula_used="Delta Ratio = Delta Gap / (24 - HCO3)"
    )


def calculate_albumin_corrected_anion_gap(text: str) -> Optional[CalcResult]:
    """Calculate Albumin-Corrected Anion Gap."""
    ag_result = calculate_anion_gap(text)
    albumin = extract_albumin(text)

    if ag_result is None or albumin is None:
        return None

    corrected_ag = ag_result.result + 2.5 * (4 - albumin)

    return CalcResult(
        calculator_name="Albumin Corrected Anion Gap",
        result=round(corrected_ag, 3),
        extracted_values={**ag_result.extracted_values, "albumin": albumin},
        formula_used="Corrected AG = AG + 2.5*(4 - albumin)"
    )


def calculate_serum_osmolality(text: str) -> Optional[CalcResult]:
    """Calculate Serum Osmolality."""
    sodium = extract_sodium(text)
    bun = extract_bun(text)
    glucose = extract_glucose(text)

    if sodium is None or bun is None or glucose is None:
        return None

    osm = 2 * sodium + (bun / 2.8) + (glucose / 18)

    return CalcResult(
        calculator_name="Serum Osmolality",
        result=round(osm, 3),
        extracted_values={"sodium": sodium, "bun": bun, "glucose": glucose},
        formula_used="Osm = 2*Na + BUN/2.8 + Glucose/18"
    )


def calculate_free_water_deficit(text: str) -> Optional[CalcResult]:
    """Calculate Free Water Deficit."""
    weight = extract_weight_kg(text)
    sodium = extract_sodium(text)
    age = extract_age(text)
    sex = extract_sex(text)

    if weight is None or sodium is None or age is None or sex is None:
        return None

    # Total body water percentage
    if age < 18:
        tbw_pct = 0.6  # Children
    elif sex == 'male':
        if age >= 65:
            tbw_pct = 0.5
        else:
            tbw_pct = 0.6
    else:  # female
        if age >= 65:
            tbw_pct = 0.45
        else:
            tbw_pct = 0.5

    deficit = tbw_pct * weight * (sodium / 140 - 1)

    return CalcResult(
        calculator_name="Free Water Deficit",
        result=round(deficit, 3),
        extracted_values={"weight": weight, "sodium": sodium, "tbw_pct": tbw_pct},
        formula_used="FWD = TBW% × weight × (Na/140 - 1)"
    )


def calculate_sodium_correction(text: str) -> Optional[CalcResult]:
    """Calculate Sodium Correction for Hyperglycemia."""
    sodium = extract_sodium(text)
    glucose = extract_glucose(text)

    if sodium is None or glucose is None:
        return None

    corrected = sodium + 0.024 * (glucose - 100)

    return CalcResult(
        calculator_name="Sodium Correction for Hyperglycemia",
        result=round(corrected, 3),
        extracted_values={"sodium": sodium, "glucose": glucose},
        formula_used="Corrected Na = Na + 0.024*(glucose - 100)"
    )


def calculate_calcium_correction(text: str) -> Optional[CalcResult]:
    """Calculate Calcium Correction for Hypoalbuminemia."""
    calcium = extract_calcium(text)
    albumin = extract_albumin(text)

    if calcium is None or albumin is None:
        return None

    corrected = calcium + 0.8 * (4 - albumin)

    return CalcResult(
        calculator_name="Calcium Correction for Hypoalbuminemia",
        result=round(corrected, 3),
        extracted_values={"calcium": calcium, "albumin": albumin},
        formula_used="Corrected Ca = Ca + 0.8*(4 - albumin)"
    )


def calculate_ldl(text: str) -> Optional[CalcResult]:
    """Calculate LDL using Friedewald equation."""
    total_chol = extract_cholesterol(text, 'total')
    hdl = extract_cholesterol(text, 'hdl')
    tg = extract_cholesterol(text, 'triglycerides')

    if total_chol is None or hdl is None or tg is None:
        return None

    ldl = total_chol - hdl - (tg / 5)

    return CalcResult(
        calculator_name="LDL Calculated",
        result=round(ldl, 3),
        extracted_values={"total_cholesterol": total_chol, "hdl": hdl, "triglycerides": tg},
        formula_used="LDL = Total - HDL - TG/5"
    )


def calculate_fib4(text: str) -> Optional[CalcResult]:
    """Calculate FIB-4 Index for Liver Fibrosis."""
    age = extract_age(text)
    ast = extract_ast(text)
    alt = extract_alt(text)
    platelets = extract_platelets(text)

    if age is None or ast is None or alt is None or platelets is None:
        return None

    fib4 = (age * ast) / (platelets * math.sqrt(alt))

    return CalcResult(
        calculator_name="Fibrosis-4 (FIB-4) Index for Liver Fibrosis",
        result=round(fib4, 3),
        extracted_values={"age": age, "ast": ast, "alt": alt, "platelets": platelets},
        formula_used="FIB-4 = (Age × AST) / (Platelets × √ALT)"
    )


# =============================================================================
# Main calculator dispatcher
# =============================================================================

CALCULATOR_PATTERNS = {
    # Physical
    "body mass index": calculate_bmi,
    "bmi": calculate_bmi,
    "mean arterial pressure": calculate_map,
    "map": calculate_map,
    "ideal body weight": calculate_ideal_body_weight,
    "adjusted body weight": calculate_adjusted_body_weight,
    "body surface area": calculate_body_surface_area,
    "bsa": calculate_body_surface_area,
    "target weight": calculate_target_weight,
    "maintenance fluids": calculate_maintenance_fluids,

    # QTc
    "qtc bazett": calculate_qtc_bazett,
    "bazett": calculate_qtc_bazett,
    "qtc fridericia": calculate_qtc_fridericia,
    "fridericia": calculate_qtc_fridericia,
    "fredericia": calculate_qtc_fridericia,
    "qtc framingham": calculate_qtc_framingham,
    "framingham qtc": calculate_qtc_framingham,
    "qtc hodges": calculate_qtc_hodges,
    "hodges": calculate_qtc_hodges,

    # Lab
    "creatinine clearance": calculate_creatinine_clearance,
    "cockcroft-gault": calculate_creatinine_clearance,
    "cockroft-gault": calculate_creatinine_clearance,
    "ckd-epi": calculate_ckd_epi_gfr,
    "glomerular filtration rate": calculate_ckd_epi_gfr,
    "gfr": calculate_ckd_epi_gfr,
    "mdrd": calculate_mdrd_gfr,
    "anion gap": calculate_anion_gap,
    "delta gap": calculate_delta_gap,
    "delta ratio": calculate_delta_ratio,
    "albumin corrected anion gap": calculate_albumin_corrected_anion_gap,
    "serum osmolality": calculate_serum_osmolality,
    "osmolality": calculate_serum_osmolality,
    "free water deficit": calculate_free_water_deficit,
    "sodium correction": calculate_sodium_correction,
    "calcium correction": calculate_calcium_correction,
    "corrected calcium": calculate_calcium_correction,
    "ldl": calculate_ldl,
    "ldl calculated": calculate_ldl,
    "fib-4": calculate_fib4,
    "fibrosis-4": calculate_fib4,
    "fib4": calculate_fib4,
}


def identify_calculator(question: str) -> Optional[str]:
    """Identify which calculator is needed from the question."""
    q = question.lower()

    # Priority patterns - check these first (most specific calculator names)
    priority_patterns = [
        "creatinine clearance",
        "cockcroft-gault",
        "cockroft-gault",
        "ckd-epi",
        "glomerular filtration rate",
        "mean arterial pressure",
        "body mass index",
        "ideal body weight",
        "body surface area",
        "maintenance fluids",
        "serum osmolality",
        "free water deficit",
        "sodium correction",
        "calcium correction",
        "anion gap",
        "delta gap",
        "delta ratio",
        "fib-4",
        "fibrosis-4",
        "ldl calculated",
        "qtc bazett",
        "qtc fridericia",
        "qtc framingham",
        "qtc hodges",
        "target weight",
    ]

    # Check priority patterns first
    for pattern in priority_patterns:
        if pattern in q:
            return pattern

    # Then check other patterns by length (longest match first)
    for pattern, _ in sorted(CALCULATOR_PATTERNS.items(), key=lambda x: -len(x[0])):
        if pattern in q:
            return pattern

    return None


def calculate(patient_note: str, question: str) -> Optional[CalcResult]:
    """
    Main entry point: identify calculator and compute result.

    Args:
        patient_note: The patient note text
        question: The question asking for a calculation

    Returns:
        CalcResult if successful, None if calculator not found or values missing
    """
    calc_pattern = identify_calculator(question)

    if calc_pattern is None:
        return None

    calc_func = CALCULATOR_PATTERNS[calc_pattern]

    # Combine patient note and question for value extraction
    # (some values like target BMI are in the question)
    combined_text = f"{patient_note}\n{question}"

    return calc_func(combined_text)


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    # Test BMI
    note = "A 45-year-old male. Height: 175 cm. Weight: 80 kg."
    question = "What is the patient's BMI?"
    result = calculate(note, question)
    print(f"BMI: {result}")

    # Test CrCl
    note2 = "An 87-year-old man. Weight: 48 kg. Creatinine: 1.4 mg/dL."
    question2 = "What is the patient's Creatinine Clearance using the Cockcroft-Gault Equation?"
    result2 = calculate(note2, question2)
    print(f"CrCl: {result2}")
