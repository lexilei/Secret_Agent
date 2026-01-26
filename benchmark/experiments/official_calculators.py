"""
Wrapper for official MedCalc-Bench calculator implementations.

Provides a unified interface to load and execute the official calculators
with their expected input format.

Expected Input Format Examples:
- age: [45, "years"] or [6, "months"]
- height: [170, "cm"] or [5, "ft", 10, "in"]
- weight: [80, "kg"] or [180, "lbs"]
- sex: "Male" or "Female"
- boolean conditions: True/False (e.g., chf, hypertension, diabetes)
"""

import os
import sys
import json
import importlib.util
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from pathlib import Path


# Path to calculator implementations
CALC_DIR = Path(__file__).parent / "calculator_implementations"


# Mapping from calculator_simple.py names to official MedCalc-Bench names
NAME_MAPPING = {
    # QTc calculators
    "QTc (Bazett)": "QTc Bazett Calculator",
    "QTc (Framingham)": "QTc Framingham Calculator",
    "QTc (Fridericia)": "QTc Fridericia Calculator",
    "QTc (Hodges)": "QTc Hodges Calculator",
    "QTc (Rautaharju)": "QTc Rautaharju Calculator",
    "qt_corrected_interval_bazett_formula": "QTc Bazett Calculator",
    "qtc_bazett": "QTc Bazett Calculator",
    "qtc bazett": "QTc Bazett Calculator",

    # Scoring systems
    "CHA2DS2-VASc Score": "CHA2DS2-VASc Score for Atrial Fibrillation Stroke Risk",
    "cha2ds2-vasc": "CHA2DS2-VASc Score for Atrial Fibrillation Stroke Risk",
    "HEART Score": "HEART Score for Major Cardiac Events",
    "heart_score": "HEART Score for Major Cardiac Events",
    "HAS-BLED Score": "HAS-BLED Score for Major Bleeding Risk",
    "has-bled": "HAS-BLED Score for Major Bleeding Risk",
    "Revised Cardiac Risk Index (RCRI)": "Revised Cardiac Risk Index for Pre-Operative Risk",
    "rcri": "Revised Cardiac Risk Index for Pre-Operative Risk",
    "SOFA Score": "Sequential Organ Failure Assessment (SOFA) Score",
    "sofa": "Sequential Organ Failure Assessment (SOFA) Score",

    # Wells criteria
    "Wells' Criteria for PE": "Wells' Criteria for Pulmonary Embolism",
    "wells_pe": "Wells' Criteria for Pulmonary Embolism",
    "wells_dvt": "Wells' Criteria for DVT",

    # Pulmonary
    "CURB-65 Score": "CURB-65 Score for Pneumonia Severity",
    "curb-65": "CURB-65 Score for Pneumonia Severity",
    "curb65": "CURB-65 Score for Pneumonia Severity",
    "PSI/PORT Score": "PSI Score: Pneumonia Severity Index for CAP",
    "psi_port": "PSI Score: Pneumonia Severity Index for CAP",
    "PERC Rule": "PERC Rule for Pulmonary Embolism",
    "perc": "PERC Rule for Pulmonary Embolism",

    # Renal
    "Creatinine Clearance (Cockcroft-Gault)": "Creatinine Clearance (Cockcroft-Gault Equation)",
    "crcl": "Creatinine Clearance (Cockcroft-Gault Equation)",
    "CKD-EPI GFR (2021)": "CKD-EPI Equations for Glomerular Filtration Rate",
    "ckd-epi": "CKD-EPI Equations for Glomerular Filtration Rate",
    "MDRD GFR": "MDRD GFR Equation",

    # Hepatic
    "FIB-4 Index": "Fibrosis-4 (FIB-4) Index for Liver Fibrosis",
    "fib-4": "Fibrosis-4 (FIB-4) Index for Liver Fibrosis",
    "MELD-Na Score": "MELD Na (UNOS/OPTN)",
    "meld-na": "MELD Na (UNOS/OPTN)",
    "Child-Pugh Score": "Child-Pugh Score for Cirrhosis Mortality",

    # Infectious
    "Centor Score (McIsaac)": "Centor Score (Modified/McIsaac) for Strep Pharyngitis",
    "centor": "Centor Score (Modified/McIsaac) for Strep Pharyngitis",
    "FeverPAIN Score": "FeverPAIN Score for Strep Pharyngitis",

    # Hematologic
    "Caprini VTE Score": "Caprini Score for Venous Thromboembolism (2005)",
    "caprini": "Caprini Score for Venous Thromboembolism (2005)",
    "Glasgow-Blatchford Score (GBS)": "Glasgow-Blatchford Bleeding Score (GBS)",
    "gbs": "Glasgow-Blatchford Bleeding Score (GBS)",

    # Misc
    "Glasgow Coma Scale (GCS)": "Glasgow Coma Score (GCS)",
    "gcs": "Glasgow Coma Score (GCS)",
    "Body Surface Area (Mosteller)": "Body Surface Area Calculator",
    "bsa": "Body Surface Area Calculator",
    "Maintenance Fluids (4-2-1 Rule)": "Maintenance Fluids Calculations",
    "Gestational Age": "Estimated Gestational Age",
    "Date of Conception": "Estimated of Conception",
    "Morphine Milligram Equivalents (MME)": "Morphine Milligram Equivalents (MME) Calculator",
    "mme": "Morphine Milligram Equivalents (MME) Calculator",
    "Target Weight": "Target weight",
    "Steroid Conversion": "Steroid Conversion Calculator",
    "Framingham Risk Score": "Framingham Risk Score for Hard Coronary Heart Disease",
    "HOMA-IR": "HOMA-IR (Homeostatic Model Assessment for Insulin Resistance)",
    "Ideal Body Weight (Devine)": "Ideal Body Weight",
    "LDL Calculated (Friedewald)": "LDL Calculated",
}


@dataclass
class OfficialCalculator:
    """Metadata for an official calculator."""
    name: str
    file_path: str
    calculator_id: int
    func: Optional[Callable] = None
    required_params: List[str] = None
    optional_params: List[str] = None


# Registry of loaded calculators
OFFICIAL_REGISTRY: Dict[str, OfficialCalculator] = {}


def _load_calculator_module(file_path: str) -> Any:
    """Dynamically load a calculator module."""
    full_path = CALC_DIR / Path(file_path).name
    if not full_path.exists():
        return None

    spec = importlib.util.spec_from_file_location("calc_module", full_path)
    module = importlib.util.module_from_spec(spec)

    # Add calculator_implementations to path for helper imports
    old_path = sys.path.copy()
    sys.path.insert(0, str(CALC_DIR))

    try:
        spec.loader.exec_module(module)
    finally:
        sys.path = old_path

    return module


def _find_calculator_function(module: Any) -> Optional[Callable]:
    """Find the main calculator function in a module."""
    # Look for common function name patterns
    for name in dir(module):
        if 'explanation' in name.lower() and callable(getattr(module, name)):
            func = getattr(module, name)
            # Check if it takes params/input_variables
            return func
    return None


def load_calculators():
    """Load all official calculators from calc_path.json."""
    global OFFICIAL_REGISTRY

    calc_path_file = CALC_DIR / "calc_path.json"
    if not calc_path_file.exists():
        print(f"Warning: calc_path.json not found at {calc_path_file}")
        return

    with open(calc_path_file) as f:
        calc_paths = json.load(f)

    for name, info in calc_paths.items():
        file_path = info["File Path"]
        calc_id = info["Calculator ID"]

        module = _load_calculator_module(file_path)
        if module is None:
            continue

        func = _find_calculator_function(module)

        calc = OfficialCalculator(
            name=name,
            file_path=file_path,
            calculator_id=calc_id,
            func=func,
        )
        OFFICIAL_REGISTRY[name] = calc
        # Also register by lowercase
        OFFICIAL_REGISTRY[name.lower()] = calc


def get_calculator(name: str) -> Optional[OfficialCalculator]:
    """Get a calculator by name (case-insensitive fuzzy match)."""
    if not OFFICIAL_REGISTRY:
        load_calculators()

    # First check name mapping
    if name in NAME_MAPPING:
        mapped_name = NAME_MAPPING[name]
        if mapped_name in OFFICIAL_REGISTRY:
            return OFFICIAL_REGISTRY[mapped_name]

    # Case-insensitive name mapping check
    name_lower = name.lower()
    for mapping_key, mapped_name in NAME_MAPPING.items():
        if name_lower == mapping_key.lower():
            if mapped_name in OFFICIAL_REGISTRY:
                return OFFICIAL_REGISTRY[mapped_name]

    # Exact match
    if name in OFFICIAL_REGISTRY:
        return OFFICIAL_REGISTRY[name]

    # Case-insensitive match
    if name_lower in OFFICIAL_REGISTRY:
        return OFFICIAL_REGISTRY[name_lower]

    # Fuzzy match - check each registered calculator
    for reg_name, calc in OFFICIAL_REGISTRY.items():
        reg_lower = reg_name.lower()
        # Skip lowercase duplicates (we registered them as aliases)
        if reg_name != calc.name and reg_lower == reg_name:
            continue
        if name_lower in reg_lower or reg_lower in name_lower:
            return calc

    return None


def compute_official(calculator_name: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Compute using official calculator implementation.

    Args:
        calculator_name: Name of the calculator
        params: Dictionary of parameters in official format

    Returns:
        {"Explanation": str, "Answer": float} or None if failed
    """
    calc = get_calculator(calculator_name)
    if calc is None or calc.func is None:
        return None

    try:
        result = calc.func(params)
        return result
    except Exception as e:
        print(f"  [DEBUG] Official calculator error: {e}")
        return None


def get_all_calculator_names() -> List[str]:
    """Get list of all available calculator names."""
    if not OFFICIAL_REGISTRY:
        load_calculators()

    # Return unique formal names
    seen = set()
    names = []
    for calc in OFFICIAL_REGISTRY.values():
        if calc.name not in seen:
            names.append(calc.name)
            seen.add(calc.name)
    return sorted(names)


# =============================================================================
# Format conversion helpers
# =============================================================================

def format_age(age_value: float, unit: str = "years") -> List:
    """Format age for official calculator input."""
    return [age_value, unit]


def format_height(value: float, unit: str = "cm") -> List:
    """Format height for official calculator input."""
    if unit == "ft_in" and isinstance(value, (list, tuple)) and len(value) == 2:
        feet, inches = value
        return [feet, "ft", inches, "in"]
    return [value, unit]


def format_weight(value: float, unit: str = "kg") -> List:
    """Format weight for official calculator input."""
    return [value, unit]


def convert_extracted_to_official(extracted: Dict[str, Any], calculator_name: str) -> Dict[str, Any]:
    """
    Convert L4 extracted values to official calculator format.

    If values are already in official format (lists, booleans), pass through.
    Otherwise convert common formats.
    """
    params = {}

    # Process each extracted value
    for key, value in extracted.items():
        if value is None:
            continue

        # If value is already a list (official format), pass through directly
        if isinstance(value, list):
            params[key] = value
            continue

        key_lower = key.lower().strip()

        # === Age ===
        if key_lower == "age":
            if isinstance(value, (int, float)):
                params["age"] = [value, "years"]
            elif isinstance(value, list):
                params["age"] = value
            elif isinstance(value, str):
                # Handle categorical age formats like "< 45", "45 - 65", "> 65"
                # Try to extract a reasonable numeric value
                if "< 45" in value or "<45" in value:
                    params["age"] = [40, "years"]  # Use middle of range
                elif "45" in value and "65" in value:
                    params["age"] = [55, "years"]  # Use middle of range
                elif "> 65" in value or ">65" in value or "≥65" in value or ">=65" in value:
                    params["age"] = [70, "years"]  # Use reasonable value
                else:
                    # Try to parse as number
                    import re
                    match = re.search(r'(\d+)', value)
                    if match:
                        params["age"] = [int(match.group(1)), "years"]

        # === Height ===
        elif key_lower in ("height", "height_cm"):
            if isinstance(value, (int, float)):
                params["height"] = [value, "cm"]
            elif isinstance(value, list):
                params["height"] = value
        elif key_lower == "height_m":
            params["height"] = [value, "m"]
        elif key_lower == "height_in":
            params["height"] = [value, "in"]

        # === Weight ===
        elif key_lower in ("weight", "weight_kg"):
            if isinstance(value, (int, float)):
                params["weight"] = [value, "kg"]
            elif isinstance(value, list):
                params["weight"] = value
        elif key_lower == "weight_lbs":
            params["weight"] = [value, "lbs"]

        # === Sex ===
        elif key_lower == "sex":
            if isinstance(value, str):
                params["sex"] = value.capitalize()

        # === Lab values that need [value, unit] format ===
        elif key_lower in ("heart_rate", "hr"):
            params["heart_rate"] = [value, "bpm"] if isinstance(value, (int, float)) else value
        elif key_lower in ("qt_interval", "qt"):
            params["qt_interval"] = [value, "msec"] if isinstance(value, (int, float)) else value
        elif key_lower in ("systolic_bp", "sbp", "sys_bp"):
            params["sys_bp"] = [value, "mmHg"] if isinstance(value, (int, float)) else value
        elif key_lower in ("diastolic_bp", "dbp", "dia_bp"):
            params["dia_bp"] = [value, "mmHg"] if isinstance(value, (int, float)) else value
        elif key_lower in ("creatinine", "creatinine_mg_dl", "serum_creatinine"):
            params["creatinine"] = [value, "mg/dL"] if isinstance(value, (int, float)) else value
        elif key_lower in ("bun", "blood_urea_nitrogen"):
            params["bun"] = [value, "mg/dL"] if isinstance(value, (int, float)) else value
        elif key_lower in ("sodium", "na"):
            params["sodium"] = [value, "mEq/L"] if isinstance(value, (int, float)) else value
        elif key_lower in ("potassium", "k"):
            params["potassium"] = [value, "mEq/L"] if isinstance(value, (int, float)) else value
        elif key_lower in ("chloride", "cl"):
            params["chloride"] = [value, "mEq/L"] if isinstance(value, (int, float)) else value
        elif key_lower in ("bicarbonate", "hco3", "co2"):
            params["bicarbonate"] = [value, "mEq/L"] if isinstance(value, (int, float)) else value
        elif key_lower in ("glucose", "blood_glucose"):
            params["glucose"] = [value, "mg/dL"] if isinstance(value, (int, float)) else value
        elif key_lower in ("albumin", "serum_albumin"):
            params["albumin"] = [value, "g/dL"] if isinstance(value, (int, float)) else value
        elif key_lower in ("calcium", "serum_calcium"):
            params["calcium"] = [value, "mg/dL"] if isinstance(value, (int, float)) else value
        elif key_lower in ("bilirubin", "total_bilirubin"):
            params["bilirubin"] = [value, "mg/dL"] if isinstance(value, (int, float)) else value
        elif key_lower in ("inr",):
            params["inr"] = [value, ""] if isinstance(value, (int, float)) else value
        elif key_lower in ("ast",):
            params["ast"] = [value, "U/L"] if isinstance(value, (int, float)) else value
        elif key_lower in ("alt",):
            params["alt"] = [value, "U/L"] if isinstance(value, (int, float)) else value
        elif key_lower in ("platelets", "platelet_count"):
            params["platelets"] = [value, "10^9/L"] if isinstance(value, (int, float)) else value

        # === Boolean conditions - handle has_X prefix ===
        elif key_lower.startswith("has_"):
            condition = key_lower.replace("has_", "")
            bool_value = bool(value)

            # Handle combined conditions that need to be split
            if condition in ("stroke_tia", "stroke/tia"):
                # CHA2DS2-VASc expects stroke, tia, thromboembolism separately
                params["stroke"] = bool_value
                params["tia"] = bool_value
                params["thromboembolism"] = bool_value
            elif condition in ("chf", "congestive_heart_failure"):
                params["chf"] = bool_value
            elif condition in ("hypertension", "htn"):
                params["hypertension"] = bool_value
            elif condition in ("diabetes", "diabetes_mellitus"):
                params["diabetes"] = bool_value
            elif condition in ("vascular_disease", "atherosclerotic_disease"):
                params["vascular_disease"] = bool_value
            else:
                params[condition] = bool_value

        # === Handle combined conditions without has_ prefix ===
        elif key_lower in ("stroke_tia", "stroke/tia"):
            bool_value = bool(value)
            params["stroke"] = bool_value
            params["tia"] = bool_value
            params["thromboembolism"] = bool_value

        # === HEART Score specific categorical parameters ===
        elif key_lower in ("history", "history_suspicious"):
            # Normalize history to expected categorical values
            if isinstance(value, str):
                val_lower = value.lower().strip()
                if any(kw in val_lower for kw in ['highly', 'typical', 'classic', 'definite', 'high']):
                    params["history"] = "Highly suspicious"
                elif any(kw in val_lower for kw in ['moderately', 'moderate', 'somewhat', 'possible']):
                    params["history"] = "Moderately suspicious"
                elif any(kw in val_lower for kw in ['slightly', 'atypical', 'low', 'non-specific', 'vague', 'unlikely']):
                    params["history"] = "Slightly suspicious"
                else:
                    # Default based on common patterns
                    params["history"] = value
            else:
                params["history"] = value
        elif key_lower in ("electrocardiogram", "ecg", "ekg"):
            # Normalize ECG to expected categorical values
            if isinstance(value, str):
                val_lower = value.lower().strip()
                if any(kw in val_lower for kw in ['st elevation', 'st depression', 'significant st', 'st deviation', 'stemi', 'nstemi']):
                    params["electrocardiogram"] = "Significant ST deviation"
                elif any(kw in val_lower for kw in ['non-specific', 'nonspecific', 'repolarization', 'minor', 't wave']):
                    params["electrocardiogram"] = "Non-specific repolarization disturbance"
                elif any(kw in val_lower for kw in ['normal', 'no changes', 'unremarkable', 'nsr', 'sinus rhythm']):
                    params["electrocardiogram"] = "Normal"
                else:
                    params["electrocardiogram"] = value
            else:
                params["electrocardiogram"] = value
        elif key_lower in ("initial_troponin", "troponin"):
            # Normalize troponin to expected categorical values
            if isinstance(value, str):
                val_lower = value.lower().strip()
                if any(kw in val_lower for kw in ['greater than three', '>3', 'more than 3', 'high', 'markedly elevated']):
                    params["initial_troponin"] = "greater than three times normal limit"
                elif any(kw in val_lower for kw in ['between', '1-3', '1 to 3', 'one to three', 'slightly elevated', 'mildly elevated']):
                    params["initial_troponin"] = "between the normal limit or up to three times the normal limit"
                elif any(kw in val_lower for kw in ['normal', 'less than', '≤normal', 'negative', 'within normal', 'not elevated']):
                    params["initial_troponin"] = "less than or equal to normal limit"
                else:
                    params["initial_troponin"] = value
            else:
                params["initial_troponin"] = value

        # === HAS-BLED specific parameters ===
        elif key_lower in ("alcoholic_drinks", "alcohol_drinks", "drinks_per_week"):
            params["alcoholic_drinks"] = int(value) if isinstance(value, (int, float, str)) else 0
        elif key_lower in ("liver_disease_has_bled", "liver_disease_hasbled"):
            params["liver_disease_has_bled"] = bool(value)
        elif key_lower in ("renal_disease_has_bled", "renal_disease_hasbled"):
            params["renal_disease_has_bled"] = bool(value)
        elif key_lower in ("medications_for_bleeding", "bleeding_medications"):
            params["medications_for_bleeding"] = bool(value)
        elif key_lower in ("prior_bleeding", "bleeding_history"):
            params["prior_bleeding"] = bool(value)
        elif key_lower in ("labile_inr",):
            params["labile_inr"] = bool(value)

        # === Glasgow-Blatchford specific parameters ===
        elif key_lower in ("hepatic_disease_history", "liver_disease_history"):
            params["hepatic_disease_history"] = bool(value)
        elif key_lower in ("melena_present", "melena"):
            params["melena_present"] = bool(value)
        elif key_lower in ("cardiac_failure", "heart_failure_gbs"):
            params["cardiac_failure"] = bool(value)
        elif key_lower in ("syncope", "recent_syncope"):
            params["syncope"] = bool(value)
        elif key_lower in ("hemoglobin", "hgb", "hb"):
            params["hemoglobin"] = [value, "g/dL"] if isinstance(value, (int, float)) else value

        # === Charlson CCI specific parameters ===
        elif key_lower in ("peptic_ulcer_disease", "peptic_ucler_disease", "peptic_ulcer"):
            # Note: Official code has typo "peptic_ucler_disease"
            params["peptic_ucler_disease"] = bool(value)
        elif key_lower in ("liver_disease", "liver_disease_cci"):
            # Charlson expects categorical: "none", "mild", "moderate to severe"
            if isinstance(value, bool):
                params["liver_disease"] = "mild" if value else "none"
            else:
                params["liver_disease"] = value
        elif key_lower in ("diabetes_mellitus", "diabetes_cci"):
            # Charlson expects categorical: "none or diet-controlled", "uncomplicated", "end-organ damage"
            if isinstance(value, bool):
                params["diabetes_mellitus"] = "uncomplicated" if value else "none or diet-controlled"
            else:
                params["diabetes_mellitus"] = value
        elif key_lower in ("solid_tumor", "tumor"):
            # Charlson expects categorical: "none", "localized", "metastatic"
            if isinstance(value, bool):
                params["solid_tumor"] = "localized" if value else "none"
            else:
                params["solid_tumor"] = value
        elif key_lower in ("mi", "myocardial_infarction"):
            params["mi"] = bool(value)
        elif key_lower in ("peripheral_vascular_disease", "pvd"):
            params["peripheral_vascular_disease"] = bool(value)
        elif key_lower in ("cva", "cerebrovascular_accident"):
            params["cva"] = bool(value)
        elif key_lower == "tia":
            params["tia"] = bool(value)
        elif key_lower in ("connective_tissue_disease", "ctd"):
            params["connective_tissue_disease"] = bool(value)
        elif key_lower == "dementia":
            params["dementia"] = bool(value)
        elif key_lower == "copd":
            params["copd"] = bool(value)
        elif key_lower == "hemiplegia":
            params["hemiplegia"] = bool(value)
        elif key_lower in ("moderate_to_severe_ckd", "ckd"):
            params["moderate_to_severe_ckd"] = bool(value)
        elif key_lower == "leukemia":
            params["leukemia"] = bool(value)
        elif key_lower == "lymphoma":
            params["lymphoma"] = bool(value)
        elif key_lower == "aids":
            params["aids"] = bool(value)

        # === Direct pass-through ===
        else:
            # Handle boolean-like values
            if isinstance(value, str):
                val_lower = value.lower()
                if val_lower in ("true", "yes", "1", "present"):
                    value = True
                elif val_lower in ("false", "no", "0", "absent"):
                    value = False
            params[key_lower] = value

    return params


def get_official_source(calculator_name: str) -> Optional[str]:
    """Get the source code of an official calculator for use in extraction prompts."""
    calc = get_calculator(calculator_name)
    if calc is None:
        return None

    full_path = CALC_DIR / Path(calc.file_path).name
    if not full_path.exists():
        return None

    try:
        with open(full_path) as f:
            return f.read()
    except Exception:
        return None


def get_expected_params(calculator_name: str) -> List[str]:
    """
    Extract expected parameter names from official calculator source.
    Looks for patterns like: params['xxx'] or input_variables['xxx']
    """
    import re
    source = get_official_source(calculator_name)
    if not source:
        return []

    # Find all parameter accesses
    patterns = [
        r"params\['(\w+)'\]",
        r'params\["(\w+)"\]',
        r"input_variables\['(\w+)'\]",
        r'input_variables\["(\w+)"\]',
        r"variables\['(\w+)'\]",
        r'variables\["(\w+)"\]',
        r"input_parameters\.get\('(\w+)'",
        r'input_parameters\.get\("(\w+)"',
        r"input_parameters\['(\w+)'\]",
    ]

    params = set()
    for pattern in patterns:
        matches = re.findall(pattern, source)
        params.update(matches)

    return sorted(list(params))


# Auto-load on import
load_calculators()


if __name__ == "__main__":
    print(f"Loaded {len(get_all_calculator_names())} official calculators:")
    for name in get_all_calculator_names()[:10]:
        calc = get_calculator(name)
        print(f"  - {name}: {calc.file_path}")
    print("  ...")
