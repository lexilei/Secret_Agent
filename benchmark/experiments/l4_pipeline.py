"""
L4 Experiment: Python-Orchestrated Pipeline with Specialist LLM Agents.

Combines the best of L2 (Python calculation) and L3 (LLM reasoning):
- Stage 1: LLM identifies which calculator is needed
- Stage 2: LLM extracts required values from patient note
- Stage 3: Python validates extracted values
- Stage 4: Python computes result (NO LLM)

Key insight: Python controls the workflow, LLMs only handle "understanding" tasks.
This tests William's hypothesis: "Python programs calling LLMs, not LLMs calling tools."

Expected benefits over L3:
- Higher accuracy: Each stage specialized and validated before proceeding
- Lower cost: Python calculation = $0
- Better debuggability: Clear failure attribution (which stage failed?)
- Cleaner distillation: Each stage can be independently distilled to Python
"""

import re
import time
from typing import Dict, Any, List, Optional, Tuple
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

# Import official MedCalc-Bench calculator implementations
from .official_calculators import (
    get_all_calculator_names,
    get_calculator,
    compute_official,
    convert_extracted_to_official,
    get_official_source,
    get_expected_params,
    OFFICIAL_REGISTRY,
)

# Keep calculator_simple for signatures/docstrings (metadata)
from .calculator_simple import (
    CALCULATOR_REGISTRY,
    CalculatorSpec,
    get_calculator_signatures,
    get_calculator_names,
    get_extraction_hints,
    get_calculator_docstring,
)

# Import old calculators for fallback pattern matching
from .calculators import (
    CalcResult,
    identify_calculator as python_identify_calculator,
    calculate as python_calculate,
)

# Import L1 ptool for fallback
from .l1_ptool import calculate_medical_value as l1_calculate


# =============================================================================
# Stage 1: Calculator Identification (LLM)
# =============================================================================

@ptool(model="deepseek-v3-0324", output_mode="structured")
def identify_calculator_l4(
    question: str,
    available_calculators: List[str],
) -> Dict[str, Any]:
    """
    Identify which medical calculator is needed based on the question.

    Analyze the question and match it to one of the available calculators.

    Return:
    {
        "calculator_name": "exact name from the list",
        "confidence": 0.0-1.0,
        "reasoning": "brief explanation"
    }
    """
    ...


# =============================================================================
# Stage 2: Value Extraction (LLM)
# =============================================================================

def extract_values_two_stage(
    patient_note: str,
    calculator_name: str,
    required_values: List[str],
    optional_values: List[str],
) -> Dict[str, Any]:
    """
    Two-stage extraction with medical reasoning for robust value extraction.

    Stage 1: Medical reasoning - analyze clinical findings and infer conditions
    Stage 2: Extract actual values with proper format
    """
    import json
    import re

    docstring = get_calculator_docstring(calculator_name)
    all_values = required_values + optional_values

    # Get official calculator source for parameter format guidance
    official_source = get_official_source(calculator_name)
    expected_params = get_expected_params(calculator_name)

    # Determine if this is a scoring system (needs medical reasoning)
    is_scoring_system = any(kw in calculator_name.lower() for kw in [
        'score', 'criteria', 'index', 'risk', 'cha2ds2', 'heart', 'wells',
        'curb', 'sofa', 'apache', 'child-pugh', 'meld', 'centor', 'fever',
        'has-bled', 'rcri', 'charlson', 'caprini', 'blatchford', 'perc'
    ])

    # =========================================================================
    # STAGE 1: Medical Reasoning (for scoring systems)
    # =========================================================================
    if is_scoring_system:
        reasoning_prompt = f"""You are a medical expert analyzing a patient note for the {calculator_name}.

PATIENT NOTE:
{patient_note}

TASK: Carefully analyze this note and identify ALL conditions/criteria relevant to this calculator.
Think step-by-step like a physician reviewing the chart.

CRITICAL - You must identify conditions from BOTH:
1. **Explicit mentions**: "history of diabetes", "has hypertension", "prior stroke"
2. **Clinical findings that IMPLY conditions** - this is crucial:
   - DWI/MRI showing high signal intensity, infarcts → STROKE
   - Dysarthria, facial palsy, hemiparesis, Babinski sign → STROKE/TIA symptoms
   - Troponin elevation, ST changes, chest pain → possible MI (vascular disease)
   - Claudication, ABI <0.9, peripheral pulses absent → peripheral arterial disease
   - Cancer, malignancy, tumor → may count as vascular disease in some scores
   - Ascites, encephalopathy, varices → liver disease
   - Edema, JVD, S3 gallop, reduced EF → CHF
   - On warfarin/anticoagulation → consider why (prior DVT/PE/AF?)
   - Elevated creatinine, dialysis → renal disease

3. **Negations** - if explicitly denied, note as FALSE:
   - "no history of diabetes" → diabetes = false
   - "denies chest pain" → relevant for some scores

REASONING FORMAT:
First, reason through each finding:
- Finding: [what you see in the note]
- Implies: [what condition this indicates]
- Confidence: [high/medium/low]

Then provide your conclusions.

Return JSON:
{{
    "reasoning": "Your step-by-step medical reasoning here",
    "conditions_present": ["list of conditions that ARE present based on explicit mentions OR clinical findings"],
    "conditions_absent": ["list of conditions explicitly denied or clearly absent"],
    "demographics": {{"age": number, "sex": "male/female"}}
}}
"""

        try:
            reasoning_response = call_llm(prompt=reasoning_prompt, model="deepseek-v3-0324", max_tokens=1500)
            try:
                reasoning_result = json.loads(reasoning_response)
            except json.JSONDecodeError:
                match = re.search(r'\{[\s\S]*\}', reasoning_response, re.DOTALL)
                reasoning_result = json.loads(match.group()) if match else {}
        except Exception:
            reasoning_result = {}

        # Build context from reasoning
        conditions_present = reasoning_result.get("conditions_present", [])
        conditions_absent = reasoning_result.get("conditions_absent", [])
        demographics = reasoning_result.get("demographics", {})
        reasoning_text = reasoning_result.get("reasoning", "")
    else:
        conditions_present = []
        conditions_absent = []
        demographics = {}
        reasoning_text = ""

    # =========================================================================
    # STAGE 2: Extract values with proper format
    # =========================================================================

    # Build the extraction prompt
    reasoning_context = ""
    if is_scoring_system and (conditions_present or demographics):
        reasoning_context = f"""
MEDICAL ANALYSIS (from Stage 1 reasoning):
- Conditions PRESENT: {', '.join(conditions_present) if conditions_present else 'None identified'}
- Conditions ABSENT: {', '.join(conditions_absent) if conditions_absent else 'None explicitly denied'}
- Demographics: Age={demographics.get('age', 'unknown')}, Sex={demographics.get('sex', 'unknown')}
- Reasoning: {reasoning_text[:500]}

Use this analysis to inform your extraction. The conditions identified above should be reflected in your boolean parameters.
"""

    stage2_prompt = f"""Extract values from the patient note for the {calculator_name} calculator.
{reasoning_context}
OFFICIAL CALCULATOR IMPLEMENTATION (reference for parameter names):
```python
{official_source[:2500] if official_source else "Not available"}
```

PATIENT NOTE:
{patient_note}

EXTRACTION INSTRUCTIONS:

1. **Parameter Names**: Use EXACT parameter names from the official implementation
2. **NUMERIC values ALWAYS with units**:
   - age → [actual_age_number, "years"] e.g., [33, "years"], [68, "years"]
   - weight → [value, "kg"] or convert lbs to kg
   - height → [value, "cm"] or convert feet/inches to cm
   - Lab values → [value, "unit"] e.g., [1.2, "mg/dL"], [140, "mmol/L"]
   - IMPORTANT: Always extract the ACTUAL numeric value, NOT categorical descriptions like "< 45" or "> 65"
3. **Boolean conditions**: Return True if present (explicitly or inferred from findings), False if absent
4. **Categorical values for scoring**: For categorical scores like history suspicion level:
   - HEART history:
     * "Slightly suspicious" (0 pts): vague/atypical chest pain, non-specific symptoms
     * "Moderately suspicious" (1 pt): typical anginal symptoms but some atypical features
     * "Highly suspicious" (2 pts): typical/classic angina, pressure, radiation to arm/jaw
   - HEART ECG:
     * "Normal" (0 pts): normal ECG, sinus rhythm, no ST changes
     * "Non-specific repolarization disturbance" (1 pt): minor T wave changes, non-specific ST changes
     * "Significant ST deviation" (2 pts): new ST elevation/depression ≥1mm
   - HEART troponin (BE CONSERVATIVE - only mark elevated if explicitly stated):
     * "less than or equal to normal limit" (0 pts): DEFAULT if troponin not mentioned or is negative/normal
     * "between the normal limit or up to three times the normal limit" (1 pt): only if troponin explicitly slightly elevated
     * "greater than three times normal limit" (2 pts): only if troponin EXPLICITLY stated as significantly elevated (>3x)

COMMON CONVERSIONS:
- Weight: lbs × 0.453592 = kg
- Height: (feet × 30.48) + (inches × 2.54) = cm
- Creatinine: µmol/L ÷ 88.4 = mg/dL
- Platelets: "150,000/µL" or "150K" → 150 (in 10^9/L)
- Sex: "man/male/he/him/Mr." → "Male", "woman/female/she/her" → "Female"

FOR SCORING SYSTEMS (extract risk factors as flat booleans, not nested):
- Map identified conditions to the exact parameter names in the implementation
- Example mappings:
  - stroke/TIA/CVA/cerebral infarct → stroke: true (or tia: true)
  - CHF/heart failure/reduced EF → chf: true
  - HTN/hypertension/elevated BP → hypertension: true
  - DM/diabetes/diabetic → diabetes_mellitus: true
  - MI/CAD/PCI/CABG/PAD → atherosclerotic_disease: true
  - High cholesterol/hyperlipidemia → hypercholesterolemia: true
  - Obese/BMI>30 → obesity: true
  - Smoker → smoking: true

Return ONLY valid JSON:
{{
    "extracted": {{
        "age": [numeric_value, "years"],
        "param_name": value,
        ...
    }},
    "missing": ["required values not found"],
    "inferred": ["conditions inferred from clinical findings"]
}}
"""

    try:
        stage2_response = call_llm(prompt=stage2_prompt, model="deepseek-v3-0324", max_tokens=1200)

        try:
            result = json.loads(stage2_response)
        except json.JSONDecodeError:
            match = re.search(r'\{[\s\S]*\}', stage2_response, re.DOTALL)
            if match:
                try:
                    result = json.loads(match.group())
                except json.JSONDecodeError:
                    result = {"extracted": {}, "missing": required_values}
            else:
                result = {"extracted": {}, "missing": required_values}

        # Ensure missing list is accurate
        extracted = result.get("extracted", {})

        # CRITICAL: Inject conditions from Stage 1 reasoning if Stage 2 missed them
        # This ensures medical inferences are reflected in the extraction
        # ONLY apply to calculators where clinical inference adds value
        calculators_needing_inference = [
            "cha2ds2-vasc",  # Stroke can be inferred from DWI/neurological findings
            "charlson",      # Comorbidities may be implied
            "rcri",          # Risk factors may be implied from conditions
        ]

        calc_lower = calculator_name.lower()
        should_inject = is_scoring_system and conditions_present and any(
            kw in calc_lower for kw in calculators_needing_inference
        )

        if should_inject:
            condition_mapping = {
                # CHA2DS2-VASc mappings
                "stroke": ["stroke", "stroke_tia", "has_stroke_tia", "prior_stroke"],
                "tia": ["tia", "stroke_tia", "has_stroke_tia"],
                "stroke/tia": ["stroke", "tia", "stroke_tia", "has_stroke_tia"],
                "stroke/tia/thromboembolism": ["stroke", "tia", "stroke_tia", "has_stroke_tia", "thromboembolism"],
                "hypertension": ["hypertension", "has_hypertension", "htn"],
                "diabetes": ["diabetes", "has_diabetes", "diabetes_mellitus"],
                "chf": ["chf", "has_chf", "heart_failure", "congestive_heart_failure"],
                "heart failure": ["chf", "has_chf", "heart_failure"],
                "vascular disease": ["vascular_disease", "has_vascular_disease", "atherosclerotic_disease"],
                # RCRI mappings
                "coronary artery disease": ["ischemic_heart_disease", "coronary_artery_disease"],
                "prior mi": ["ischemic_heart_disease"],
                "renal insufficiency": ["creatinine_over_2", "renal_insufficiency"],
                "cerebrovascular disease": ["history_of_cerebrovascular_disease"],
                # General
                "age ≥65": [],  # Age handled separately
                "age ≥75": [],
                "female": [],  # Sex handled separately
                "male": [],
            }

            for condition in conditions_present:
                cond_lower = condition.lower().strip()

                # Try to find matching parameter names
                matched = False
                for pattern, param_names in condition_mapping.items():
                    if pattern in cond_lower or cond_lower in pattern:
                        for param in param_names:
                            # Only set if not already set to True
                            if param not in extracted or not extracted.get(param):
                                extracted[param] = True
                                matched = True
                        break

                # Skip fuzzy matching - only use explicit mappings to avoid over-extraction

        actual_missing = [v for v in required_values if v not in extracted or extracted[v] is None]
        result["missing"] = actual_missing
        result["extracted"] = extracted

        return result

    except Exception as e:
        return {"extracted": {}, "missing": required_values, "error": str(e)}


# =============================================================================
# Stage 2.5: Normalize extracted values
# =============================================================================

# Condition aliases for scoring systems
CONDITION_ALIASES = {
    # CHA2DS2-VASc
    "chf": "has_chf", "congestive heart failure": "has_chf", "heart failure": "has_chf",
    "htn": "has_hypertension", "hypertension": "has_hypertension", "high blood pressure": "has_hypertension",
    "dm": "has_diabetes", "diabetes": "has_diabetes", "diabetes mellitus": "has_diabetes",
    "stroke": "has_stroke_tia", "tia": "has_stroke_tia", "stroke/tia": "has_stroke_tia", "cva": "has_stroke_tia",
    "vascular disease": "has_vascular_disease", "mi": "has_vascular_disease", "pad": "has_vascular_disease",
    # HEART Score
    "suspicious history": "history_suspicious", "moderately suspicious": "history_suspicious",
    "st deviation": "ecg_findings", "ecg changes": "ecg_findings",
    # HAS-BLED
    "renal disease": "has_renal_disease", "liver disease": "has_liver_disease",
    "bleeding": "has_bleeding_history", "labile inr": "has_labile_inr",
    "alcohol": "has_alcohol_use", "drugs": "has_drug_use",
    # General
    "age_65_74": "age_65_74", "age_over_65": "age_over_65",
}


def normalize_extracted_values(extracted: Dict[str, Any], calculator_name: str) -> Dict[str, Any]:
    """
    Normalize extracted values to handle:
    - Conditions list → individual boolean parameters
    - Alias names → canonical parameter names
    - Boolean-like values → actual booleans
    """
    normalized = {}

    # Handle conditions list if present
    conditions = extracted.pop("conditions", None)
    if conditions and isinstance(conditions, list):
        for condition in conditions:
            cond_lower = condition.lower().strip()
            # Map to canonical name
            if cond_lower in CONDITION_ALIASES:
                normalized[CONDITION_ALIASES[cond_lower]] = True
            else:
                # Try has_ prefix
                param_name = f"has_{cond_lower.replace(' ', '_')}"
                normalized[param_name] = True

    # Process each extracted value
    for key, value in extracted.items():
        key_lower = key.lower().strip()

        # Map aliases
        if key_lower in CONDITION_ALIASES:
            key = CONDITION_ALIASES[key_lower]

        # Normalize boolean-like values
        if isinstance(value, str):
            val_lower = value.lower().strip()
            if val_lower in ("true", "yes", "1", "present", "positive"):
                value = True
            elif val_lower in ("false", "no", "0", "absent", "negative"):
                value = False
        elif isinstance(value, (int, float)) and key.startswith("has_"):
            value = bool(value)

        normalized[key] = value

    return normalized


# =============================================================================
# Stage 3: Value Validation (Python)
# =============================================================================

def validate_extracted_values(
    extracted: Dict[str, Any],
    calculator_name: str,
) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Validate extracted values are complete and reasonable.

    Returns:
        (is_valid, missing_required, cleaned_values)
    """
    # First flatten any nested dicts (e.g., risk_factors: {hypercholesterolemia: true})
    flattened = {}
    for key, value in extracted.items():
        if isinstance(value, dict):
            # Flatten nested dict - its keys become top-level parameters
            for subkey, subvalue in value.items():
                flattened[subkey] = subvalue
        else:
            flattened[key] = value

    # First normalize the extracted values
    extracted = normalize_extracted_values(flattened.copy(), calculator_name)

    signatures = get_calculator_signatures()
    sig = signatures.get(calculator_name, {})
    required = sig.get("required", [])
    optional = sig.get("optional", [])

    missing = []
    cleaned = {}

    # Check required values
    for key in required:
        if key in extracted:
            value = extracted[key]
            # Handle None values as missing
            if value is None:
                missing.append(key)
                continue
            # Clean up value
            if isinstance(value, str):
                # Try to parse numeric strings
                try:
                    cleaned[key] = float(value)
                except ValueError:
                    # Keep as string (e.g., "male"/"female")
                    cleaned[key] = value.lower().strip()
            else:
                cleaned[key] = value
        else:
            missing.append(key)

    # Add optional values if present (from simple signatures)
    for key in optional:
        if key in extracted and extracted[key] is not None:
            value = extracted[key]
            if isinstance(value, str):
                try:
                    cleaned[key] = float(value)
                except ValueError:
                    cleaned[key] = value.lower().strip()
            else:
                cleaned[key] = value

    # IMPORTANT: Pass through ALL other extracted values for official calculators
    # The official calculators use different parameter names than calculator_simple.py
    # So we need to pass through values like 'bedridden_for_atleast_3_days', 'previous_dvt_documented', etc.
    known_keys = set(required) | set(optional)
    for key, value in extracted.items():
        if key not in cleaned and value is not None:
            # Pass through boolean values directly
            if isinstance(value, bool):
                cleaned[key] = value
            # Pass through lists (official format like [value, "unit"])
            elif isinstance(value, list):
                cleaned[key] = value
            # Pass through numeric values
            elif isinstance(value, (int, float)):
                cleaned[key] = value
            # Clean string values
            elif isinstance(value, str):
                # Keep original case for categorical values
                if any(kw in key.lower() for kw in ['history', 'electrocardiogram', 'troponin', 'ecg', 'ekg']):
                    cleaned[key] = value
                else:
                    try:
                        cleaned[key] = float(value)
                    except ValueError:
                        cleaned[key] = value.lower().strip()

    # Validate ranges - with null safety checks
    if "age" in cleaned:
        age_val = cleaned["age"]
        if isinstance(age_val, (int, float)) and (age_val is None or not (0 < age_val <= 120)):
            return False, ["age out of range or missing"], cleaned
    if "weight_kg" in cleaned:
        weight_val = cleaned["weight_kg"]
        if isinstance(weight_val, (int, float)) and (weight_val is None or not (1 < weight_val < 500)):
            return False, ["weight_kg out of range or missing"], cleaned
    if "height_cm" in cleaned:
        height_val = cleaned["height_cm"]
        if isinstance(height_val, (int, float)) and (height_val is None or not (30 < height_val < 300)):
            return False, ["height_cm out of range or missing"], cleaned

    return len(missing) == 0, missing, cleaned


# =============================================================================
# Stage 4: Python Calculation (NO LLM)
# =============================================================================

def compute_with_python(
    calculator_name: str,
    values: Dict[str, Any],
) -> Optional[CalcResult]:
    """
    Compute result using pure Python (no LLM).

    Uses official MedCalc-Bench calculator implementations.
    Falls back to pattern-based calculation if needed.
    """
    # Convert extracted values to official format
    official_params = convert_extracted_to_official(values, calculator_name)

    # Try official calculator first
    result = compute_official(calculator_name, official_params)
    if result is not None and "Answer" in result:
        return CalcResult(
            calculator_name=calculator_name,
            result=result["Answer"],
            formula_used=f"Official MedCalc-Bench implementation",
            extracted_values=values,
        )

    # Fallback: reconstruct a pseudo patient note and use full pipeline
    note_parts = []
    if "age" in values:
        sex_str = values.get("sex", "person")
        note_parts.append(f"A {values['age']}-year-old {sex_str}")
    if "weight_kg" in values:
        note_parts.append(f"weighing {values['weight_kg']} kg")
    if "height_cm" in values:
        note_parts.append(f"height {values['height_cm']} cm")
    if "creatinine_mg_dl" in values:
        note_parts.append(f"creatinine {values['creatinine_mg_dl']} mg/dL")

    # Add boolean flags for risk scores
    for key, value in values.items():
        if key.startswith("has_") and value:
            condition = key.replace("has_", "").replace("_", " ")
            note_parts.append(f"with history of {condition}")

    pseudo_note = " ".join(note_parts)
    pseudo_question = f"Calculate the {calculator_name}"

    return python_calculate(pseudo_note, pseudo_question)


# =============================================================================
# L4 Pipeline Experiment
# =============================================================================

class L4PipelineExperiment(BaseExperiment):
    """
    L4 Experiment: Python-orchestrated pipeline with specialist LLM agents.

    Pipeline stages:
    1. Identify calculator (LLM)
    2. Extract values (LLM)
    3. Validate values (Python)
    4. Compute result (Python)

    Key difference from L3:
    - L3: Single ReAct agent figures out workflow autonomously
    - L4: Python explicitly controls the pipeline, LLMs only do "understanding"
    """

    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        self.model = config.model
        self.max_repair_attempts = getattr(config, "max_repair_attempts", 1)

        # Stage statistics
        self.stage_stats = {
            "identify_success": 0,
            "identify_fail": 0,
            "extract_success": 0,
            "extract_fail": 0,
            "validate_success": 0,
            "validate_fail": 0,
            "compute_success": 0,
            "compute_fail": 0,
        }

    def setup(self):
        """Initialize ptools."""
        # Force ptools to register
        _ = identify_calculator_l4
        # Note: extraction now uses extract_values_with_hints (direct LLM call)
        self._setup_complete = True

    def _identify_calculator(
        self,
        question: str,
    ) -> Tuple[Optional[str], float, Dict[str, int]]:
        """
        Stage 1: Identify which calculator to use.

        Returns: (calculator_name, confidence, token_counts)
        """
        # Get all 55 calculator signatures from calculator_simple
        signatures = get_calculator_signatures()
        available = list(signatures.keys())

        try:
            result = identify_calculator_l4(
                question=question,
                available_calculators=available,
            )

            if isinstance(result, dict):
                # Handle nested {"result": {...}} structure from ptool
                inner = result.get("result", result)
                if isinstance(inner, dict):
                    calc_name = inner.get("calculator_name")
                    confidence = inner.get("confidence", 0.5)
                else:
                    calc_name = result.get("calculator_name")
                    confidence = result.get("confidence", 0.5)

                # Validate calculator name exists
                if calc_name in signatures:
                    return calc_name, confidence, {"input": 500, "output": 100}

                # Also check CALCULATOR_REGISTRY which includes aliases
                if calc_name in CALCULATOR_REGISTRY:
                    # Get canonical name
                    spec = CALCULATOR_REGISTRY[calc_name]
                    return spec.name, confidence, {"input": 500, "output": 100}

                # Try fuzzy matching
                calc_lower = calc_name.lower() if calc_name else ""
                for sig_name in signatures:
                    if calc_lower in sig_name.lower() or sig_name.lower() in calc_lower:
                        return sig_name, confidence * 0.8, {"input": 500, "output": 100}

                # Also check aliases for fuzzy match
                for name, spec in CALCULATOR_REGISTRY.items():
                    if isinstance(spec, CalculatorSpec):
                        for alias in spec.aliases:
                            if calc_lower in alias.lower() or alias.lower() in calc_lower:
                                return spec.name, confidence * 0.8, {"input": 500, "output": 100}

            # LLM failed, try Python fallback
            calc_pattern = python_identify_calculator(question)
            if calc_pattern:
                # Map pattern to full name via registry
                for name, spec in CALCULATOR_REGISTRY.items():
                    if isinstance(spec, CalculatorSpec):
                        if calc_pattern.lower() in name.lower() or calc_pattern.lower() in spec.name.lower():
                            return spec.name, 0.6, {"input": 0, "output": 0}
                        for alias in spec.aliases:
                            if calc_pattern.lower() in alias.lower():
                                return spec.name, 0.6, {"input": 0, "output": 0}

            return None, 0.0, {"input": 500, "output": 100}

        except Exception as e:
            # Fall back to Python identification
            calc_pattern = python_identify_calculator(question)
            if calc_pattern:
                for name, spec in CALCULATOR_REGISTRY.items():
                    if isinstance(spec, CalculatorSpec):
                        if calc_pattern.lower() in name.lower() or calc_pattern.lower() in spec.name.lower():
                            return spec.name, 0.5, {"input": 0, "output": 0}
            return None, 0.0, {"input": 0, "output": 0}

    def _extract_values(
        self,
        patient_note: str,
        calculator_name: str,
    ) -> Tuple[Dict[str, Any], List[str], Dict[str, int]]:
        """
        Stage 2: Extract required values from patient note.

        Uses two-stage extraction:
        - Stage 1: Identify which values are present
        - Stage 2: Extract with proper unit/date handling

        Returns: (extracted_values, missing_values, token_counts)
        """
        signatures = get_calculator_signatures()
        sig = signatures.get(calculator_name, {})
        required = sig.get("required", [])
        optional = sig.get("optional", [])

        try:
            # Use two-stage extraction for robust handling
            result = extract_values_two_stage(
                patient_note=patient_note,
                calculator_name=calculator_name,
                required_values=required,
                optional_values=optional,
            )

            if isinstance(result, dict):
                extracted = result.get("extracted", {})
                missing = result.get("missing", [])
                # Two stages = ~1500 input tokens, ~500 output tokens
                return extracted, missing, {"input": 1500, "output": 500}

            return {}, required, {"input": 1500, "output": 500}

        except Exception as e:
            print(f"  [DEBUG] extract_values_two_stage error: {e}")
            return {}, required, {"input": 0, "output": 0}

    def _repair_extraction(
        self,
        patient_note: str,
        calculator_name: str,
        current_values: Dict[str, Any],
        missing: List[str],
    ) -> Tuple[Dict[str, Any], List[str], Dict[str, int]]:
        """
        Attempt to repair extraction by re-prompting with feedback.
        """
        import json

        # Build repair prompt with explicit feedback and docstring
        docstring = get_calculator_docstring(calculator_name)

        repair_prompt = f"""Previous extraction was incomplete for {calculator_name}.

MISSING REQUIRED VALUES: {', '.join(missing)}
ALREADY EXTRACTED: {current_values}

CALCULATOR DESCRIPTION:
{docstring if docstring else "No description available."}

PATIENT NOTE:
{patient_note}

INSTRUCTIONS:
1. Re-read the patient note CAREFULLY to find the missing values.
2. For sex/gender: "man"/"male"/"he"/"him"/"gentleman" → "male", "woman"/"female"/"she"/"her"/"lady" → "female"
3. Convert units as needed (lbs→kg, feet/inches→cm)
4. For scoring criteria: Check for each condition mentioned in the calculator description.

Return ONLY JSON with the missing values:
{{"extracted": {{"value_name": value, ...}}, "missing": ["values still not found"]}}
"""

        try:
            # call_llm takes a prompt string and returns a string
            response = call_llm(
                prompt=repair_prompt,
                model="deepseek-v3-0324",
                max_tokens=500,
            )

            try:
                parsed = json.loads(response)
            except json.JSONDecodeError:
                import re
                json_match = re.search(r'\{[\s\S]*\}', response)
                if json_match:
                    try:
                        parsed = json.loads(json_match.group())
                    except json.JSONDecodeError:
                        parsed = {"extracted": {}}
                else:
                    parsed = {"extracted": {}}

            new_extracted = parsed.get("extracted", {})
            # Merge with existing values
            merged = {**current_values, **new_extracted}
            # Check what's still missing
            still_missing = [k for k in missing if k not in new_extracted]
            return merged, still_missing, {"input": 1200, "output": 200}

        except Exception:
            return current_values, missing, {"input": 0, "output": 0}

    def run_instance(self, instance: MedCalcInstance) -> ExperimentResult:
        """
        Run the L4 pipeline on a single instance.

        Pipeline:
        1. Identify calculator (LLM)
        2. Extract values (LLM)
        3. Validate values (Python)
        4. Compute result (Python)
        """
        start_time = time.time()
        total_input_tokens = 0
        total_output_tokens = 0

        trace = {
            "stages": [],
            "method": "l4_pipeline",
        }

        try:
            # =================================================================
            # Stage 1: Identify Calculator (LLM)
            # =================================================================
            calc_name, confidence, tokens = self._identify_calculator(instance.question)
            total_input_tokens += tokens["input"]
            total_output_tokens += tokens["output"]

            trace["stages"].append({
                "stage": "identify",
                "calculator": calc_name,
                "confidence": confidence,
            })

            if calc_name is None:
                self.stage_stats["identify_fail"] += 1
                raise ValueError("Could not identify calculator")

            self.stage_stats["identify_success"] += 1

            # =================================================================
            # Stage 2: Extract Values (LLM)
            # =================================================================
            extracted, missing, tokens = self._extract_values(
                patient_note=instance.patient_note,
                calculator_name=calc_name,
            )
            total_input_tokens += tokens["input"]
            total_output_tokens += tokens["output"]

            trace["stages"].append({
                "stage": "extract",
                "extracted": extracted,
                "missing": missing,
            })

            if extracted:
                self.stage_stats["extract_success"] += 1
            else:
                self.stage_stats["extract_fail"] += 1

            # =================================================================
            # Stage 3: Validate Values (Python)
            # =================================================================
            is_valid, validation_missing, cleaned_values = validate_extracted_values(
                extracted=extracted,
                calculator_name=calc_name,
            )

            trace["stages"].append({
                "stage": "validate",
                "is_valid": is_valid,
                "missing": validation_missing,
                "cleaned": cleaned_values,
            })

            # Attempt repair if values missing
            if not is_valid and self.max_repair_attempts > 0:
                for attempt in range(self.max_repair_attempts):
                    repaired, still_missing, tokens = self._repair_extraction(
                        patient_note=instance.patient_note,
                        calculator_name=calc_name,
                        current_values=cleaned_values,
                        missing=validation_missing,
                    )
                    total_input_tokens += tokens["input"]
                    total_output_tokens += tokens["output"]

                    is_valid, validation_missing, cleaned_values = validate_extracted_values(
                        extracted=repaired,
                        calculator_name=calc_name,
                    )

                    trace["stages"].append({
                        "stage": f"repair_{attempt+1}",
                        "extracted": repaired,
                        "is_valid": is_valid,
                        "missing": validation_missing,
                    })

                    if is_valid:
                        break

            if not is_valid:
                self.stage_stats["validate_fail"] += 1
                # Try to proceed anyway with partial values
            else:
                self.stage_stats["validate_success"] += 1

            # =================================================================
            # Stage 4: Compute Result (Python - NO LLM)
            # =================================================================
            result = compute_with_python(calc_name, cleaned_values)

            if result is None:
                # Python failed - try L1 ptool fallback
                trace["stages"].append({
                    "stage": "compute_python",
                    "success": False,
                    "error": "Python calculation failed, trying L1 fallback",
                })

                try:
                    l1_result = l1_calculate(
                        patient_note=instance.patient_note,
                        question=instance.question,
                    )
                    # L1 returns {"result": float} structure
                    if isinstance(l1_result, dict):
                        inner = l1_result.get("result", l1_result)
                        if isinstance(inner, (int, float)):
                            predicted = float(inner)
                            self.stage_stats["compute_success"] += 1
                            trace["stages"].append({
                                "stage": "compute_l1_fallback",
                                "success": True,
                                "result": predicted,
                                "method": "l1_fallback",
                            })
                            # Add L1 tokens
                            total_input_tokens += 800
                            total_output_tokens += 100
                        else:
                            predicted = None
                            self.stage_stats["compute_fail"] += 1
                    elif isinstance(l1_result, (int, float)):
                        predicted = float(l1_result)
                        self.stage_stats["compute_success"] += 1
                        total_input_tokens += 800
                        total_output_tokens += 100
                    else:
                        predicted = None
                        self.stage_stats["compute_fail"] += 1
                except Exception as e:
                    predicted = None
                    self.stage_stats["compute_fail"] += 1
                    trace["stages"].append({
                        "stage": "compute_l1_fallback",
                        "success": False,
                        "error": str(e),
                    })
            else:
                self.stage_stats["compute_success"] += 1
                predicted = result.result
                trace["stages"].append({
                    "stage": "compute",
                    "success": True,
                    "result": predicted,
                    "formula": result.formula_used,
                })

            latency_ms = (time.time() - start_time) * 1000

            # Calculate cost (only for LLM stages)
            cost_metrics = calculate_cost(
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
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
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                total_tokens=total_input_tokens + total_output_tokens,
                cost_usd=cost_metrics.cost_usd,
                num_steps=len(trace["stages"]),
                raw_response=str(predicted),
                trace=trace,
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
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                total_tokens=total_input_tokens + total_output_tokens,
                cost_usd=0.0,
                error=str(e),
                error_type=type(e).__name__,
                trace=trace,
            )

    def get_summary(self) -> Dict[str, Any]:
        """Get summary with stage-level statistics."""
        summary = super().get_summary()
        summary["stage_stats"] = self.stage_stats

        # Calculate stage success rates
        for stage in ["identify", "extract", "validate", "compute"]:
            success = self.stage_stats.get(f"{stage}_success", 0)
            fail = self.stage_stats.get(f"{stage}_fail", 0)
            total = success + fail
            summary[f"{stage}_success_rate"] = success / total if total > 0 else 0

        return summary


# =============================================================================
# Standalone execution
# =============================================================================

if __name__ == "__main__":
    """
    Run L4 Pipeline experiment standalone.

    Usage:
        export TOGETHER_API_KEY="your-key"
        python -m benchmark.experiments.l4_pipeline
        python -m benchmark.experiments.l4_pipeline --n 10
        python -m benchmark.experiments.l4_pipeline --instance 42
    """
    import argparse

    parser = argparse.ArgumentParser(description="Run L4 Pipeline Experiment")
    parser.add_argument("-n", "--num", type=int, default=5, help="Number of instances")
    parser.add_argument("--instance", type=int, help="Run specific instance")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    # Load dataset
    print("Loading MedCalc-Bench dataset...")
    from benchmark.dataset.loader import MedCalcDataset
    from benchmark.config import ExperimentConfig

    dataset = MedCalcDataset()

    if args.instance is not None:
        all_instances = dataset.load("test")
        instances = [i for i in all_instances if i.row_number == args.instance]
        if not instances:
            print(f"Instance {args.instance} not found")
            sys.exit(1)
    else:
        instances = dataset.get_debug_subset(args.num)

    print(f"Running L4 Pipeline on {len(instances)} instances...")
    print("Pipeline: Identify(LLM) -> Extract(LLM) -> Validate(Py) -> Compute(Py)")
    print()

    # Create experiment
    from benchmark.config import ABLATION_CONFIGS
    config = ABLATION_CONFIGS["l4_pipeline"]
    experiment = L4PipelineExperiment(config)
    experiment.setup()

    # Run instances
    correct = 0
    total = 0
    total_cost = 0.0

    for i, instance in enumerate(instances):
        print(f"[{i+1}/{len(instances)}] {instance.calculator_name}")
        print(f"  Q: {instance.question[:60]}...")

        result = experiment.run_instance(instance)
        total += 1
        total_cost += result.cost_usd

        if result.error:
            status = f"ERROR: {result.error}"
        elif result.is_correct_tolerance:
            correct += 1
            status = "CORRECT"
        else:
            status = f"WRONG (pred={result.predicted_answer}, exp={result.ground_truth})"

        print(f"  {status}")
        print(f"  Latency: {result.latency_ms:.0f}ms, Cost: ${result.cost_usd:.4f}")

        if args.verbose and result.trace:
            for stage in result.trace.get("stages", []):
                print(f"    - {stage.get('stage')}: {stage}")

        print()

    # Summary
    print("=" * 60)
    print(f"L4 PIPELINE RESULTS: {correct}/{total} ({correct/total*100:.1f}%)")
    print(f"Total cost: ${total_cost:.4f}")
    print(f"Avg cost:   ${total_cost/total:.4f}/instance")
    print()
    print("Stage statistics:")
    summary = experiment.get_summary()
    for stage in ["identify", "extract", "validate", "compute"]:
        rate = summary.get(f"{stage}_success_rate", 0)
        print(f"  {stage}: {rate*100:.1f}% success")
    print("=" * 60)
