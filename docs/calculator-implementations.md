# Calculator Implementations

Documentation for the medical calculator systems used in the benchmark experiments.

**Last Updated**: 2025-01-08

---

## Overview

The framework includes two calculator registries:

1. **Simplified Registry** (`calculator_simple.py`) - Handwritten, 55 calculators with clean interfaces
2. **Official Registry** (`calculator_implementations/`) - Original MedCalc-Bench implementations, 48+ calculators

Both registries are used by different experiment layers:
- L1, L2 use the simplified registry
- L1O, L2O, L4 use the official implementations

---

## Simplified Calculator Registry

**File**: `benchmark/experiments/calculator_simple.py`

A handwritten registry with explicit type signatures and consistent interfaces.

### Architecture

```python
@calculator
def bmi(weight_kg: float, height_cm: float) -> float:
    """Calculate Body Mass Index.

    Formula: weight / (height_m)^2
    """
    height_m = height_cm / 100
    return weight_kg / (height_m ** 2)
```

The `@calculator` decorator:
- Registers the function in `CALCULATOR_REGISTRY`
- Introspects parameter types from signature
- Extracts formula from docstring

### Calculator Categories (55 total)

| Category | Count | Examples |
|----------|-------|----------|
| Physical/Anthropometric | 7 | BMI, IBW, ABW, BSA, MAP, Target Weight, Maintenance Fluids |
| Renal Function | 3 | CrCl (Cockcroft-Gault), CKD-EPI, MDRD |
| Electrolytes | 10 | Anion Gap, Delta Gap, Osmolality, Sodium Correction, FENa |
| Cardiac | 9 | QTc (5 formulas), CHA2DS2-VASc, HEART, RCRI, Wells PE |
| Hepatic | 4 | FIB-4, MELD-Na, Child-Pugh, Steroid Conversion |
| Pulmonary | 4 | CURB-65, PSI, PERC, SOFA |
| Infectious | 4 | Centor, FeverPAIN, SIRS, Glasgow-Blatchford |
| Hematologic | 4 | HAS-BLED, Wells DVT, Caprini, MME |
| ICU Scoring | 2 | APACHE II, Charlson CCI |
| Obstetric | 3 | Gestational Age, Due Date, Conception Date |
| Other | 3 | GCS, HOMA-IR, Framingham Risk |

### Using the Registry

```python
from benchmark.experiments.calculator_simple import (
    CALCULATOR_REGISTRY,
    get_calculator,
    compute
)

# List all calculators
for name, spec in CALCULATOR_REGISTRY.items():
    print(f"{name}: {spec.required_params}")

# Get calculator spec
spec = get_calculator("bmi")
print(spec.formula)  # "weight / (height_m)^2"

# Compute result
result = compute("bmi", weight_kg=80, height_cm=180)
```

---

## Official Calculator Implementations

**Directory**: `benchmark/experiments/calculator_implementations/`

The original MedCalc-Bench calculator implementations with comprehensive unit handling and explanation generation.

### Structure

```
calculator_implementations/
├── calc_path.json           # Registry mapping names to files
├── rounding.py              # Rounding utilities
├── unit_converter_new.py    # Unit conversion system
├── height_conversion.py     # Height unit conversions
├── weight_conversion.py     # Weight unit conversions
├── age_conversion.py        # Age parsing utilities
├── convert_temperature.py   # Temperature conversions
├── bmi_calculator.py        # BMI implementation
├── creatinine_clearance.py  # CrCl (Cockcroft-Gault)
├── cha2ds2_vasc_score.py    # Stroke risk score
├── heart_score.py           # Cardiac events risk
├── apache_ii.py             # ICU severity score
└── ... (60+ files)
```

### Calculator Registry (calc_path.json)

Maps official calculator names to file paths and IDs:

```json
{
  "Body Mass Index (BMI)": {
    "path": "bmi_calculator.py",
    "id": 6
  },
  "Creatinine Clearance (Cockcroft-Gault Equation)": {
    "path": "creatinine_clearance.py",
    "id": 2
  }
}
```

### Standard Function Interface

All calculators follow this pattern:

```python
def calculator_name_explanation(input_variables) -> dict:
    """
    Compute calculator and return explanation with answer.

    Returns:
        {"Explanation": str, "Answer": float}
    """
    # ... calculation logic ...
    return {
        "Explanation": explanation_text,
        "Answer": result
    }
```

### Input Format Conventions

| Type | Format | Example |
|------|--------|---------|
| Numeric with unit | `[value, "unit"]` | `[80, "kg"]`, `[170, "cm"]` |
| Age | `[value, "years"]` or `[value, "months"]` | `[45, "years"]` |
| Height (imperial) | `[feet, "ft", inches, "in"]` | `[5, "ft", 10, "in"]` |
| Boolean | `True` or `False` | `True` |
| Categorical | String | `"Highly suspicious"`, `"Normal"` |

### Calculator Categories

#### Scoring Systems (17 calculators)

| Calculator | File | ID | Purpose |
|------------|------|-----|---------|
| CHA2DS2-VASc | `cha2ds2_vasc_score.py` | 4 | AFib stroke risk |
| APACHE II | `apache_ii.py` | 28 | ICU severity |
| HEART Score | `heart_score.py` | 18 | Chest pain evaluation |
| HAS-BLED | `has_bled_score.py` | 25 | Anticoagulation bleeding risk |
| RCRI | `cardiac_risk_index.py` | 17 | Pre-operative cardiac risk |
| SOFA | `sofa.py` | 43 | Organ failure assessment |
| Wells' PE | `wells_criteria_pe.py` | 8 | Pulmonary embolism risk |
| Wells' DVT | `wells_criteria_dvt.py` | 16 | DVT risk |
| CURB-65 | `curb-65.py` | 45 | Pneumonia severity |
| PSI | `psi_score.py` | 29 | CAP severity index |
| PERC | `perc_rule.py` | 48 | PE exclusion |
| Caprini | `caprini_score.py` | 36 | VTE risk |
| Centor | `centor_score.py` | 20 | Strep pharyngitis |
| FeverPAIN | `feverpain.py` | 33 | Strep screening |
| Glasgow-Blatchford | `glasgow_bleeding_score.py` | 27 | Upper GI bleeding |
| Charlson CCI | `cci.py` | 32 | Comorbidity index |
| GCS | `glasgow_coma_score.py` | 21 | Consciousness level |

#### Renal Function (3 calculators)

| Calculator | File | ID |
|------------|------|-----|
| CrCl (Cockcroft-Gault) | `creatinine_clearance.py` | 2 |
| CKD-EPI (2021) | `ckd-epi_2021_creatinine.py` | 3 |
| MDRD GFR | `mdrd_gfr.py` | 9 |

#### Electrolyte & Acid-Base (9 calculators)

| Calculator | File | ID |
|------------|------|-----|
| Anion Gap | `anion_gap.py` | 39 |
| Delta Gap | `delta_gap.py` | 63 |
| Delta Ratio | `delta_ratio.py` | 64 |
| Albumin Corrected Anion Gap | `albumin_corrected_anion.py` | 65 |
| Albumin Corrected Delta Gap | `albumin_corrected_delta_gap.py` | 66 |
| Albumin Corrected Delta Ratio | `albumin_delta_ratio.py` | 67 |
| Serum Osmolality | `sOsm.py` | 30 |
| Sodium Correction | `sodium_correction_hyperglycemia.py` | 26 |
| FENa | `compute_fena.py` | 40 |

#### Hepatic (3 calculators)

| Calculator | File | ID |
|------------|------|-----|
| Child-Pugh | `child_pugh_score.py` | 15 |
| FIB-4 | `fibrosis_4.py` | 19 |
| MELD-Na | `meldna.py` | 23 |

#### Cardiovascular (7 calculators)

| Calculator | File | ID |
|------------|------|-----|
| MAP | `mean_arterial_pressure.py` | 5 |
| QTc Bazett | `qt_calculator_bazett.py` | 11 |
| QTc Fridericia | `qt_calculator_fredericia.py` | 56 |
| QTc Framingham | `qt_calculator_framingham.py` | 57 |
| QTc Hodges | `qt_calculator_hodges.py` | 58 |
| QTc Rautaharju | `qt_calculator_rautaharju.py` | 59 |
| Framingham Risk | `framingham_risk_score.py` | 46 |

#### Anthropometric (5 calculators)

| Calculator | File | ID |
|------------|------|-----|
| BMI | `bmi_calculator.py` | 6 |
| BSA (Mosteller) | `bsa_calculator.py` | 60 |
| Ideal Body Weight | `ideal_body_weight.py` | 10 |
| Adjusted Body Weight | `adjusted_body_weight.py` | 62 |
| Target Weight | `target_weight.py` | 61 |

#### Other Categories

| Category | Calculators |
|----------|-------------|
| Medication/Dosing | MME, Steroid Conversion |
| Metabolic | HOMA-IR, LDL (Friedewald) |
| Fluid/Nutritional | Free Water Deficit, Maintenance Fluids |
| Obstetric | Gestational Age, Due Date, Conception Date |

---

## Official Calculators Wrapper

**File**: `benchmark/experiments/official_calculators.py`

Unified interface for accessing official calculator implementations.

### Key Functions

```python
from benchmark.experiments.official_calculators import (
    load_calculators,
    get_calculator,
    compute_official,
    convert_extracted_to_official,
    get_official_source,
    get_expected_params,
    NAME_MAPPING
)
```

| Function | Purpose |
|----------|---------|
| `load_calculators()` | Load all calculators into `OFFICIAL_REGISTRY` |
| `get_calculator(name)` | Retrieve by name with fuzzy matching |
| `compute_official(name, params)` | Execute calculator |
| `convert_extracted_to_official(params)` | Convert LLM-extracted formats |
| `get_official_source(name)` | Get source code for prompts |
| `get_expected_params(name)` | Get parameter specification |

### Name Mapping

The `NAME_MAPPING` dictionary maps 100+ alternative names to official names:

```python
NAME_MAPPING = {
    "bmi": "Body Mass Index (BMI)",
    "body mass index": "Body Mass Index (BMI)",
    "crcl": "Creatinine Clearance (Cockcroft-Gault Equation)",
    "cockcroft-gault": "Creatinine Clearance (Cockcroft-Gault Equation)",
    "creatinine clearance": "Creatinine Clearance (Cockcroft-Gault Equation)",
    "cha2ds2": "CHA2DS2-VASc Score for Atrial Fibrillation Stroke Risk",
    # ... 100+ mappings
}
```

### Parameter Conversion

The `convert_extracted_to_official()` function handles format normalization:

```python
# LLM extracts: {"weight": 80, "height": 180}
# Converts to: {"weight": [80, "kg"], "height": [180, "cm"]}

# Handles:
# - Age: 45 → [45, "years"]
# - Height: 170 → [170, "cm"]
# - Weight: 80 → [80, "kg"]
# - Lab values: {"sodium": 140} → {"sodium": [140, "mEq/L"]}
# - Booleans: "yes" → True, "no" → False
# - Categoricals: Normalized to expected options
```

### Usage Example

```python
from benchmark.experiments.official_calculators import (
    load_calculators,
    compute_official,
    convert_extracted_to_official
)

# Initialize registry
load_calculators()

# LLM-extracted values
extracted = {
    "weight": 80,
    "height": 180
}

# Convert to official format
official_params = convert_extracted_to_official(extracted, "bmi")
# Result: {"weight": [80, "kg"], "height": [180, "cm"]}

# Compute
result = compute_official("bmi", official_params)
# Result: {"Explanation": "...", "Answer": 24.69}
```

---

## Unit Conversion Utilities

**File**: `calculator_implementations/unit_converter_new.py`

Comprehensive unit conversion system used by calculators.

### Available Conversions

| Function | Purpose |
|----------|---------|
| `vol_to_vol_explanation()` | Volume (L, dL, mL, uL) |
| `molg_to_molg_explanation()` | Molar/gram conversions |
| `mol_g_explanation()` | Moles to grams |
| `g_to_mol_explanation()` | Grams to moles |
| `mEq_to_mol_explanation()` | Milliequivalent to mol |
| `mol_to_mEq_explanation()` | Mol to milliequivalent |
| `conversion_explanation()` | General concentration conversions |
| `convert_to_units_per_liter_explanation()` | Cell count conversions |
| `mmHg_to_kPa_explanation()` | Pressure (mmHg to kPa) |
| `kPa_to_mmHg_explanation()` | Pressure (kPa to mmHg) |

### Supplementary Converters

| File | Purpose |
|------|---------|
| `height_conversion.py` | m, cm, ft, in conversions |
| `weight_conversion.py` | kg, lbs conversions |
| `age_conversion.py` | Age parsing and normalization |
| `convert_temperature.py` | Fahrenheit to Celsius |

---

## Rounding

**File**: `calculator_implementations/rounding.py`

Consistent rounding for medical calculations:

```python
def round_number(value: float) -> float:
    """Round based on magnitude to preserve precision.

    - Large numbers: 5 decimal places
    - Small numbers: 5 significant digits
    """
```

---

## Adding a New Calculator

### To Simplified Registry

```python
# In calculator_simple.py

@calculator
def new_calculator(
    param1: float,
    param2: float,
    optional_param: Optional[float] = None
) -> float:
    """Calculate something useful.

    Formula: param1 * param2 / 100
    """
    result = param1 * param2 / 100
    if optional_param:
        result *= optional_param
    return result
```

### To Official Registry

1. Create `calculator_implementations/new_calculator.py`:

```python
def new_calculator_explanation(param1, param2):
    """
    Compute new calculator.

    Args:
        param1: [value, "unit"]
        param2: [value, "unit"]

    Returns:
        {"Explanation": str, "Answer": float}
    """
    val1 = param1[0]
    val2 = param2[0]

    explanation = f"Using param1={val1} and param2={val2}.\n"
    result = val1 * val2 / 100
    explanation += f"Result = {val1} * {val2} / 100 = {result}"

    return {
        "Explanation": explanation,
        "Answer": result
    }
```

2. Add to `calc_path.json`:

```json
{
  "New Calculator Name": {
    "path": "new_calculator.py",
    "id": 70
  }
}
```

3. Add name mappings to `official_calculators.py`:

```python
NAME_MAPPING["new calc"] = "New Calculator Name"
NAME_MAPPING["new_calculator"] = "New Calculator Name"
```

---

## See Also

- [Experiments](experiments.md) - How calculators are used in benchmark layers
- [Critic-Repair System](critic-repair.md) - Python calculator fallback for repairs
- [MedCalc Domain](medcalc-domain.md) - Domain-specific audits
