# Benchmark Experiments

Documentation for the benchmark experiment layers (L0-L5) used to evaluate the ptool_framework on medical calculation tasks.

**Last Updated**: 2025-01-08

---

## Overview

The experiment layers implement William Cohen's research hypothesis: **"Python programs calling LLMs, not LLMs calling tools."**

Each layer represents a different balance between:
- **LLM autonomy** vs **Python control**
- **Flexibility** vs **Predictability**
- **Cost** vs **Accuracy**

The progression from L0 to L5 shows how structured approaches can improve accuracy while reducing costs.

---

## Layer Hierarchy

```
L0: Vanilla Baseline
 │   └─ Pure LLM, no structure
 │
L0+: Baseline with Formulae
 │   └─ LLM with formula hints in prompt
 │
L1: @ptool Decorator
 │   └─ Single LLM call for extraction + calculation
 │   └─ L1O: Extraction only, Python computes
 │
L2: @distilled Decorator
 │   └─ Python-first with LLM fallback
 │   └─ L2O: Using official MedCalc implementations
 │
L3: ReAct Agent
 │   └─ Autonomous multi-step reasoning
 │   └─ L3+: With structured audits
 │
L4: Python-Orchestrated Pipeline
 │   ├─ L4-Pipeline: 4-stage identify→extract→validate→compute
 │   ├─ L4-Adaptive: Difficulty-aware routing
 │   └─ L4-Orchestrator: Multi-agent coordination
 │
L5: Self-Improving Agents
    └─ Learning from traces, ICL injection
```

---

## Experiment Layers

### L0: Vanilla Baseline

**File**: `benchmark/experiments/baseline.py`

**Approach**: Direct LLM call with no structure. The model receives the patient note and calculator name, and must return the calculated value.

**When to Use**: As a baseline to measure improvements from structured approaches.

```python
class BaselineExperiment(BaseExperiment):
    """L0: Pure LLM baseline - no ptools, no structure."""

    def run_instance(self, instance: Dict) -> ExperimentResult:
        response = self.llm.generate(
            prompt=self.format_prompt(instance),
            model=self.model
        )
        return self.parse_response(response)
```

---

### L0+: Baseline with Formulae

**File**: `benchmark/experiments/baseline_with_formulae.py`

**Approach**: Enhanced baseline that includes medical formula references in the prompt. The LLM sees the actual formulas it should use.

**Key Feature**: `get_formula_reference()` extracts formulas from `calculators.py` and injects them into the prompt.

**When to Use**: When you want a slightly better baseline without full structural changes.

---

### L1: @ptool Decorator

**File**: `benchmark/experiments/l1_ptool.py`

**Approach**: Uses the `@ptool` decorator to define a single LLM function that extracts parameters AND performs the calculation.

```python
@ptool(model="deepseek-v3-0324", output_mode="structured")
def calculate_medical_value(
    patient_note: str,
    calculator_name: str
) -> Dict[str, Any]:
    """Extract values from patient note and calculate the result.

    Return {"calculator": str, "extracted_values": dict, "result": float}
    """
    ...
```

**Characteristics**:
- Single LLM call
- LLM responsible for both extraction and calculation
- Prone to arithmetic errors

---

### L1O: Official Calculator Variant (NEW)

**File**: `benchmark/experiments/l1o_official.py`

**Approach**: LLM extracts parameters, but Python performs the calculation using official MedCalc implementations.

**Pipeline**:
1. **LLM**: Extract calculator name and parameter values
2. **Python**: Call official calculator function

```python
@ptool(model="deepseek-v3-0324", output_mode="structured")
def extract_for_calculator(patient_note: str, calculator_name: str) -> Dict:
    """Extract parameter values from patient note.

    Return {"calculator": str, "values": dict}
    """
    ...

class L1OOfficialExperiment(BaseExperiment):
    def run_instance(self, instance: Dict) -> ExperimentResult:
        # Stage 1: LLM extraction
        extracted = extract_for_calculator(...)

        # Stage 2: Python calculation
        result = compute_official(extracted["calculator"], extracted["values"])
        return result
```

**Advantages over L1**:
- No arithmetic errors (Python computes)
- Uses validated calculator implementations
- Separation of concerns

---

### L2: @distilled Decorator

**File**: `benchmark/experiments/l2_distilled.py`

**Approach**: Python-first with LLM fallback. Attempts pattern matching before falling back to LLM.

```python
@distilled(fallback_ptool="calculate_medical_value")
def calculate_fast(patient_note: str, calculator_name: str) -> Dict:
    if matches_bmi_pattern(patient_note):
        return compute_bmi(...)
    if matches_map_pattern(patient_note):
        return compute_map(...)
    raise DistillationFallback("No pattern matched")
```

**When to Use**: After collecting traces that reveal common patterns.

---

### L2O: Official Implementation Variant (NEW)

**File**: `benchmark/experiments/l2o_official.py`

**Approach**: Uses official MedCalc-Bench calculator implementations with LLM-based parameter extraction.

**Pipeline**:
1. **LLM**: Identify which calculator is needed
2. **LLM**: Extract parameters for that specific calculator
3. **Python**: Call official calculator function
4. **Fallback**: L1 ptool if official fails

**Key Features**:
- `OfficialCalculatorRegistry` - Dynamically loads from `calculator_implementations/`
- `extract_parameters_for_calculator()` - LLM extraction with format guidance
- `normalize_official_params()` - Handles unit normalization (Unicode mu → micro)
- Comprehensive fallback chain

```python
class L2OOfficialExperiment(BaseExperiment):
    def run_instance(self, instance: Dict) -> ExperimentResult:
        # Stage 1: Identify calculator
        calc_name = identify_calculator(patient_note, calculator_name)

        # Stage 2: Extract parameters
        params = extract_parameters_for_calculator(patient_note, calc_name)

        # Stage 3: Compute with official implementation
        result = compute_official(calc_name, params)
        return result
```

---

### L3: ReAct Agent

**File**: `benchmark/experiments/l3_react.py`

**Approach**: Autonomous agent that reasons about the problem and takes actions in a loop.

**Pattern**: Thought → Action → Observation → Repeat

```python
class L3ReactExperiment(BaseExperiment):
    def run_instance(self, instance: Dict) -> ExperimentResult:
        agent = ReActAgent(
            available_ptools=[calculate_medical_value, ...],
            max_steps=10
        )
        trajectory = agent.run(goal=f"Calculate {calculator_name}...")
        return trajectory.final_answer
```

**When to Use**: Complex problems requiring multi-step reasoning.

---

### L3+: ReAct with Audits

**File**: `benchmark/experiments/l3_audit.py`

**Approach**: L3 ReAct agent enhanced with structured audits that verify intermediate steps.

**Features**:
- Step location tracking
- Typicality audits
- Repair mechanism for failed audits

---

### L4: Python-Orchestrated Pipeline (Refactored)

**File**: `benchmark/experiments/l4_pipeline.py`

**Approach**: Python orchestrates a 4-stage pipeline with specialist LLM agents at each stage.

**Core Philosophy**: "Python programs calling LLMs, not LLMs calling tools"

**Pipeline Stages**:

| Stage | Type | Purpose |
|-------|------|---------|
| 1 | LLM | Identify which calculator is needed |
| 2 | LLM | Extract values (two-stage: reasoning + extraction) |
| 3 | Python | Validate extracted values (completeness, ranges) |
| 4 | Python | Compute result (NO LLM, pure Python) |

```python
class L4PipelineExperiment(BaseExperiment):
    def run_instance(self, instance: Dict) -> ExperimentResult:
        # Stage 1: Identify calculator
        calc_name = identify_calculator_l4(patient_note, calculator_name)

        # Stage 2: Two-stage extraction
        # 2a: Medical reasoning (identify conditions from findings)
        # 2b: Extract values with proper formats
        values = extract_values_two_stage(patient_note, calc_name)

        # Stage 3: Python validation
        is_valid, feedback = validate_extracted_values(values, calc_name)
        if not is_valid:
            values = repair_extraction(patient_note, values, feedback)

        # Stage 4: Python computation (NO LLM)
        result = compute_with_python(calc_name, values)
        return result
```

**Key Innovations**:
- **Two-stage extraction**: Medical reasoning first, then value extraction
- **Condition injection**: Stage 1 reasoning informs Stage 2 extraction
- **Explicit condition mapping**: Handles stroke/TIA/thromboembolism for CHA2DS2-VASc, etc.
- **Per-stage success tracking**: Metrics for each pipeline stage

---

### L4-Adaptive: Difficulty-Aware Routing (NEW)

**File**: `benchmark/experiments/l4_adaptive.py`

**Approach**: Routes problems to different layers based on estimated difficulty.

**Research Question**: "Can automatic level selection achieve L3 accuracy at L2 cost?"

**Routing Strategy**:

| Difficulty | Target Level | Rationale |
|------------|--------------|-----------|
| SIMPLE | L2 | Python-first, cheapest |
| MEDIUM | L4 Pipeline | Balanced accuracy/cost |
| HARD | L3 ReAct | Most thorough |

**Difficulty Classification**:
- **SIMPLE** (9 calculators): BMI, MAP, QTc variants, etc.
- **MEDIUM** (11 calculators): CrCl, GFR, Anion Gap, etc.
- **HARD** (18+ calculators): CHA2DS2, CURB-65, HEART, Wells, APACHE, SOFA, Charlson, etc.

```python
class DifficultyRouter(BaseRouter):
    """Routes based on calculator difficulty."""

    difficulty_levels = {
        "SIMPLE": ["bmi", "map", "qtc_bazett", ...],
        "MEDIUM": ["creatinine_clearance", "anion_gap", ...],
        "HARD": ["cha2ds2_vasc", "heart_score", "apache_ii", ...]
    }

    def route(self, calculator_name: str) -> str:
        difficulty = self.estimate_difficulty(calculator_name)
        return self.level_mapping[difficulty]

class L4AdaptiveExperiment(BaseExperiment):
    def run_instance(self, instance: Dict) -> ExperimentResult:
        # Determine difficulty
        level = self.router.route(instance["calculator"])

        # Route to appropriate experiment
        if level == "L2":
            return self.l2_experiment.run_instance(instance)
        elif level == "L4":
            return self.l4_experiment.run_instance(instance)
        else:  # L3
            return self.l3_experiment.run_instance(instance)
```

**Statistics Tracked**:
- Routing decisions per difficulty level
- Accuracy per level
- Cost breakdown

---

### L4-Orchestrator: Multi-Agent Coordination

**File**: `benchmark/experiments/l4_orchestrator.py`

**Approach**: Multiple specialized agents coordinated by an orchestrator.

**Features**:
- Agent handoff protocols
- Parallel execution for independent tasks
- Experience-based routing

---

### L5: Self-Improving Agents

**Files**: `benchmark/experiments/l5_icl.py`, `benchmark/experiments/l5_improving.py`

**Approach**: Agents that learn from their execution traces and improve over time.

**Key Features**:
- Pattern extraction from successful trajectories
- ICL example injection from trace store
- Negative example learning from failures
- Pattern memory with persistence and decay

---

## Official Calculators Infrastructure

**File**: `benchmark/experiments/official_calculators.py`

The unified wrapper for official MedCalc-Bench calculator implementations.

**Key Functions**:

| Function | Purpose |
|----------|---------|
| `load_calculators()` | Load all 48 calculators from `calc_path.json` |
| `get_calculator(name)` | Retrieve calculator with fuzzy matching |
| `compute_official(name, params)` | Execute calculator function |
| `convert_extracted_to_official(params)` | Convert parameter formats |
| `get_official_source(name)` | Get calculator source code |
| `get_expected_params(name)` | Get parameter specification |

**Name Mapping**: Maps 100+ alternative calculator names to official names.

---

## Running Experiments

```bash
# Run specific experiment
python -m benchmark.runner --experiment l4_pipeline --dataset medcalc

# Run multiple experiments for comparison
python -m benchmark.runner --experiments l1,l2,l3,l4_pipeline --dataset medcalc

# Run with specific model
python -m benchmark.runner --experiment l4_adaptive --model deepseek-v3-0324
```

---

## Experiment Comparison

| Layer | LLM Calls | Python Control | Accuracy | Cost |
|-------|-----------|----------------|----------|------|
| L0 | 1 | None | Low | $ |
| L1 | 1 | None | Medium | $ |
| L1O | 1 | Calculation | Medium-High | $ |
| L2 | 0-1 | Pattern matching | Medium | $-$$ |
| L2O | 2 | Validation + Calc | High | $$ |
| L3 | N (variable) | None | High | $$$ |
| L4 | 2-3 | Orchestration | High | $$ |
| L4-Adaptive | Variable | Routing | High | $-$$$ |
| L5 | N + learning | Pattern memory | Highest | $$$$ |

---

## See Also

- [Calculator Implementations](calculator-implementations.md) - Medical calculator system
- [Mode 1: @ptool Decorator](mode-1-ptool-decorator.md) - Basic ptool usage
- [Mode 3: ReAct Agent](mode-3-react-agent.md) - Agent architecture
- [Critic-Repair System](critic-repair.md) - Error detection and correction
