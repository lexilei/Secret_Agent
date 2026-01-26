# Applying ptool_framework to William's Research Tasks

This document maps our framework to the specific tasks and research questions in William's 2026 agent research plan.

**Last Updated**: 2024-12-30

---

## Current Implementation Status (December 2024)

### Task Implementation Summary

| Task | Implementation | Status | Files |
|------|---------------|--------|-------|
| MedCalc | Full pipeline with benchmarks | ✓ DONE | `benchmark/experiments/`, `examples/medcalc_*.py` |
| HotPotQA | Framework supports, needs dataset adapter | ◐ PARTIAL | `react.py`, `trace_builder.py` |
| TravelPlanner | Critique+repair loop ready | ◐ PARTIAL | `critic.py`, `repair.py` |
| DABStep | Python+LLM workflow pattern ready | ◐ PARTIAL | `ptool.py`, `llm_backend.py` |
| Verbalized Algorithms | Framework supports trace collection | ◐ PARTIAL | `traces.py`, `trace_store.py` |

### Component Implementation Status

| Component | Status | Lines | File |
|-----------|--------|-------|------|
| @ptool decorator | ✓ DONE | ~300 | `ptool.py` |
| @distilled decorator | ✓ DONE | ~200 | `distilled.py` |
| ReActAgent | ✓ DONE | ~500 | `react.py` |
| TraceCritic | ✓ DONE | 735 | `critic.py` |
| RepairAgent | ✓ DONE | 790 | `repair.py` |
| Audit System (SSRM) | ✓ DONE | ~1000 | `audit/` |
| ModelSelector | ✓ DONE | 1263 | `model_selector.py` |
| AgentOrchestrator | ✓ DONE | 500+ | `orchestrator.py` |
| SelfImprovingAgent | ✓ DONE | 600+ | `self_improving.py` |
| Benchmark Suite | ✓ DONE | ~800 | `benchmark/` |

### The Feedback Loop Gap

```
Current State:
@ptool execution → trace_store.log_execution() → traces saved to ~/.ptool_traces/
                                                          ↓
                                              (NEVER READ BACK FOR ICL)

William's Vision:
@ptool execution → trace_store.log_execution() → traces saved
         ↑                                              ↓
         └──── ICL examples injected ←── trace_store.get_icl_examples()
```

**Gap**: Traces are collected but NOT automatically used to improve future executions.

**Solution** (In Progress): Add `trace_path` and `write_to_memories` parameters to `@ptool` decorator.

---

## Running Benchmarks

### L1-L5 Experiments

```bash
# Navigate to benchmark directory
cd benchmark/

# L1: Basic ptool execution
python experiments/l1_ptool.py

# L2: Distilled function performance
python experiments/l2_distilled.py

# L3: ReAct agent execution
python experiments/l3_react.py

# L3+: ReAct with Critic + Repair
python experiments/l3_audit.py

# L4: Multi-agent orchestration
python experiments/l4_orchestrator.py

# L5: Self-improving agents
python experiments/l5_improving.py

# L5 ICL: In-context learning patterns
python experiments/l5_icl.py
```

### MedCalc Benchmark

```bash
# Run full MedCalc pipeline
python examples/medcalc_full_pipeline.py

# Run showcase demo
python examples/medcalc_showcase.py
```

### Running All Tests

```bash
# Unit tests (no API calls)
pytest tests/ -v -k "not integration"

# Integration tests (requires API key)
export TOGETHER_API_KEY="..."
pytest tests/test_react_integration.py -v
```

---

## How the Framework Maps to William's Vision

| William's Concept | Our Implementation | File:Line |
|-------------------|-------------------|-----------|
| ptool (pseudo-tool) | `@ptool` decorator | `ptool.py:195` |
| Python workflow calling ptools | Regular Python functions calling `@ptool` functions | Examples in `examples/` |
| Workflow trace | `WorkflowTrace` / `ExecutionTrace` | `traces.py`, `trace_store.py:33` |
| Trace builder | `TraceBuilder` / `ProgramGenerator` | `trace_builder.py`, `program_generator.py` |
| Behavior distillation | `BehaviorDistiller` + `@distilled` decorator | `distiller.py`, `distilled.py` |
| "Back off to more agentic" | `DistillationFallback` exception | `distilled.py:15` |
| ICL demos from traces | L5: `ICLExample` in `self_improving.py` | `self_improving.py:159` |
| Audits/unit tests | SSRM-style `audit/` subsystem | `audit/structured_audit.py` |
| Typicality audits | Unigram/Bigram/Trigram/HMM models | `audit/typicality/` |
| Critique + Repair (Approach 3) | `TraceCritic` + `RepairAgent` | `critic.py`, `repair.py` |
| L0-L3 autonomy spectrum | Mode 1 (@ptool), Mode 2 (generator), Mode 3 (ReAct) | See docs/ |
| L4 multi-agent | `AgentOrchestrator` | `orchestrator.py` |
| L5 self-modifying | `SelfImprovingAgent` | `self_improving.py` |

---

## Task 1: MedCalc (Medical Calculations)

**Why it fits**: Medical calculations have clear structure (extract values → apply formula → interpret), but edge cases need LLM reasoning.

### Implementation

```python
# medcalc_agent.py
"""
MedCalc: Calculate medical scores from clinical notes.
Demonstrates: ptools for extraction, Python for calculation, distillation for common cases.
"""

from typing import Dict, List, Literal, Optional
from ptool_framework import ptool, distilled, DistillationFallback

# =============================================================================
# PTOOLS - LLM handles extraction and interpretation
# =============================================================================

@ptool()
def extract_clinical_values(
    note: str,
    required_fields: List[str]
) -> Dict[str, Optional[float]]:
    """Extract clinical values from a medical note.

    Parse the note and extract numeric values for each required field.
    Handle various formats: "BP 120/80", "blood pressure: 120 over 80", etc.

    Return a dict mapping field names to values (None if not found).
    Example: {"systolic_bp": 120, "diastolic_bp": 80, "heart_rate": None}
    """
    ...

@ptool()
def interpret_score(
    score_name: str,
    score_value: float,
    patient_context: Dict[str, any]
) -> Dict[str, str]:
    """Interpret a medical score in clinical context.

    Return:
    {
        "risk_level": "low" | "moderate" | "high" | "critical",
        "interpretation": "Plain English explanation",
        "recommendations": "Clinical recommendations"
    }
    """
    ...

@ptool()
def identify_calculator(task_description: str) -> str:
    """Identify which medical calculator to use.

    Given a task description, return the calculator name.
    Options: "CHA2DS2-VASc", "MELD", "Wells", "CURB-65", "GFR", "BMI", etc.
    """
    ...

# =============================================================================
# PYTHON - Deterministic calculations (the "distilled" part)
# =============================================================================

def calculate_cha2ds2_vasc(values: Dict[str, float]) -> int:
    """CHA2DS2-VASc score for stroke risk in AFib."""
    score = 0
    if values.get("age", 0) >= 75:
        score += 2
    elif values.get("age", 0) >= 65:
        score += 1
    if values.get("chf"):
        score += 1
    if values.get("hypertension"):
        score += 1
    if values.get("diabetes"):
        score += 1
    if values.get("stroke_history"):
        score += 2
    if values.get("vascular_disease"):
        score += 1
    if values.get("female"):
        score += 1
    return score

def calculate_bmi(values: Dict[str, float]) -> float:
    """BMI calculation."""
    weight_kg = values.get("weight_kg")
    height_m = values.get("height_m")
    if weight_kg and height_m:
        return round(weight_kg / (height_m ** 2), 1)
    raise ValueError("Missing weight or height")

def calculate_gfr(values: Dict[str, float]) -> float:
    """eGFR using CKD-EPI equation."""
    # Simplified version
    creatinine = values.get("creatinine")
    age = values.get("age")
    female = values.get("female", False)

    if not (creatinine and age):
        raise ValueError("Missing creatinine or age")

    # CKD-EPI formula (simplified)
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

    return round(gfr, 1)

CALCULATORS = {
    "CHA2DS2-VASc": calculate_cha2ds2_vasc,
    "BMI": calculate_bmi,
    "GFR": calculate_gfr,
    # Add more...
}

# =============================================================================
# MAIN WORKFLOW - Python orchestrates, ptools reason
# =============================================================================

def medcalc_workflow(clinical_note: str, task: str) -> Dict:
    """
    Main MedCalc workflow.

    This is William's L0/L1: Python controls flow, ptools do reasoning.
    """
    # Step 1: Identify calculator (ptool - needs reasoning)
    calculator_name = identify_calculator(task)

    # Step 2: Get required fields (pure Python - deterministic)
    field_requirements = {
        "CHA2DS2-VASc": ["age", "chf", "hypertension", "diabetes",
                        "stroke_history", "vascular_disease", "female"],
        "BMI": ["weight_kg", "height_m"],
        "GFR": ["creatinine", "age", "female"],
    }
    required_fields = field_requirements.get(calculator_name, [])

    # Step 3: Extract values (ptool - needs NLP)
    values = extract_clinical_values(clinical_note, required_fields)

    # Step 4: Calculate (pure Python - deterministic)
    calculator_fn = CALCULATORS.get(calculator_name)
    if not calculator_fn:
        raise ValueError(f"Unknown calculator: {calculator_name}")

    score = calculator_fn(values)

    # Step 5: Interpret (ptool - needs clinical reasoning)
    interpretation = interpret_score(
        score_name=calculator_name,
        score_value=score,
        patient_context=values
    )

    return {
        "calculator": calculator_name,
        "extracted_values": values,
        "score": score,
        "interpretation": interpretation
    }

# =============================================================================
# DISTILLED VERSION - After collecting traces
# =============================================================================

@distilled(fallback_ptool="extract_clinical_values")
def extract_clinical_values_fast(
    note: str,
    required_fields: List[str]
) -> Dict[str, Optional[float]]:
    """Distilled extraction - regex for common patterns, LLM for complex cases."""
    import re

    values = {}
    note_lower = note.lower()

    # Common patterns learned from traces
    patterns = {
        "age": r"(\d+)\s*(?:year|yr|y/?o)",
        "weight_kg": r"(\d+(?:\.\d+)?)\s*kg",
        "height_m": r"(\d+(?:\.\d+)?)\s*m(?:eter)?s?\b",
        "creatinine": r"creatinine[:\s]+(\d+(?:\.\d+)?)",
        "systolic_bp": r"(?:bp|blood pressure)[:\s]+(\d+)/",
        "diastolic_bp": r"(?:bp|blood pressure)[:\s]+\d+/(\d+)",
    }

    for field in required_fields:
        if field in patterns:
            match = re.search(patterns[field], note_lower)
            if match:
                values[field] = float(match.group(1))
            else:
                values[field] = None
        else:
            # Can't handle this field with regex
            raise DistillationFallback(f"No pattern for field: {field}")

    # If we couldn't extract required fields, fall back
    missing = [f for f in required_fields if values.get(f) is None]
    if len(missing) > len(required_fields) * 0.5:
        raise DistillationFallback(f"Too many missing fields: {missing}")

    return values
```

### Research Value

1. **Behavior Distillation Demo**: `extract_clinical_values` → `extract_clinical_values_fast`
2. **Measurable**: Can compare accuracy and speed on MedCalc benchmark
3. **Clear L0/L1 structure**: Python does math, LLM does NLP

---

## Task 2: HotPotQA (Multi-hop Reasoning)

**Why it fits**: Multi-hop QA requires decomposition (trace builder) and can be distilled for common patterns.

### Implementation

```python
# hotpotqa_agent.py
"""
HotPotQA: Multi-hop question answering.
Demonstrates: Trace builders, workflow traces, incremental distillation.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from ptool_framework import ptool, WorkflowTrace, TraceStep

# =============================================================================
# PTOOLS
# =============================================================================

@ptool()
def decompose_question(question: str) -> List[str]:
    """Decompose a multi-hop question into single-hop sub-questions.

    Example:
    "What is the capital of the country where the Eiffel Tower is located?"
    → ["Where is the Eiffel Tower located?", "What is the capital of France?"]

    Return the sub-questions in order of dependency.
    """
    ...

@ptool()
def answer_simple_question(
    question: str,
    context: str
) -> str:
    """Answer a simple factual question given context.

    Return just the answer, no explanation.
    """
    ...

@ptool()
def retrieve_context(
    query: str,
    corpus: List[str]
) -> str:
    """Find the most relevant passage for a query.

    Return the passage that best answers the query.
    """
    ...

@ptool()
def synthesize_answer(
    original_question: str,
    sub_answers: List[Tuple[str, str]]  # [(sub_q, answer), ...]
) -> str:
    """Synthesize final answer from sub-question answers.

    Combine the sub-answers to answer the original question.
    """
    ...

# =============================================================================
# TRACE BUILDER - Instance-specific workflow generation
# =============================================================================

@ptool()
def build_reasoning_trace(
    question: str,
    corpus: List[str]
) -> WorkflowTrace:
    """Build an instance-specific reasoning trace for a question.

    This is William's "trace builder" - a PoT-like ptool that produces
    a workflow trace (sequence of tool calls) for a specific instance.

    Analyze the question and produce a trace that:
    1. Decomposes the question if multi-hop
    2. Retrieves relevant context for each sub-question
    3. Answers sub-questions in dependency order
    4. Synthesizes the final answer

    Return a WorkflowTrace that can be executed.
    """
    ...

# =============================================================================
# MAIN WORKFLOW - Fixed structure (L0)
# =============================================================================

def hotpotqa_fixed_workflow(question: str, corpus: List[str]) -> Dict:
    """
    Fixed two-hop workflow (L0 in William's hierarchy).

    Simple, predictable, but limited to 2-hop questions.
    """
    # Step 1: Decompose
    sub_questions = decompose_question(question)

    # Step 2: Answer each sub-question
    sub_answers = []
    accumulated_context = ""

    for sub_q in sub_questions:
        # Retrieve context
        context = retrieve_context(sub_q, corpus)
        accumulated_context += f"\n{context}"

        # Answer with accumulated context
        answer = answer_simple_question(sub_q, accumulated_context)
        sub_answers.append((sub_q, answer))

    # Step 3: Synthesize
    final_answer = synthesize_answer(question, sub_answers)

    return {
        "question": question,
        "decomposition": sub_questions,
        "sub_answers": sub_answers,
        "final_answer": final_answer
    }

# =============================================================================
# ADAPTIVE WORKFLOW - Uses trace builder (L1)
# =============================================================================

def hotpotqa_adaptive_workflow(question: str, corpus: List[str]) -> Dict:
    """
    Adaptive workflow using trace builder (L1 in William's hierarchy).

    More flexible - trace builder decides the reasoning structure.
    Falls back to fixed workflow if trace execution fails.
    """
    from ptool_framework import TraceExecutor

    # Build instance-specific trace
    trace = build_reasoning_trace(question, corpus)

    # Execute with error detection and retry (like Reflexion)
    executor = TraceExecutor(max_retries=2, validate_outputs=True)

    try:
        result = executor.execute(trace)
        if result.success:
            return {
                "question": question,
                "trace": trace,
                "final_answer": result.final_output,
                "method": "adaptive"
            }
    except Exception as e:
        print(f"Trace execution failed: {e}, falling back to fixed workflow")

    # Fallback to fixed workflow
    return hotpotqa_fixed_workflow(question, corpus)

# =============================================================================
# DISTILLATION: Trace patterns → Python
# =============================================================================

# After running adaptive workflow on many examples, we can identify patterns:
# Pattern 1: "What is X of the Y that Z?" → decompose to [find Y that Z, find X of Y]
# Pattern 2: "Who/What/When did X after Y?" → decompose to [when Y, what X after that]

@distilled(fallback_ptool="decompose_question")
def decompose_question_fast(question: str) -> List[str]:
    """Distilled decomposition for common question patterns."""
    import re
    q = question.lower()

    # Pattern: "What is the X of the country/city/person where/who Y?"
    match = re.match(
        r"what is the (\w+) of the (\w+) (?:where|who|that) (.+)\?",
        q
    )
    if match:
        attr, entity_type, condition = match.groups()
        return [
            f"Which {entity_type} {condition}?",
            f"What is the {attr} of that {entity_type}?"
        ]

    # Pattern: "When did X happen after Y?"
    match = re.match(r"when did (.+) (?:happen )?after (.+)\?", q)
    if match:
        event1, event2 = match.groups()
        return [
            f"When did {event2}?",
            f"When did {event1}?"
        ]

    # Can't handle - fall back to LLM
    raise DistillationFallback("Unknown question pattern")
```

### Research Value

1. **Trace Builder Demo**: `build_reasoning_trace` generates instance-specific workflows
2. **L0 vs L1 comparison**: Fixed vs adaptive workflow on same benchmark
3. **Distillation of decomposition patterns**: Common question structures → regex

---

## Task 3: TravelPlanner

**Why it fits**: Complex constraints, needs both reasoning (ptools) and constraint solving (Python).

### Implementation

```python
# travelplanner_agent.py
"""
TravelPlanner: Plan multi-city trips with constraints.
Demonstrates: Constraint handling in Python, reasoning in ptools, repair agents.
"""

from typing import Dict, List, Optional, Literal
from dataclasses import dataclass
from datetime import date, timedelta
from ptool_framework import ptool, distilled, DistillationFallback

@dataclass
class TripConstraints:
    budget: float
    start_date: date
    end_date: date
    must_visit: List[str]
    preferences: List[str]  # "museums", "beaches", "nightlife", etc.

@dataclass
class FlightOption:
    origin: str
    destination: str
    price: float
    departure: str
    arrival: str

@dataclass
class HotelOption:
    city: str
    name: str
    price_per_night: float
    rating: float

# =============================================================================
# PTOOLS - Reasoning tasks
# =============================================================================

@ptool()
def understand_request(user_request: str) -> TripConstraints:
    """Parse a natural language travel request into structured constraints.

    Extract: budget, dates, required cities, preferences.
    Example: "I want to visit Paris and Rome in March, budget $3000, love art"
    """
    ...

@ptool()
def suggest_itinerary_order(
    cities: List[str],
    constraints: TripConstraints
) -> List[str]:
    """Suggest optimal order to visit cities.

    Consider: geography, flight availability, city characteristics.
    Return cities in suggested visit order.
    """
    ...

@ptool()
def select_activities(
    city: str,
    days: int,
    preferences: List[str],
    budget_remaining: float
) -> List[Dict]:
    """Select activities for a city visit.

    Return list of activities with estimated costs and times.
    """
    ...

@ptool()
def critique_plan(
    plan: Dict,
    constraints: TripConstraints
) -> Dict[str, any]:
    """Critique a travel plan for issues.

    Check: budget overrun, impossible timings, preference mismatches.
    Return: {"valid": bool, "issues": [...], "suggestions": [...]}
    """
    ...

@ptool()
def repair_plan(
    plan: Dict,
    issues: List[str],
    constraints: TripConstraints
) -> Dict:
    """Repair a plan to fix identified issues.

    Modify the plan to address each issue while staying within constraints.
    """
    ...

# =============================================================================
# PYTHON - Constraint enforcement & optimization
# =============================================================================

def calculate_trip_cost(plan: Dict) -> float:
    """Calculate total trip cost (deterministic)."""
    total = 0
    for segment in plan.get("segments", []):
        total += segment.get("flight_cost", 0)
        total += segment.get("hotel_cost", 0) * segment.get("nights", 0)
        total += sum(a.get("cost", 0) for a in segment.get("activities", []))
    return total

def validate_constraints(plan: Dict, constraints: TripConstraints) -> List[str]:
    """Check plan against hard constraints (deterministic)."""
    issues = []

    # Budget check
    total_cost = calculate_trip_cost(plan)
    if total_cost > constraints.budget:
        issues.append(f"Over budget: ${total_cost:.0f} > ${constraints.budget:.0f}")

    # Date check
    plan_days = sum(s.get("nights", 0) for s in plan.get("segments", []))
    available_days = (constraints.end_date - constraints.start_date).days
    if plan_days > available_days:
        issues.append(f"Too many days: {plan_days} > {available_days}")

    # Must-visit check
    visited = {s.get("city") for s in plan.get("segments", [])}
    missing = set(constraints.must_visit) - visited
    if missing:
        issues.append(f"Missing required cities: {missing}")

    return issues

def optimize_flights(
    cities: List[str],
    available_flights: List[FlightOption]
) -> List[FlightOption]:
    """Select cheapest valid flight sequence (deterministic optimization)."""
    # Simple greedy selection - could use more sophisticated optimization
    selected = []
    for i in range(len(cities) - 1):
        origin, dest = cities[i], cities[i + 1]
        options = [f for f in available_flights
                   if f.origin == origin and f.destination == dest]
        if options:
            selected.append(min(options, key=lambda f: f.price))
    return selected

# =============================================================================
# MAIN WORKFLOW - With critic and repair (L2/L3 features)
# =============================================================================

def plan_trip(user_request: str, max_repair_attempts: int = 3) -> Dict:
    """
    Main trip planning workflow with critique and repair loop.

    This demonstrates William's "behavior distillation approach 3":
    - Generate initial plan
    - Critique for issues
    - Repair if needed
    """
    # Step 1: Understand request (ptool)
    constraints = understand_request(user_request)

    # Step 2: Suggest city order (ptool - reasoning about geography/preferences)
    city_order = suggest_itinerary_order(constraints.must_visit, constraints)

    # Step 3: Build initial plan
    plan = {"segments": [], "constraints": constraints}

    days_per_city = (constraints.end_date - constraints.start_date).days // len(city_order)

    for city in city_order:
        # Select activities (ptool - needs preference reasoning)
        budget_remaining = constraints.budget - calculate_trip_cost(plan)
        activities = select_activities(
            city, days_per_city, constraints.preferences, budget_remaining
        )

        plan["segments"].append({
            "city": city,
            "nights": days_per_city,
            "activities": activities
        })

    # Step 4: Validate with Python (hard constraints)
    python_issues = validate_constraints(plan, constraints)

    # Step 5: Critique with LLM (soft constraints, preferences)
    critique = critique_plan(plan, constraints)
    all_issues = python_issues + critique.get("issues", [])

    # Step 6: Repair loop
    for attempt in range(max_repair_attempts):
        if not all_issues:
            break

        print(f"Repair attempt {attempt + 1}: {len(all_issues)} issues")

        # Repair (ptool - needs creative problem solving)
        plan = repair_plan(plan, all_issues, constraints)

        # Re-validate
        python_issues = validate_constraints(plan, constraints)
        critique = critique_plan(plan, constraints)
        all_issues = python_issues + critique.get("issues", [])

    return {
        "plan": plan,
        "total_cost": calculate_trip_cost(plan),
        "valid": len(all_issues) == 0,
        "remaining_issues": all_issues
    }
```

### Research Value

1. **Critique + Repair Loop**: Demonstrates approach 3 from William's plan
2. **Python for constraints**: Hard constraints checked deterministically
3. **LLM for soft reasoning**: Preferences, geography, activity selection

---

## Task 4: DABStep (Data Analysis)

**Why it fits**: Data analysis has clear structure but needs LLM for interpretation.

```python
# dabstep_agent.py
"""
DABStep: Data analysis benchmark.
Demonstrates: Python for data ops, ptools for interpretation, trace collection.
"""

import pandas as pd
from typing import Dict, List, Any, Literal
from ptool_framework import ptool, enable_tracing

# =============================================================================
# PTOOLS - Analysis interpretation
# =============================================================================

@ptool()
def understand_analysis_request(
    request: str,
    available_columns: List[str]
) -> Dict[str, Any]:
    """Parse a data analysis request.

    Return:
    {
        "operation": "aggregate" | "filter" | "correlate" | "compare" | "trend",
        "columns": [columns to use],
        "groupby": column to group by (optional),
        "filters": [{"column": ..., "op": ..., "value": ...}],
        "aggregations": ["sum", "mean", "count", etc.]
    }
    """
    ...

@ptool()
def interpret_results(
    request: str,
    results: Dict[str, Any],
    context: str
) -> str:
    """Interpret analysis results in natural language.

    Provide insights, highlight interesting findings, suggest follow-ups.
    """
    ...

@ptool()
def suggest_visualizations(
    data_summary: Dict,
    analysis_type: str
) -> List[Dict]:
    """Suggest appropriate visualizations.

    Return list of {type, x, y, title, description}.
    """
    ...

# =============================================================================
# PYTHON - Data operations (deterministic, fast)
# =============================================================================

def execute_analysis(df: pd.DataFrame, parsed_request: Dict) -> Dict:
    """Execute parsed analysis request on dataframe."""
    result_df = df.copy()

    # Apply filters
    for f in parsed_request.get("filters", []):
        col, op, val = f["column"], f["op"], f["value"]
        if op == "==":
            result_df = result_df[result_df[col] == val]
        elif op == ">":
            result_df = result_df[result_df[col] > val]
        elif op == "<":
            result_df = result_df[result_df[col] < val]
        elif op == "contains":
            result_df = result_df[result_df[col].str.contains(val, na=False)]

    # Apply groupby and aggregations
    if parsed_request.get("groupby"):
        grouped = result_df.groupby(parsed_request["groupby"])
        agg_results = {}
        for agg in parsed_request.get("aggregations", ["count"]):
            for col in parsed_request.get("columns", []):
                if col != parsed_request["groupby"]:
                    agg_results[f"{col}_{agg}"] = getattr(grouped[col], agg)()
        return {"type": "grouped", "data": agg_results}

    # Simple aggregation
    if parsed_request.get("aggregations"):
        results = {}
        for agg in parsed_request["aggregations"]:
            for col in parsed_request.get("columns", []):
                results[f"{col}_{agg}"] = getattr(result_df[col], agg)()
        return {"type": "aggregated", "data": results}

    return {"type": "filtered", "data": result_df.to_dict(), "shape": result_df.shape}

def compute_statistics(df: pd.DataFrame, columns: List[str]) -> Dict:
    """Compute basic statistics (pure Python/pandas)."""
    stats = {}
    for col in columns:
        if df[col].dtype in ['int64', 'float64']:
            stats[col] = {
                "mean": df[col].mean(),
                "std": df[col].std(),
                "min": df[col].min(),
                "max": df[col].max(),
                "median": df[col].median()
            }
        else:
            stats[col] = {
                "unique": df[col].nunique(),
                "top": df[col].mode().iloc[0] if len(df[col].mode()) > 0 else None,
                "freq": df[col].value_counts().iloc[0] if len(df) > 0 else 0
            }
    return stats

# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def analyze_data(df: pd.DataFrame, request: str) -> Dict:
    """
    Main data analysis workflow.

    Python handles data operations, LLM handles understanding and interpretation.
    """
    enable_tracing(True)  # Collect traces for distillation

    # Step 1: Parse request (ptool)
    parsed = understand_analysis_request(request, df.columns.tolist())

    # Step 2: Execute analysis (pure Python - fast, deterministic)
    results = execute_analysis(df, parsed)

    # Step 3: Compute additional stats (pure Python)
    stats = compute_statistics(df, parsed.get("columns", []))

    # Step 4: Interpret results (ptool)
    interpretation = interpret_results(
        request=request,
        results=results,
        context=f"Dataset has {len(df)} rows, columns: {df.columns.tolist()}"
    )

    # Step 5: Suggest visualizations (ptool)
    viz_suggestions = suggest_visualizations(
        data_summary=stats,
        analysis_type=parsed["operation"]
    )

    return {
        "parsed_request": parsed,
        "results": results,
        "statistics": stats,
        "interpretation": interpretation,
        "visualizations": viz_suggestions
    }
```

---

## Task 5: Verbalized Algorithms (Sort, Cluster)

**Why it fits**: Perfect for studying trace generalization - algorithms have clear structure.

```python
# verbalized_algorithms.py
"""
Verbalized Algorithms: LLM executes algorithms step-by-step.
Demonstrates: Trace-based learning, ptool ICL, algorithm distillation.
"""

from typing import List, Tuple, Any
from ptool_framework import ptool, WorkflowTrace, TraceStep

# =============================================================================
# PTOOLS - Algorithm steps as reasoning
# =============================================================================

@ptool()
def compare_elements(a: Any, b: Any, criterion: str) -> Literal["less", "equal", "greater"]:
    """Compare two elements according to a criterion.

    Criterion examples: "alphabetically", "by length", "by numeric value",
                       "by date", "by importance"

    Return "less" if a < b, "equal" if a == b, "greater" if a > b.
    """
    ...

@ptool()
def find_pivot(elements: List[Any], strategy: str) -> int:
    """Select a pivot element for partitioning.

    Strategy: "first", "last", "median", "random"
    Return the index of the chosen pivot.
    """
    ...

@ptool()
def assign_cluster(
    element: Any,
    centroids: List[Any],
    distance_metric: str
) -> int:
    """Assign an element to the nearest cluster.

    Distance metrics: "semantic", "numeric", "categorical"
    Return the index of the nearest centroid.
    """
    ...

@ptool()
def compute_centroid(
    elements: List[Any],
    element_type: str
) -> Any:
    """Compute the centroid of a cluster.

    For text: most representative/central concept
    For numbers: mean
    For categories: mode
    """
    ...

# =============================================================================
# VERBALIZED QUICKSORT
# =============================================================================

def verbalized_quicksort(
    elements: List[Any],
    criterion: str = "alphabetically"
) -> Tuple[List[Any], List[TraceStep]]:
    """
    Quicksort with LLM-based comparisons.

    This produces a trace that can be:
    1. Used as ICL demo for trace builders
    2. Analyzed for distillation
    3. Converted to audits/unit tests
    """
    trace = []

    def partition(arr: List[Any], low: int, high: int) -> int:
        # Select pivot (could be ptool for complex criteria)
        pivot_idx = high
        pivot = arr[pivot_idx]
        trace.append(TraceStep(
            ptool_name="select_pivot",
            args={"elements": arr[low:high+1], "strategy": "last"},
            expected_output=pivot_idx - low
        ))

        i = low - 1
        for j in range(low, high):
            # Compare using ptool
            comparison = compare_elements(arr[j], pivot, criterion)
            trace.append(TraceStep(
                ptool_name="compare_elements",
                args={"a": arr[j], "b": pivot, "criterion": criterion},
                expected_output=comparison
            ))

            if comparison == "less":
                i += 1
                arr[i], arr[j] = arr[j], arr[i]

        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        return i + 1

    def quicksort_recursive(arr: List[Any], low: int, high: int):
        if low < high:
            pi = partition(arr, low, high)
            quicksort_recursive(arr, low, pi - 1)
            quicksort_recursive(arr, pi + 1, high)

    result = elements.copy()
    quicksort_recursive(result, 0, len(result) - 1)

    return result, trace

# =============================================================================
# VERBALIZED K-MEANS
# =============================================================================

def verbalized_kmeans(
    elements: List[Any],
    k: int,
    distance_metric: str = "semantic",
    max_iterations: int = 10
) -> Tuple[List[int], List[Any], List[TraceStep]]:
    """
    K-means clustering with LLM-based distance/centroid computation.
    """
    trace = []

    # Initialize centroids (first k elements)
    centroids = elements[:k]
    assignments = [0] * len(elements)

    for iteration in range(max_iterations):
        # Assignment step
        new_assignments = []
        for i, elem in enumerate(elements):
            cluster_idx = assign_cluster(elem, centroids, distance_metric)
            trace.append(TraceStep(
                ptool_name="assign_cluster",
                args={"element": elem, "centroids": centroids, "metric": distance_metric},
                expected_output=cluster_idx
            ))
            new_assignments.append(cluster_idx)

        # Check convergence
        if new_assignments == assignments:
            break
        assignments = new_assignments

        # Update step
        new_centroids = []
        for cluster_idx in range(k):
            cluster_elements = [e for i, e in enumerate(elements) if assignments[i] == cluster_idx]
            if cluster_elements:
                centroid = compute_centroid(cluster_elements, "text")
                trace.append(TraceStep(
                    ptool_name="compute_centroid",
                    args={"elements": cluster_elements, "element_type": "text"},
                    expected_output=centroid
                ))
                new_centroids.append(centroid)
            else:
                new_centroids.append(centroids[cluster_idx])
        centroids = new_centroids

    return assignments, centroids, trace

# =============================================================================
# TRACE → AUDIT GENERATION (from SSRM paper)
# =============================================================================

@ptool()
def generate_algorithm_audit(
    algorithm_name: str,
    trace: List[TraceStep],
    input_elements: List[Any]
) -> List[Dict]:
    """Generate reasoning audits from an algorithm trace.

    Audits are unit tests that describe properties of algorithmic behavior.
    Examples:
    - "After sorting, element X should come before element Y"
    - "Elements in cluster 1 should all be more similar to centroid 1 than centroid 2"

    Return list of audit checks that should pass.
    """
    ...
```

---

## Summary: Framework Coverage of William's Ideas

| William's Concept | Implementation | Example Task |
|-------------------|----------------|--------------|
| ptools as LLM prompts | `@ptool` decorator | All tasks |
| Python workflows calling ptools | Main workflow functions | All tasks |
| Trace builders | `build_reasoning_trace` ptool | HotPotQA |
| Behavior distillation approach 1 | `BehaviorDistiller` | MedCalc extraction |
| Behavior distillation approach 2 | ReAct → trace → distill | HotPotQA adaptive |
| Behavior distillation approach 3 | Critique + repair loop | TravelPlanner |
| Fallback to agentic | `@distilled` + `DistillationFallback` | All tasks |
| ICL from traces | Trace store → prompt examples | Verbalized algorithms |
| Audits from traces | `generate_algorithm_audit` | Verbalized algorithms |
| L0 fixed workflow | `hotpotqa_fixed_workflow` | HotPotQA |
| L1 LLM as router | `identify_calculator` | MedCalc |
| L2/L3 adaptive | `hotpotqa_adaptive_workflow` | HotPotQA |

---

## Next Steps for Research

1. **Implement one task end-to-end** (suggest: MedCalc or HotPotQA)
2. **Collect traces** on benchmark dataset
3. **Measure distillation success rate** (how much can become Python?)
4. **Compare L0/L1/L2** performance and reliability
5. **Generate audits** from traces and test generalization

---

## Integration with William's 2026 Agent Research Vision

### Key Research Questions from William's Plan

| Question | Framework Support | Status |
|----------|-------------------|--------|
| "How to trade off generality vs predictability?" | L0-L5 spectrum implemented | ✓ Ready |
| "Can we distill agent behavior to Python?" | `BehaviorDistiller` + `@distilled` | ✓ Ready |
| "How to validate agent reasoning?" | SSRM audits (structured + typicality) | ✓ Ready |
| "How to make agents self-improving?" | `SelfImprovingAgent` with memory | ✓ Ready |
| "How to select the right model?" | `ModelSelector` with experience tracking | ✓ Ready (not integrated) |

### William's Three Distillation Approaches

| Approach | Description | Implementation | Status |
|----------|-------------|----------------|--------|
| **Approach 1** | Trace patterns → Python rules | `BehaviorDistiller.analyze_trace_patterns()` | ✓ Implemented |
| **Approach 2** | ReAct → WorkflowTrace → distill | `ReActTrajectory.to_workflow_trace()` | ✓ Implemented |
| **Approach 3** | Generate + Critique + Repair loop | `TraceCritic` + `RepairAgent` | ✓ Implemented |

### The Autonomy Spectrum (L0-L5)

```
L0: Fixed Python workflow calling LLMs          ← Mode 1 (@ptool)
L1: LLMs as routers (modify workflow)           ← Mode 2 (ProgramGenerator)
L2: State graphs (LangGraph-style)              ← Not implemented
L3: ReAct agents (thought-action loops)         ← Mode 3 (ReActAgent)
L4: Multi-agent systems                         ← AgentOrchestrator
L5: Self-modifying systems                      ← SelfImprovingAgent
```

### The Core Innovation: "Python Calling LLMs"

Traditional agents: LLM decides what tool to call → unpredictable
ptool_framework: Python decides what ptool to call → predictable, testable

```python
# Traditional (unpredictable)
agent.run("Calculate BMI")  # LLM decides tools internally

# ptool_framework (predictable)
def calculate_bmi(note: str) -> Dict:
    values = extract_values(note)      # ptool - LLM extracts
    bmi = values["weight"] / values["height"]**2  # Python - deterministic
    risk = interpret_bmi(bmi)          # ptool - LLM interprets
    return {"bmi": bmi, "risk": risk}
```

### Research Milestones

| Milestone | Description | Status |
|-----------|-------------|--------|
| Core ptools | @ptool decorator working | ✓ Done |
| Trace collection | All executions logged | ✓ Done |
| ReAct agent | Thought-action loops | ✓ Done |
| Distillation | @distilled with fallback | ✓ Done |
| Audits | SSRM-style validation | ✓ Done |
| Critic/Repair | Reflexion-style loop | ✓ Done |
| Multi-agent | Orchestration | ✓ Done |
| Self-improving | Pattern learning | ✓ Done |
| **Auto-ICL** | Traces → ICL examples | ⏳ Gap 1 |
| **Model selection** | Auto model choice | ⏳ Gap 2 |

---

## Recommended Research Experiments

### Experiment 1: Distillation Effectiveness
```python
# 1. Run ReAct on 100 MedCalc problems
# 2. Collect all traces
# 3. Attempt distillation for each ptool
# 4. Measure: Python coverage %, accuracy vs LLM, speedup

from benchmark.experiments.l2_distilled import run_distillation_experiment
results = run_distillation_experiment(dataset="medcalc", n_samples=100)
```

### Experiment 2: Audit Correlation
```python
# 1. Run agent on dataset
# 2. Compute audit scores (structured + typicality)
# 3. Correlate with correctness
# 4. Measure: Can audits predict failures?

from ptool_framework.audit import AuditRunner
runner = AuditRunner()
correlation = runner.compute_audit_outcome_correlation(traces)
```

### Experiment 3: Self-Improvement Over Time
```python
# 1. Run SelfImprovingAgent on 50 problems
# 2. Track pattern extraction
# 3. Measure accuracy improvement over time
# 4. Analyze: Which patterns help most?

from benchmark.experiments.l5_improving import run_improvement_experiment
results = run_improvement_experiment(n_problems=50, track_patterns=True)
```

### Experiment 4: Model Selection Value
```python
# 1. Run with fixed model (expensive)
# 2. Run with ModelSelector (adaptive)
# 3. Compare: accuracy, cost, latency
# 4. Measure: Cost savings with auto-selection

from ptool_framework.model_selector import ModelSelector
selector = ModelSelector()
# Compare model="gpt-4" vs model="auto"
```
