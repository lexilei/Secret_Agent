# Demo Pipeline: Step-by-Step Walkthrough

This folder contains a complete walkthrough of the ptool_framework pipeline.

## Files

### Step 1: Template-Based Generation
**File:** `step1_template_generated.py`

Generated using the `analyzer` template. Shows the basic structure:
- 3 @ptool functions (extract, analyze, summarize)
- 1 main workflow function
- CLI entry point

**Command used:**
```python
from ptool_framework.program_generator import generate_from_template
generate_from_template('analyzer', 'Analyze restaurant reviews', 'step1_template_generated.py')
```

---

### Step 2: Original Example Program
**File:** `step2_original_program.py`

Copy of `example_program.py` - a meal analyzer with:
- `extract_food_items(text) -> List[str]` - @ptool
- `is_healthy(food) -> bool` - @ptool
- `suggest_alternative(food) -> str` - @ptool
- `analyze_meal(description) -> dict` - pure Python workflow

---

### Step 3-4: Trace Collection
**File:** `step4_collected_traces.json`

After running the program 3 times, we collected 17 traces:
- `extract_food_items`: 3 traces
- `is_healthy`: 9 traces
- `suggest_alternative`: 3 traces (only called for unhealthy foods)

**Sample trace:**
```json
{
  "ptool_name": "is_healthy",
  "inputs": {"food": "pizza"},
  "output": false,
  "success": true,
  "execution_time_ms": 705.5,
  "model_used": "deepseek-v3"
}
```

---

### Step 5: Refactoring Analysis
**File:** `step5_refactoring_analysis.txt`

AST analysis of the program showing:
- Function types (ptool vs pure Python)
- Line numbers
- Parameter types
- Return types
- Trace counts per function

---

### Step 6: Distilled Version
**File:** `step6_distilled_version.py`

**This is the key output!** Shows the program after distillation:

1. **Original @ptool functions** - kept for fallback
2. **@distilled versions** - try Python first:
   - `extract_food_items_distilled`: Regex matching against known foods
   - `is_healthy_distilled`: Lookup table of healthy/unhealthy foods
   - `suggest_alternative_distilled`: Lookup table of substitutions

3. **Fallback behavior**:
   - If Python version can handle it → fast, no LLM call
   - If Python version fails → raises `DistillationFallback` → calls original @ptool

**Performance Results (from running step6):**
```
extract_food_items_distilled:
  Python success: 75%
  LLM fallbacks: 25%

is_healthy_distilled:
  Python success: 100%
  LLM fallbacks: 0%

suggest_alternative_distilled:
  Python success: 100%
  LLM fallbacks: 0%
```

---

## Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│  Step 1-2: Start with @ptool program                            │
│  (All reasoning done by LLM)                                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 3-4: Run program, collect traces                          │
│  (17 traces captured to ~/.ptool_traces/)                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 5: Analyze traces for patterns                            │
│  (Find common inputs/outputs that Python can handle)            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 6: Distilled version                                      │
│  (Python handles 85%+ of cases, LLM only for edge cases)        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Insights

1. **Incremental Optimization**: Start fully agentic, become deterministic over time
2. **Safe Fallback**: @distilled always has LLM backup - never fails silently
3. **Trace-Driven**: Patterns come from real execution data, not guesses
4. **Measurable**: Can track exactly what % of cases Python handles

---

## How to Run

```bash
# Run original (all LLM)
python step2_original_program.py

# Run distilled (Python first, LLM fallback)
python step6_distilled_version.py
```

---

## Next Steps

With more traces (50+), the BehaviorDistiller can:
1. Automatically identify more patterns
2. Generate Python code via LLM
3. Validate against held-out traces
4. Produce optimized @distilled functions

Run:
```bash
ptool refactor step2_original_program.py --mode distill -o step7_auto_distilled.py
```
