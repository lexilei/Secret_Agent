# StrategyQA Experiment Levels

This document explains how L0-L3 experiments are built for StrategyQA benchmark.

## Overview

Based on William Cohen's research plan, experiments are organized by the level of control:
- **Lower levels (L0-L1)**: Fixed workflows, more predictable
- **Higher levels (L2-L3)**: Dynamic workflows, more flexible

The distillation path goes: **L3 → L2 → L1 → L0** (trading flexibility for predictability).

```
┌─────────────────────────────────────────────────────────────────────┐
│                     StrategyQA Experiment Levels                     │
├─────────────────────────────────────────────────────────────────────┤
│  L0: Baseline          │ Fixed prompt → LLM → Answer                │
│  L1: Structured        │ Fixed prompt + structure → LLM → Answer    │
│  L2: Trace Builder     │ Python controls flow, LLM executes ptools  │
│  L3: ReAct Agent       │ LLM controls flow, chooses actions         │
└─────────────────────────────────────────────────────────────────────┘
```

---

## L0: Baseline (`levels.py:26-70`)

**Concept**: Direct yes/no prompt with no structure.

**How it works**:
```
Question → [LLM: Answer Yes/No] → Answer
```

**Implementation**:
```python
class L0_Baseline(StrategyQAExperiment):
    PROMPT = """Answer the following yes/no question.
    Question: {question}
    Answer with only "Yes" or "No":"""

    def run_instance(self, instance):
        response = call_llm(prompt=self.PROMPT.format(...))
        return self.extract_boolean(response)
```

**Characteristics**:
- Single LLM call
- No reasoning structure
- Fastest but least accurate
- Cost: ~$0.0001/question

**Run**:
```bash
python -m benchmark.experiments.strategyqa.run_experiments -e L0 -n 10
```

---

## L1: Structured Prompting (`levels.py:77-308`)

**Concept**: Add structure to prompt to guide LLM reasoning.

### L1-CoT: Chain of Thought

**How it works**:
```
Question → [LLM: Think step by step] → Reasoning → [LLM: Answer based on reasoning] → Answer
```

**Implementation** (`levels.py:77-138`):
```python
class L1_CoT(StrategyQAExperiment):
    PROMPT = """Answer the following yes/no question by thinking step by step.
    Question: {question}
    Let's think step by step:
    1. First, I need to understand what the question is asking.
    2. Then, I'll consider relevant facts.
    3. Finally, I'll determine the answer.
    Reasoning:"""

    def run_instance(self, instance):
        # Step 1: Get reasoning
        reasoning = call_llm(prompt=self.PROMPT.format(...))
        # Step 2: Get final answer based on reasoning
        answer = call_llm(prompt=..., reasoning)
        return self.extract_boolean(answer)
```

**Characteristics**:
- 2 LLM calls
- Explicit reasoning chain
- Better accuracy than L0

### L1-CoC: Chain of Code

**How it works**:
```
Question → [LLM: Write pseudocode logic] → Execute mentally → Answer
```

**Implementation** (`levels.py:141-210`):
```python
class L1_CoC(StrategyQAExperiment):
    PROMPT = """Answer the following yes/no question by writing out the logical steps as pseudocode.
    Question: {question}
    Write pseudocode that would determine the answer:
    ```
    # Define the problem
    question = "{question}"
    # Step 1: Identify key facts needed
    facts_needed = [...]
    # Step 2: Look up or reason about each fact
    facts = {...}
    # Step 3: Apply logic to determine answer
    answer = determine_answer(facts)
    ```
    Final Answer (Yes or No):"""
```

**Characteristics**:
- Single LLM call with code structure
- Combines reasoning with pseudo-execution
- Good for logical/computational questions

### L1-PTP: Program Trace Prompting

**How it works** (from [doctest-prompting](https://github.com/wwcohen/doctest-prompting)):
```
Question → [LLM: Complete Python doctest trace] → Extract True/False from trace
```

**Implementation** (`levels.py:213-308`):
```python
class L1_PTP(StrategyQAExperiment):
    # Uses Python doctest format with traced function execution
    PROMPT = '''def answer_yes_no(question: str) -> bool:
    """Answer a yes/no question by identifying key facts and reasoning.

    >>> answer_yes_no("Are more people related to Genghis Khan than Julius Caesar?")
    # Step 1: Identify what we need to know
    key_facts_needed = ["descendants of Genghis Khan", "descendants of Julius Caesar"]
    # Step 2: Recall relevant facts
    fact_1 = "Genghis Khan had many children, ~16 million descendants"
    fact_2 = "Julius Caesar had few children, lineage died out"
    # Step 3: Compare and reason
    comparison = "16 million >> few"
    # Step 4: Determine answer
    answer = True  # More people are related to Genghis Khan
    True

    >>> answer_yes_no("{question}")
    # Step 1: Identify what we need to know
    key_facts_needed = '''
```

**Characteristics**:
- Reasoning as code trace (doctest format)
- Steps are observable and auditable
- ICL-ready (can use as few-shot examples)

**Run L1 variants**:
```bash
python -m benchmark.experiments.strategyqa.run_experiments -e L1-cot -n 10
python -m benchmark.experiments.strategyqa.run_experiments -e L1-coc -n 10
python -m benchmark.experiments.strategyqa.run_experiments -e L1-ptp -n 10
```

---

## L2: Trace Builder (`trace_builder.py`)

**Concept**: Python controls the workflow, LLM executes individual "ptools" (thinking tools).

**How it works**:
```
Question
    → [ptool: decompose] → Sub-questions
    → [ptool: answer_factual] × N → Facts
    → [ptool: retrieve] → Retrieved paragraphs (RAG)
    → [ptool: combine_facts] → Final Answer
```

**Key Data Structures** (`trace_builder.py:32-89`):
```python
@dataclass
class TraceStep:
    ptool: str           # "decompose", "answer_factual", "retrieve", etc.
    goal: str            # What this step aims to achieve
    input: Dict          # Input parameters
    output: Any          # Output from the ptool
    success: bool

@dataclass
class WorkflowTrace:
    question: str
    steps: List[TraceStep]
    final_answer: Optional[bool]

    def to_icl_demo(self) -> str:  # Format as ICL demonstration
    def get_audit_info(self) -> Dict:  # For auditing
```

**Available ptools** (`trace_builder.py:96-243`):

| ptool | Input | Output | Purpose |
|-------|-------|--------|---------|
| `decompose` | question | List[{type, question}] | Break into sub-questions |
| `answer_factual` | question, context? | str | Answer factual question |
| `retrieve` | query, top_k | List[{title, content}] | RAG from paragraphs |
| `evaluate_condition` | condition, facts | bool | Evaluate True/False |
| `combine_facts` | question, facts | bool | Final yes/no answer |

**Workflow** (`trace_builder.py:268-338`):
```python
class StrategyQATraceBuilder:
    def build_trace(self, question: str) -> WorkflowTrace:
        # Step 1: Decompose
        sub_questions = self.executor.execute("decompose", ...)

        # Step 2: Answer each sub-question
        for sub_q in sub_questions:
            if sub_q.type == "factual":
                # Optional: retrieve relevant paragraphs first
                retrieved = self.executor.execute("retrieve", query=sub_q.question)
                answer = self.executor.execute("answer_factual",
                    question=sub_q.question,
                    context=retrieved
                )
            else:
                # comparison/evaluation
                answer = self.executor.execute("evaluate_condition", ...)

        # Step 3: Combine to get final answer
        final = self.executor.execute("combine_facts", ...)
        return trace
```

**RAG Integration** (`trace_builder.py:244-280`):

The `retrieve` ptool uses TF-IDF similarity to find relevant paragraphs from `strategyqa_train_paragraphs.json` (9,251 paragraphs):

```python
def _retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
    """Retrieve relevant paragraphs using TF-IDF similarity."""
    # Returns: [{"title": "...", "content": "...", "score": 0.85}, ...]
```

**Characteristics**:
- **Python controls flow**: Workflow is deterministic (decompose → answer → combine)
- **LLM executes ptools**: Each ptool is a "thinking" operation
- **Traces are auditable**: Can check each step
- **Traces are ICL-ready**: Can format as demonstrations
- **RAG-enabled**: Can retrieve facts from knowledge base

**Run**:
```bash
python -m benchmark.experiments.strategyqa.run_experiments -e L2 -n 10
# or with RAG enabled
python -m benchmark.experiments.strategyqa.run_experiments -e L2-rag -n 10
```

---

## L3: ReAct Agent (`l3_react.py`)

**Concept**: LLM decides what action to take at each step (Think → Act → Observe loop).

**How it works**:
```
Question
    → [LLM: What should I do?] → Thought + Action
    → [Execute Action] → Observation
    → [LLM: Based on observation, what next?] → Thought + Action
    → ... repeat until finish(Yes/No)
```

**Key Difference from L2**:

| Aspect | L2 (Trace Builder) | L3 (ReAct) |
|--------|-------------------|------------|
| Control | Python decides flow | LLM decides flow |
| Workflow | Fixed: decompose→answer→combine | Dynamic: LLM chooses |
| Steps | Predictable (3-5 steps) | Variable (up to max_steps) |
| Auditability | High (fixed structure) | Medium (flexible) |

**Available Actions** (`l3_react.py:46-72`):
```python
SYSTEM_PROMPT = """You are a reasoning agent...
Available actions:
1. decompose(question) - Break a complex question into simpler sub-questions
2. lookup(question) - Look up a factual piece of information
3. evaluate(condition) - Evaluate if a condition is True or False
4. finish(Yes/No) - Give your final answer when confident

Format:
Thought: <reasoning about what to do next>
Action: <action_name>(<argument>)
"""
```

**Implementation** (`l3_react.py:88-198`):
```python
class L3_ReAct(StrategyQAExperiment):
    def run_instance(self, instance):
        history = []
        context = {}  # Accumulated facts

        for step in range(self.max_steps):
            # Build prompt with history
            prompt = SYSTEM_PROMPT + STEP_PROMPT.format(
                question=instance.question,
                history=self._format_history(history)
            )

            # Get LLM decision
            response = call_llm(prompt)
            thought, action_name, action_arg = self._parse_response(response)

            # Check for finish
            if action_name == "finish":
                return action_arg  # "Yes" or "No"

            # Execute action
            observation = self._execute_action(action_name, action_arg, context)

            # Add to history for next iteration
            history.append({
                "thought": thought,
                "action": f"{action_name}({action_arg})",
                "observation": observation
            })
```

**Simplified Variant** (`l3_react.py:316-407`):
```python
class L3_ReActSimple(StrategyQAExperiment):
    """Single-turn: just numbered thoughts then answer."""
    PROMPT = """Answer this yes/no question using step-by-step reasoning.
    Question: {question}

    Use this format:
    Thought 1: <first reasoning step>
    Thought 2: <second reasoning step>
    ...
    Answer: Yes or No
    """
```

**Characteristics**:
- **LLM controls flow**: Agent decides what to do
- **Dynamic reasoning**: Can adapt to problem
- **Context accumulation**: Facts build up across steps
- **Max steps limit**: Prevents infinite loops
- **Most flexible but least predictable**

**Run**:
```bash
python -m benchmark.experiments.strategyqa.run_experiments -e L3 -n 10
python -m benchmark.experiments.strategyqa.run_experiments -e L3-simple -n 10
```

---

## Experiment Registry (`levels.py:353-372`)

```python
EXPERIMENTS = {
    # L0 - Baseline
    "L0": L0_Baseline,
    "L0-baseline": L0_Baseline,

    # L1 - Structured Prompts
    "L1-cot": L1_CoT,
    "L1-coc": L1_CoC,
    "L1-ptp": L1_PTP,

    # L2 - Trace Builder (Python controls flow)
    "L2": L2_TraceBuilder,
    "L2-trace": L2_TraceBuilder,
    "L2-rag": L2_TraceBuilderRAG,  # With retrieval

    # L3 - ReAct Agent (LLM controls flow)
    "L3": L3_ReAct,
    "L3-react": L3_ReAct,
    "L3-simple": L3_ReActSimple,
}
```

---

---

## L4: Adaptive / Pipeline (`l4_adaptive.py`)

**Two variants with different approaches to orchestration.**

### L4-Adaptive: Complexity-Based Routing

**Concept**: Route questions to different levels based on estimated complexity.

```
Question → [Estimate Complexity] → Route to L1/L2/L3
```

**Complexity levels**:
- **SIMPLE**: Single fact lookup → Route to L1-CoT (cheap)
- **MEDIUM**: 2-3 facts, some reasoning → Route to L2 (balanced)
- **HARD**: Multi-hop, complex logic → Route to L3 (thorough)

**Implementation** (`l4_adaptive.py:178-331`):
```python
class ComplexityRouter:
    def _estimate_complexity_rules(self, question: str) -> Complexity:
        # Heuristics: word count, complexity indicators, question marks
        score = 0
        if word_count > 20: score += 2
        score += complex_indicator_count
        score += multi_hop_indicator_count * 2

        if score >= 4: return Complexity.HARD
        elif score >= 2: return Complexity.MEDIUM
        else: return Complexity.SIMPLE

class L4_Adaptive(StrategyQAExperiment):
    def run_instance(self, instance):
        decision = self.router.route(instance.question)
        experiment = self._get_experiment(decision.target_level)
        return experiment.run_instance(instance)
```

**Research question**: Can automatic level selection achieve L3 accuracy at L1/L2 cost?

### L4-Pipeline: Python-Controlled Stages

**Concept**: Python orchestrates workflow, LLM handles understanding.

```
Stage 1: [LLM] Analyze question structure
Stage 2: [LLM] Decompose into sub-questions
Stage 3: [LLM/RAG] Answer each sub-question
Stage 4: [Python] Combine answers logically
```

**Implementation** (`l4_adaptive.py:340-480`):
```python
class L4_Pipeline(StrategyQAExperiment):
    def run_instance(self, instance):
        # Stage 1: Analyze
        analysis = self._stage1_analyze(instance.question)

        # Stage 2: Decompose
        sub_questions = self._stage2_decompose(instance.question, analysis)

        # Stage 3: Answer sub-questions (with optional RAG)
        sub_answers = [self._stage3_answer_sub(sq) for sq in sub_questions]

        # Stage 4: Combine (final LLM call for yes/no)
        predicted = self._stage4_combine(instance.question, sub_answers)
```

**Run**:
```bash
python -m benchmark.experiments.strategyqa.run_experiments -e L4 -n 10
python -m benchmark.experiments.strategyqa.run_experiments -e L4-adaptive -n 10
python -m benchmark.experiments.strategyqa.run_experiments -e L4-pipeline -n 10
```

---

## L5: Learning Agents (`l5_learning.py`)

**Concept**: Learn from execution experience to improve over time.

### L5-Improving: Online Learning

**How it works**:
```
Run N:
    1. Retrieve relevant patterns from memory
    2. Augment prompt with patterns
    3. Execute
    4. If correct: Extract positive pattern → Store
       If wrong: Extract negative pattern → Store
Run N+1:
    - Uses learned patterns from Run N
    - Performance improves over time
```

**Components** (`l5_learning.py:30-150`):
```python
@dataclass
class LearnedPattern:
    pattern_id: str
    pattern_type: str      # "positive" or "negative"
    question_type: str     # "comparison", "temporal", "causal", etc.
    content: str           # The pattern/example
    confidence: float
    use_count: int
    success_count: int

class PatternMemory:
    def store_pattern(self, pattern: LearnedPattern)
    def get_relevant_patterns(self, question, pattern_type, max_patterns) -> List[LearnedPattern]
    def update_pattern_stats(self, pattern_id, success: bool)
```

**Question type classification**:
- **comparison**: "more than", "less than", "greater"
- **temporal**: "before", "after", "when", "during"
- **causal**: "because", "cause", "result", "lead to"
- **possibility**: "can", "could", "able", "possible"
- **similarity**: "same", "similar", "different"
- **factual**: default

**Implementation** (`l5_learning.py:210-350`):
```python
class L5_Improving(StrategyQAExperiment):
    def run_instance(self, instance):
        # Build prompt with relevant patterns
        prompt = self._build_prompt_with_patterns(instance.question)

        # Execute
        response = call_llm(prompt=prompt, model=self.model)
        predicted = self.extract_boolean(response)

        # Learn from this execution
        if is_correct and self.learn_from_success:
            pattern = extract_pattern_from_trace(...)
            self.pattern_memory.store_pattern(pattern)
        elif not is_correct and self.learn_from_failure:
            pattern = extract_pattern_from_trace(...)
            self.pattern_memory.store_pattern(pattern)

        # Update pattern statistics
        for pid in self.patterns_used:
            self.pattern_memory.update_pattern_stats(pid, is_correct)
```

### L5-ICL: In-Context Learning from Training

**Concept**: Pre-populate pattern memory with examples from training data.

**How it works**:
```
Setup:
    1. Load training data
    2. Group by question type
    3. Create ICL patterns from good examples
    4. Store in pattern memory

Run:
    - Start with knowledge (vs L5-Improving which starts from scratch)
    - Continue learning from execution
```

**Implementation** (`l5_learning.py:380-450`):
```python
class L5_ICL(L5_Improving):
    def _load_icl_examples(self):
        # Group training by question type
        by_type = group_by_question_type(self.train_data)

        for qtype, instances in by_type.items():
            # Select instances with decomposition (better examples)
            good = [i for i in instances if i.decomposition][:self.icl_per_type]

            for inst in good:
                content = f"""Question type: {qtype}
Question: {inst.question}
Sub-questions: {inst.decomposition}
Key facts: {inst.facts}
Answer: {"Yes" if inst.answer else "No"}"""

                pattern = LearnedPattern(
                    pattern_type="positive",
                    question_type=qtype,
                    content=content,
                    confidence=1.0,  # High confidence for training examples
                )
                self.pattern_memory.store_pattern(pattern)
```

**Run**:
```bash
python -m benchmark.experiments.strategyqa.run_experiments -e L5 -n 10
python -m benchmark.experiments.strategyqa.run_experiments -e L5-improving -n 10
python -m benchmark.experiments.strategyqa.run_experiments -e L5-icl -n 10
```

---

## Experiment Registry (`levels.py`)

```python
EXPERIMENTS = {
    # L0 - Baseline
    "L0": L0_Baseline,

    # L1 - Structured Prompts
    "L1-cot": L1_CoT,
    "L1-coc": L1_CoC,
    "L1-ptp": L1_PTP,

    # L2 - Trace Builder (Python controls flow)
    "L2": L2_TraceBuilder,
    "L2-rag": L2_TraceBuilderRAG,

    # L3 - ReAct Agent (LLM controls flow)
    "L3": L3_ReAct,
    "L3-simple": L3_ReActSimple,

    # L4 - Adaptive Routing / Pipeline
    "L4": L4_Adaptive,
    "L4-adaptive": L4_Adaptive,
    "L4-pipeline": L4_Pipeline,

    # L5 - Learning Agents
    "L5": L5_Improving,
    "L5-improving": L5_Improving,
    "L5-icl": L5_ICL,
}
```

---

## Summary Table

| Level | Name | Control | Key Feature | Cost |
|-------|------|---------|-------------|------|
| L0 | Baseline | Fixed | Direct prompt | Lowest |
| L1 | Structured | Fixed | CoT/CoC/PTP prompts | Low |
| L2 | Trace Builder | Python | Decompose→Answer→Combine | Medium |
| L2-RAG | + Retrieval | Python | + TF-IDF paragraph retrieval | Medium |
| L3 | ReAct | LLM | Think→Act→Observe loop | High |
| L4-Adaptive | Routing | Router | Route by complexity | Variable |
| L4-Pipeline | Stages | Python | 4-stage pipeline | Medium |
| L5-Improving | Learning | LLM | Online pattern learning | High |
| L5-ICL | + Training | LLM | Pre-loaded ICL examples | High |

---

## Distillation Path

The key insight from William Cohen's plan: Start with flexible agents, progressively optimize to deterministic code.

```
L5 (Learning Agents)
    ↓ Extract successful patterns
L3 (ReAct Agent)
    ↓ Collect traces, find common patterns
L2 (Trace Builder)
    ↓ Generalize workflow templates
L1 (Structured Prompts)
    ↓ Extract rules, distill to code
L0 (Fixed Prompts / Pure Python)
```

Each level trades **flexibility** for **predictability** and **cost efficiency**.
