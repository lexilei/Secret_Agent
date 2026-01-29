"""
StrategyQA Experiments organized by level.

L0: Standard baseline - simple yes/no prompt
L1: Structured prompting variants (CoT, CoC, PTP)
L2: 2-step decomposition with ReAct control flow

All experiments use DeepSeek by default.
"""

import time
import re
from typing import Optional, Dict, Any, List

from .base import StrategyQAExperiment, StrategyQAResult
from ...dataset.strategyqa_loader import StrategyQAInstance
from ...metrics.cost import calculate_cost

from ptool_framework.llm_backend import call_llm


# ============================================================================
# L0: Standard Baseline
# ============================================================================

class L0_Baseline(StrategyQAExperiment):
    """
    L0: Standard baseline - direct yes/no prompt.

    No structure, no reasoning steps. Just ask for the answer.
    """

    PROMPT = """Answer the following yes/no question.

Question: {question}

Answer with only "Yes" or "No":"""

    def __init__(self, model: str = "deepseek-v3-0324"):
        super().__init__(model)

    def run_instance(self, instance: StrategyQAInstance) -> StrategyQAResult:
        prompt = self.PROMPT.format(question=instance.question)

        start_time = time.time()
        try:
            response = call_llm(prompt=prompt, model=self.model)
            latency_ms = (time.time() - start_time) * 1000

            predicted = self.extract_boolean(response)
            input_tokens = len(prompt) // 4
            output_tokens = len(response) // 4 if response else 0

            cost = calculate_cost(input_tokens, output_tokens, self.model)

            return StrategyQAResult(
                qid=instance.qid,
                question=instance.question,
                predicted_answer=predicted,
                ground_truth=instance.answer,
                is_correct=predicted == instance.answer if predicted is not None else False,
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                cost_usd=cost.cost_usd,
                raw_response=response,
            )
        except Exception as e:
            return self._error_result(instance, str(e), type(e).__name__, time.time() - start_time)


# ============================================================================
# L1: Structured Prompting Variants
# ============================================================================

class L1_CoT(StrategyQAExperiment):
    """
    L1-CoT: Chain-of-Thought prompting.

    Ask the model to think step by step before answering.
    """

    PROMPT = """Answer the following yes/no question by thinking step by step.

Question: {question}

Let's think step by step:
1. First, I need to understand what the question is asking.
2. Then, I'll consider relevant facts.
3. Finally, I'll determine the answer.

Reasoning:"""

    ANSWER_PROMPT = """Based on the reasoning above, the answer to "{question}" is (Yes or No):"""

    def __init__(self, model: str = "deepseek-v3-0324"):
        super().__init__(model)

    def run_instance(self, instance: StrategyQAInstance) -> StrategyQAResult:
        prompt = self.PROMPT.format(question=instance.question)

        start_time = time.time()
        total_input = 0
        total_output = 0

        try:
            # Step 1: Get reasoning
            reasoning = call_llm(prompt=prompt, model=self.model)
            total_input += len(prompt) // 4
            total_output += len(reasoning) // 4 if reasoning else 0

            # Step 2: Get final answer
            answer_prompt = prompt + reasoning + "\n\n" + self.ANSWER_PROMPT.format(question=instance.question)
            response = call_llm(prompt=answer_prompt, model=self.model)
            total_input += len(answer_prompt) // 4
            total_output += len(response) // 4 if response else 0

            latency_ms = (time.time() - start_time) * 1000
            predicted = self.extract_boolean(response)
            cost = calculate_cost(total_input, total_output, self.model)

            return StrategyQAResult(
                qid=instance.qid,
                question=instance.question,
                predicted_answer=predicted,
                ground_truth=instance.answer,
                is_correct=predicted == instance.answer if predicted is not None else False,
                latency_ms=latency_ms,
                input_tokens=total_input,
                output_tokens=total_output,
                total_tokens=total_input + total_output,
                cost_usd=cost.cost_usd,
                raw_response=f"Reasoning: {reasoning}\n\nAnswer: {response}",
                trace={"reasoning": reasoning, "answer": response},
            )
        except Exception as e:
            return self._error_result(instance, str(e), type(e).__name__, time.time() - start_time)


class L1_CoC(StrategyQAExperiment):
    """
    L1-CoC: Chain-of-Code prompting.

    Ask the model to write pseudocode/logic to solve the problem.
    """

    PROMPT = """Answer the following yes/no question by writing out the logical steps as pseudocode.

Question: {question}

Write pseudocode that would determine the answer:
```
# Define the problem
question = "{question}"

# Step 1: Identify key facts needed
facts_needed = [
    # List what facts we need to answer this
]

# Step 2: Look up or reason about each fact
facts = {{
    # fact_name: value
}}

# Step 3: Apply logic to determine answer
def determine_answer(facts):
    # Logic here
    pass

# Step 4: Return final answer
answer = determine_answer(facts)  # Should be True or False
```

Now execute this logic mentally and provide the final answer.

Final Answer (Yes or No):"""

    def __init__(self, model: str = "deepseek-v3-0324"):
        super().__init__(model)

    def run_instance(self, instance: StrategyQAInstance) -> StrategyQAResult:
        prompt = self.PROMPT.format(question=instance.question)

        start_time = time.time()
        try:
            response = call_llm(prompt=prompt, model=self.model)
            latency_ms = (time.time() - start_time) * 1000

            predicted = self.extract_boolean(response)
            input_tokens = len(prompt) // 4
            output_tokens = len(response) // 4 if response else 0
            cost = calculate_cost(input_tokens, output_tokens, self.model)

            return StrategyQAResult(
                qid=instance.qid,
                question=instance.question,
                predicted_answer=predicted,
                ground_truth=instance.answer,
                is_correct=predicted == instance.answer if predicted is not None else False,
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                cost_usd=cost.cost_usd,
                raw_response=response,
            )
        except Exception as e:
            return self._error_result(instance, str(e), type(e).__name__, time.time() - start_time)


class L1_PTP(StrategyQAExperiment):
    """
    L1-PTP: Program Trace Prompting (William Cohen).

    Based on: https://github.com/wwcohen/doctest-prompting
    Paper: "Watch Your Steps: Observable and Modular Chains of Thought"

    Uses Python doctest-style traces to demonstrate reasoning steps.
    Functions are "traced" showing input → intermediate steps → output.
    """

    # PTP prompt with traced function execution examples
    PROMPT = '''def answer_yes_no(question: str) -> bool:
    """Answer a yes/no question by identifying key facts and reasoning.

    >>> answer_yes_no("Are more people today related to Genghis Khan than Julius Caesar?")
    # Step 1: Identify what we need to know
    key_facts_needed = ["descendants of Genghis Khan", "descendants of Julius Caesar"]
    # Step 2: Recall relevant facts
    fact_1 = "Genghis Khan had many children and genetic studies show ~16 million men are his descendants"
    fact_2 = "Julius Caesar had few legitimate children, lineage largely died out"
    # Step 3: Compare and reason
    comparison = "16 million descendants >> few descendants"
    # Step 4: Determine answer
    answer = True  # More people are related to Genghis Khan
    True

    >>> answer_yes_no("Could a llama win a marathon?")
    # Step 1: Identify what we need to know
    key_facts_needed = ["marathon distance", "llama running capabilities"]
    # Step 2: Recall relevant facts
    fact_1 = "A marathon is 26.2 miles (42.2 km)"
    fact_2 = "Llamas can run about 35 mph but only in short bursts, not endurance runners"
    # Step 3: Compare and reason
    comparison = "Llamas lack endurance for 26+ miles, would need to stop/rest"
    # Step 4: Determine answer
    answer = False  # Llamas cannot sustain marathon distance
    False

    >>> answer_yes_no("{question}")
    # Step 1: Identify what we need to know
    key_facts_needed = '''

    def __init__(self, model: str = "deepseek-v3-0324"):
        super().__init__(model)

    def run_instance(self, instance: StrategyQAInstance) -> StrategyQAResult:
        prompt = self.PROMPT.format(question=instance.question)

        start_time = time.time()
        try:
            response = call_llm(prompt=prompt, model=self.model)
            latency_ms = (time.time() - start_time) * 1000

            # Extract the final True/False from the trace
            predicted = self._extract_answer_from_trace(response)
            input_tokens = len(prompt) // 4
            output_tokens = len(response) // 4 if response else 0
            cost = calculate_cost(input_tokens, output_tokens, self.model)

            return StrategyQAResult(
                qid=instance.qid,
                question=instance.question,
                predicted_answer=predicted,
                ground_truth=instance.answer,
                is_correct=predicted == instance.answer if predicted is not None else False,
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                cost_usd=cost.cost_usd,
                raw_response=response,
                trace={"format": "ptp", "steps": response},
            )
        except Exception as e:
            return self._error_result(instance, str(e), type(e).__name__, time.time() - start_time)

    def _extract_answer_from_trace(self, response: str) -> Optional[bool]:
        """Extract True/False from PTP trace output."""
        if not response:
            return None

        # Look for final True/False in the trace
        # Pattern: "answer = True" or just "True" at end of trace
        lines = response.strip().split('\n')

        # Check last few lines for True/False
        for line in reversed(lines[-5:]):
            line = line.strip()
            if line == 'True' or 'answer = True' in line:
                return True
            if line == 'False' or 'answer = False' in line:
                return False

        # Fallback to standard extraction
        return self.extract_boolean(response)


# ============================================================================
# L2: Trace Builder (from William Cohen's research plan)
# ============================================================================

# Import the full TraceBuilder implementation
from .trace_builder import L2_TraceBuilder, WorkflowTrace, TraceStep

# Import L3 ReAct
from .l3_react import L3_ReAct, L3_ReActSimple

# Alias for backward compatibility
L2_Decompose = L2_TraceBuilder


# ============================================================================
# Helper method for error results (add to base class ideally)
# ============================================================================

def _error_result(self, instance, error_msg, error_type, elapsed):
    return StrategyQAResult(
        qid=instance.qid,
        question=instance.question,
        predicted_answer=None,
        ground_truth=instance.answer,
        is_correct=False,
        latency_ms=elapsed * 1000,
        input_tokens=0,
        output_tokens=0,
        total_tokens=0,
        cost_usd=0.0,
        error=error_msg,
        error_type=error_type,
    )

# Add to base class
StrategyQAExperiment._error_result = _error_result


# ============================================================================
# Experiment Registry
# ============================================================================

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
    "L2-decompose": L2_Decompose,  # Alias

    # L3 - ReAct Agent (LLM controls flow)
    "L3": L3_ReAct,
    "L3-react": L3_ReAct,
    "L3-simple": L3_ReActSimple,
}
