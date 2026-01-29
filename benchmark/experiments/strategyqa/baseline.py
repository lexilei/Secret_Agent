"""
L0 Baseline Experiment for StrategyQA: Direct LLM call without framework.

This is the vanilla baseline - a direct prompt to the LLM asking for
a yes/no answer. No decomposition, no multi-step reasoning.
"""

import time
from typing import Optional

from .base import StrategyQAExperiment, StrategyQAResult
from ...dataset.strategyqa_loader import StrategyQAInstance
from ...metrics.cost import calculate_cost

from ptool_framework.llm_backend import call_llm


class StrategyQABaseline(StrategyQAExperiment):
    """
    L0 Baseline: Direct LLM call without any framework.

    Simply prompts the LLM with the question, asking for a yes/no answer.
    No structure, no multi-step reasoning, no decomposition.
    """

    PROMPT_TEMPLATE = """Answer the following yes/no question.

Question: {question}

Instructions:
1. Think about the question carefully
2. Consider what facts would be relevant
3. Determine if the answer is yes or no

Important: Return ONLY "Yes" or "No" as your final answer. Nothing else.

Answer:"""

    def __init__(self, model: str = "deepseek-v3-0324"):
        super().__init__(model)

    def run_instance(self, instance: StrategyQAInstance) -> StrategyQAResult:
        """
        Run baseline on a single instance.

        Args:
            instance: StrategyQA problem instance

        Returns:
            StrategyQAResult with prediction and metrics
        """
        # Build prompt
        prompt = self.PROMPT_TEMPLATE.format(question=instance.question)

        # Call LLM
        start_time = time.time()
        try:
            response = call_llm(
                prompt=prompt,
                model=self.model,
            )
            latency_ms = (time.time() - start_time) * 1000

            # Extract answer
            predicted = self.extract_boolean(response)

            # Estimate tokens (rough)
            input_tokens = len(prompt) // 4
            output_tokens = len(response) // 4 if response else 0

            # Calculate cost
            cost_metrics = calculate_cost(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model=self.model,
            )
            cost_usd = cost_metrics.cost_usd

            # Evaluate accuracy (exact match for boolean)
            is_correct = predicted == instance.answer if predicted is not None else False

            return StrategyQAResult(
                qid=instance.qid,
                question=instance.question,
                predicted_answer=predicted,
                ground_truth=instance.answer,
                is_correct=is_correct,
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                cost_usd=cost_usd,
                raw_response=response,
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return StrategyQAResult(
                qid=instance.qid,
                question=instance.question,
                predicted_answer=None,
                ground_truth=instance.answer,
                is_correct=False,
                latency_ms=latency_ms,
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                cost_usd=0.0,
                error=str(e),
                error_type=type(e).__name__,
            )


class StrategyQABaselineCoT(StrategyQAExperiment):
    """
    L0+ Baseline with Chain-of-Thought: Direct LLM call with CoT prompting.

    Prompts the LLM to think step by step before answering.
    """

    PROMPT_TEMPLATE = """Answer the following yes/no question by thinking step by step.

Question: {question}

Instructions:
1. Break down what the question is asking
2. Consider what facts you know that are relevant
3. Reason through the answer step by step
4. Give your final answer as Yes or No

Let's think step by step:"""

    def __init__(self, model: str = "deepseek-v3-0324"):
        super().__init__(model)

    def run_instance(self, instance: StrategyQAInstance) -> StrategyQAResult:
        """Run CoT baseline on a single instance."""
        prompt = self.PROMPT_TEMPLATE.format(question=instance.question)

        start_time = time.time()
        try:
            response = call_llm(
                prompt=prompt,
                model=self.model,
            )
            latency_ms = (time.time() - start_time) * 1000

            predicted = self.extract_boolean(response)

            input_tokens = len(prompt) // 4
            output_tokens = len(response) // 4 if response else 0

            cost_metrics = calculate_cost(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model=self.model,
            )
            cost_usd = cost_metrics.cost_usd

            is_correct = predicted == instance.answer if predicted is not None else False

            return StrategyQAResult(
                qid=instance.qid,
                question=instance.question,
                predicted_answer=predicted,
                ground_truth=instance.answer,
                is_correct=is_correct,
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                cost_usd=cost_usd,
                raw_response=response,
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return StrategyQAResult(
                qid=instance.qid,
                question=instance.question,
                predicted_answer=None,
                ground_truth=instance.answer,
                is_correct=False,
                latency_ms=latency_ms,
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                cost_usd=0.0,
                error=str(e),
                error_type=type(e).__name__,
            )
