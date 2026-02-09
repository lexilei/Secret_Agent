"""
L0 Baseline Experiment for StrategyQA: Direct LLM call without framework.
Call once.

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
"""

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

    PROMPT_TEMPLATE = """
    Q: Do hamsters provide food for any animals?
A: Hamsters are prey animals. Prey are food for predators. Thus, hamsters provide food for some animals. So
the answer is yes.
Q: Could Brooke Shields succeed at University of Pennsylvania?
A: Brooke Shields went to Princeton University. Princeton University is about as academically rigorous as the
University of Pennsylvania. Thus, Brooke Shields could also succeed at the University of Pennsylvania. So the
answer is yes.
Q: Yes or no: Hydrogen’s atomic number squared exceeds number of Spice Girls?
A: Hydrogen has an atomic number of 1. 1 squared is 1. There are 5 Spice Girls. Thus, Hydrogen’s atomic
number squared is less than 5. So the answer is no.
Q: Yes or no: Is it common to see frost during some college commencements?
A: College commencement ceremonies can happen in December, May, and June. December is in the winter, so
there can be frost. Thus, there could be frost at some commencements. So the answer is yes.
Q: Yes or no: Could a llama birth twice during War in Vietnam (1945-46)?
A: The War in Vietnam was 6 months. The gestation period for a llama is 11 months, which is more than 6
months. Thus, a llama could not give birth twice during the War in Vietnam. So the answer is no.
Q: Yes or no: Would a pear sink in water?
A: The density of a pear is about 0.6g/cm3
, which is less than water. Objects less dense than water float. Thus,
a pear would float. So the answer is no.
Q: {question}
A:"""

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
