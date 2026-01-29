"""
ReAct Experiment for StrategyQA.

Uses ReAct agent with custom ptools for multi-step reasoning.
"""

import time
from typing import List, Optional, Dict, Any, Literal

from .base import StrategyQAExperiment, StrategyQAResult
from ...dataset.strategyqa_loader import StrategyQAInstance
from ...metrics.cost import calculate_cost

from ptool_framework import ptool, PToolSpec, get_registry
from ptool_framework.react import ReActAgent, ReActResult
from ptool_framework.llm_backend import call_llm


# ============================================================================
# StrategyQA-specific ptools
# ============================================================================

@ptool(model="deepseek-v3-0324", output_mode="text")
def decompose_question(question: str) -> str:
    """
    Break down a yes/no question into simpler sub-questions.

    Args:
        question: The main yes/no question to decompose

    Returns:
        A list of 2-4 sub-questions, one per line, that when answered
        will help determine the answer to the main question.
        Use #N to refer to answer of sub-question N.

    Example:
        Input: "Are more people related to Genghis Khan than Julius Caesar?"
        Output:
        1. How many descendants does Genghis Khan have?
        2. How many descendants does Julius Caesar have?
        3. Is #1 greater than #2?
    """
    ...


@ptool(model="deepseek-v3-0324", output_mode="text")
def answer_factual_question(question: str) -> str:
    """
    Answer a factual question with a brief response.

    Args:
        question: A factual question (not yes/no)

    Returns:
        A concise factual answer (1-2 sentences).

    Example:
        Input: "What is the population of Tokyo?"
        Output: "Tokyo has approximately 14 million people in the city proper."
    """
    ...


@ptool(model="deepseek-v3-0324", output_mode="text")
def determine_yes_no(
    question: str,
    facts: str,
) -> Literal["Yes", "No"]:
    """
    Determine if the answer to a yes/no question is Yes or No based on given facts.

    Args:
        question: The yes/no question to answer
        facts: Relevant facts that help answer the question

    Returns:
        Either "Yes" or "No" based on the facts provided.
    """
    ...


# ============================================================================
# Experiment class
# ============================================================================

class StrategyQAReAct(StrategyQAExperiment):
    """
    ReAct experiment for StrategyQA.

    Uses ReAct agent with custom ptools:
    - decompose_question: Break question into sub-questions
    - answer_factual_question: Answer sub-questions
    - determine_yes_no: Combine facts into final answer
    """

    def __init__(
        self,
        model: str = "deepseek-v3-0324",
        max_steps: int = 10,
        echo: bool = False,
    ):
        super().__init__(model)
        self.max_steps = max_steps
        self.echo = echo

        # Register our ptools
        self.ptools = [
            decompose_question.spec,
            answer_factual_question.spec,
            determine_yes_no.spec,
        ]

    def setup(self) -> None:
        """Initialize the ReAct agent."""
        super().setup()
        # Ptools are already registered via the @ptool decorator

    def run_instance(self, instance: StrategyQAInstance) -> StrategyQAResult:
        """Run ReAct agent on a single instance."""
        start_time = time.time()

        # Create goal for the agent
        goal = f"""Answer this yes/no question: {instance.question}

Use the available tools to:
1. First decompose the question into simpler sub-questions
2. Answer each sub-question to gather facts
3. Use determine_yes_no to combine the facts into a final Yes or No answer

Return your final answer as ANSWER: Yes or ANSWER: No"""

        try:
            # Create and run agent
            agent = ReActAgent(
                available_ptools=self.ptools,
                model=self.model,
                max_steps=self.max_steps,
                echo=self.echo,
            )

            result: ReActResult = agent.run(goal)

            latency_ms = (time.time() - start_time) * 1000

            # Extract answer from trajectory
            predicted = self._extract_answer_from_result(result)

            # Estimate tokens from trajectory
            total_tokens = self._estimate_tokens_from_trajectory(result)
            input_tokens = total_tokens // 2  # Rough estimate
            output_tokens = total_tokens - input_tokens

            cost_metrics = calculate_cost(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model=self.model,
            )

            is_correct = predicted == instance.answer if predicted is not None else False

            # Build trace for debugging
            trace = {
                "trajectory_id": result.trajectory.trajectory_id if result.trajectory else None,
                "steps": [
                    {
                        "thought": step.thought.content if step.thought else None,
                        "action": step.action.ptool_name if step.action else None,
                        "action_args": step.action.args if step.action else None,
                        "observation": str(step.observation.result)[:200] if step.observation else None,
                    }
                    for step in (result.trajectory.steps if result.trajectory else [])
                ],
                "final_answer": result.answer,
                "success": result.success,
            }

            return StrategyQAResult(
                qid=instance.qid,
                question=instance.question,
                predicted_answer=predicted,
                ground_truth=instance.answer,
                is_correct=is_correct,
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cost_usd=cost_metrics.cost_usd,
                num_steps=len(result.trajectory.steps) if result.trajectory else 0,
                raw_response=str(result.answer),
                trace=trace,
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

    def _extract_answer_from_result(self, result: ReActResult) -> Optional[bool]:
        """Extract boolean answer from ReAct result."""
        if result.answer is not None:
            # Try to parse the answer
            return self.extract_boolean(str(result.answer))

        # Look through the trajectory for final answer
        if result.trajectory and result.trajectory.steps:
            for step in reversed(result.trajectory.steps):
                if step.observation and step.observation.result:
                    parsed = self.extract_boolean(str(step.observation.result))
                    if parsed is not None:
                        return parsed

        return None

    def _estimate_tokens_from_trajectory(self, result: ReActResult) -> int:
        """Estimate total tokens used from trajectory."""
        total = 0
        if result.trajectory:
            for step in result.trajectory.steps:
                if step.thought:
                    total += len(step.thought.content) // 4
                if step.action:
                    total += len(str(step.action.args)) // 4
                if step.observation and step.observation.result:
                    total += len(str(step.observation.result)) // 4
        return max(total, 100)  # Minimum estimate


class StrategyQAReActSimple(StrategyQAExperiment):
    """
    Simplified ReAct-style experiment without full agent framework.

    Uses a simple think-act-observe loop with direct LLM calls.
    """

    REACT_PROMPT = """You are solving a yes/no question through step-by-step reasoning.

Question: {question}

You have the following actions available:
- THINK: Reason about what you know and what you need to find out
- SEARCH[query]: Look up a factual piece of information
- ANSWER[Yes/No]: Provide your final answer

Format each step as:
Thought: <your reasoning>
Action: <THINK or SEARCH[query] or ANSWER[Yes/No]>

Begin solving the question step by step. When you're confident in your answer, use ANSWER[Yes] or ANSWER[No].

Step 1:"""

    CONTINUE_PROMPT = """Continue from where you left off. Previous steps:
{history}

Next step:"""

    def __init__(
        self,
        model: str = "deepseek-v3-0324",
        max_steps: int = 6,
    ):
        super().__init__(model)
        self.max_steps = max_steps

    def run_instance(self, instance: StrategyQAInstance) -> StrategyQAResult:
        """Run simplified ReAct on a single instance."""
        start_time = time.time()
        total_input_tokens = 0
        total_output_tokens = 0
        trace = {"steps": []}

        try:
            # Initial prompt
            prompt = self.REACT_PROMPT.format(question=instance.question)
            history = ""

            for step in range(self.max_steps):
                response = call_llm(prompt=prompt, model=self.model)

                total_input_tokens += len(prompt) // 4
                total_output_tokens += len(response) // 4 if response else 0

                trace["steps"].append({
                    "step": step + 1,
                    "response": response,
                })

                # Check for final answer
                if response:
                    import re
                    answer_match = re.search(r'ANSWER\s*\[\s*(Yes|No)\s*\]', response, re.IGNORECASE)
                    if answer_match:
                        final_answer = answer_match.group(1).lower() == "yes"
                        latency_ms = (time.time() - start_time) * 1000

                        cost_metrics = calculate_cost(
                            input_tokens=total_input_tokens,
                            output_tokens=total_output_tokens,
                            model=self.model,
                        )

                        is_correct = final_answer == instance.answer

                        return StrategyQAResult(
                            qid=instance.qid,
                            question=instance.question,
                            predicted_answer=final_answer,
                            ground_truth=instance.answer,
                            is_correct=is_correct,
                            latency_ms=latency_ms,
                            input_tokens=total_input_tokens,
                            output_tokens=total_output_tokens,
                            total_tokens=total_input_tokens + total_output_tokens,
                            cost_usd=cost_metrics.cost_usd,
                            num_steps=step + 1,
                            raw_response=response,
                            trace=trace,
                        )

                # Continue with next step
                history += f"\nStep {step + 1}:\n{response}\n"
                prompt = self.CONTINUE_PROMPT.format(history=history)

            # Max steps reached without answer
            latency_ms = (time.time() - start_time) * 1000
            cost_metrics = calculate_cost(
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                model=self.model,
            )

            # Try to extract answer from last response
            predicted = self.extract_boolean(trace["steps"][-1]["response"] if trace["steps"] else "")

            return StrategyQAResult(
                qid=instance.qid,
                question=instance.question,
                predicted_answer=predicted,
                ground_truth=instance.answer,
                is_correct=predicted == instance.answer if predicted is not None else False,
                latency_ms=latency_ms,
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                total_tokens=total_input_tokens + total_output_tokens,
                cost_usd=cost_metrics.cost_usd,
                num_steps=self.max_steps,
                raw_response=str(trace["steps"][-1] if trace["steps"] else ""),
                trace=trace,
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
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                total_tokens=total_input_tokens + total_output_tokens,
                cost_usd=0.0,
                error=str(e),
                error_type=type(e).__name__,
                trace=trace,
            )
