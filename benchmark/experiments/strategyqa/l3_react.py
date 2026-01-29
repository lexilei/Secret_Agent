"""
L3 ReAct Agent for StrategyQA.

Based on William Cohen's research plan:
- L3 = ReAct agentic pipeline
- LLM decides what to do next (vs L2 where Python controls flow)
- Uses same ptools as L2, but agent chooses when/how to call them

Key difference from L2:
- L2 (Trace Builder): Python controls flow → decompose → answer_subs → combine
- L3 (ReAct Agent): LLM thinks → chooses action → observes → repeats until done
"""

import time
import re
import json
from typing import Optional, Dict, Any, List

from .base import StrategyQAExperiment, StrategyQAResult
from ...dataset.strategyqa_loader import StrategyQAInstance
from ...metrics.cost import calculate_cost

from ptool_framework.llm_backend import call_llm


# ============================================================================
# ReAct Agent for StrategyQA
# ============================================================================

class L3_ReAct(StrategyQAExperiment):
    """
    L3: ReAct Agent for StrategyQA.

    Uses think-act-observe loop where the LLM decides:
    - What to think about
    - Which action (ptool) to call
    - When to give final answer

    Available actions:
    - decompose(question) → break into sub-questions
    - lookup(question) → answer a factual question
    - evaluate(condition) → evaluate True/False
    - finish(answer) → give final Yes/No answer
    """

    SYSTEM_PROMPT = """You are a reasoning agent that answers yes/no questions by breaking them down and looking up facts.

Available actions:
1. decompose(question) - Break a complex question into simpler sub-questions
2. lookup(question) - Look up a factual piece of information
3. evaluate(condition) - Evaluate if a condition is True or False based on what you know
4. finish(Yes/No) - Give your final answer when you're confident

Format your response as:
Thought: <your reasoning about what to do next>
Action: <action_name>(<argument>)

Examples:
Thought: I need to break down this question into simpler parts.
Action: decompose(Are more people related to Genghis Khan than Julius Caesar?)

Thought: I need to find out how many descendants Genghis Khan has.
Action: lookup(How many descendants does Genghis Khan have?)

Thought: Based on the facts, Genghis Khan has more descendants.
Action: finish(Yes)

IMPORTANT:
- Always start with a Thought
- Call exactly ONE action per turn
- When you have enough information, use finish(Yes) or finish(No)
- Maximum {max_steps} steps allowed
"""

    STEP_PROMPT = """Question: {question}

Previous steps:
{history}

What's your next step?

Thought:"""

    def __init__(self, model: str = "deepseek-v3-0324", max_steps: int = 8):
        super().__init__(model)
        self.max_steps = max_steps

    def run_instance(self, instance: StrategyQAInstance) -> StrategyQAResult:
        """Run ReAct agent on a single instance."""
        start_time = time.time()
        total_input_tokens = 0
        total_output_tokens = 0

        history = []
        trace = {"steps": [], "question": instance.question}
        context = {}  # Store facts and intermediate results

        try:
            for step_num in range(self.max_steps):
                # Build prompt
                history_text = self._format_history(history)
                prompt = self.SYSTEM_PROMPT.format(max_steps=self.max_steps) + "\n\n"
                prompt += self.STEP_PROMPT.format(
                    question=instance.question,
                    history=history_text if history_text else "(none yet)"
                )

                # Get LLM response
                response = call_llm(prompt=prompt, model=self.model)
                total_input_tokens += len(prompt) // 4
                total_output_tokens += len(response) // 4 if response else 0

                # Parse thought and action
                thought, action_name, action_arg = self._parse_response(response)

                trace["steps"].append({
                    "step": step_num + 1,
                    "thought": thought,
                    "action": action_name,
                    "action_arg": action_arg,
                })

                # Check for finish action
                if action_name == "finish":
                    final_answer = action_arg.lower().strip() in ["yes", "true"]
                    latency_ms = (time.time() - start_time) * 1000

                    cost = calculate_cost(total_input_tokens, total_output_tokens, self.model)

                    return StrategyQAResult(
                        qid=instance.qid,
                        question=instance.question,
                        predicted_answer=final_answer,
                        ground_truth=instance.answer,
                        is_correct=final_answer == instance.answer,
                        latency_ms=latency_ms,
                        input_tokens=total_input_tokens,
                        output_tokens=total_output_tokens,
                        total_tokens=total_input_tokens + total_output_tokens,
                        cost_usd=cost.cost_usd,
                        num_steps=step_num + 1,
                        raw_response=response,
                        trace=trace,
                    )

                # Execute action and get observation
                observation = self._execute_action(action_name, action_arg, context)
                total_input_tokens += observation.get("input_tokens", 0)
                total_output_tokens += observation.get("output_tokens", 0)

                trace["steps"][-1]["observation"] = observation.get("result")

                # Add to history
                history.append({
                    "thought": thought,
                    "action": f"{action_name}({action_arg})",
                    "observation": observation.get("result"),
                })

            # Max steps reached - try to extract answer from context
            latency_ms = (time.time() - start_time) * 1000
            predicted = self._guess_answer_from_context(context, instance.question)

            cost = calculate_cost(total_input_tokens, total_output_tokens, self.model)

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
                cost_usd=cost.cost_usd,
                num_steps=self.max_steps,
                raw_response="Max steps reached",
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

    def _format_history(self, history: List[Dict]) -> str:
        """Format history for prompt."""
        if not history:
            return ""

        lines = []
        for i, h in enumerate(history):
            lines.append(f"Step {i+1}:")
            lines.append(f"  Thought: {h['thought']}")
            lines.append(f"  Action: {h['action']}")
            lines.append(f"  Observation: {h['observation']}")
        return "\n".join(lines)

    def _parse_response(self, response: str) -> tuple:
        """Parse thought and action from LLM response."""
        thought = ""
        action_name = "unknown"
        action_arg = ""

        if not response:
            return thought, action_name, action_arg

        # Extract thought
        thought_match = re.search(r'Thought:\s*(.+?)(?=Action:|$)', response, re.DOTALL)
        if thought_match:
            thought = thought_match.group(1).strip()

        # Extract action
        action_match = re.search(r'Action:\s*(\w+)\((.+?)\)', response, re.DOTALL)
        if action_match:
            action_name = action_match.group(1).lower()
            action_arg = action_match.group(2).strip().strip('"\'')

        return thought, action_name, action_arg

    def _execute_action(self, action_name: str, action_arg: str, context: Dict) -> Dict:
        """Execute an action and return observation."""
        input_tokens = 0
        output_tokens = 0

        if action_name == "decompose":
            prompt = f"""Break down this question into 2-3 simpler sub-questions:

Question: {action_arg}

Return as a numbered list:
1."""
            result = call_llm(prompt=prompt, model=self.model)
            input_tokens = len(prompt) // 4
            output_tokens = len(result) // 4 if result else 0

            # Store in context
            context["sub_questions"] = result
            return {"result": result, "input_tokens": input_tokens, "output_tokens": output_tokens}

        elif action_name == "lookup":
            prompt = f"""Answer this factual question briefly (1-2 sentences):

Question: {action_arg}

Answer:"""
            result = call_llm(prompt=prompt, model=self.model)
            input_tokens = len(prompt) // 4
            output_tokens = len(result) // 4 if result else 0

            # Store in context
            context[action_arg] = result
            return {"result": result, "input_tokens": input_tokens, "output_tokens": output_tokens}

        elif action_name == "evaluate":
            # Use context to evaluate
            context_str = "\n".join(f"- {k}: {v}" for k, v in context.items() if k != "sub_questions")
            prompt = f"""Based on these facts, evaluate the condition:

Facts:
{context_str}

Condition: {action_arg}

Is this True or False? Answer with only True or False:"""
            result = call_llm(prompt=prompt, model=self.model)
            input_tokens = len(prompt) // 4
            output_tokens = len(result) // 4 if result else 0

            return {"result": result, "input_tokens": input_tokens, "output_tokens": output_tokens}

        else:
            return {"result": f"Unknown action: {action_name}", "input_tokens": 0, "output_tokens": 0}

    def _guess_answer_from_context(self, context: Dict, question: str) -> Optional[bool]:
        """Try to guess answer from accumulated context."""
        if not context:
            return None

        # Use LLM to make final judgment
        context_str = "\n".join(f"- {k}: {v}" for k, v in context.items() if k != "sub_questions")
        prompt = f"""Based on these facts, answer the yes/no question:

Question: {question}

Facts:
{context_str}

Answer Yes or No:"""

        try:
            result = call_llm(prompt=prompt, model=self.model)
            return "yes" in result.lower()
        except:
            return None


# ============================================================================
# Simplified ReAct (fewer actions, more direct)
# ============================================================================

class L3_ReActSimple(StrategyQAExperiment):
    """
    L3 Simplified: ReAct with minimal action set.

    Only two actions:
    - think(reasoning) - Record reasoning step
    - answer(Yes/No) - Give final answer

    The LLM does all reasoning in "think" steps, then answers.
    """

    PROMPT = """Answer this yes/no question using step-by-step reasoning.

Question: {question}

Use this format:
Thought 1: <first reasoning step>
Thought 2: <second reasoning step>
...
Answer: Yes or No

Begin:
Thought 1:"""

    def __init__(self, model: str = "deepseek-v3-0324"):
        super().__init__(model)

    def run_instance(self, instance: StrategyQAInstance) -> StrategyQAResult:
        """Run simplified ReAct on a single instance."""
        start_time = time.time()

        prompt = self.PROMPT.format(question=instance.question)

        try:
            response = call_llm(prompt=prompt, model=self.model)
            latency_ms = (time.time() - start_time) * 1000

            input_tokens = len(prompt) // 4
            output_tokens = len(response) // 4 if response else 0

            # Extract answer
            predicted = self._extract_answer(response)

            # Count thoughts
            num_thoughts = len(re.findall(r'Thought \d+:', response))

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
                num_steps=num_thoughts,
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

    def _extract_answer(self, response: str) -> Optional[bool]:
        """Extract final answer from response."""
        if not response:
            return None

        # Look for "Answer: Yes/No"
        match = re.search(r'Answer:\s*(Yes|No)', response, re.IGNORECASE)
        if match:
            return match.group(1).lower() == "yes"

        # Fallback
        return self.extract_boolean(response)
