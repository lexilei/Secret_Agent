"""
2-Step Decomposition Experiment for StrategyQA.

Decomposes questions into sub-questions, answers each, then combines.
Two variants:
- DecomposeOracle: Uses dataset's provided decomposition
- DecomposeLLM: LLM generates the decomposition
"""

import time
import json
from typing import List, Optional, Dict, Any

from .base import StrategyQAExperiment, StrategyQAResult
from ...dataset.strategyqa_loader import StrategyQAInstance
from ...metrics.cost import calculate_cost

from ptool_framework.llm_backend import call_llm


class StrategyQADecomposeOracle(StrategyQAExperiment):
    """
    2-Step Decomposition using Oracle (dataset-provided) decomposition.

    Uses the decomposition sub-questions from the dataset, then:
    1. Answers each sub-question
    2. Combines sub-answers to produce final yes/no answer
    """

    SUB_QUESTION_TEMPLATE = """Answer the following question concisely.

Question: {question}

Give a brief factual answer (1-2 sentences):"""

    COMBINE_TEMPLATE = """Based on the following sub-questions and their answers, determine the answer to the main question.

Main Question: {main_question}

Sub-questions and answers:
{sub_qa_text}

Based on these answers, is the answer to the main question Yes or No?
Return ONLY "Yes" or "No":"""

    def __init__(self, model: str = "deepseek-v3-0324"):
        super().__init__(model)

    def run_instance(self, instance: StrategyQAInstance) -> StrategyQAResult:
        """Run oracle decomposition on a single instance."""
        start_time = time.time()
        total_input_tokens = 0
        total_output_tokens = 0
        trace = {"steps": [], "decomposition": instance.decomposition}

        try:
            # Step 1: Answer each sub-question
            sub_answers = []
            for i, sub_q in enumerate(instance.decomposition):
                # Replace #N references with previous answers
                resolved_q = self._resolve_references(sub_q, sub_answers)

                prompt = self.SUB_QUESTION_TEMPLATE.format(question=resolved_q)
                response = call_llm(prompt=prompt, model=self.model)

                sub_answers.append(response.strip() if response else "Unknown")

                total_input_tokens += len(prompt) // 4
                total_output_tokens += len(response) // 4 if response else 0

                trace["steps"].append({
                    "step": i + 1,
                    "type": "sub_question",
                    "question": resolved_q,
                    "answer": sub_answers[-1],
                })

            # Step 2: Combine answers
            sub_qa_text = "\n".join(
                f"{i+1}. Q: {q}\n   A: {a}"
                for i, (q, a) in enumerate(zip(instance.decomposition, sub_answers))
            )

            combine_prompt = self.COMBINE_TEMPLATE.format(
                main_question=instance.question,
                sub_qa_text=sub_qa_text,
            )
            final_response = call_llm(prompt=combine_prompt, model=self.model)

            total_input_tokens += len(combine_prompt) // 4
            total_output_tokens += len(final_response) // 4 if final_response else 0

            trace["steps"].append({
                "step": len(instance.decomposition) + 1,
                "type": "combine",
                "prompt": combine_prompt,
                "response": final_response,
            })

            # Extract final answer
            predicted = self.extract_boolean(final_response)
            latency_ms = (time.time() - start_time) * 1000

            cost_metrics = calculate_cost(
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                model=self.model,
            )

            is_correct = predicted == instance.answer if predicted is not None else False

            return StrategyQAResult(
                qid=instance.qid,
                question=instance.question,
                predicted_answer=predicted,
                ground_truth=instance.answer,
                is_correct=is_correct,
                latency_ms=latency_ms,
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                total_tokens=total_input_tokens + total_output_tokens,
                cost_usd=cost_metrics.cost_usd,
                num_steps=len(instance.decomposition) + 1,
                decomposition_used=True,
                raw_response=final_response,
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

    def _resolve_references(self, question: str, previous_answers: List[str]) -> str:
        """Replace #N references with previous answers."""
        import re
        result = question
        for match in re.finditer(r'#(\d+)', question):
            idx = int(match.group(1)) - 1
            if 0 <= idx < len(previous_answers):
                result = result.replace(match.group(0), previous_answers[idx])
        return result


class StrategyQADecomposeLLM(StrategyQAExperiment):
    """
    2-Step Decomposition where LLM generates the decomposition.

    1. LLM decomposes the question into sub-questions
    2. Answers each sub-question
    3. Combines sub-answers to produce final yes/no answer
    """

    DECOMPOSE_TEMPLATE = """Break down this yes/no question into simpler sub-questions that can help answer it.

Question: {question}

Instructions:
- Generate 2-4 sub-questions that, when answered, will help determine the final answer
- Each sub-question should be answerable with factual information
- Use #N to refer to the answer of sub-question N (e.g., "Is #1 greater than #2?")

Return the sub-questions as a JSON list:
["sub-question 1", "sub-question 2", ...]

Sub-questions:"""

    SUB_QUESTION_TEMPLATE = """Answer the following question concisely.

Question: {question}

Give a brief factual answer (1-2 sentences):"""

    COMBINE_TEMPLATE = """Based on the following sub-questions and their answers, determine the answer to the main question.

Main Question: {main_question}

Sub-questions and answers:
{sub_qa_text}

Based on these answers, is the answer to the main question Yes or No?
Return ONLY "Yes" or "No":"""

    def __init__(self, model: str = "deepseek-v3-0324"):
        super().__init__(model)

    def run_instance(self, instance: StrategyQAInstance) -> StrategyQAResult:
        """Run LLM decomposition on a single instance."""
        start_time = time.time()
        total_input_tokens = 0
        total_output_tokens = 0
        trace = {"steps": []}

        try:
            # Step 0: Decompose the question
            decompose_prompt = self.DECOMPOSE_TEMPLATE.format(question=instance.question)
            decompose_response = call_llm(prompt=decompose_prompt, model=self.model)

            total_input_tokens += len(decompose_prompt) // 4
            total_output_tokens += len(decompose_response) // 4 if decompose_response else 0

            # Parse decomposition
            sub_questions = self._parse_decomposition(decompose_response)
            trace["decomposition"] = sub_questions
            trace["steps"].append({
                "step": 0,
                "type": "decompose",
                "response": decompose_response,
                "parsed": sub_questions,
            })

            if not sub_questions:
                # Fallback: use original question
                sub_questions = [instance.question]

            # Step 1: Answer each sub-question
            sub_answers = []
            for i, sub_q in enumerate(sub_questions):
                # Replace #N references with previous answers
                resolved_q = self._resolve_references(sub_q, sub_answers)

                prompt = self.SUB_QUESTION_TEMPLATE.format(question=resolved_q)
                response = call_llm(prompt=prompt, model=self.model)

                sub_answers.append(response.strip() if response else "Unknown")

                total_input_tokens += len(prompt) // 4
                total_output_tokens += len(response) // 4 if response else 0

                trace["steps"].append({
                    "step": i + 1,
                    "type": "sub_question",
                    "question": resolved_q,
                    "answer": sub_answers[-1],
                })

            # Step 2: Combine answers
            sub_qa_text = "\n".join(
                f"{i+1}. Q: {q}\n   A: {a}"
                for i, (q, a) in enumerate(zip(sub_questions, sub_answers))
            )

            combine_prompt = self.COMBINE_TEMPLATE.format(
                main_question=instance.question,
                sub_qa_text=sub_qa_text,
            )
            final_response = call_llm(prompt=combine_prompt, model=self.model)

            total_input_tokens += len(combine_prompt) // 4
            total_output_tokens += len(final_response) // 4 if final_response else 0

            trace["steps"].append({
                "step": len(sub_questions) + 1,
                "type": "combine",
                "response": final_response,
            })

            # Extract final answer
            predicted = self.extract_boolean(final_response)
            latency_ms = (time.time() - start_time) * 1000

            cost_metrics = calculate_cost(
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                model=self.model,
            )

            is_correct = predicted == instance.answer if predicted is not None else False

            return StrategyQAResult(
                qid=instance.qid,
                question=instance.question,
                predicted_answer=predicted,
                ground_truth=instance.answer,
                is_correct=is_correct,
                latency_ms=latency_ms,
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                total_tokens=total_input_tokens + total_output_tokens,
                cost_usd=cost_metrics.cost_usd,
                num_steps=len(sub_questions) + 2,  # decompose + sub-qs + combine
                decomposition_used=True,
                raw_response=final_response,
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

    def _parse_decomposition(self, response: str) -> List[str]:
        """Parse LLM response to extract sub-questions."""
        import re

        if not response:
            return []

        # Try to find JSON array
        json_match = re.search(r'\[.*?\]', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        # Fallback: extract numbered lines
        lines = []
        for line in response.split('\n'):
            line = line.strip()
            # Match "1. question" or "- question" patterns
            match = re.match(r'^(?:\d+[\.\)]\s*|-\s*|•\s*)(.+)', line)
            if match:
                lines.append(match.group(1))

        return lines if lines else []

    def _resolve_references(self, question: str, previous_answers: List[str]) -> str:
        """Replace #N references with previous answers."""
        import re
        result = question
        for match in re.finditer(r'#(\d+)', question):
            idx = int(match.group(1)) - 1
            if 0 <= idx < len(previous_answers):
                result = result.replace(match.group(0), previous_answers[idx])
        return result
