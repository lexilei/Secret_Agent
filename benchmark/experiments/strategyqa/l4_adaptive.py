"""
L4 Adaptive Experiment for StrategyQA.

Routes tasks to different levels based on question complexity:
- SIMPLE → L0/L1 (direct prompt, cheap)
- MEDIUM → L2 (trace builder, balanced)
- HARD → L3 (ReAct agent, thorough)

Key insight: Not all questions need the same sophistication.
Simple factual questions can use cheap approaches;
complex multi-hop reasoning justifies more expensive approaches.
"""

import time
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional

from .base import StrategyQAExperiment, StrategyQAResult
from ...dataset.strategyqa_loader import StrategyQAInstance
from ...metrics.cost import calculate_cost

from ptool_framework.llm_backend import call_llm


# =============================================================================
# Routing Infrastructure
# =============================================================================

class Complexity(Enum):
    """Question complexity levels."""
    SIMPLE = "simple"      # Single fact, direct answer
    MEDIUM = "medium"      # 2-3 facts, some reasoning
    HARD = "hard"          # Multi-hop, complex logic


@dataclass
class RoutingDecision:
    """Result of routing decision."""
    complexity: Complexity
    target_level: str
    confidence: float = 1.0
    reasoning: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseRouter(ABC):
    """Abstract base for routing strategies."""

    @abstractmethod
    def route(self, question: str) -> RoutingDecision:
        """Make routing decision for a question."""
        pass


# =============================================================================
# Complexity Estimation
# =============================================================================

# Keywords that suggest different complexity levels
SIMPLE_INDICATORS = [
    "is", "are", "was", "were", "did", "does", "can", "could",
    "has", "have", "will", "would",
]

COMPLEX_INDICATORS = [
    "more", "less", "greater", "fewer", "larger", "smaller",
    "more than", "less than", "compared to", "before", "after",
    "both", "either", "neither", "if", "when", "while",
    "simultaneously", "relationship", "connection",
]

MULTI_HOP_INDICATORS = [
    "and", "but", "however", "therefore", "because", "since",
    "although", "despite", "whereas", "while also",
]


class ComplexityRouter(BaseRouter):
    """
    Routes based on question complexity estimation.

    Uses heuristics:
    1. Question length
    2. Number of entities/concepts
    3. Presence of comparison/temporal words
    4. Syntactic complexity
    """

    def __init__(
        self,
        level_mapping: Optional[Dict[Complexity, str]] = None,
        use_llm_estimation: bool = False,
        model: str = "deepseek-v3-0324",
    ):
        self.level_mapping = level_mapping or {
            Complexity.SIMPLE: "L1-cot",
            Complexity.MEDIUM: "L2",
            Complexity.HARD: "L3",
        }
        self.use_llm_estimation = use_llm_estimation
        self.model = model

        # Statistics
        self.routing_stats: Dict[str, int] = {
            "simple": 0, "medium": 0, "hard": 0,
            "rule_based": 0, "llm_based": 0,
        }

    def _estimate_complexity_rules(self, question: str) -> Complexity:
        """Estimate complexity using rule-based heuristics."""
        question_lower = question.lower()
        words = question_lower.split()
        word_count = len(words)

        # Count indicators
        complex_count = sum(1 for ind in COMPLEX_INDICATORS if ind in question_lower)
        multi_hop_count = sum(1 for ind in MULTI_HOP_INDICATORS if ind in question_lower)

        # Heuristic scoring
        score = 0

        # Length-based (longer questions tend to be harder)
        if word_count > 15:
            score += 2
        elif word_count > 8:
            score += 1

        # Complexity indicators (comparison, temporal, etc.)
        score += complex_count * 2  # Each indicator adds 2 points

        # Multi-hop indicators (and, but, because, etc.)
        score += multi_hop_count

        # Question mark count (multiple questions = harder)
        question_marks = question.count("?")
        if question_marks > 1:
            score += 3

        # Classify with adjusted thresholds
        if score >= 5:
            return Complexity.HARD
        elif score >= 2:
            return Complexity.MEDIUM
        else:
            return Complexity.SIMPLE

    def _estimate_complexity_llm(self, question: str) -> Complexity:
        """Estimate complexity using LLM."""
        prompt = f"""Estimate the complexity of answering this yes/no question.

Question: {question}

Consider:
- How many facts need to be looked up?
- Is there comparison or temporal reasoning?
- Are multiple concepts connected?

Complexity levels:
- simple: Single fact lookup, direct answer
- medium: 2-3 facts, some reasoning needed
- hard: Multiple facts, complex multi-hop reasoning

Answer with only one word: simple, medium, or hard"""

        try:
            response = call_llm(prompt=prompt, model=self.model)
            response_lower = response.lower().strip()

            if "hard" in response_lower:
                return Complexity.HARD
            elif "medium" in response_lower:
                return Complexity.MEDIUM
            else:
                return Complexity.SIMPLE
        except Exception:
            return Complexity.MEDIUM

    def route(self, question: str) -> RoutingDecision:
        """Route based on complexity estimation."""
        # Try rule-based first (fast, free)
        complexity = self._estimate_complexity_rules(question)
        reasoning = "Rule-based estimation"

        # Optionally use LLM for uncertain cases
        if self.use_llm_estimation and complexity == Complexity.MEDIUM:
            complexity = self._estimate_complexity_llm(question)
            reasoning = "LLM-based estimation"
            self.routing_stats["llm_based"] += 1
        else:
            self.routing_stats["rule_based"] += 1

        self.routing_stats[complexity.value] += 1

        return RoutingDecision(
            complexity=complexity,
            target_level=self.level_mapping[complexity],
            confidence=0.8,
            reasoning=reasoning,
        )


# =============================================================================
# L4 Adaptive Experiment
# =============================================================================

class L4_Adaptive(StrategyQAExperiment):
    """
    L4 Adaptive: Routes questions to appropriate levels based on complexity.

    Simple questions → L1 (CoT, cheap)
    Medium questions → L2 (Trace Builder, balanced)
    Hard questions → L3 (ReAct, thorough)
    """

    def __init__(
        self,
        model: str = "deepseek-v3-0324",
        use_llm_routing: bool = False,
    ):
        super().__init__(model)
        self.router = ComplexityRouter(
            use_llm_estimation=use_llm_routing,
            model=model,
        )

        # Lazy-loaded experiments
        self._experiments: Dict[str, StrategyQAExperiment] = {}

        # Statistics
        self.routing_decisions: List[RoutingDecision] = []
        self.level_stats: Dict[str, Dict[str, int]] = {}

    def _get_experiment(self, level: str) -> StrategyQAExperiment:
        """Lazy-load experiment for a level."""
        if level not in self._experiments:
            if level == "L0":
                from .levels import L0_Baseline
                self._experiments[level] = L0_Baseline(self.model)
            elif level == "L1-cot":
                from .levels import L1_CoT
                self._experiments[level] = L1_CoT(self.model)
            elif level == "L1-ptp":
                from .levels import L1_PTP
                self._experiments[level] = L1_PTP(self.model)
            elif level == "L2":
                from .trace_builder import L2_TraceBuilder
                self._experiments[level] = L2_TraceBuilder(self.model)
            elif level == "L2-rag":
                from .trace_builder import L2_TraceBuilderRAG
                self._experiments[level] = L2_TraceBuilderRAG(self.model)
            elif level == "L3":
                from .l3_react import L3_ReAct
                self._experiments[level] = L3_ReAct(self.model)
            else:
                raise ValueError(f"Unknown level: {level}")

            self._experiments[level].setup()

            # Initialize stats
            if level not in self.level_stats:
                self.level_stats[level] = {"calls": 0, "correct": 0}

        return self._experiments[level]

    def run_instance(self, instance: StrategyQAInstance) -> StrategyQAResult:
        """Run with adaptive routing."""
        start_time = time.time()

        # Route to appropriate level
        decision = self.router.route(instance.question)
        self.routing_decisions.append(decision)

        target_level = decision.target_level

        # Get and run experiment
        experiment = self._get_experiment(target_level)
        result = experiment.run_instance(instance)

        # Update stats
        self.level_stats[target_level]["calls"] += 1
        if result.is_correct:
            self.level_stats[target_level]["correct"] += 1

        # Add routing info to result
        result.trace = result.trace or {}
        if isinstance(result.trace, dict):
            result.trace["routing"] = {
                "complexity": decision.complexity.value,
                "target_level": decision.target_level,
                "confidence": decision.confidence,
                "reasoning": decision.reasoning,
            }

        return result

    def get_summary(self) -> Dict[str, Any]:
        """Get summary with routing statistics."""
        summary = super().get_summary()

        # Routing stats
        summary["routing_stats"] = self.router.routing_stats

        # Level performance
        summary["level_stats"] = {}
        for level, stats in self.level_stats.items():
            calls = stats["calls"]
            correct = stats["correct"]
            summary["level_stats"][level] = {
                "calls": calls,
                "correct": correct,
                "accuracy": correct / calls if calls > 0 else 0,
            }

        return summary


# =============================================================================
# L4 Pipeline (Python-controlled stages)
# =============================================================================

class L4_Pipeline(StrategyQAExperiment):
    """
    L4 Pipeline: Python controls workflow, LLM handles understanding.

    Stage 1: [LLM] Analyze question and identify key concepts
    Stage 2: [LLM] Decompose into sub-questions
    Stage 3: [LLM/RAG] Answer each sub-question
    Stage 4: [Python] Combine answers logically
    Stage 5: [Python] Determine final yes/no

    Key insight: Python orchestrates, LLM reasons.
    """

    def __init__(self, model: str = "deepseek-v3-0324", use_rag: bool = False):
        super().__init__(model)
        self.use_rag = use_rag

        # Optional: RAG retriever
        self._retriever = None

    @property
    def retriever(self):
        """Lazy load retriever."""
        if self._retriever is None and self.use_rag:
            from .trace_builder import get_paragraph_retriever
            self._retriever = get_paragraph_retriever()
            self._retriever.load()
        return self._retriever

    def _stage1_analyze(self, question: str) -> Dict[str, Any]:
        """Stage 1: Analyze question structure."""
        prompt = f"""Analyze this yes/no question and identify:
1. Main subject/topic
2. Key concepts that need to be verified
3. Type of reasoning needed (comparison, temporal, causal, etc.)

Question: {question}

Return as JSON:
{{"subject": "...", "concepts": ["...", "..."], "reasoning_type": "..."}}"""

        response = call_llm(prompt=prompt, model=self.model)

        # Parse JSON
        try:
            match = re.search(r'\{[\s\S]*\}', response)
            if match:
                import json
                return json.loads(match.group())
        except:
            pass

        return {"subject": question, "concepts": [], "reasoning_type": "unknown"}

    def _stage2_decompose(self, question: str, analysis: Dict) -> List[str]:
        """Stage 2: Decompose into sub-questions."""
        prompt = f"""Decompose this yes/no question into 2-3 simpler factual sub-questions.

Question: {question}
Analysis: {analysis}

Return as JSON array of strings:
["sub-question 1?", "sub-question 2?"]"""

        response = call_llm(prompt=prompt, model=self.model)

        # Parse JSON
        try:
            match = re.search(r'\[[\s\S]*\]', response)
            if match:
                import json
                return json.loads(match.group())
        except:
            pass

        return [question]

    def _stage3_answer_sub(self, sub_question: str) -> str:
        """Stage 3: Answer a sub-question (with optional RAG)."""
        context = ""
        if self.use_rag and self.retriever:
            results = self.retriever.retrieve(sub_question, top_k=2)
            if results:
                context = "\n".join(f"[{r['title']}]: {r['content'][:200]}" for r in results)
                context = f"\nRelevant context:\n{context}\n"

        prompt = f"""Answer this factual question briefly (1-2 sentences).
{context}
Question: {sub_question}

Answer:"""

        response = call_llm(prompt=prompt, model=self.model)
        return response.strip()

    def _stage4_combine(self, question: str, sub_answers: List[Dict]) -> bool:
        """Stage 4: Combine sub-answers to determine final answer."""
        # Format sub-answers
        sub_text = "\n".join(
            f"Q: {sa['question']}\nA: {sa['answer']}"
            for sa in sub_answers
        )

        prompt = f"""Based on these facts, answer the original yes/no question.

Original Question: {question}

Facts:
{sub_text}

Think step by step, then answer Yes or No.
Final Answer:"""

        response = call_llm(prompt=prompt, model=self.model)

        # Extract yes/no
        response_lower = response.lower()
        if "yes" in response_lower:
            return True
        elif "no" in response_lower:
            return False
        else:
            # Default based on sentiment
            return "yes" in response_lower or "true" in response_lower

    def run_instance(self, instance: StrategyQAInstance) -> StrategyQAResult:
        """Run pipeline on a single instance."""
        start_time = time.time()
        total_input = 0
        total_output = 0

        try:
            # Stage 1: Analyze
            analysis = self._stage1_analyze(instance.question)
            total_input += len(instance.question) // 4 + 100
            total_output += 50

            # Stage 2: Decompose
            sub_questions = self._stage2_decompose(instance.question, analysis)
            total_input += 100
            total_output += len(str(sub_questions)) // 4

            # Stage 3: Answer sub-questions
            sub_answers = []
            for sq in sub_questions:
                answer = self._stage3_answer_sub(sq)
                sub_answers.append({"question": sq, "answer": answer})
                total_input += len(sq) // 4 + 50
                total_output += len(answer) // 4

            # Stage 4: Combine
            predicted = self._stage4_combine(instance.question, sub_answers)
            total_input += 100
            total_output += 20

            latency_ms = (time.time() - start_time) * 1000
            cost = calculate_cost(total_input, total_output, self.model)

            return StrategyQAResult(
                qid=instance.qid,
                question=instance.question,
                predicted_answer=predicted,
                ground_truth=instance.answer,
                is_correct=predicted == instance.answer,
                latency_ms=latency_ms,
                input_tokens=total_input,
                output_tokens=total_output,
                total_tokens=total_input + total_output,
                cost_usd=cost.cost_usd,
                num_steps=4,  # 4 stages
                trace={
                    "analysis": analysis,
                    "sub_questions": sub_questions,
                    "sub_answers": sub_answers,
                    "stages": ["analyze", "decompose", "answer_subs", "combine"],
                },
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
                input_tokens=total_input,
                output_tokens=total_output,
                total_tokens=total_input + total_output,
                cost_usd=0.0,
                error=str(e),
                error_type=type(e).__name__,
            )
