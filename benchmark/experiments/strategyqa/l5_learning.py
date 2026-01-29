"""
L5 Learning Experiments for StrategyQA.

Two variants:
- L5_Improving: Learns patterns from execution experience
- L5_ICL: Pre-populates with ICL examples from training set

Key insight: Use successful traces to improve future performance.
"""

import time
import re
import json
import uuid
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from .base import StrategyQAExperiment, StrategyQAResult
from ...dataset.strategyqa_loader import StrategyQAInstance, StrategyQADataset
from ...metrics.cost import calculate_cost

from ptool_framework.llm_backend import call_llm


# =============================================================================
# Pattern Memory (Simplified version for StrategyQA)
# =============================================================================

@dataclass
class LearnedPattern:
    """A learned pattern from execution."""
    pattern_id: str
    pattern_type: str  # "positive" or "negative"
    question_type: str  # Category of question
    content: str  # The pattern content
    confidence: float = 1.0
    use_count: int = 0
    success_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "pattern_type": self.pattern_type,
            "question_type": self.question_type,
            "content": self.content,
            "confidence": self.confidence,
            "use_count": self.use_count,
            "success_count": self.success_count,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "LearnedPattern":
        return cls(**data)


class PatternMemory:
    """
    Simple pattern memory for StrategyQA.

    Stores:
    - Positive patterns: Successful reasoning traces
    - Negative patterns: Failed attempts to avoid
    """

    def __init__(self, path: Optional[Path] = None):
        self.path = path or Path(__file__).parent / ".strategyqa_patterns"
        self.patterns: Dict[str, LearnedPattern] = {}
        self._loaded = False

    def load(self):
        """Load patterns from disk."""
        if self._loaded:
            return

        patterns_file = self.path / "patterns.json"
        if patterns_file.exists():
            with open(patterns_file, 'r') as f:
                data = json.load(f)
                for pid, pdata in data.items():
                    self.patterns[pid] = LearnedPattern.from_dict(pdata)

        self._loaded = True

    def save(self):
        """Save patterns to disk."""
        self.path.mkdir(parents=True, exist_ok=True)
        patterns_file = self.path / "patterns.json"

        data = {pid: p.to_dict() for pid, p in self.patterns.items()}
        with open(patterns_file, 'w') as f:
            json.dump(data, f, indent=2)

    def store_pattern(self, pattern: LearnedPattern):
        """Store a pattern."""
        self.patterns[pattern.pattern_id] = pattern

    def get_relevant_patterns(
        self,
        question: str,
        pattern_type: Optional[str] = None,
        max_patterns: int = 3,
    ) -> List[LearnedPattern]:
        """Get patterns relevant to a question."""
        # Simple keyword matching for relevance
        question_words = set(question.lower().split())

        scored = []
        for pattern in self.patterns.values():
            if pattern_type and pattern.pattern_type != pattern_type:
                continue

            # Score by keyword overlap
            pattern_words = set(pattern.content.lower().split())
            overlap = len(question_words & pattern_words)
            score = overlap * pattern.confidence

            if score > 0:
                scored.append((score, pattern))

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)

        return [p for _, p in scored[:max_patterns]]

    def update_pattern_stats(self, pattern_id: str, success: bool):
        """Update pattern usage statistics."""
        if pattern_id in self.patterns:
            pattern = self.patterns[pattern_id]
            pattern.use_count += 1
            if success:
                pattern.success_count += 1
            # Update confidence based on success rate
            if pattern.use_count > 0:
                pattern.confidence = 0.5 + 0.5 * (pattern.success_count / pattern.use_count)

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        positive = sum(1 for p in self.patterns.values() if p.pattern_type == "positive")
        negative = sum(1 for p in self.patterns.values() if p.pattern_type == "negative")
        return {
            "total_patterns": len(self.patterns),
            "positive_patterns": positive,
            "negative_patterns": negative,
        }

    def clear(self):
        """Clear all patterns."""
        self.patterns.clear()


# =============================================================================
# Pattern Extraction
# =============================================================================

def extract_question_type(question: str) -> str:
    """Extract question type/category for pattern matching."""
    question_lower = question.lower()

    # Common question types
    if any(w in question_lower for w in ["more", "less", "greater", "fewer", "larger", "smaller"]):
        return "comparison"
    elif any(w in question_lower for w in ["before", "after", "when", "during", "while"]):
        return "temporal"
    elif any(w in question_lower for w in ["because", "cause", "result", "lead to", "effect"]):
        return "causal"
    elif any(w in question_lower for w in ["can", "could", "able", "possible"]):
        return "possibility"
    elif any(w in question_lower for w in ["same", "similar", "different", "like"]):
        return "similarity"
    else:
        return "factual"


def extract_pattern_from_trace(
    question: str,
    trace: Dict[str, Any],
    answer: bool,
    is_correct: bool,
    model: str = "deepseek-v3-0324",
) -> Optional[LearnedPattern]:
    """Extract a pattern from an execution trace."""
    question_type = extract_question_type(question)

    # Build pattern content
    if is_correct:
        # Positive pattern: capture successful reasoning
        pattern_type = "positive"

        # Extract key reasoning steps
        steps = trace.get("steps", [])
        if steps:
            reasoning = []
            for step in steps:
                if isinstance(step, dict):
                    goal = step.get("goal", "")
                    output = step.get("output", "")
                    if goal and output:
                        reasoning.append(f"- {goal}: {str(output)[:100]}")

            content = f"""Question type: {question_type}
Question: {question}
Reasoning steps:
{chr(10).join(reasoning)}
Answer: {"Yes" if answer else "No"}"""
        else:
            content = f"""Question type: {question_type}
Question: {question}
Answer: {"Yes" if answer else "No"}"""
    else:
        # Negative pattern: capture failed approach
        pattern_type = "negative"
        content = f"""Question type: {question_type}
Failed approach for: {question}
Predicted: {"Yes" if answer else "No"} (incorrect)
Lesson: Double-check reasoning for {question_type} questions."""

    return LearnedPattern(
        pattern_id=f"{pattern_type}_{uuid.uuid4().hex[:8]}",
        pattern_type=pattern_type,
        question_type=question_type,
        content=content,
        confidence=0.8 if is_correct else 0.5,
        metadata={
            "question": question,
            "answer": answer,
            "is_correct": is_correct,
        },
    )


# =============================================================================
# L5 Improving: Learns from execution
# =============================================================================

class L5_Improving(StrategyQAExperiment):
    """
    L5 Improving: Learns patterns from execution experience.

    After each run:
    - Successful: Extract positive pattern
    - Failed: Extract negative pattern (what to avoid)

    Before each run:
    - Retrieve relevant patterns
    - Augment prompt with patterns
    """

    def __init__(
        self,
        model: str = "deepseek-v3-0324",
        max_patterns: int = 3,
        learn_from_success: bool = True,
        learn_from_failure: bool = True,
    ):
        super().__init__(model)
        self.max_patterns = max_patterns
        self.learn_from_success = learn_from_success
        self.learn_from_failure = learn_from_failure

        self.pattern_memory = PatternMemory()
        self.pattern_memory.load()

        # Track learning
        self.patterns_used: List[str] = []
        self.patterns_learned = 0
        self.accuracy_history: List[bool] = []

    def _build_prompt_with_patterns(self, question: str) -> str:
        """Build prompt augmented with relevant patterns."""
        # Get relevant positive patterns
        patterns = self.pattern_memory.get_relevant_patterns(
            question,
            pattern_type="positive",
            max_patterns=self.max_patterns,
        )

        # Get negative patterns to avoid
        negative = self.pattern_memory.get_relevant_patterns(
            question,
            pattern_type="negative",
            max_patterns=1,
        )

        prompt_parts = []

        # Add positive examples
        if patterns:
            prompt_parts.append("## Similar successful examples:")
            for p in patterns:
                prompt_parts.append(f"\n{p.content}\n")
                self.patterns_used.append(p.pattern_id)

        # Add warnings from negative patterns
        if negative:
            prompt_parts.append("\n## Lessons from past mistakes:")
            for p in negative:
                prompt_parts.append(f"- {p.content}")
                self.patterns_used.append(p.pattern_id)

        # Main question
        prompt_parts.append(f"""
## Your task:
Answer this yes/no question by thinking step by step.

Question: {question}

Think through the key facts needed, then provide your reasoning and final answer.

Reasoning:""")

        return "\n".join(prompt_parts)

    def run_instance(self, instance: StrategyQAInstance) -> StrategyQAResult:
        """Run with pattern augmentation and learning."""
        start_time = time.time()
        self.patterns_used = []

        try:
            # Build augmented prompt
            prompt = self._build_prompt_with_patterns(instance.question)

            # Call LLM
            response = call_llm(prompt=prompt, model=self.model)
            latency_ms = (time.time() - start_time) * 1000

            # Extract answer
            predicted = self.extract_boolean(response)

            # Calculate metrics
            input_tokens = len(prompt) // 4
            output_tokens = len(response) // 4 if response else 0
            cost = calculate_cost(input_tokens, output_tokens, self.model)

            is_correct = predicted == instance.answer if predicted is not None else False
            self.accuracy_history.append(is_correct)

            # Learn from this execution
            if is_correct and self.learn_from_success:
                pattern = extract_pattern_from_trace(
                    question=instance.question,
                    trace={"response": response},
                    answer=predicted,
                    is_correct=True,
                    model=self.model,
                )
                if pattern:
                    self.pattern_memory.store_pattern(pattern)
                    self.patterns_learned += 1

            elif not is_correct and self.learn_from_failure:
                pattern = extract_pattern_from_trace(
                    question=instance.question,
                    trace={"response": response},
                    answer=predicted if predicted is not None else False,
                    is_correct=False,
                    model=self.model,
                )
                if pattern:
                    self.pattern_memory.store_pattern(pattern)
                    self.patterns_learned += 1

            # Update pattern stats
            for pid in self.patterns_used:
                self.pattern_memory.update_pattern_stats(pid, is_correct)

            # Save periodically
            if len(self.accuracy_history) % 10 == 0:
                self.pattern_memory.save()

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
                cost_usd=cost.cost_usd,
                raw_response=response,
                trace={
                    "patterns_used": len(self.patterns_used),
                    "patterns_total": len(self.pattern_memory.patterns),
                },
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self.accuracy_history.append(False)

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

    def get_summary(self) -> Dict[str, Any]:
        """Get summary with learning stats."""
        summary = super().get_summary()

        # Pattern stats
        summary["patterns_learned"] = self.patterns_learned
        summary["pattern_memory"] = self.pattern_memory.get_stats()

        # Learning curve
        batch_size = 10
        learning_curve = []
        for i in range(0, len(self.accuracy_history), batch_size):
            batch = self.accuracy_history[i:i + batch_size]
            if batch:
                learning_curve.append({
                    "batch": i // batch_size + 1,
                    "accuracy": sum(batch) / len(batch),
                })
        summary["learning_curve"] = learning_curve

        return summary

    def teardown(self):
        """Save patterns on teardown."""
        self.pattern_memory.save()
        super().teardown()


# =============================================================================
# L5 ICL: Pre-populated with training examples
# =============================================================================

class L5_ICL(L5_Improving):
    """
    L5 ICL: Pre-populates pattern memory with ICL examples from training.

    Starts with knowledge from training data, then continues learning.
    """

    def __init__(
        self,
        model: str = "deepseek-v3-0324",
        max_patterns: int = 3,
        icl_per_type: int = 2,
        train_data: Optional[List[StrategyQAInstance]] = None,
    ):
        super().__init__(model, max_patterns)
        self.icl_per_type = icl_per_type
        self.train_data = train_data
        self.icl_loaded = 0

        # Clear existing patterns and load ICL
        self.pattern_memory.clear()
        self._load_icl_examples()

    def _load_icl_examples(self):
        """Load ICL examples from training data."""
        # Load training data if not provided
        if self.train_data is None:
            try:
                dataset = StrategyQADataset()
                train, _ = dataset.load_with_split(val_ratio=0.2)
                self.train_data = train
            except Exception as e:
                print(f"Could not load training data: {e}")
                return

        # Group by question type
        by_type: Dict[str, List[StrategyQAInstance]] = {}
        for instance in self.train_data:
            qtype = extract_question_type(instance.question)
            if qtype not in by_type:
                by_type[qtype] = []
            by_type[qtype].append(instance)

        # Select examples per type
        for qtype, instances in by_type.items():
            # Prefer instances with decomposition
            good = [i for i in instances if i.decomposition][:self.icl_per_type]
            if len(good) < self.icl_per_type:
                good = instances[:self.icl_per_type]

            for inst in good:
                # Build ICL pattern
                decomp_text = ""
                if inst.decomposition:
                    decomp_text = "\nSub-questions:\n" + "\n".join(f"- {q}" for q in inst.decomposition)

                facts_text = ""
                if inst.facts:
                    facts_text = "\nKey facts:\n" + "\n".join(f"- {f}" for f in inst.facts)

                content = f"""Question type: {qtype}
Question: {inst.question}
{decomp_text}
{facts_text}
Answer: {"Yes" if inst.answer else "No"}"""

                pattern = LearnedPattern(
                    pattern_id=f"icl_{qtype}_{uuid.uuid4().hex[:8]}",
                    pattern_type="positive",
                    question_type=qtype,
                    content=content,
                    confidence=1.0,  # High confidence for training examples
                    metadata={
                        "from_training": True,
                        "qid": inst.qid,
                    },
                )

                self.pattern_memory.store_pattern(pattern)
                self.icl_loaded += 1

        print(f"Loaded {self.icl_loaded} ICL examples from {len(by_type)} question types")

    def get_summary(self) -> Dict[str, Any]:
        """Get summary with ICL stats."""
        summary = super().get_summary()
        summary["icl_examples_loaded"] = self.icl_loaded
        return summary
