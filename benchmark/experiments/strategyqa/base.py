"""
Base experiment class and result types for StrategyQA.

All StrategyQA experiments inherit from StrategyQAExperiment.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime
import re

from ...dataset.strategyqa_loader import StrategyQAInstance


@dataclass
class StrategyQAResult:
    """Result from running an experiment on a single StrategyQA instance."""

    # Instance identification
    qid: str
    question: str

    # Answer
    predicted_answer: Optional[bool]
    ground_truth: bool

    # Correctness (exact match for boolean)
    is_correct: bool

    # Performance metrics
    latency_ms: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float

    # Framework-specific metrics
    num_steps: int = 0              # ReAct/decomposition steps taken
    decomposition_used: bool = False
    facts_used: bool = False

    # Error handling
    error: Optional[str] = None
    error_type: Optional[str] = None

    # Raw outputs
    raw_response: Optional[str] = None
    trace: Optional[Dict[str, Any]] = None

    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "qid": self.qid,
            "question": self.question,
            "predicted_answer": self.predicted_answer,
            "ground_truth": self.ground_truth,
            "is_correct": self.is_correct,
            "latency_ms": self.latency_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
            "num_steps": self.num_steps,
            "decomposition_used": self.decomposition_used,
            "facts_used": self.facts_used,
            "error": self.error,
            "error_type": self.error_type,
            "raw_response": self.raw_response,
            "timestamp": self.timestamp,
        }

    @property
    def success(self) -> bool:
        """Whether the prediction was correct."""
        return self.is_correct and self.error is None


class StrategyQAExperiment(ABC):
    """
    Abstract base class for all StrategyQA experiments.

    Subclasses must implement run_instance() to process a single instance.
    """

    def __init__(self, model: str = "deepseek-v3-0324"):
        """
        Initialize experiment.

        Args:
            model: LLM model to use
        """
        self.model = model
        self.results: List[StrategyQAResult] = []
        self._setup_complete = False

    def setup(self) -> None:
        """Perform one-time setup. Override in subclasses if needed."""
        self._setup_complete = True

    def teardown(self) -> None:
        """Clean up resources. Override in subclasses if needed."""
        pass

    @abstractmethod
    def run_instance(self, instance: StrategyQAInstance) -> StrategyQAResult:
        """
        Run experiment on a single instance.

        Args:
            instance: StrategyQA problem instance

        Returns:
            StrategyQAResult with prediction and metrics
        """
        pass

    def run_batch(
        self,
        instances: List[StrategyQAInstance],
        progress_callback: Optional[callable] = None,
    ) -> List[StrategyQAResult]:
        """
        Run experiment on a batch of instances.

        Args:
            instances: List of StrategyQA instances
            progress_callback: Optional callback(current, total) for progress

        Returns:
            List of StrategyQAResult objects
        """
        if not self._setup_complete:
            self.setup()

        self.results = []
        total = len(instances)

        for i, instance in enumerate(instances):
            try:
                result = self.run_instance(instance)
            except Exception as e:
                # Create error result
                result = StrategyQAResult(
                    qid=instance.qid,
                    question=instance.question,
                    predicted_answer=None,
                    ground_truth=instance.answer,
                    is_correct=False,
                    latency_ms=0.0,
                    input_tokens=0,
                    output_tokens=0,
                    total_tokens=0,
                    cost_usd=0.0,
                    error=str(e),
                    error_type=type(e).__name__,
                )

            self.results.append(result)

            if progress_callback:
                progress_callback(i + 1, total)

        return self.results

    def extract_boolean(self, text: str) -> Optional[bool]:
        """
        Extract boolean answer from LLM response.

        Args:
            text: Raw LLM response text

        Returns:
            Extracted boolean or None
        """
        if not text:
            return None

        text_lower = text.strip().lower()

        # Direct patterns
        if text_lower in ['true', 'yes', 'correct', 'right']:
            return True
        if text_lower in ['false', 'no', 'incorrect', 'wrong']:
            return False

        # Answer patterns
        true_patterns = [
            r'\b(?:answer|result)(?:\s*(?:is|=|:))?\s*(?:true|yes)\b',
            r'\byes[,.]?\s*(?:the answer is|that is correct)',
            r'\btrue[,.]?\s*(?:the answer is)',
            r'\bthe answer is\s*(?:true|yes)\b',
        ]

        false_patterns = [
            r'\b(?:answer|result)(?:\s*(?:is|=|:))?\s*(?:false|no)\b',
            r'\bno[,.]?\s*(?:the answer is|that is incorrect)',
            r'\bfalse[,.]?\s*(?:the answer is)',
            r'\bthe answer is\s*(?:false|no)\b',
        ]

        for pattern in true_patterns:
            if re.search(pattern, text_lower):
                return True

        for pattern in false_patterns:
            if re.search(pattern, text_lower):
                return False

        # Look for true/false or yes/no at the end
        last_word_match = re.search(r'\b(true|false|yes|no)\s*[.!]?\s*$', text_lower)
        if last_word_match:
            word = last_word_match.group(1)
            return word in ['true', 'yes']

        # Look for true/false or yes/no anywhere (last occurrence)
        all_matches = re.findall(r'\b(true|false|yes|no)\b', text_lower)
        if all_matches:
            last = all_matches[-1]
            return last in ['true', 'yes']

        return None

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for completed results.

        Returns:
            Dictionary with accuracy, cost, latency stats
        """
        if not self.results:
            return {}

        total = len(self.results)
        correct = sum(1 for r in self.results if r.is_correct)
        errors = sum(1 for r in self.results if r.error is not None)

        # Count predictions
        pred_true = sum(1 for r in self.results if r.predicted_answer is True)
        pred_false = sum(1 for r in self.results if r.predicted_answer is False)
        pred_none = sum(1 for r in self.results if r.predicted_answer is None)

        latencies = [r.latency_ms for r in self.results if r.error is None]
        costs = [r.cost_usd for r in self.results]
        tokens = [r.total_tokens for r in self.results]

        return {
            "experiment": self.__class__.__name__,
            "model": self.model,
            "total_instances": total,
            "correct": correct,
            "errors": errors,
            "accuracy": correct / total if total > 0 else 0,
            "error_rate": errors / total if total > 0 else 0,
            "predictions": {
                "true": pred_true,
                "false": pred_false,
                "none": pred_none,
            },
            "total_cost_usd": sum(costs),
            "avg_cost_usd": sum(costs) / total if total > 0 else 0,
            "total_tokens": sum(tokens),
            "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
        }
